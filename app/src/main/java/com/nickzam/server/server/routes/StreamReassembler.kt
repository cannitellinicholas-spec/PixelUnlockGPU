package com.nickzam.server.server.routes

/**
 * Reconstructs net text from a stream of LiteRT-LM emissions.
 *
 * A producer is either:
 *  - cumulative: each emission repeats the whole text so far
 *    ("The", "The quick", "The quick brown", …)
 *  - incremental: each emission is only the new token
 *    ("The", " quick", " brown", …)
 *
 * One rule handles both against the already-emitted buffer, with no mode
 * to calibrate and no way to silently drop a chunk:
 *  - emission extends what we emitted  -> emit the extension
 *  - emission equals what we emitted   -> emit nothing (pure repeat)
 *  - anything else                     -> emit it raw (new token / partial)
 *
 * Comparing to the whole emitted buffer (not the previous message) is what
 * makes a cumulative stream safe: a repeated cumulative prefix collapses to
 * "" instead of being appended again, which is how the earlier per-message
 * "calibrate the mode" version could duplicate text.
 *
 * Verified 2026-10-01: LiteRT-LM 0.12.0 `sendMessageAsync` is incremental;
 * the old length-sliced delta chopped spans (the streamed gibberish).
 */
object StreamReassembler {
    fun next(emitted: String, emission: String): String = when {
        emission.isEmpty() -> ""
        emission.length > emitted.length && emission.startsWith(emitted) ->
            emission.substring(emitted.length)
        emission == emitted -> ""
        else -> emission
    }
}

/**
 * Client-side stop-sequence emulation for the streaming path (issue #3).
 *
 * The pinned LiteRT-LM engine exposes no per-request stop-sequence API —
 * generation runs to EOS within the window — so stop semantics are
 * emulated here, on the server, over the text the route has already
 * reassembled. The match is a post-hoc text match on the *joined* text
 * (the full accumulated string, handed in on every call), never on
 * individual emissions, so a stop string split across token boundaries
 * still matches: the pieces only have to meet in the joined text.
 *
 * Feed each new joined text to [onJoined]; it returns only the new text
 * that is safe to forward to the client now:
 *  - If a full stop string occurs in the joined text, [stopped] flips to
 *    true and the text up to (never past) the first occurrence is
 *    returned; every later call returns "".
 *  - If no full stop occurs, the longest suffix of the joined text that
 *    is a *proper prefix* of some stop string is held back — it might
 *    complete into a stop when the next chunk lands — and the rest is
 *    forwarded. That holdback is the emulation's tiny latency cost: at
 *    most (longest stop − 1) trailing characters wait for one more chunk.
 *  - [finish] flushes any held-back suffix when the stream ends without
 *    a stop, so no-match text is delivered whole, just possibly a chunk
 *    late at the tail.
 *
 * JVM-pure (no Android deps) so the streaming route's exact behaviour is
 * unit-testable off-device.
 */
class StopFilter(stopSequences: List<String>) {

    /** Requested sequences with empties removed (an empty stop matches everywhere). */
    private val stops: List<String> = stopSequences.filter { it.isNotEmpty() }

    /** Chars of the joined text already cleared for the client. */
    var forwarded: Int = 0
        private set

    /** True once a stop string has matched; nothing further is ever forwarded. */
    var stopped: Boolean = false
        private set

    /**
     * Returns the new text safe to forward given the full joined text so
     * far. See the class doc for the holdback and stop rules.
     */
    fun onJoined(joined: String): String {
        if (stopped) return ""
        val start = minOf(forwarded, joined.length)

        val firstStop = firstStopIndex(joined)
        if (firstStop >= 0) {
            stopped = true
            if (firstStop <= start) return ""
            val out = joined.substring(start, firstStop)
            forwarded = firstStop
            return out
        }

        val safeEnd = joined.length - holdbackLength(joined)
        if (safeEnd <= start) return ""
        val out = joined.substring(start, safeEnd)
        forwarded = safeEnd
        return out
    }

    /**
     * Flushes any held-back suffix at end of stream. Only meaningful when
     * the stream ended unstopped; after a stop it returns "".
     */
    fun finish(joined: String): String {
        if (stopped) return ""
        if (forwarded >= joined.length) return ""
        val out = joined.substring(forwarded)
        forwarded = joined.length
        return out
    }

    /** Index of the earliest occurrence of any stop string, or -1. */
    private fun firstStopIndex(joined: String): Int {
        var best = -1
        for (s in stops) {
            val i = joined.indexOf(s)
            if (i >= 0 && (best < 0 || i < best)) best = i
        }
        return best
    }

    /**
     * Length of the longest suffix of [joined] that is a proper prefix of
     * some stop string — the part that might still grow into a stop.
     */
    private fun holdbackLength(joined: String): Int {
        var best = 0
        for (s in stops) {
            val maxK = minOf(s.length - 1, joined.length)
            for (k in maxK downTo 1) {
                if (joined.endsWith(s.substring(0, k))) {
                    if (k > best) best = k
                    break
                }
            }
        }
        return best
    }

    companion object {
        /**
         * [text] truncated at the first occurrence of any stop string, or
         * unchanged when none occurs. Used for the non-streaming path,
         * where the whole response text exists at once.
         */
        fun truncate(text: String, stops: List<String>): String {
            var best = -1
            for (s in stops) {
                if (s.isEmpty()) continue
                val i = text.indexOf(s)
                if (i >= 0 && (best < 0 || i < best)) best = i
            }
            return if (best >= 0) text.substring(0, best) else text
        }
    }
}
