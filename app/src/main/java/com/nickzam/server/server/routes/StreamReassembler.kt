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
