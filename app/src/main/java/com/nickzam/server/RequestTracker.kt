// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicLong

/**
 * Process-wide tracker for LLM requests. Exposes StateFlows the UI can observe.
 *
 * Lifecycle:
 *   enqueue() → (waiting in queue) → markStarted() → recordChunk()* → markCompleted()
 *
 * Concurrency model:
 *   - Compound mutations (move from queue → current, current → history) are serialized
 *     by [mutex] so the four StateFlows stay consistent with each other.
 *   - Per-chunk updates use `StateFlow.update` (lock-free CAS) — they're on the hot
 *     streaming path, so we don't want to contend with the slower compound writes.
 *
 * Memory:
 *   - History is capped at [HISTORY_CAP]; oldest entries are dropped.
 *   - Stats are cumulative and survive service restarts (in-memory only).
 */
object RequestTracker {

    enum class State { QUEUED, RUNNING, COMPLETED, ERRORED, CANCELLED }

    data class Entry(
        val id: String,
        val model: String,
        val stream: Boolean,
        val messageCount: Int,
        val promptChars: Int,
        val state: State,
        val enqueuedAt: Long,
        val startedAt: Long? = null,
        val completedAt: Long? = null,
        /**
         * Wall-clock ms at which the FIRST output chunk arrived. The span
         * startedAt→firstChunkAtMs is the prefill (prompt-eval) window, which
         * also absorbs a cold engine build + full-history prefill; the span
         * firstChunkAtMs→completedAt is the token-by-token decode that people
         * mean when they say "tok/s". Null until the first chunk lands.
         */
        val firstChunkAtMs: Long? = null,
        val chunkCount: Int = 0,
        val outputChars: Int = 0,
        val error: String? = null,
        /**
         * Client identity for fairness, rate limiting, and per-app metrics.
         * Derived from the request's `User-Agent` header at enqueue time;
         * unknown / no-UA requests fall back to `"anonymous"`.
         */
        val client: String = "anonymous",
    ) {
        /** Time spent in the queue waiting for the inference mutex. */
        fun queueWaitMs(now: Long = System.currentTimeMillis()): Long =
            (startedAt ?: now) - enqueuedAt

        /** Time spent in inference (live for RUNNING, frozen once completed). */
        fun inferenceMs(now: Long = System.currentTimeMillis()): Long {
            val start = startedAt ?: return 0L
            val end = completedAt ?: now
            return (end - start).coerceAtLeast(0L)
        }

        /**
         * End-to-end speed including cold-engine build + prefill. Matches
         * what a client sees between "sent request" and "got all tokens".
         */
        val tokensPerSec: Float
            get() {
                val ms = inferenceMs()
                return if (ms > 0 && outputChars > 0)
                    (outputChars / 4f) * 1000f / ms else 0f
            }

        /**
         * Steady-state decode window (first token → end), or null when it
         * isn't measurable. Non-streaming calls record one chunk at
         * completion, so a near-zero window means "no streaming samples" —
         * treated as unmeasurable rather than absurdly fast.
         */
        val decodeWindowMs: Long?
            get() {
                val first = firstChunkAtMs ?: return null
                val end = completedAt ?: System.currentTimeMillis()
                val ms = end - first
                return if (ms >= 100 && outputTokensEst > 1) ms else null
            }

        /** Steady-state decode speed: measured from the FIRST token onward. */
        val decodeTokensPerSec: Float
            get() {
                val ms = decodeWindowMs ?: return 0f
                return (outputTokensEst - 1).toFloat() * 1000f / ms
            }

        /** Prefill (+ cold-engine build, if any): request start → first token. */
        val prefillMs: Long?
            get() {
                val s = startedAt ?: return null
                return firstChunkAtMs?.minus(s)
            }

        /**
         * Prompt-eval speed: prompt tokens (estimated ~4 chars/token) over the
         * prefill window. Like decode tok/s it needs a measurable window; the
         * first-token latency also absorbs any cold-engine build, so the
         * number is a floor for prefill on cold starts, exact when warm.
         */
        val prefillTokensPerSec: Float
            get() {
                val ms = prefillMs?.takeIf { it >= 100 } ?: return 0f
                val tok = promptChars / 4f
                return if (tok > 0) tok * 1000f / ms else 0f
            }

        /** Chunks per second, computed only when we have a completed duration. */
        val chunksPerSec: Float
            get() {
                val ms = inferenceMs()
                return if (ms > 0 && chunkCount > 0) chunkCount.toFloat() * 1000f / ms else 0f
            }

        /** Estimated tokens (LiteRT-LM has no token counter; ~4 chars/token). */
        val outputTokensEst: Long get() = outputChars.toLong() / 4L
    }

    data class Stats(
        val totalRequests: Long = 0,
        val totalCompleted: Long = 0,
        val totalErrors: Long = 0,
        val totalCancelled: Long = 0,
        val totalChunks: Long = 0,
        val totalInferenceMs: Long = 0,
        val totalOutputChars: Long = 0,
        /** Sum of decode-window ms (first token → last) over completed requests. */
        val totalDecodeMs: Long = 0,
        /** Sum of tokens generated after the first token, same set. */
        val totalDecodeTokensEst: Long = 0,
        /** Sum of prefill-window ms (start → first token) over completed requests. */
        val totalPrefillMs: Long = 0,
        /** Sum of prompt tokens estimated over those same prefill windows. */
        val totalPrefillTokensEst: Long = 0,
    ) {
        val avgLatencyMs: Long
            get() = if (totalCompleted > 0) totalInferenceMs / totalCompleted else 0L

        val avgChunksPerSec: Float
            get() = if (totalInferenceMs > 0) totalChunks.toFloat() * 1000f / totalInferenceMs else 0f

        val errorRate: Float
            get() = if (totalRequests > 0) (totalErrors + totalCancelled).toFloat() / totalRequests else 0f

        // LiteRT-LM 0.12.0 exposes no token/usage API, so throughput is
        // estimated from output characters at ~4 chars/token (BPE English
        // average), which lands on the ~14 tok/s the GPU is measured at.
        val totalTokensEst: Long
            get() = totalOutputChars / 4

        val avgTokensPerSec: Float
            get() = if (totalInferenceMs > 0)
                (totalOutputChars / 4f) * 1000f / totalInferenceMs else 0f

        /**
         * Steady-state decode speed across completed requests, measured from
         * each request's first token onward. This excludes cold-engine build
         * and prompt prefill, so it's the number to compare against published
         * GPU tok/s. Zero until at least one multi-token request completes.
         */
        val avgDecodeTokensPerSec: Float
            get() = if (totalDecodeMs > 0 && totalDecodeTokensEst > 0)
                totalDecodeTokensEst.toFloat() * 1000f / totalDecodeMs else 0f

        /**
         * Prompt-eval speed across completed requests: estimated prompt tokens
         * over each request's prefill window (start → first token). Because a
         * cold-engine build can land in that window, this is a floor for true
         * prefill speed; on warm turns (engine + KV cache reused) it is exact.
         */
        val avgPrefillTokensPerSec: Float
            get() = if (totalPrefillMs > 0 && totalPrefillTokensEst > 0)
                totalPrefillTokensEst.toFloat() * 1000f / totalPrefillMs else 0f
    }

    private const val HISTORY_CAP = 50

    private val mutex = Mutex()
    private val idCounter = AtomicLong(0)

    private val _queue = MutableStateFlow<List<Entry>>(emptyList())
    val queue = _queue.asStateFlow()

    private val _current = MutableStateFlow<Entry?>(null)
    val current = _current.asStateFlow()

    private val _history = MutableStateFlow<List<Entry>>(emptyList())
    val history = _history.asStateFlow()

    private val _stats = MutableStateFlow(Stats())
    val stats = _stats.asStateFlow()

    /** Append a new request to the queue and bump totalRequests. */
    suspend fun enqueue(model: String, stream: Boolean, messageCount: Int, promptChars: Int): Entry {
        val entry = Entry(
            id = idCounter.incrementAndGet().toString(),
            model = model,
            stream = stream,
            messageCount = messageCount,
            promptChars = promptChars,
            state = State.QUEUED,
            enqueuedAt = System.currentTimeMillis()
        )
        mutex.withLock {
            _queue.update { it + entry }
            _stats.update { it.copy(totalRequests = it.totalRequests + 1) }
        }
        return entry
    }

    /**
     * Atomically check queue capacity and enqueue. Returns null if the queue
     * is full ([maxDepth] reached) — the caller should respond 429.
     */
    suspend fun tryEnqueue(
        model: String,
        stream: Boolean,
        messageCount: Int,
        promptChars: Int,
        maxDepth: Int,
        client: String = "anonymous",
    ): Entry? = mutex.withLock {
        if (_queue.value.size >= maxDepth) return@withLock null
        val entry = Entry(
            id = idCounter.incrementAndGet().toString(),
            model = model,
            stream = stream,
            messageCount = messageCount,
            promptChars = promptChars,
            state = State.QUEUED,
            enqueuedAt = System.currentTimeMillis(),
            client = client,
        )
        _queue.update { it + entry }
        _stats.update { it.copy(totalRequests = it.totalRequests + 1) }
        entry
    }

    /**
     * Promote [id] from queue to current. If it's no longer in the queue
     * (already completed or cancelled before reaching the inference mutex)
     * this is a no-op.
     */
    suspend fun markStarted(id: String) {
        mutex.withLock {
            val q = _queue.value
            val idx = q.indexOfFirst { it.id == id }
            if (idx < 0) return@withLock
            val entry = q[idx].copy(state = State.RUNNING, startedAt = System.currentTimeMillis())
            _queue.update { it.toMutableList().also { l -> l.removeAt(idx) } }
            _current.update { entry }
        }
    }

    /**
     * Hot-path: called once per streamed chunk. Lock-free; only updates [_current].
     * For streaming responses prefer [accumulatorFor], which batches multiple
     * chunks into one [Entry.copy]. This non-batching variant is retained for
     * the non-streaming code paths and tests that record a single final chunk.
     */
    fun recordChunk(id: String, chunkText: String) {
        recordChunkBatch(id, chunks = 1, chars = chunkText.length)
    }

    /**
     * Bulk version of [recordChunk]. Adds [chunks] events totalling [chars]
     * output characters to the current request. One [Entry.copy] regardless
     * of batch size.
     */
    fun recordChunkBatch(id: String, chunks: Int, chars: Int) {
        if (chunks <= 0 && chars <= 0) return
        _current.update { cur ->
            if (cur != null && cur.id == id) {
                cur.copy(
                    chunkCount = cur.chunkCount + chunks,
                    outputChars = cur.outputChars + chars,
                    firstChunkAtMs = cur.firstChunkAtMs ?: System.currentTimeMillis(),
                )
            } else cur
        }
    }

    /**
     * Build a batching accumulator scoped to one streaming request. The route
     * calls [ChunkAccumulator.add] once per token (cheap — plain counters)
     * and [ChunkAccumulator.flush] when it's time to push a fresh snapshot.
     */
    fun accumulatorFor(
        id: String,
        flushEveryChunks: Int = 16,
        flushEveryMs: Long = 100L,
    ): ChunkAccumulator = ChunkAccumulator(id, flushEveryChunks, flushEveryMs)

    class ChunkAccumulator internal constructor(
        private val id: String,
        private val flushEveryChunks: Int,
        private val flushEveryMs: Long,
    ) {
        private var pendingChunks: Int = 0
        private var pendingChars: Int = 0
        private var lastFlushNanos: Long = System.nanoTime()

        fun add(deltaChars: Int) {
            pendingChunks += 1
            pendingChars += deltaChars
            if (pendingChunks >= flushEveryChunks) { flush(); return }
            val elapsedMs = (System.nanoTime() - lastFlushNanos) / 1_000_000L
            if (elapsedMs >= flushEveryMs) flush()
        }

        fun flush() {
            if (pendingChunks == 0 && pendingChars == 0) return
            recordChunkBatch(id, pendingChunks, pendingChars)
            pendingChunks = 0
            pendingChars = 0
            lastFlushNanos = System.nanoTime()
        }
    }

    /**
     * Finalize the request: move it from current to history, update stats.
     * Pass [error] for failures, [cancelled]=true for client/server cancellations.
     */
    suspend fun markCompleted(id: String, error: String? = null, cancelled: Boolean = false) {
        mutex.withLock {
            // If the request was queued and never started (rare race), remove it from the queue.
            val queued = _queue.value
            val qIdx = queued.indexOfFirst { it.id == id }
            if (qIdx >= 0) {
                val finalState = when {
                    cancelled -> State.CANCELLED
                    error != null -> State.ERRORED
                    else -> State.COMPLETED
                }
                val finalized = queued[qIdx].copy(
                    state = finalState,
                    completedAt = System.currentTimeMillis(),
                    error = error
                )
                _queue.update { it.toMutableList().also { l -> l.removeAt(qIdx) } }
                pushHistory(finalized)
                bumpStats(finalized)
                return@withLock
            }

            val cur = _current.value
            if (cur == null || cur.id != id) return@withLock

            val finalState = when {
                cancelled -> State.CANCELLED
                error != null -> State.ERRORED
                else -> State.COMPLETED
            }
            val finalized = cur.copy(
                state = finalState,
                completedAt = System.currentTimeMillis(),
                error = error
            )
            _current.update { null }
            pushHistory(finalized)
            bumpStats(finalized)
        }
    }

    /** Caller must hold [mutex]. */
    private fun pushHistory(entry: Entry) {
        _history.update { existing ->
            val next = ArrayList<Entry>(minOf(existing.size + 1, HISTORY_CAP))
            next.add(entry)
            for (i in 0 until minOf(existing.size, HISTORY_CAP - 1)) next.add(existing[i])
            next
        }
    }

    /** Caller must hold [mutex]. */
    private fun bumpStats(entry: Entry) {
        val infMs = entry.inferenceMs()
        val decodeMs = if (entry.state == State.COMPLETED) entry.decodeWindowMs else null
        val prefillMs = if (entry.state == State.COMPLETED) entry.prefillMs?.takeIf { it >= 100 } else null
        _stats.update { s ->
            s.copy(
                totalCompleted = s.totalCompleted + if (entry.state == State.COMPLETED) 1 else 0,
                totalErrors = s.totalErrors + if (entry.state == State.ERRORED) 1 else 0,
                totalCancelled = s.totalCancelled + if (entry.state == State.CANCELLED) 1 else 0,
                totalChunks = s.totalChunks + entry.chunkCount,
                totalOutputChars = s.totalOutputChars + entry.outputChars,
                totalInferenceMs = s.totalInferenceMs + if (entry.state == State.COMPLETED) infMs else 0L,
                totalDecodeMs = s.totalDecodeMs + (decodeMs ?: 0L),
                totalDecodeTokensEst = s.totalDecodeTokensEst +
                    (if (decodeMs != null) entry.outputTokensEst - 1 else 0L),
                totalPrefillMs = s.totalPrefillMs + (prefillMs ?: 0L),
                totalPrefillTokensEst = s.totalPrefillTokensEst +
                    (if (prefillMs != null) entry.promptChars / 4L else 0L),
            )
        }
    }

    /** Wipe history; stats and any in-flight work are untouched. */
    suspend fun clearHistory() {
        mutex.withLock { _history.update { emptyList() } }
    }

    /** Reset cumulative counters. Useful after a config change. */
    suspend fun resetStats() {
        mutex.withLock { _stats.update { Stats() } }
    }

    /** Called when the service is torn down — any pending work is moot. */
    suspend fun resetAll() {
        mutex.withLock {
            _queue.update { emptyList() }
            _current.update { null }
        }
    }
}
