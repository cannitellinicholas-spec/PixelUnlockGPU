package com.nickzam.server

import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class RequestTrackerTest {

    @Test fun `tryEnqueue enforces max depth`() = runTest {
        RequestTracker.resetAll()
        RequestTracker.clearHistory()
        val first = RequestTracker.tryEnqueue("m", false, 1, 10, maxDepth = 1)
        assertNotNull(first)
        assertNull(RequestTracker.tryEnqueue("m", false, 1, 10, maxDepth = 1))
        RequestTracker.markCompleted(first!!.id)
        val after = RequestTracker.tryEnqueue("m", false, 1, 10, maxDepth = 1)
        assertNotNull(after)
        RequestTracker.markCompleted(after!!.id)
    }

    @Test fun `lifecycle moves queue to current to history`() = runTest {
        RequestTracker.resetAll()
        RequestTracker.clearHistory()
        val e = RequestTracker.enqueue("m", true, 2, 20)
        assertEquals(1, RequestTracker.queue.value.size)
        RequestTracker.markStarted(e.id)
        assertEquals(0, RequestTracker.queue.value.size)
        assertEquals(e.id, RequestTracker.current.value!!.id)
        RequestTracker.recordChunk(e.id, "hello")
        RequestTracker.markCompleted(e.id)
        assertNull(RequestTracker.current.value)
        assertEquals(1, RequestTracker.history.value.size)
        val done = RequestTracker.history.value.single()
        assertEquals(5, done.outputChars)
        assertEquals(RequestTracker.State.COMPLETED, done.state)
    }

    @Test fun `errors and cancellations are recorded`() = runTest {
        RequestTracker.resetAll()
        RequestTracker.clearHistory()
        RequestTracker.resetStats()
        val e1 = RequestTracker.enqueue("m", false, 1, 5)
        RequestTracker.markCompleted(e1.id, error = "boom")
        val e2 = RequestTracker.enqueue("m", false, 1, 5)
        RequestTracker.markCompleted(e2.id, cancelled = true)
        val stats = RequestTracker.stats.value
        assertEquals(1L, stats.totalErrors)
        assertEquals(1L, stats.totalCancelled)
    }

    // --- prefill / decode split (regression for the "3.8 tok/s" confusion) ---
    // The blended rate folds a ~8s cold-engine build + prefill into a short
    // generation and reads absurdly slow. These pin the honest split.

    private fun streamingEntry(
        startedAt: Long, firstChunkAtMs: Long?, completedAt: Long, outputChars: Int,
    ) = com.nickzam.server.RequestTracker.Entry(
        id = "t", model = "m", stream = true, messageCount = 1, promptChars = 10,
        state = com.nickzam.server.RequestTracker.State.COMPLETED,
        enqueuedAt = startedAt, startedAt = startedAt, completedAt = completedAt,
        firstChunkAtMs = firstChunkAtMs, chunkCount = outputChars, outputChars = outputChars,
    )

    @Test fun `decode rate excludes prefill from the speed`() {
        // 8s cold build+prefill, then 100 tokens written over the next 10s.
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 8_000, completedAt = 18_000, outputChars = 400)
        assertEquals(8_000L, e.prefillMs)
        // 100 tokens est, minus the first = 99 over 10s = 9.9 tok/s decode.
        assertEquals(9.9f, e.decodeTokensPerSec, 0.01f)
        // Blended end-to-end is far lower (100 tok / 18s), which is the number
        // that looked "broken" — assert it's strictly below decode.
        val blended = e.tokensPerSec
        assert(blended < e.decodeTokensPerSec) { "blended $blended should trail decode ${e.decodeTokensPerSec}" }
    }

    @Test fun `non-streaming single-chunk completion has no measurable decode window`() {
        // Non-streaming records one chunk at completion → first≈end. Must read
        // unmeasurable (0), never a fake multi-thousand tok/s.
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 18_000, completedAt = 18_000, outputChars = 400)
        assertNull(e.decodeWindowMs)
        assertEquals(0f, e.decodeTokensPerSec, 0f)
    }

    @Test fun `no first-chunk stamp means no decode rate`() {
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = null, completedAt = 5_000, outputChars = 400)
        assertNull(e.prefillMs)
        assertNull(e.decodeWindowMs)
        assertEquals(0f, e.decodeTokensPerSec, 0f)
    }

    @Test fun `single-token output has no decode window`() {
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 8_000, completedAt = 18_000, outputChars = 4)
        assertNull(e.decodeWindowMs)
    }

    // --- prefill tok/s counter (UI side-by-side with decode) ---

    @Test fun `prefill rate is prompt tokens over the prefill window`() {
        // promptChars=4000 → 1000 est tokens; first chunk at 8s → 125 tok/s.
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 8_000, completedAt = 18_000, outputChars = 400)
            .copy(promptChars = 4_000)
        assertEquals(125f, e.prefillTokensPerSec, 0.01f)
    }

    @Test fun `tiny prefill windows read unmeasurable`() {
        // Sub-100ms window (non-streaming completion stamps first≈start) must
        // return 0 rather than an absurd rate.
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 20, completedAt = 18_000, outputChars = 400)
            .copy(promptChars = 4_000)
        assertEquals(0f, e.prefillTokensPerSec, 0f)
    }

    // --- engine-build split (regression for "prefill jumps to 111 tok/s
    // while nothing is generating yet", 2026-10-02) ---
    // A 12 s cold engine build used to land inside the prefill window, so
    // prefill tok/s averaged 1-ish on cold starts and the UI showed stale
    // averages as if live. Phase + engineReadyAtMs make both honest.

    private fun runningEntry(
        startedAt: Long, engineReadyAtMs: Long?, firstChunkAtMs: Long?, outputChars: Int = 0,
    ) = streamingEntry(
        startedAt = startedAt, firstChunkAtMs = firstChunkAtMs,
        completedAt = startedAt, outputChars = outputChars,
    ).copy(
        state = com.nickzam.server.RequestTracker.State.RUNNING,
        completedAt = null, engineReadyAtMs = engineReadyAtMs,
    )

    @Test fun `prefill window excludes the engine build`() {
        // 12 s cold build (0→12_000), then 2_000 est prompt tokens over 16s.
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 28_000, completedAt = 30_000, outputChars = 400)
            .copy(promptChars = 8_000, engineReadyAtMs = 12_000)
        assertEquals(12_000L, e.engineStartMs)
        assertEquals(16_000L, e.prefillMs)
        assertEquals(125f, e.prefillTokensPerSec, 0.01f)
    }

    @Test fun `entries without an engine stamp fall back to the old window`() {
        // Back-compat: sessions that never called markEngineReady keep the
        // startedAt→firstChunk semantics instead of going null.
        val e = streamingEntry(startedAt = 0, firstChunkAtMs = 8_000, completedAt = 18_000, outputChars = 400)
        assertNull(e.engineStartMs)
        assertEquals(8_000L, e.prefillMs)
    }

    @Test fun `phase reads queued engine-start prefill decode`() {
        val q = streamingEntry(startedAt = 0, firstChunkAtMs = null, completedAt = 0, outputChars = 0)
            .copy(state = com.nickzam.server.RequestTracker.State.QUEUED, completedAt = null)
        assertEquals(com.nickzam.server.RequestTracker.Phase.QUEUED, q.phase)
        assertEquals(com.nickzam.server.RequestTracker.Phase.ENGINE_START, runningEntry(startedAt = 0, engineReadyAtMs = null, firstChunkAtMs = null).phase)
        assertEquals(com.nickzam.server.RequestTracker.Phase.PREFILL, runningEntry(startedAt = 0, engineReadyAtMs = 1_000, firstChunkAtMs = null).phase)
        assertEquals(com.nickzam.server.RequestTracker.Phase.DECODE, runningEntry(startedAt = 0, engineReadyAtMs = 1_000, firstChunkAtMs = 5_000, outputChars = 40).phase)
    }

    @Test fun `live prefill rate is zero until the engine is ready`() {
        // Queued or still building: no fake live number.
        assertEquals(0f, runningEntry(startedAt = 0, engineReadyAtMs = null, firstChunkAtMs = null).livePrefillTokensPerSec, 0f)
        // Engine ready, first token at 12 s with an 8k-char prompt: 50 tok/s.
        val e = runningEntry(startedAt = 0, engineReadyAtMs = 4_000, firstChunkAtMs = 12_000)
            .copy(promptChars = 1_600)
        assertEquals(50f, e.livePrefillTokensPerSec, 0.01f)
    }

    @Test fun `markEngineReady stamps once and only the current request`() = runTest {
        RequestTracker.resetAll()
        RequestTracker.clearHistory()
        val e = RequestTracker.enqueue("m", true, 1, 100)
        RequestTracker.markStarted(e.id)
        RequestTracker.markEngineReady(e.id)
        val stamped = RequestTracker.current.value!!
        assertNotNull(stamped.engineReadyAtMs)
        // Idempotent: a second call never re-stamps the same request.
        kotlinx.coroutines.delay(1)
        RequestTracker.markEngineReady(e.id)
        assertEquals(stamped.engineReadyAtMs, RequestTracker.current.value!!.engineReadyAtMs)
        // Unknown id is a no-op.
        RequestTracker.markEngineReady("nope")
        assertEquals(stamped.engineReadyAtMs, RequestTracker.current.value!!.engineReadyAtMs)
        RequestTracker.markCompleted(e.id)
    }
}
