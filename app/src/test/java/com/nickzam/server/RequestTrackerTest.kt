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
}
