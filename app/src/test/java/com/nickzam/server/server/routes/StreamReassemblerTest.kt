package com.nickzam.server.server.routes

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Regression net for the 2026-10-01 streamed-gibberish bug: the old code
 * assumed cumulative emissions and sliced by length, so incremental
 * per-token chunks were chopped and equal-length emissions dropped.
 */
class StreamReassemblerTest {

    private fun drain(emissions: List<String>): String {
        val acc = StringBuilder()
        for (e in emissions) acc.append(StreamReassembler.next(acc.toString(), e))
        return acc.toString()
    }

    @Test
    fun `incremental per-token stream passes through untouched`() {
        val toks = listOf("The", " quick", " brown", " fox")
        assertEquals("The quick brown fox", drain(toks))
    }

    @Test
    fun `incremental equal-length tokens are never dropped`() {
        // The old code skipped emissions whose length matched the previous
        // one (e.g. "\n2" after "\n1"), losing whole lines.
        val toks = listOf("\n1", "\n2", "\n3", "40")
        assertEquals("\n1\n2\n340", drain(toks))
    }

    @Test
    fun `cumulative stream emits each extension exactly once`() {
        val cum = listOf("The", "The quick", "The quick brown")
        assertEquals("The quick brown", drain(cum))
    }

    @Test
    fun `repeated cumulative prefix collapses to nothing`() {
        // Same cumulative text delivered twice must not duplicate.
        val acc1 = StringBuilder()
        acc1.append(StreamReassembler.next("", "The quick"))
        assertEquals("", StreamReassembler.next(acc1.toString(), "The quick"))
    }

    @Test
    fun `empty emissions are no-ops`() {
        assertEquals("abc", drain(listOf("", "abc", "")))
    }

    @Test
    fun `first emission always passes through`() {
        assertEquals("hello", StreamReassembler.next("", "hello"))
    }

    @Test
    fun `emission shorter than emitted and unrelated is appended raw`() {
        // Never lose text: unknown shapes go out raw rather than being
        // silently sliced away.
        assertEquals("Thequick", drain(listOf("The", "quick")))
    }

    @Test
    fun `reassembly of the fox regression fixture`() {
        // Real LiteRT-LM 0.12.0 sequence shape (incremental).
        val sentence = "The quick brown fox jumps over the lazy dog."
        val toks = sentence.split(" ").mapIndexed { i, w -> if (i == 0) w else " $w" }
        assertEquals(sentence, drain(toks))
    }

    // ---- StopFilter (issue #3: client-side stop-sequence emulation) ----

    /**
     * Mirrors the streaming route: reassemble each emission into the
     * accumulator, feed the joined text to the filter per delta, stop
     * feeding once the filter reports a stop, and flush with finish()
     * when the stream ends unstopped. Returns (forwardedText, stopped).
     */
    private fun drainWithStops(
        emissions: List<String>,
        stops: List<String>,
    ): Pair<String, Boolean> {
        val acc = StringBuilder()
        val filter = StopFilter(stops)
        val forwarded = StringBuilder()
        for (e in emissions) {
            val delta = StreamReassembler.next(acc.toString(), e)
            if (delta.isNotEmpty()) {
                acc.append(delta)
                forwarded.append(filter.onJoined(acc.toString()))
                if (filter.stopped) break
            }
        }
        if (!filter.stopped) forwarded.append(filter.finish(acc.toString()))
        return forwarded.toString() to filter.stopped
    }

    @Test
    fun `stop at chunk boundary forwards nothing past it`() {
        val (text, stopped) = drainWithStops(
            listOf("Hello", "\n\n", "World"), listOf("\n\n"))
        assertEquals("Hello", text)
        assertTrue(stopped)
    }

    @Test
    fun `stop split across two chunks still matches`() {
        // Neither chunk alone contains "\n\n"; the joined text does.
        val (text, stopped) = drainWithStops(
            listOf("Hello\n", "\nWorld"), listOf("\n\n"))
        assertEquals("Hello", text)
        assertTrue(stopped)
    }

    @Test
    fun `no stop match forwards everything unstopped`() {
        val (text, stopped) = drainWithStops(
            listOf("Hello", " world"), listOf("\n\n"))
        assertEquals("Hello world", text)
        assertFalse(stopped)
    }

    @Test
    fun `match in the very first chunk truncates it`() {
        val (text, stopped) = drainWithStops(
            listOf("Stop here\n\nand more"), listOf("\n\n"))
        assertEquals("Stop here", text)
        assertTrue(stopped)
    }

    @Test
    fun `held-back stop prefix is flushed at end of stream`() {
        // The trailing "\n" is a proper prefix of "\n\n", so onJoined
        // holds it back; finish() must deliver it when no stop completes.
        val (text, stopped) = drainWithStops(
            listOf("Hello\n"), listOf("\n\n"))
        assertEquals("Hello\n", text)
        assertFalse(stopped)
    }

    @Test
    fun `truncate cuts at the first stop occurrence`() {
        assertEquals("a", StopFilter.truncate("a\n\nb", listOf("\n\n")))
    }

    @Test
    fun `truncate leaves no-stop text unchanged`() {
        assertEquals("a\nb", StopFilter.truncate("a\nb", listOf("\n\n")))
    }

    @Test
    fun `truncate with stop at position zero is empty`() {
        assertEquals("", StopFilter.truncate("\n\nb", listOf("\n\n")))
    }
}
