package com.nickzam.server.server.routes

import org.junit.Assert.assertEquals
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
}
