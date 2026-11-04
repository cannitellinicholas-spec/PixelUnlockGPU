package com.nickzam.server

import com.google.gson.Gson
import com.nickzam.server.server.routes.StreamReassembler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UsageEstimateTest {

    private val gson = Gson()

    @Test fun `estimateUsage divides chars by four`() {
        val usage = estimateUsage(promptChars = 16, completionChars = 8)
        assertEquals(4, usage.promptTokens)
        assertEquals(2, usage.completionTokens)
        assertEquals(6, usage.totalTokens)
    }

    @Test fun `estimateUsage floors sub-four char counts to zero`() {
        val usage = estimateUsage(promptChars = 3, completionChars = 2)
        assertEquals(0, usage.promptTokens)
        assertEquals(0, usage.completionTokens)
        assertEquals(0, usage.totalTokens)
    }

    @Test fun `estimateUsage floors partial tokens`() {
        val usage = estimateUsage(promptChars = 7, completionChars = 5)
        assertEquals(1, usage.promptTokens)
        assertEquals(1, usage.completionTokens)
        assertEquals(2, usage.totalTokens)
    }

    @Test fun `estimateUsage total is sum of prompt and completion`() {
        for ((promptChars, completionChars) in listOf(0 to 0, 1 to 100, 40 to 41, 123 to 456)) {
            val usage = estimateUsage(promptChars, completionChars)
            assertEquals(usage.promptTokens + usage.completionTokens, usage.totalTokens)
        }
    }

    @Test fun `streamed usage completion tokens match reassembled text length over four`() {
        // Mirrors runInferenceStreaming in ChatRoute: each emission goes
        // through StreamReassembler.next against the text accumulated so
        // far, and only non-empty deltas are appended to acc — the same
        // acc whose length feeds estimateUsage for the final chunk.
        val cases = listOf(
            // Incremental: each emission is only the new token.
            listOf("The", " quick", " brown", " fox") to "The quick brown fox",
            // Cumulative: each emission repeats the whole text so far.
            listOf("The", "The quick", "The quick brown") to "The quick brown",
        )
        for ((emissions, expected) in cases) {
            val acc = StringBuilder()
            for (full in emissions) {
                val delta = StreamReassembler.next(acc.toString(), full)
                if (delta.isNotEmpty()) {
                    acc.append(delta)
                }
            }
            assertEquals(expected, acc.toString())

            val usage = estimateUsage(promptChars = 20, completionChars = acc.length)
            assertEquals(acc.length / 4, usage.completionTokens)

            // The final chunk runInferenceStreaming actually emits.
            val finalChunk = StreamResponse(
                id = "chatcmpl-1",
                `object` = "chat.completion.chunk",
                created = 123L,
                model = "test-model",
                choices = listOf(StreamChoice(0, StreamDelta(), "stop")),
                usage = usage,
            )
            val json = Gson().toJson(finalChunk)
            assertTrue(json.contains("\"completion_tokens\":${acc.length / 4}"))
        }
    }

    @Test fun `chat response serializes usage with openai keys`() {
        val response = ChatResponse(
            id = "chatcmpl-1",
            `object` = "chat.completion",
            created = 123L,
            model = "test-model",
            choices = listOf(
                Choice(
                    index = 0,
                    message = Message(role = "assistant", content = stringContent("hello")),
                    finishReason = "stop",
                )
            ),
            usage = estimateUsage(promptChars = 16, completionChars = 8),
        )
        val json = gson.toJson(response)
        assertTrue(json.contains("\"usage\""))
        assertTrue(json.contains("\"prompt_tokens\":4"))
        assertTrue(json.contains("\"completion_tokens\":2"))
        assertTrue(json.contains("\"total_tokens\":6"))
    }

    @Test fun `stream final chunk serializes usage with openai keys`() {
        val response = StreamResponse(
            id = "chatcmpl-1",
            `object` = "chat.completion.chunk",
            created = 123L,
            model = "test-model",
            choices = listOf(StreamChoice(0, StreamDelta(), "stop")),
            usage = estimateUsage(promptChars = 20, completionChars = 12),
        )
        val json = gson.toJson(response)
        assertTrue(json.contains("\"usage\""))
        assertTrue(json.contains("\"prompt_tokens\":5"))
        assertTrue(json.contains("\"completion_tokens\":3"))
        assertTrue(json.contains("\"total_tokens\":8"))
    }

    @Test fun `stream chunk without usage omits usage key`() {
        val response = StreamResponse(
            id = "chatcmpl-1",
            `object` = "chat.completion.chunk",
            created = 123L,
            model = "test-model",
            choices = listOf(StreamChoice(0, StreamDelta(content = "hi"), null)),
        )
        val json = gson.toJson(response)
        assertFalse(json.contains("\"usage\""))
    }

    @Test fun `chat response without usage omits usage key`() {
        val response = ChatResponse(
            id = "chatcmpl-1",
            `object` = "chat.completion",
            created = 123L,
            model = "test-model",
            choices = listOf(
                Choice(
                    index = 0,
                    message = Message(role = "assistant", content = stringContent("hello")),
                    finishReason = "stop",
                )
            ),
        )
        val json = gson.toJson(response)
        assertFalse(json.contains("\"usage\""))
    }
}
