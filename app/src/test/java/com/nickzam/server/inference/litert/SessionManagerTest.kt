package com.nickzam.server.inference.litert

import com.nickzam.server.Backend
import com.nickzam.server.GPU_MODEL_ID
import com.nickzam.server.Message
import com.nickzam.server.STABLE_MODEL_ID
import com.nickzam.server.inference.EngineKey
import com.nickzam.server.messagesPrefixHash
import com.nickzam.server.stringContent
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Pure JVM tests for [SessionReuse.decide], the KV prefix-reuse rule
 * extracted from `SessionManager.resolve`. Fixtures are built with the
 * real [Message] constructor and the real `messagesPrefixHash`, so a
 * matching [CachedState] is exactly what `SessionManager.commit` would
 * have recorded for the cached prefix.
 */
class SessionManagerTest {

    private val requestEngineKey = EngineKey(GPU_MODEL_ID, null, Backend.LITERT_GPU)
    private val requestTemperature = 0.8f
    private val requestTopK = 40
    private val requestTopP = 0.95f

    private fun user(text: String) = Message("user", stringContent(text))
    private fun assistant(text: String) = Message("assistant", stringContent(text))
    private fun system(text: String) = Message("system", stringContent(text))
    private fun tool(text: String) = Message("tool", stringContent(text))

    /** A cached state whose prefixHash genuinely matches [messages] up to [seenCount]. */
    private fun cachedState(messages: List<Message>, seenCount: Int): CachedState =
        CachedState(
            engineKey = requestEngineKey,
            temperature = requestTemperature,
            topK = requestTopK,
            topP = requestTopP,
            prefixHash = messagesPrefixHash(messages, seenCount),
            seenCount = seenCount,
        )

    private fun decide(cached: CachedState?, messages: List<Message>): Decision =
        SessionReuse.decide(
            cached = cached,
            messages = messages,
            engineKey = requestEngineKey,
            temperature = requestTemperature,
            topK = requestTopK,
            topP = requestTopP,
        )

    /** One cached exchange; requests below extend it with various tails. */
    private val prior = listOf(user("hi"), assistant("hello"))

    @Test fun `no cached state is fresh`() {
        val messages = prior + user("more")
        assertEquals(Decision.FRESH, decide(null, messages))
    }

    @Test fun `engine mismatch is fresh`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = prior.size).copy(
            engineKey = EngineKey(STABLE_MODEL_ID, null, Backend.LITERT_NPU),
        )
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `engine mismatch on model id only is fresh`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = prior.size).copy(
            engineKey = EngineKey(STABLE_MODEL_ID, null, Backend.LITERT_GPU),
        )
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `engine mismatch on backend only is fresh`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = prior.size).copy(
            engineKey = EngineKey(GPU_MODEL_ID, null, Backend.LITERT_NPU),
        )
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `engine mismatch on maxTokens only is fresh`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = prior.size).copy(
            engineKey = EngineKey(GPU_MODEL_ID, 1024, Backend.LITERT_GPU),
        )
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `sampler mismatch is fresh`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached.copy(temperature = 0.1f), messages))
        assertEquals(Decision.FRESH, decide(cached.copy(topK = 1), messages))
        assertEquals(Decision.FRESH, decide(cached.copy(topP = 0.5f), messages))
    }

    @Test fun `seenCount beyond the message list is fresh`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = messages.size + 1)
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `prefix hash mismatch is fresh`() {
        val messages = prior + user("more")
        val divergent = listOf(user("different"), assistant("other"))
        val cached = cachedState(messages, seenCount = prior.size).copy(
            prefixHash = messagesPrefixHash(divergent, divergent.size),
        )
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `exactly the cached prefix with no tail is fresh`() {
        val cached = cachedState(prior, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached, prior))
    }

    @Test fun `single new user message reuses`() {
        val messages = prior + user("more")
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.REUSE, decide(cached, messages))
    }

    @Test fun `assistant then user tail reuses`() {
        val messages = prior + listOf(assistant("replayed reply"), user("more"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.REUSE, decide(cached, messages))
    }

    @Test fun `two new user messages reuse`() {
        val messages = prior + listOf(user("first new"), user("second new"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.REUSE, decide(cached, messages))
    }

    @Test fun `user assistant user tail reuses`() {
        val messages = prior + listOf(user("first new"), assistant("replayed reply"), user("second new"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.REUSE, decide(cached, messages))
    }

    @Test fun `user then assistant tail is fresh`() {
        // Deliberate edge change: this tail reused under the old
        // last-driving-message rule, but committing it would advance
        // seenCount past an assistant message the engine never saw.
        val messages = prior + listOf(user("more"), assistant("replayed reply"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `assistant only tail is fresh`() {
        val messages = prior + assistant("replayed reply")
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `new system message in tail is fresh`() {
        val messages = prior + listOf(system("new instructions"), user("more"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `user then system tail is fresh`() {
        val messages = prior + listOf(user("more"), system("trailing instructions"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached, messages))
    }

    @Test fun `tool then user tail is fresh`() {
        // decide() is pure and sits below the API validation layer
        // (which rejects the "tool" role upstream), so the role is
        // constructed directly here to pin decide()'s own rule.
        val messages = prior + listOf(tool("tool result"), user("more"))
        val cached = cachedState(messages, seenCount = prior.size)
        assertEquals(Decision.FRESH, decide(cached, messages))
    }
}
