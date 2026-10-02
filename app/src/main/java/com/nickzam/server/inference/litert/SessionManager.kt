// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference.litert

import android.util.LruCache
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Message as LlmMessage
import com.nickzam.server.ChatRequest
import com.nickzam.server.LogManager
import com.nickzam.server.Message
import com.nickzam.server.contentString
import com.nickzam.server.inference.EngineKey
import com.nickzam.server.inference.EngineRegistry
import com.nickzam.server.messagesPrefixHash

/**
 * Owns LiteRT-LM [Conversation] lifecycle and caching. Separated from the
 * chat route so the route can stay focused on HTTP wiring while this class
 * holds the bookkeeping that lets the engine reuse KV-cached conversations
 * across turns.
 *
 * Two tables of state:
 *   - [sessions]: cached `Conversation`s keyed by `session_id + engineKey`,
 *     reused across turns when the client replays a matching prefix.
 *   - [EngineRegistry.activeConversations]: at most one live [Conversation]
 *     per engine — LiteRT-LM enforces this on the native side, so we keep a
 *     marker here to close any prior conversation before creating a new one.
 */
class SessionManager(private val registry: EngineRegistry) {

    /** What [resolve] returns. Caller commits or invalidates after inference. */
    data class Resolved(
        val conversation: Conversation,
        /** The freshly-arrived message to send via sendMessage[Async]. */
        val prompt: LlmMessage,
        val cacheKey: String?,
        val engineKey: EngineKey,
        val temperature: Float,
        val topK: Int,
        val topP: Float,
    ) {
        val isCached: Boolean get() = cacheKey != null
    }

    private data class CachedSession(
        val conversation: Conversation,
        val engineKey: EngineKey,
        val temperature: Float,
        val topK: Int,
        val topP: Float,
        val prefixHash: Long,
        val seenCount: Int,
        val createdAt: Long,
    )

    private val sessions = object : LruCache<String, CachedSession>(4) {
        override fun entryRemoved(
            evicted: Boolean,
            key: String?,
            oldValue: CachedSession?,
            newValue: CachedSession?,
        ) {
            if (oldValue != null && oldValue.conversation !== newValue?.conversation) {
                try { oldValue.conversation.close() } catch (_: Exception) {}
            }
        }
    }

    init {
        // Wire the engine-eviction listener: when an engine entry leaves
        // the registry's LRU, every conversation tied to it MUST be closed
        // first (a conversation outliving its engine is undefined on the
        // native side).
        registry.onLiteRtEvicted = { engineKey: EngineKey ->
            val staleKeys = sessions.snapshot().filter { it.value.engineKey == engineKey }.keys
            staleKeys.forEach { sessions.remove(it) }
            registry.activeConversations.remove(engineKey)?.let { conv ->
                try { conv.close() } catch (_: Exception) {}
            }
        }
    }

    /**
     * Decide whether to reuse a cached conversation or build a fresh one,
     * and compute the prompt fragment to send accordingly.
     */
    fun resolve(
        req: ChatRequest,
        acquired: EngineRegistry.AcquiredEngine.LiteRt,
        temperature: Float,
        topK: Int,
        topP: Float,
    ): Resolved {
        val systemText = req.messages.firstOrNull { it.role == "system" }?.contentString()
        val nonSystem = req.messages.filter { it.role != "system" }
        if (nonSystem.isEmpty()) {
            throw IllegalArgumentException("Request has no non-system messages")
        }
        val last = nonSystem.last()
        if (last.role != "user") {
            throw IllegalArgumentException("Last message must have role=user")
        }
        val lastPrompt = LlmMessageConverter.apiToLlmMessage(last)
        val prior = nonSystem.dropLast(1)

        // Auto-prefix path for clients that don't send session_id (TypingMind
        // and most OpenAI clients replay the full history every turn). The
        // replayed history grows and its hash changes each turn, so there is
        // no stable key to look up — we SCAN the (tiny) session LRU for a
        // warmed conversation on the same engine that already ingested this
        // request's prefix, mirroring the explicit-session reuse rules: same
        // engine + sampler params, seenCount < size, stored prefixHash
        // matches hash(messages, seenCount), and exactly one trailing
        // non-assistant (user) message. The matched conversation is
        // re-keyed forward by commit() after a successful generation.
        if (req.sessionId.isNullOrEmpty()) {
            val hit = sessions.snapshot().entries.firstOrNull { (_, s) ->
                s.engineKey == acquired.cacheKey &&
                    s.temperature == temperature && s.topK == topK && s.topP == topP &&
                    s.seenCount < req.messages.size &&
                    s.prefixHash == messagesPrefixHash(req.messages, s.seenCount) &&
                    run {
                        val newRange = req.messages.subList(s.seenCount, req.messages.size)
                        val driving = newRange.filter { it.role != "assistant" }
                        driving.size == 1 && driving[0].role == "user"
                    }
            }
            if (hit != null) {
                LogManager.i("SessionManager", "Auto-prefix cache hit (sending 1 new user turn)")
                // Keep the slot under its original key: commit() refreshes
                // seenCount/prefixHash in place after a successful
                // generation, so the conversation is never re-keyed,
                // duplicated, or closed under our feet.
                return Resolved(
                    conversation = hit.value.conversation,
                    prompt = LlmMessageConverter.apiToLlmMessage(
                        req.messages.subList(hit.value.seenCount, req.messages.size)
                            .first { it.role != "assistant" }
                    ),
                    cacheKey = hit.key,
                    engineKey = acquired.cacheKey,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                )
            }
            val freshKey = "auto:${messagesPrefixHash(req.messages, req.messages.size).toString(16)}_${acquired.cacheKey.asString()}"
            val conversation = createConversation(acquired, temperature, topK, topP, systemText, prior)
            return Resolved(conversation, lastPrompt, freshKey, acquired.cacheKey, temperature, topK, topP)
        }

        val cacheKey = "${req.sessionId}_${acquired.cacheKey.asString()}"
        val cached = sessions.get(cacheKey)

        val canReuse = cached != null &&
            cached.temperature == temperature &&
            cached.topK == topK &&
            cached.topP == topP &&
            cached.seenCount < req.messages.size &&
            cached.prefixHash == messagesPrefixHash(req.messages, cached.seenCount) &&
            run {
                val newRange = req.messages.subList(cached.seenCount, req.messages.size)
                val driving = newRange.filter { it.role != "assistant" }
                driving.size == 1 && driving[0].role == "user"
            }

        if (canReuse) {
            cached!!
            val newDriving = req.messages.subList(cached.seenCount, req.messages.size)
                .first { it.role != "assistant" }
            LogManager.i("SessionManager", "Session $cacheKey reused (sending 1 new ${newDriving.role} turn)")
            return Resolved(
                conversation = cached.conversation,
                prompt = LlmMessageConverter.apiToLlmMessage(newDriving),
                cacheKey = cacheKey,
                engineKey = acquired.cacheKey,
                temperature = temperature,
                topK = topK,
                topP = topP,
            )
        }

        if (cached != null) sessions.remove(cacheKey)
        val conversation = createConversation(acquired, temperature, topK, topP, systemText, prior)
        return Resolved(conversation, lastPrompt, cacheKey, acquired.cacheKey, temperature, topK, topP)
    }

    private fun createConversation(
        acquired: EngineRegistry.AcquiredEngine.LiteRt,
        temperature: Float,
        topK: Int,
        topP: Float,
        systemText: String?,
        prior: List<Message>,
    ): Conversation {
        purgeConversationsOnEngine(acquired.cacheKey)
        val conv = try {
            LlmMessageConverter.createConversation(
                engine = acquired.engine.native,
                temperature = temperature,
                topK = topK,
                topP = topP,
                systemText = systemText,
                initial = prior,
            )
        } catch (e: Exception) {
            if (e.message?.contains("session already exists", ignoreCase = true) == true) {
                LogManager.w("SessionManager", "Engine ${acquired.cacheKey.asString()} stuck; evicting.")
                registry.dropLiteRt(acquired.cacheKey)
                throw IllegalStateException("Engine had a stuck conversation; evicted. Please retry the request.", e)
            }
            throw e
        }
        registry.activeConversations[acquired.cacheKey] = conv
        return conv
    }

    private fun purgeConversationsOnEngine(engineKey: EngineKey) {
        val staleKeys = sessions.snapshot().filter { it.value.engineKey == engineKey }.keys
        staleKeys.forEach { sessions.remove(it) }
        registry.activeConversations.remove(engineKey)?.let { prior ->
            try { prior.close() } catch (_: Exception) {}
        }
    }

    /** Successful generation — commit (or refresh) the cached conversation. */
    fun commit(resolved: Resolved, messages: List<Message>) {
        val cacheKey = resolved.cacheKey ?: return
        sessions.put(cacheKey, CachedSession(
            conversation = resolved.conversation,
            engineKey = resolved.engineKey,
            temperature = resolved.temperature,
            topK = resolved.topK,
            topP = resolved.topP,
            prefixHash = messagesPrefixHash(messages, messages.size),
            seenCount = messages.size,
            createdAt = sessions.get(cacheKey)?.createdAt ?: System.currentTimeMillis(),
        ))
        registry.activeConversations[resolved.engineKey] = resolved.conversation
    }

    /** Failure path — drop the (possibly half-initialized) conversation. */
    fun invalidate(resolved: Resolved) {
        registry.activeConversations.remove(resolved.engineKey, resolved.conversation)
        val cacheKey = resolved.cacheKey
        if (cacheKey != null) {
            sessions.remove(cacheKey)
        } else {
            try { resolved.conversation.close() } catch (_: Exception) {}
        }
    }

    /** Stateless cleanup helper. */
    fun closeIfStateless(resolved: Resolved) {
        if (!resolved.isCached) {
            registry.activeConversations.remove(resolved.engineKey, resolved.conversation)
            try { resolved.conversation.close() } catch (_: Exception) {}
        }
    }

    /** Drop every cached session. Memory-pressure path. */
    fun evictAll() {
        sessions.evictAll()
    }

    fun size(): Int = sessions.size()
}
