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
 * The cached-conversation facts [SessionReuse.decide] judges a request
 * against, detached from any live `Conversation` so the decision is
 * unit-testable on the JVM. Mirrors the fields of `SessionManager`'s
 * private cached-session record that the reuse rule reads.
 *
 * [prefixHash] is a [Long]: that is what `messagesPrefixHash` returns.
 */
data class CachedState(
    val engineKey: EngineKey,
    val temperature: Float,
    val topK: Int,
    val topP: Float,
    val prefixHash: Long,
    val seenCount: Int,
)

/** Outcome of [SessionReuse.decide]. */
enum class Decision { REUSE, FRESH }

/**
 * Pure KV prefix-reuse decision: no Android, no engine, no
 * `SessionManager` instance state — only [Message], `messagesPrefixHash`
 * and value comparisons — so it can be exercised directly from JVM tests.
 */
object SessionReuse {

    /**
     * Decide whether the request described by [messages] plus the
     * request's own [engineKey] / [temperature] / [topK] / [topP] may
     * reuse the conversation described by [cached], or must replay fresh.
     *
     * Contract — [Decision.FRESH] unless ALL of the following hold:
     *  - [cached] is non-null;
     *  - the cached engine key and sampler parameters (temperature, topK,
     *    topP) equal the request's — a conversation warmed under another
     *    engine or sampler is never reused;
     *  - `cached.seenCount < messages.size` — there is at least one new
     *    message beyond what the cached conversation already ingested;
     *  - `cached.prefixHash == messagesPrefixHash(messages, cached.seenCount)`
     *    — the request really does extend the cached prefix, rather than
     *    replaying a divergent history under the same count;
     *  - the tail (messages after `seenCount`) drives the conversation
     *    forward with user turns only, and the request ends in a user
     *    message — see below.
     *
     * Tail rule: let the "driving" messages be the tail messages whose
     * role is not `"assistant"`. Reuse requires at least one driving
     * message, every driving message to have role `"user"`, and the last
     * message of the full list to have role `"user"`.
     *
     * (a) Intermediate new user turns are client-asserted context only
     * and are NEVER fed to the engine: on reuse the caller sends only the
     * final user message (which, under the ends-in-user rule, is
     * `messages.last()`). Earlier new user turns in a multi-message tail
     * are taken on the client's word as part of the replayed history;
     * the engine's KV state is not updated with their text. Accepted
     * cost, in plain terms: those skipped intermediate user turns are
     * permanently absent from the engine's context, and no later reuse
     * check can detect the gap — `prefixHash` is computed over the
     * client's messages only, so the history `commit` records matches
     * the client's list whether or not the engine ever saw every turn
     * in it. This is the mirror image of the issue-#3 stop case: there
     * the engine's KV ran ahead of the client, so the route invalidates
     * the session; here the client's history runs ahead of the engine,
     * accepted deliberately for prefix-reuse latency.
     *
     * (b) Ends-in-user rule: a tail must end in a user message, so that
     * reuse has a well-defined send target: `messages.last()` is the
     * final user turn, the prompt the generation answers, and that is
     * the one message the caller sends. A tail ending in an assistant
     * message (e.g. `[user, assistant]`, which the previous
     * last-driving-message rule reused) has no user prompt after it to
     * answer — there is nothing for the reused conversation to respond
     * to. The check is also defence-in-depth: `resolve`'s preamble
     * already rejects any request whose last non-system message is not
     * a user message; this rule additionally pins the last message of
     * the full list, system messages included, inside the pure decision
     * itself.
     *
     * (c) Assistant messages in the tail are replay-only, never fed to
     * the engine (pre-existing invariant): they are excluded from the
     * driving set, and only the final user message is sent on reuse.
     * An assistant-only tail therefore has no driving message and is
     * [Decision.FRESH], as is any tail carrying a non-user, non-assistant
     * role (e.g. a new `system` message) among its driving messages.
     */
    fun decide(
        cached: CachedState?,
        messages: List<Message>,
        engineKey: EngineKey,
        temperature: Float,
        topK: Int,
        topP: Float,
    ): Decision {
        if (cached == null) return Decision.FRESH
        if (cached.engineKey != engineKey) return Decision.FRESH
        if (cached.temperature != temperature || cached.topK != topK || cached.topP != topP) {
            return Decision.FRESH
        }
        if (cached.seenCount >= messages.size) return Decision.FRESH
        if (cached.prefixHash != messagesPrefixHash(messages, cached.seenCount)) {
            return Decision.FRESH
        }
        val tail = messages.subList(cached.seenCount, messages.size)
        val driving = tail.filter { it.role != "assistant" }
        if (driving.isEmpty()) return Decision.FRESH
        if (driving.any { it.role != "user" }) return Decision.FRESH
        if (messages.last().role != "user") return Decision.FRESH
        return Decision.REUSE
    }
}

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
        // request's prefix. The reuse rule itself lives in
        // SessionReuse.decide (shared with the explicit-session path):
        // same engine + sampler params, seenCount < size, stored prefixHash
        // matches hash(messages, seenCount), and a tail whose driving
        // (non-assistant) messages are all user, ending in a user message.
        // The matched conversation is re-keyed forward by commit() after a
        // successful generation.
        if (req.sessionId.isNullOrEmpty()) {
            val hit = sessions.snapshot().entries.firstOrNull { (_, s) ->
                SessionReuse.decide(
                    cached = CachedState(
                        engineKey = s.engineKey,
                        temperature = s.temperature,
                        topK = s.topK,
                        topP = s.topP,
                        prefixHash = s.prefixHash,
                        seenCount = s.seenCount,
                    ),
                    messages = req.messages,
                    engineKey = acquired.cacheKey,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                ) == Decision.REUSE
            }
            if (hit != null) {
                LogManager.i("SessionManager", "Auto-prefix cache hit (sending final new user turn)")
                // Keep the slot under its original key: commit() refreshes
                // seenCount/prefixHash in place after a successful
                // generation, so the conversation is never re-keyed,
                // duplicated, or closed under our feet.
                // On reuse only the final user message is sent (decide()
                // guarantees messages.last() is that user message); any
                // intermediate new user turns are client-asserted context
                // and are never fed to the engine.
                return Resolved(
                    conversation = hit.value.conversation,
                    prompt = LlmMessageConverter.apiToLlmMessage(req.messages.last()),
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

        // Same SessionReuse.decide rule as the auto-prefix scan above.
        // The explicit path historically carried no engineKey comparison
        // of its own: the engine key is encoded in cacheKey, so a lookup
        // hit is already on this engine. decide() still receives it (and
        // the cached entry's own key) so both sites apply one rule.
        // Equivalence with that lookup-key scheme relies on
        // EngineKey.asString() being injective over the reachable
        // catalog, which holds because catalog model IDs contain no "_"
        // (both ids in ModelCatalog are hyphen-separated) and maxTokens
        // renders as a bare integer or "model": the first two "_"
        // boundaries are therefore unambiguous and the backend name is
        // the remaining suffix.
        val canReuse = SessionReuse.decide(
            cached = cached?.let {
                CachedState(
                    engineKey = it.engineKey,
                    temperature = it.temperature,
                    topK = it.topK,
                    topP = it.topP,
                    prefixHash = it.prefixHash,
                    seenCount = it.seenCount,
                )
            },
            messages = req.messages,
            engineKey = acquired.cacheKey,
            temperature = temperature,
            topK = topK,
            topP = topP,
        ) == Decision.REUSE

        if (canReuse) {
            cached!!
            // Only the final user message is sent (decide() guarantees
            // messages.last() is that user message); intermediate new
            // user turns are client-asserted context, never fed.
            LogManager.i("SessionManager", "Session $cacheKey reused (sending final new user turn)")
            return Resolved(
                conversation = cached.conversation,
                prompt = LlmMessageConverter.apiToLlmMessage(req.messages.last()),
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
