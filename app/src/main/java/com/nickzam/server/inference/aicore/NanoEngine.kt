// Gemini Nano lane: generation is delegated to the on-device AICore service
// through the ML Kit GenAI Prompt API (com.google.mlkit:genai-prompt).
// FAST preference only — the FULL preview bundle fails inference on this
// firmware (verified 2026-10-05 in the sibling PixelChat integration, where
// even a clean AICore model re-download did not repair it).

package com.nickzam.server.inference.aicore

import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.StreamingCallback
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.ModelPreference
import com.google.mlkit.genai.prompt.ModelReleaseStage
import com.google.mlkit.genai.prompt.PromptPrefix
import com.google.mlkit.genai.prompt.TextPart
import com.google.mlkit.genai.prompt.generateContentRequest
import com.google.mlkit.genai.prompt.generationConfig
import com.google.mlkit.genai.prompt.modelConfig
import com.nickzam.server.ChatRequest
import com.nickzam.server.LogManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

/** Structured failure so routes can map Nano errors to honest HTTP codes. */
class NanoUnavailableException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** Prompt larger than AICore's own token window — a client error, not a server fault. */
class NanoPromptTooLargeException(message: String) : Exception(message)

/** What the producer publishes: streamed deltas, the terminal response, or a failure. */
private sealed class NanoEmit {
    data class Delta(val text: String) : NanoEmit()
    data class Final(val text: String) : NanoEmit()
    data class Failed(val error: Throwable) : NanoEmit()
}

/**
 * Host engine for the AICORE backend: no in-process weights, no files —
 * one cached FAST-preference client talking to the AICore system service.
 *
 * Availability is probed (never assumed): [status] returns AICore's own
 * FeatureStatus, cached briefly so health polls and request admission don't
 * hammer the binder. The model download itself belongs to AICore; this app
 * never fetches Nano weights.
 */
class NanoEngine {

    @Volatile private var cachedStatus: Int? = null
    @Volatile private var cachedStatusAtMs: Long = 0L

    /** The single FAST client (preemptive creation; no network on construction). */
    private val client by lazy {
        Generation.getClient(generationConfig {
            modelConfig = modelConfig {
                // PREVIEW stage + FAST preference — exactly the pin the sibling
                // PixelChat lane verified on this device.
                releaseStage = ModelReleaseStage.PREVIEW
                preference = ModelPreference.FAST
            }
        })
    }

    /**
     * Current FeatureStatus, cached for [ttlMs]. [FeatureStatus.AVAILABLE]
     * is the only value that permits generation. Probe failures surface as
     * UNAVAILABLE with the cause logged, so /health degrades honestly.
     */
    suspend fun status(ttlMs: Long = 15_000L): Int {
        val now = System.currentTimeMillis()
        cachedStatus?.let { if (now - cachedStatusAtMs < ttlMs) return it }
        val fresh = try {
            client.checkStatus()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            LogManager.e("NanoEngine", "checkStatus failed", e)
            FeatureStatus.UNAVAILABLE
        }
        cachedStatus = fresh
        cachedStatusAtMs = now
        return fresh
    }

    fun invalidateStatusCache() { cachedStatusAtMs = 0L }

    /** Human-readable reason a non-AVAILABLE status blocks generation. */
    fun unavailableReason(status: Int): String = when (status) {
        FeatureStatus.DOWNLOADING -> "Gemini Nano is downloading in AICore"
        FeatureStatus.DOWNLOADABLE -> "Gemini Nano needs a model download in AICore (open AICore settings or connect Wi-Fi)"
        else -> "Gemini Nano unavailable in AICore (is the AICore service installed and enabled?)"
    }

    /** AICore's advertised token window for this client, or null if unknown. */
    suspend fun tokenLimit(): Int? = try {
        client.getTokenLimit()
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        LogManager.e("NanoEngine", "getTokenLimit failed", e)
        null
    }

    /**
     * Run one chat completion. Streams deltas through [onText] as AICore
     * emits them; an implementation that returns only final text is handled
     * by emitting it once at the end (never zero text, never twice).
     *
     * Throws [NanoUnavailableException] when the host isn't AVAILABLE and
     * [NanoPromptTooLargeException] when the prompt exceeds AICore's window.
     */
    suspend fun generate(
        chat: ChatRequest,
        temperature: Float?,
        topK: Int?,
        maxTokens: Int?,
        onText: suspend (String) -> Unit,
    ): String = coroutineScope {
        val hostStatus = status()
        if (hostStatus != FeatureStatus.AVAILABLE) {
            throw NanoUnavailableException(unavailableReason(hostStatus))
        }

        val request = buildRequest(chat, temperature, topK, maxTokens)

        // Fail before generation when AICore itself says the prompt can't fit.
        val limit = tokenLimit()
        val tokens = try {
            client.countTokens(request).totalTokens
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // AICore refuses to count inputs past its own limit — that's a
            // prompt-size client error (413), not host unavailability (503).
            val msg = e.message.orEmpty()
            if (msg.contains("exceeds", ignoreCase = true) && msg.contains("limit", ignoreCase = true)) {
                throw NanoPromptTooLargeException(
                    "Prompt exceeds Gemini Nano's input limit. " +
                        "Shorten the conversation or use a LiteRT model.",
                )
            }
            throw NanoUnavailableException(
                "AICore token count failed: ${msg.ifEmpty { e.javaClass.simpleName }}", e,
            )
        }
        if (limit != null && limit > 0 && tokens >= limit) {
            throw NanoPromptTooLargeException(
                "Prompt of $tokens tokens exceeds Gemini Nano's window of $limit. " +
                    "Shorten the conversation or use a LiteRT model.",
            )
        }

        val emitted = StringBuilder()
        val sink = Channel<NanoEmit>(Channel.UNLIMITED)
        launch(Dispatchers.Default) {
            try {
                val callback = object : StreamingCallback {
                    override fun onNewText(text: String) {
                        if (text.isNotEmpty()) sink.trySend(NanoEmit.Delta(text))
                    }
                }
                val response = client.generateContent(request, callback)
                // Some implementations never invoke the callback; the consumer
                // uses the final text only when nothing streamed.
                sink.trySend(NanoEmit.Final(response.candidates.firstOrNull()?.text.orEmpty()))
            } catch (e: CancellationException) {
                sink.trySend(NanoEmit.Failed(e))
            } catch (e: Exception) {
                LogManager.e("NanoEngine", "generateContent failed", e)
                sink.trySend(NanoEmit.Failed(e))
            } finally {
                sink.close()
            }
        }

        try {
            while (true) {
                when (val item = sink.receive()) {
                    is NanoEmit.Delta -> {
                        // beta4 streams incremental deltas (verified in the
                        // sibling lane); they are response text verbatim.
                        emitted.append(item.text)
                        onText(item.text)
                    }
                    is NanoEmit.Final -> {
                        if (emitted.isEmpty() && item.text.isNotEmpty()) {
                            emitted.append(item.text)
                            onText(item.text)
                        }
                        break
                    }
                    is NanoEmit.Failed -> throw item.error
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            invalidateStatusCache()
            throw NanoUnavailableException(
                "AICore generation failed: ${e.message ?: e.javaClass.simpleName}", e,
            )
        }
        val out = emitted.toString()
        if (out.isBlank()) {
            invalidateStatusCache()
            throw NanoUnavailableException("AICore returned an empty response")
        }
        out
    }

    /**
     * Map the OpenAI-shaped messages onto the Prompt API's flat request:
     * system turns become a PromptPrefix; the rest flatten to a
     * `ROLE:\n…` transcript ending in an `ASSISTANT:` cue — the exact
     * prompt shape verified against the beta4 AAR in the sibling PixelChat
     * lane (beta4 has no chat-history content type).
     */
    private fun buildRequest(
        chat: ChatRequest,
        temperature: Float?,
        topK: Int?,
        maxTokens: Int?,
    ): GenerateContentRequest {
        val system = chat.messages.filter { it.role == "system" }
            .joinToString("\n\n") { contentText(it) }
        val history = chat.messages.filter { it.role != "system" }
            .joinToString("\n\n") { m ->
                val role = if (m.role == "assistant") "ASSISTANT" else "USER"
                "$role:\n${contentText(m)}"
            }
        return generateContentRequest(TextPart(history + "\n\nASSISTANT:\n")) {
            if (system.isNotEmpty()) promptPrefix = PromptPrefix(system)
            temperature?.let { this.temperature = it }
            topK?.let { this.topK = it }
            maxTokens?.let { this.maxOutputTokens = it }
            candidateCount = 1
        }
    }

    /** Plain text of a message across the accepted content shapes. */
    private fun contentText(message: com.nickzam.server.Message): String {
        val c = message.content ?: return ""
        if (c.isJsonNull) return ""
        if (c.isJsonPrimitive) return c.asString
        // Content-part arrays: concatenate the text parts (validation already
        // rejected non-text parts).
        return c.asJsonArray.mapNotNull { el ->
            if (el.isJsonObject && el.asJsonObject.has("text") &&
                el.asJsonObject["text"].isJsonPrimitive
            ) el.asJsonObject["text"].asString else null
        }.joinToString(" ")
    }

    /** AICore owns the model; closing our client handle releases the binder only. */
    fun close() {
        try { client.close() } catch (_: Exception) {}
    }
}
