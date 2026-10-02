// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.server.routes

import android.content.Context
import com.google.ai.edge.litertlm.Message as LlmMessage
import com.google.gson.Gson
import com.nickzam.server.ChatRequest
import com.nickzam.server.ChatResponse
import com.nickzam.server.ChatValidation
import com.nickzam.server.Choice
import com.nickzam.server.ErrorDetails
import com.nickzam.server.ErrorResponse
import com.nickzam.server.LogManager
import com.nickzam.server.Message
import com.nickzam.server.RateLimiter
import com.nickzam.server.RequestTracker
import com.nickzam.server.RichErrorDetails
import com.nickzam.server.RichErrorResponse
import com.nickzam.server.Settings
import com.nickzam.server.StreamChoice
import com.nickzam.server.StreamDelta
import com.nickzam.server.StreamResponse
import com.nickzam.server.inference.EngineRegistry
import com.nickzam.server.inference.litert.LlmMessageConverter
import com.nickzam.server.inference.litert.SessionManager
import com.nickzam.server.server.auth.authorize
import com.nickzam.server.stringContent
import com.nickzam.server.textChars
import io.ktor.http.CacheControl
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.request.receive
import io.ktor.server.response.cacheControl
import io.ktor.server.response.header
import io.ktor.server.response.respond
import io.ktor.server.response.respondBytesWriter
import io.ktor.server.routing.Route
import io.ktor.server.routing.post
import io.ktor.utils.io.ByteWriteChannel
import io.ktor.utils.io.writeStringUtf8
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/**
 * `POST /v1/chat/completions` — OpenAI-compatible text chat.
 *
 * Handles request validation (strict: the model must equal the stable ID;
 * tools/images/audio/stop are rejected explicitly), per-client rate
 * limiting, bounded queue depth, engine acquisition on the catalog-declared
 * backend (no fallback), streaming SSE, non-streaming JSON, and timeout /
 * cancellation cleanup.
 *
 * LiteRT models hold the inference mutex across generation so two engines
 * never pump the JNI runtime at once. Client disconnect/cancellation
 * propagates to the LiteRT-LM generation flow and always frees the slot.
 */
fun Route.chatRoute(
    appContext: Context,
    engineRegistry: EngineRegistry,
    sessionManager: SessionManager,
    rateLimiter: RateLimiter,
    inferenceMutex: Mutex,
    serviceScope: CoroutineScope,
    lastActivityAt: AtomicLong,
    acquireWakeLock: suspend (timeoutMs: Long, block: suspend () -> Unit) -> Unit,
) {
    val gson = Gson()
    post("/v1/chat/completions") {
        if (!authorize(call, appContext)) return@post
        lastActivityAt.set(System.currentTimeMillis())

        // Per-client rate limit, keyed on User-Agent. rate=0 disables.
        val clientId = call.request.headers["User-Agent"]?.takeIf { it.isNotBlank() } ?: "anonymous"
        val rate = Settings.rateLimitPerSec(appContext)
        if (rate > 0.0) {
            rateLimiter.ratePerSec = rate
            rateLimiter.burst = Settings.rateLimitBurst(appContext)
            val retryAfter = rateLimiter.tryAcquire(clientId)
            if (retryAfter != null) {
                call.response.header("Retry-After", retryAfter.toString())
                call.response.header("X-RateLimit-Client", clientId)
                call.respond(
                    HttpStatusCode.TooManyRequests,
                    ErrorResponse(ErrorDetails(
                        message = "Rate limit for client '$clientId' exhausted; retry in ${retryAfter}s.",
                        type = "rate_limit_error",
                        code = 429,
                    )),
                )
                return@post
            }
        }

        // Body-size guard. ~2 bytes per char covers JSON overhead with margin.
        val maxChars = Settings.maxPromptChars(appContext)
        val bodyCap = maxChars.toLong() * 2L + 8_192L
        val contentLength = call.request.headers["Content-Length"]?.toLongOrNull()
        if (contentLength != null && contentLength > bodyCap) {
            call.respond(
                HttpStatusCode.PayloadTooLarge,
                ErrorResponse(ErrorDetails(
                    message = "Request body of $contentLength bytes exceeds cap of $bodyCap",
                    type = "invalid_request_error",
                    code = 413,
                )),
            )
            return@post
        }

        val req = try {
            call.receive<ChatRequest>()
        } catch (e: Exception) {
            LogManager.e("ChatRoute", "Failed to parse ChatRequest body", e)
            val rootCause = generateSequence(e as Throwable?) { it.cause }.lastOrNull() ?: e
            call.respond(
                HttpStatusCode.BadRequest,
                ErrorResponse(ErrorDetails(
                    message = "Invalid JSON body: ${rootCause.javaClass.simpleName}: ${rootCause.message ?: e.message}",
                    type = "invalid_request_error",
                    code = 400,
                )),
            )
            return@post
        }

        // Strict contract validation: unknown model, bad roles/content,
        // unsupported features, invalid sampling. Never silently ignored.
        ChatValidation.validate(req)?.let { failure ->
            call.respond(
                HttpStatusCode(failure.httpStatus, "Validation"),
                ErrorResponse(ErrorDetails(failure.message, failure.type, failure.httpStatus)),
            )
            return@post
        }

        val promptChars = req.messages.sumOf { it.textChars() }
        if (promptChars > maxChars) {
            call.respond(
                HttpStatusCode.PayloadTooLarge,
                ErrorResponse(ErrorDetails(
                    message = "Prompt of $promptChars chars exceeds limit of $maxChars",
                    type = "invalid_request_error",
                    code = 413,
                )),
            )
            return@post
        }

        val maxDepth = Settings.maxQueueDepth(appContext)
        val entry = RequestTracker.tryEnqueue(
            model = req.model,
            stream = req.stream,
            messageCount = req.messages.size,
            promptChars = promptChars,
            maxDepth = maxDepth,
            client = clientId,
        )
        if (entry == null) {
            call.response.header("Retry-After", "5")
            call.respond(
                HttpStatusCode.TooManyRequests,
                ErrorResponse(ErrorDetails(
                    message = "Queue full ($maxDepth in flight). Retry shortly.",
                    type = "rate_limit_error",
                    code = 429,
                )),
            )
            return@post
        }

        val queueDepth = RequestTracker.queue.value.size
        val queuePosition = RequestTracker.queue.value.indexOfFirst { it.id == entry.id } + 1
        val avgInfMs = RequestTracker.stats.value.avgLatencyMs
        val estimatedWaitMs = (queuePosition - 1).coerceAtLeast(0) * avgInfMs
        call.response.header("X-Queue-Position", queuePosition.toString())
        call.response.header("X-Queue-Depth", queueDepth.toString())
        call.response.header("X-Estimated-Wait-Ms", estimatedWaitMs.toString())
        call.response.header("X-Request-Id", entry.id)
        call.response.header("X-Client-Id", clientId)

        val remoteIp = call.request.local.remoteHost
        val timeoutMs = Settings.requestTimeoutMs(appContext)

        var resolved: SessionManager.Resolved? = null
        var inferenceOk = false
        var streamWriter: ByteWriteChannel? = null
        // Serialize the whole request against the engine: LiteRT-LM allows
        // one live conversation per engine, and resolve() closes/replaces
        // conversations as it books sessions — doing that while another
        // request is generating on the same engine segfaults
        // liblitertlm_jni.so (SIGSEGV observed 2026-10-01 when a second
        // request resolved mid-stream of the first). Lock waits happen
        // before the timeout starts, so long generations don't starve
        // queued clients into spurious 408s.
        inferenceMutex.withLock {
        try {
            LogManager.i(
                "ChatRoute",
                "Request #${entry.id} from $remoteIp [$clientId]: model=${req.model}, " +
                    "stream=${req.stream}, msgs=${req.messages.size}, chars=$promptChars, " +
                    "session=${req.sessionId?.ifEmpty { null } ?: "(stateless)"}",
            )

            val responseId = "chatcmpl-${entry.id}"
            // Clock starts BEFORE engine acquire + session resolve: a cold
            // engine build and a full-history prefill are user-observable
            // latency. The tracker stamps the first-chunk time when the first
            // token arrives, so the UI can show the prefill vs decode split.
            RequestTracker.markStarted(entry.id)
            val temp = req.temperature ?: Settings.temperature(appContext)
            val topK = req.topK ?: Settings.topK(appContext)
            val topP = req.topP ?: Settings.topP(appContext)

            // Acquire + resolve are serialized by the outer inferenceMutex
            // but sit outside the timeout (timeout covers generation only).
            val acquired = try {
                engineRegistry.acquire(req.model, req.maxTokens)
            } catch (e: IllegalArgumentException) {
                // Strict model resolution (defense in depth — validation
                // already rejects unknown IDs before enqueue).
                LogManager.e("ChatRoute", "Engine acquire rejected for ${req.model}", e)
                RequestTracker.markCompleted(entry.id, error = "unknown_model")
                call.respond(
                    HttpStatusCode.BadRequest,
                    ErrorResponse(ErrorDetails(
                        message = e.message ?: "Unknown model",
                        type = "invalid_request_error",
                        code = 400,
                    )),
                )
                return@post
            } catch (e: Throwable) {
                LogManager.e("ChatRoute", "Engine acquire failed for ${req.model}", e)
                RequestTracker.markCompleted(entry.id, error = "engine_acquire: ${e.message ?: e.javaClass.simpleName}")
                val (httpStatus, details) = liteRtAcquireEnvelope(req.model, e)
                call.respond(httpStatus, RichErrorResponse(details))
                return@post
            }

            val resolvedLocal = try {
                sessionManager.resolve(req, acquired, temp, topK, topP)
            } catch (e: IllegalArgumentException) {
                LogManager.e("ChatRoute", "Session resolve failed for #${entry.id}", e)
                RequestTracker.markCompleted(entry.id, error = "resolve: ${e.message ?: e.javaClass.simpleName}")
                call.respond(
                    HttpStatusCode.BadRequest,
                    ErrorResponse(ErrorDetails(
                        message = e.message ?: "Invalid request",
                        type = "invalid_request_error",
                        code = 400,
                    )),
                )
                return@post
            }
            resolved = resolvedLocal

            if (req.stream) {
                call.response.cacheControl(CacheControl.NoCache(null))
                call.respondBytesWriter(contentType = ContentType.Text.EventStream) {
                    streamWriter = this@respondBytesWriter
                    val acc = RequestTracker.accumulatorFor(entry.id)
                    withTimeout(timeoutMs) {
                    acquireWakeLock(timeoutMs) {
                        try {
                            runInferenceStreaming(
                                conversation = resolvedLocal.conversation,
                                prompt = resolvedLocal.prompt,
                                writer = this@respondBytesWriter,
                                responseId = responseId,
                                modelName = req.model,
                                gson = gson,
                                serviceScope = serviceScope,
                                onChunk = { chunk -> acc.add(chunk.length) },
                            )
                        } finally {
                            acc.flush()
                        }
                    }
                    }
                }
            } else {
                var result: LlmMessage? = null
                withTimeout(timeoutMs) {
                    acquireWakeLock(timeoutMs) {
                        result = withContext(Dispatchers.Default) {
                            resolvedLocal.conversation.sendMessage(resolvedLocal.prompt, emptyMap())
                        }
                    }
                }
                val finalMsg = result!!
                val responseText = LlmMessageConverter.messageText(finalMsg)
                RequestTracker.recordChunk(entry.id, responseText)
                call.respond(ChatResponse(
                    id = responseId,
                    `object` = "chat.completion",
                    created = System.currentTimeMillis() / 1000,
                    model = req.model,
                    choices = listOf(Choice(
                        index = 0,
                        message = Message(role = "assistant", content = stringContent(responseText)),
                        finishReason = "stop",
                    )),
                ))
            }
            inferenceOk = true
            RequestTracker.markCompleted(entry.id)
            lastActivityAt.set(System.currentTimeMillis())
        } catch (te: TimeoutCancellationException) {
            LogManager.e("ChatRoute", "Request #${entry.id} timed out after ${timeoutMs} ms")
            try { resolved?.conversation?.cancelProcess() } catch (_: Exception) {}
            RequestTracker.markCompleted(entry.id, error = "timeout after ${timeoutMs} ms")
            val w = streamWriter
            if (w != null) {
                writeSseError(w, "Inference timeout", "timeout", 408, gson)
            } else {
                try {
                    call.respond(
                        HttpStatusCode.RequestTimeout,
                        ErrorResponse(ErrorDetails("Inference timeout", "timeout", 408)),
                    )
                } catch (_: Exception) { /* stream already started */ }
            }
        } catch (ce: kotlinx.coroutines.CancellationException) {
            try { resolved?.conversation?.cancelProcess() } catch (_: Exception) {}
            RequestTracker.markCompleted(entry.id, cancelled = true)
            throw ce
        } catch (e: Exception) {
            LogManager.e("ChatRoute", "Request #${entry.id} error", e)
            RequestTracker.markCompleted(entry.id, error = e.message ?: e.javaClass.simpleName)
            val w = streamWriter
            if (w != null) {
                writeSseError(w, e.message ?: "Unknown error", "server_error", 500, gson)
            } else {
                try {
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        ErrorResponse(ErrorDetails(
                            message = e.message ?: "Unknown error",
                            type = "server_error",
                            code = 500,
                        )),
                    )
                } catch (_: Exception) { /* stream already started */ }
            }
        } finally {
            val r = resolved
            if (r != null) {
                if (r.isCached) {
                    if (inferenceOk) sessionManager.commit(r, req.messages)
                    else sessionManager.invalidate(r)
                } else {
                    sessionManager.closeIfStateless(r)
                }
            }
        }
        } // inferenceMutex.withLock
    }
}

private suspend fun runInferenceStreaming(
    conversation: com.google.ai.edge.litertlm.Conversation,
    prompt: LlmMessage,
    writer: ByteWriteChannel,
    responseId: String,
    modelName: String,
    gson: Gson,
    serviceScope: CoroutineScope,
    onChunk: (String) -> Unit = {},
) {
    val writeMutex = Mutex()
    suspend fun safeWrite(s: String) {
        writeMutex.withLock {
            writer.writeStringUtf8(s)
            writer.flush()
        }
    }
    val heartbeat = serviceScope.launch {
        while (isActive) {
            delay(10_000L)
            try { safeWrite(": ka\n\n") } catch (_: Throwable) { return@launch }
        }
    }
    try {
        val initResp = StreamResponse(
            id = responseId,
            `object` = "chat.completion.chunk",
            created = System.currentTimeMillis() / 1000,
            model = modelName,
            choices = listOf(StreamChoice(0, StreamDelta(role = "assistant"), null)),
        )
        safeWrite("data: ${gson.toJson(initResp)}\n\n")

        // Reconstruct net text from LiteRT-LM emissions (see StreamReassembler):
        // the old length-sliced "cumulative" delta chopped spans and dropped
        // equal-length emissions — the streamed gibberish of 2026-10-01.
        val acc = StringBuilder()
        conversation.sendMessageAsync(prompt, emptyMap()).collect { msg ->
            val full = LlmMessageConverter.messageText(msg)
            val delta = StreamReassembler.next(acc.toString(), full)
            if (delta.isNotEmpty()) {
                acc.append(delta)
                onChunk(delta)
                val chunkResp = StreamResponse(
                    id = responseId,
                    `object` = "chat.completion.chunk",
                    created = System.currentTimeMillis() / 1000,
                    model = modelName,
                    choices = listOf(StreamChoice(0, StreamDelta(content = delta), null)),
                )
                safeWrite("data: ${gson.toJson(chunkResp)}\n\n")
            }
        }
        val finalResp = StreamResponse(
            id = responseId,
            `object` = "chat.completion.chunk",
            created = System.currentTimeMillis() / 1000,
            model = modelName,
            choices = listOf(StreamChoice(0, StreamDelta(), "stop")),
        )
        safeWrite("data: ${gson.toJson(finalResp)}\n\n")
        safeWrite("data: [DONE]\n\n")
    } finally {
        heartbeat.cancel()
    }
}

private suspend fun writeSseError(
    writer: ByteWriteChannel,
    message: String,
    type: String,
    code: Int,
    gson: Gson,
) {
    try {
        val json = gson.toJson(ErrorResponse(ErrorDetails(message, type, code)))
        writer.writeStringUtf8("data: $json\n\n")
        writer.writeStringUtf8("data: [DONE]\n\n")
        writer.flush()
    } catch (_: java.io.IOException) {
        /* Client gone; nothing actionable. */
    } catch (_: Exception) {
        /* Defensive: never let error-reporting itself throw out of a catch arm. */
    }
}

/**
 * Build the LiteRT engine-init envelope. The acquire step throws when the
 * model file is missing, the hash mismatches, the SoC marker doesn't match,
 * or the NPU delegate isn't loadable — surface a structured response with
 * concrete next steps instead of a bare 500.
 */
internal fun liteRtAcquireEnvelope(modelId: String, t: Throwable): Pair<HttpStatusCode, RichErrorDetails> {
    return HttpStatusCode.ServiceUnavailable to RichErrorDetails(
        message = "LiteRT-LM engine failed to initialise for '$modelId': " +
            "${t.message ?: t.javaClass.simpleName}.",
        type = "litert_engine_failed",
        code = "LITERT_INIT_FAILED",
        actionable = true,
        nextSteps = listOf(
            "Open the PixelUnlockGPU app and confirm the model is downloaded and hash-verified",
            "Confirm this device is a Pixel 10 (Tensor G5) — the model only runs there",
            "Check /health last_load_error for the sanitized failure reason",
        ),
    )
}
