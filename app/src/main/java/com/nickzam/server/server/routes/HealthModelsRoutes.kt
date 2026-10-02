package com.nickzam.server.server.routes

import android.content.Context
import android.os.Build
import com.nickzam.server.AVAILABLE_MODELS
import com.nickzam.server.BuildConfig
import com.nickzam.server.LogManager
import com.nickzam.server.ModelData
import com.nickzam.server.ModelListResponse
import com.nickzam.server.findModelInfo
import com.nickzam.server.inference.EngineRegistry
import com.nickzam.server.server.auth.authorize
import io.ktor.server.response.respond
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/**
 * `GET /v1/models` — OpenAI-shaped model index.
 *
 * Lists **installed** models only: the stable catalog entry appears iff its
 * `.litertlm` bundle is on disk. Anything else on disk (unknown files) is
 * not loadable under strict model resolution, so it is not advertised.
 */
fun Route.modelsRoute(
    appContext: Context,
    engineRegistry: EngineRegistry,
) {
    get("/v1/models") {
        if (!authorize(call, appContext)) return@get
        val data = AVAILABLE_MODELS.mapNotNull { info ->
            val file = engineRegistry.modelFileFor(info)
            if (!file.exists()) return@mapNotNull null
            ModelData(
                id = info.id,
                created = file.lastModified() / 1000,
                ownedBy = "pixelunlockgpu-" + info.backend.name.lowercase(),
            )
        }
        call.respond(ModelListResponse(data = data))
    }
}

/**
 * `GET /health` — liveness + readiness + backend evidence.
 *
 * Reports server state and port, selected model installed/loaded state,
 * the confirmed backend (`GOOGLE_TENSOR_NPU` only when a healthy NPU engine
 * is cached — otherwise `unknown`), runtime version + artifact hash prefix,
 * queue state, last sanitized load error, and app/device identity.
 *
 * Auth: when an API key is configured, health requires it — the response
 * exposes device/runtime details, so remote health access is keyed. With no
 * key configured (loopback development default), health stays open for
 * monitoring probes.
 *
 * Never emits prompts, tokens, full model paths, or filesystem details.
 */
fun Route.healthRoute(
    engineRegistry: EngineRegistry,
    appContext: Context,
    inferenceMutex: kotlinx.coroutines.sync.Mutex,
) {
    get("/health") {
        if (!authorize(call, appContext)) return@get
        call.respond(healthSnapshot(engineRegistry, appContext))
    }

    /**
     * `POST /health/warm?model=<id>` — force the named engine to load and
     * run a 1-token generation so the JNI runtime is warm. Lets deploy/test
     * scripts wait for first-token-ready instead of guessing.
     *
     * `model` defaults to the selected catalog model when omitted; unknown
     * IDs fail explicitly. Bounded by a 60s timeout so a wedged engine init
     * doesn't hang a monitoring poll.
     */
    post("/health/warm") {
        if (!authorize(call, appContext)) return@post
        val modelParam = call.request.queryParameters["model"]?.trim()
        val modelId = if (modelParam.isNullOrEmpty())
            com.nickzam.server.Settings.selectedModelId(appContext) else modelParam
        val t0 = System.nanoTime()
        try {
            engineRegistry.resolveModelInfo(modelId)
        } catch (e: IllegalArgumentException) {
            call.respond(
                io.ktor.http.HttpStatusCode.BadRequest,
                mapOf(
                    "model" to modelId,
                    "status" to "unknown_model",
                    "error" to (e.message ?: "unknown model"),
                    "ms" to (System.nanoTime() - t0) / 1_000_000,
                ),
            )
            return@post
        }
        try {
            val acquired = withTimeout(60_000L) {
                engineRegistry.acquire(modelId, /*maxTokens=*/null)
            }
            // Hold the inference mutex briefly so warm-up doesn't collide
            // with a concurrent /v1/chat/completions on the same engine —
            // LiteRT-LM doesn't support two live conversations per engine.
            withTimeout(60_000L) {
                inferenceMutex.withLock {
                    val conv = com.nickzam.server.inference.litert.LlmMessageConverter.createConversation(
                        engine = acquired.engine.native,
                        temperature = 0f,
                        topK = 1,
                        topP = 1f,
                        systemText = null,
                        initial = emptyList(),
                    )
                    try {
                        withContext(Dispatchers.Default) {
                            conv.sendMessage(
                                com.google.ai.edge.litertlm.Message.user("Hi"),
                                emptyMap(),
                            )
                        }
                    } finally {
                        try { conv.close() } catch (_: Exception) {}
                    }
                }
            }
            call.respond(mapOf(
                "model" to modelId,
                "status" to "warm",
                "engine_loaded" to true,
                "ms" to (System.nanoTime() - t0) / 1_000_000,
            ))
        } catch (_: TimeoutCancellationException) {
            call.respond(
                io.ktor.http.HttpStatusCode.GatewayTimeout,
                mapOf(
                    "model" to modelId,
                    "status" to "timeout",
                    "ms" to (System.nanoTime() - t0) / 1_000_000,
                ),
            )
        } catch (e: Throwable) {
            LogManager.e("HealthRoute", "Warm-up failed for $modelId", e)
            call.respond(
                io.ktor.http.HttpStatusCode.ServiceUnavailable,
                mapOf(
                    "model" to modelId,
                    "status" to "error",
                    "error" to (e.message ?: e.javaClass.simpleName),
                    "ms" to (System.nanoTime() - t0) / 1_000_000,
                ),
            )
        }
    }
}

private fun healthSnapshot(
    engineRegistry: EngineRegistry,
    appContext: Context,
): Map<String, Any?> {
    val snapshot = engineRegistry.snapshot()
    // Engines with a fully-successful build attempt are proof of a live backend.
    fun okModelsFor(backendName: String): Set<String> = snapshot
        .filter { e -> e.backend == backendName && e.attempts.isNotEmpty() && e.attempts.all { it.result == "ok" } }
        .map { e -> e.cacheKey.substringBefore('_') }
        .toSet()
    val npuReady = okModelsFor("LITERT_NPU")
    val gpuReady = okModelsFor("LITERT_GPU")
    val cpuReady = okModelsFor("LITERT_CPU")
    val actualBackend = when {
        npuReady.isNotEmpty() -> "GOOGLE_TENSOR_NPU"
        gpuReady.isNotEmpty() -> "GPU_OPENCL"
        cpuReady.isNotEmpty() -> "CPU"
        engineRegistry.lastLoadError != null -> "error"
        else -> "unknown"
    }
    // Per-catalog-model evidence: installed/loaded + declared + proven backend.
    val models = AVAILABLE_MODELS.map { info ->
        val file = engineRegistry.modelFileFor(info)
        val installed = file.exists()
        val loaded = snapshot.any { it.cacheKey.startsWith("${info.id}_") }
        mapOf(
            "id" to info.id,
            "declared_backend" to info.backend.name,
            "installed" to installed,
            "loaded" to loaded,
            "backend_proven" to when {
                info.id in npuReady -> "GOOGLE_TENSOR_NPU"
                info.id in gpuReady -> "GPU_OPENCL"
                info.id in cpuReady -> "CPU"
                loaded -> "declared_${info.backend.name.lowercase()}"
                else -> null
            },
            "size_bytes" to if (installed) file.length() else null,
            "sha256_prefix" to info.sha256.take(16),
            // Verified context limit only; null = unverified (see manifest).
            "context_tokens" to info.contextTokens,
        )
    }
    val selectedId = com.nickzam.server.Settings.selectedModelId(appContext)
    val selected = findModelInfo(selectedId) ?: AVAILABLE_MODELS.first()
    val selectedFile = engineRegistry.modelFileFor(selected)
    val selectedInstalled = selectedFile.exists()
    val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        Build.SOC_MODEL ?: "unknown"
    } else "unknown"
    val stats = com.nickzam.server.RequestTracker.stats.value
    val cur = com.nickzam.server.RequestTracker.current.value
    val totalCharsSinceStart = stats.totalOutputChars + (cur?.outputChars?.toLong() ?: 0L)
    val thermal = runCatching { com.nickzam.server.ThermalMonitor.read(appContext) }.getOrNull()
    return mapOf(
        "status" to "ok",
        "service" to "pixelunlockgpu-android",
        "version" to BuildConfig.VERSION_NAME,
        // Exposure of the API surface: loopback / tailscale (E2E encrypted) / lan.
        "access" to mapOf(
            "mode" to com.nickzam.server.Settings.accessMode(appContext).name.lowercase(),
            "tailnet_ip" to com.nickzam.server.Settings.tailscaleIp(),
            "encrypted" to (com.nickzam.server.Settings.accessMode(appContext) ==
                com.nickzam.server.Settings.AccessMode.TAILSCALE),
        ),
        "throughput" to mapOf(
            // LiteRT-LM 0.12.0 exposes no token/usage API; tokens estimated
            // at ~4 chars/token. total_tokens is since server start.
            "note" to "estimated: ~4 chars/token (no token API in LiteRT-LM)",
            "total_tokens_since_start" to (totalCharsSinceStart / 4),
            "total_output_chars_since_start" to totalCharsSinceStart,
            // avg_tokens_per_sec is end-to-end (includes cold-engine build +
            // prefill); decode_tokens_per_sec measures first-token→last and is
            // the number comparable to published GPU tok/s figures.
            "avg_tokens_per_sec" to stats.avgTokensPerSec,
            "avg_decode_tokens_per_sec" to stats.avgDecodeTokensPerSec,
            "avg_prefill_tokens_per_sec" to stats.avgPrefillTokensPerSec,
            "live_tokens_per_sec" to (cur?.tokensPerSec ?: 0f),
            "live_decode_tokens_per_sec" to (cur?.decodeTokensPerSec ?: 0f),
            "live_prefill_ms" to (cur?.prefillMs),
            "live_prefill_tokens_per_sec" to (cur?.prefillTokensPerSec ?: 0f),
            "generating" to (cur != null),
        ),
        "thermal" to if (thermal == null) null else mapOf(
            "gpu_temp_c" to thermal.gpuTempC,
            "status" to com.nickzam.server.ThermalMonitor.statusName(thermal.status),
            "headroom" to thermal.headroom,
            "sensor_source" to thermal.sensorSource,
        ),
        "server" to mapOf(
            "port" to com.nickzam.server.Settings.port(appContext),
            "loopback_only" to (com.nickzam.server.Settings.accessMode(appContext) ==
                com.nickzam.server.Settings.AccessMode.LOOPBACK),
        ),
        "model" to mapOf(
            "id" to selected.id,
            "installed" to selectedInstalled,
            "loaded" to snapshot.any { it.cacheKey.startsWith("${selected.id}_") },
            "size_bytes" to if (selectedInstalled) selectedFile.length() else null,
            "sha256_prefix" to selected.sha256.take(16),
            "context_tokens" to selected.contextTokens,
        ),
        "models" to models,
        "engine" to mapOf(
            "actual_backend" to actualBackend,
            "litert_version" to BuildConfig.LITERT_VERSION,
            "loaded_count" to engineRegistry.engineCount(),
            "last_load_error" to engineRegistry.lastLoadError,
        ),
        "engines" to snapshot.map { e ->
            mapOf(
                "key" to e.cacheKey,
                "backend" to e.backend,
                "attempts" to e.attempts.map {
                    mapOf(
                        "backend" to it.backend,
                        "result" to it.result,
                        "duration_ms" to it.durationMs,
                    )
                },
            )
        },
        "queue" to mapOf(
            "depth" to com.nickzam.server.RequestTracker.queue.value.size,
            "active" to (com.nickzam.server.RequestTracker.current.value != null),
            "max_depth" to com.nickzam.server.Settings.maxQueueDepth(appContext),
        ),
        "stats" to mapOf(
            "total_requests" to stats.totalRequests,
            "total_completed" to stats.totalCompleted,
            "total_errors" to stats.totalErrors,
            "total_cancelled" to stats.totalCancelled,
        ),
        "app" to mapOf(
            "version" to BuildConfig.VERSION_NAME,
            "version_code" to BuildConfig.VERSION_CODE,
        ),
        "device" to mapOf(
            "model" to (Build.MODEL ?: "unknown"),
            "soc" to soc,
        ),
    )
}
