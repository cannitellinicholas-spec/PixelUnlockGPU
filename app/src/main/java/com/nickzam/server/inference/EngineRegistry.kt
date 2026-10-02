// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference

import android.content.Context
import android.util.LruCache
import com.google.ai.edge.litertlm.Conversation
import com.nickzam.server.AVAILABLE_MODELS
import com.nickzam.server.Backend
import com.nickzam.server.LogManager
import com.nickzam.server.ModelInfo
import com.nickzam.server.findModelInfo
import com.nickzam.server.inference.litert.LiteRtEngine
import com.nickzam.server.inference.litert.LiteRtEngineBuilder
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Engine cache + factory. The catalog declares the backend per-model (see
 * [Backend]); this registry honors that declaration and either returns a
 * cached engine, builds a fresh one, or throws a structured error.
 *
 * Strictness rules (spec):
 * - Unknown model IDs fail explicitly — never mapped to "the only model".
 * - No silent CPU/GPU fallback for NPU entries.
 * - SHA-256 is verified before load; mismatch refuses the load. Verified
 *   files are memoized by (path, size, mtime) so a 3.1 GB re-hash doesn't
 *   run on every request.
 * - The last sanitized load error is retained for `/health`.
 *
 * Only one MODEL is resident at a time (acquiring a different model evicts
 * the previous one's engines — each bundle is ~3 GB resident); the cache
 * holds up to [maxResidentLiteRt] engines to cover multiple KV budgets of
 * the same model.
 */
class EngineRegistry(
    private val appContext: Context,
    private val maxResidentLiteRt: Int = 2,
) {

    /**
     * One step in the engine-init record. Kept so `/health` can surface
     * per-engine init metadata: each cached engine records the single
     * attempt that built it (label = backend name; result = "ok").
     */
    data class BackendAttempt(
        val backend: String,
        val result: String,
        val durationMs: Long,
    )

    /**
     * Public view of a cached engine entry — surfaced via /health.
     * [cacheKey] stays a flat string so the `engines[].key` JSON field
     * doesn't break monitoring clients.
     */
    data class CachedEngineInfo(
        val cacheKey: String,
        val backend: String,
        val attempts: List<BackendAttempt>,
    )

    private data class LiteRtCacheEntry(
        val engine: LiteRtEngine,
        val attempts: List<BackendAttempt>,
    )

    /**
     * LiteRT-LM engine LRU. Keyed by [EngineKey] because LiteRT-LM's
     * `EngineConfig.maxNumTokens` is the *total* KV-cache budget (input +
     * output); reusing an engine built with a larger budget for a request that
     * asked for less would let the model overgenerate.
     *
     * Conversations attached to an evicted engine MUST be torn down first —
     * a conversation outliving its parent engine is undefined behavior on
     * the native side. The [onLiteRtEvicted] callback handles that
     * coordination; the session manager registers it before serving traffic.
     */
    private val liteRtEngines = object : LruCache<EngineKey, LiteRtCacheEntry>(maxResidentLiteRt) {
        override fun entryRemoved(
            evicted: Boolean,
            key: EngineKey?,
            oldValue: LiteRtCacheEntry?,
            newValue: LiteRtCacheEntry?,
        ) {
            if (oldValue == null || key == null) return
            // Notify listeners FIRST so they can close conversations
            // referencing this engine before we tear the engine down.
            if (oldValue !== newValue) {
                try { onLiteRtEvicted?.invoke(key) } catch (_: Throwable) {}
                try {
                    oldValue.engine.close()
                    if (evicted) LogManager.i("EngineRegistry", "Evicted engine: ${key.asString()}")
                } catch (e: Exception) {
                    LogManager.e("EngineRegistry", "Error closing evicted engine ${key.asString()}", e)
                }
            }
        }
    }

    /**
     * Per-engine conversation lookup. Filled by the session manager; at most
     * one live [Conversation] per engine (enforced natively by LiteRT-LM).
     */
    val activeConversations = ConcurrentHashMap<EngineKey, Conversation>()

    /**
     * Listener invoked when a LiteRT engine entry is removed (eviction or
     * explicit drop). Registered by the session manager so it can flush its
     * conversation cache before the engine is closed.
     */
    @Volatile var onLiteRtEvicted: ((cacheKey: EngineKey) -> Unit)? = null

    /** Last sanitized load failure, surfaced via `/health`. Null when clean. */
    @Volatile var lastLoadError: String? = null
        private set

    @Volatile private var lastLoadErrorAt: Long = 0L

    fun lastLoadErrorAt(): Long = lastLoadErrorAt

    /** Memoized (path, size, mtime) triples that already passed SHA-256. */
    private val verifiedFiles = ConcurrentHashMap.newKeySet<String>()

    /** Strict resolution: unknown IDs throw — the route maps this to a 400. */
    fun resolveModelInfo(modelId: String): ModelInfo {
        return findModelInfo(modelId)
            ?: throw IllegalArgumentException(
                "Unknown model '$modelId'. This server only serves " +
                    AVAILABLE_MODELS.joinToString(", ") { "\"${it.id}\"" } + "."
            )
    }

    /** File on disk backing [info]. App-private external files dir. */
    fun modelFileFor(info: ModelInfo): File =
        File(appContext.getExternalFilesDir(null), info.filename.ifBlank { "${info.id}.litertlm" })

    /**
     * Get-or-build an engine for [modelId]. The KV-cache budget is the
     * device context window ([KV_BUDGET_TOKENS]) for every engine — the
     * per-request `max_tokens` caps generation, not the cache, so one
     * maximally-sized engine serves every request (this is what lets a
     * desktop client like TypingMind drive long chats without prefill
     * rejections). [maxTokens] is accepted for call-site compatibility
     * and no longer sizes the cache.
     * The backend is whatever the catalog declares for this model. No
     * fallback; init failures propagate verbatim and are retained for
     * `/health`.
     */
    fun acquire(modelId: String, maxTokens: Int?): AcquiredEngine.LiteRt {
        val info = resolveModelInfo(modelId)
        // Context window is a user-tunable KV-cache budget (see
        // Settings.contextTokens). It is part of the engine key, so each
        // distinct budget is its own cached engine; changing it and reloading
        // builds at the new size. Defaults to the 32k ceiling.
        val kvBudget = com.nickzam.server.Settings.contextTokens(appContext)
            .coerceAtMost(KV_BUDGET_TOKENS)
        val cacheKey = EngineKey(info.id, kvBudget, info.backend)
        liteRtEngines.get(cacheKey)?.let {
            return AcquiredEngine.LiteRt(it.engine, cacheKey, it.attempts)
        }

        val modelFile = modelFileFor(info)
        if (!modelFile.exists()) {
            val msg = "Model file not found: ${modelFile.name}. Download or import it first."
            recordLoadError(msg)
            throw IllegalStateException(msg)
        }

        // Catalog-declared SoC check for NPU models: don't even try to init
        // if the device clearly doesn't match the compiled-for SoC.
        if (info.backend == Backend.LITERT_NPU) {
            val marker = info.requiredSocMarker
            if (marker != null && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                val soc = android.os.Build.SOC_MODEL?.lowercase().orEmpty()
                if (!soc.contains(marker.lowercase())) {
                    val msg = "Model '${info.id}' requires SoC marker '$marker' " +
                        "but this device reports '$soc'."
                    recordLoadError(msg)
                    throw IllegalStateException(msg)
                }
            }
            // On this app's target device (Pixel 10 / Tensor G5) an NPU
            // engine build ABORTS the process: the bundled
            // libLiteRtDispatch_GoogleTensor.so requires
            // /vendor/lib64/libedgetpu_litert.so, which is invisible to an
            // app linker namespace, and LiteRT calls abort() rather than
            // throwing ("No usable Dispatch runtime found"). The failure
            // happens natively, before anything can catch it, so the only
            // safe behavior is to refuse NPU builds outright. The entry
            // stays listed; the GPU catalog entry is the working path.
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S &&
                android.os.Build.SOC_MODEL?.lowercase()?.contains("tensor g5") == true
            ) {
                val msg = "Model '${info.id}' targets the Tensor G5 TPU, but the NPU " +
                    "delegate is not loadable from an app on stock firmware (vendor " +
                    "TPU library inaccessible). Use the GPU model instead."
                recordLoadError(msg)
                throw IllegalStateException(msg)
            }
        }

        // SHA-256 before load (memoized by path+size+mtime). Fail-closed.
        val memoKey = "${modelFile.absolutePath}|${modelFile.length()}|${modelFile.lastModified()}"
        if (!verifiedFiles.contains(memoKey)) {
            LogManager.i("EngineRegistry", "Verifying SHA-256 for ${modelFile.name} (${modelFile.length()} bytes)")
            val refusal = Sha256.verifyOrRefusal(modelFile, info.sha256)
            if (refusal != null) {
                recordLoadError(refusal)
                throw IllegalStateException(refusal)
            }
            verifiedFiles.add(memoKey)
            LogManager.i("EngineRegistry", "SHA-256 verified for ${modelFile.name}")
        }

        // One model resident at a time: each bundle is ~3 GB, so before
        // building for a different model, drop every cached engine of other
        // models. Conversations tied to them are torn down first via
        // [onLiteRtEvicted]. Same-model entries (other KV budgets) still
        // ride the LRU.
        val otherModels = liteRtEngines.snapshot().keys.filter { it.modelId != info.id }
        if (otherModels.isNotEmpty()) {
            LogManager.i("EngineRegistry", "Switching to ${info.id}: evicting ${otherModels.size} engine(s) of other models")
            otherModels.forEach { liteRtEngines.remove(it) }
        }

        val nativeLibDir = appContext.applicationInfo.nativeLibraryDir.orEmpty()
        val t0 = System.nanoTime()
        val native = try {
            LiteRtEngineBuilder.build(modelFile, kvBudget, info.backend, nativeLibDir)
        } catch (e: Exception) {
            val ms = (System.nanoTime() - t0) / 1_000_000
            val msg = "Failed to initialize ${info.backend.name} engine for ${info.id}: " +
                "${e.message ?: e.javaClass.simpleName} (${ms}ms)"
            recordLoadError(msg)
            throw IllegalStateException(msg, e)
        }
        val attempt = BackendAttempt(info.backend.name, "ok", (System.nanoTime() - t0) / 1_000_000)
        val wrapped = LiteRtEngine(info.id, info.backend, native, cacheKey)
        val entry = LiteRtCacheEntry(wrapped, listOf(attempt))
        try {
            liteRtEngines.put(cacheKey, entry)
            lastLoadError = null
            LogManager.i("EngineRegistry", "Engine ${cacheKey.asString()} ready (${info.backend.name}, ${attempt.durationMs}ms)")
        } catch (e: Exception) {
            try { wrapped.close() } catch (_: Exception) {}
            throw e
        }
        return AcquiredEngine.LiteRt(wrapped, cacheKey, listOf(attempt))
    }

    private fun recordLoadError(msg: String) {
        lastLoadError = msg
        lastLoadErrorAt = System.currentTimeMillis()
    }

    /**
     * Snapshot for /health. Converts each [EngineKey] to its stable string
     * form so the `engines[].key` field remains a flat string in JSON.
     */
    fun snapshot(): List<CachedEngineInfo> =
        liteRtEngines.snapshot().map { (key, v) ->
            CachedEngineInfo(key.asString(), v.engine.backend.name, v.attempts)
        }

    fun engineCount(): Int = liteRtEngines.size()

    /** Idle eviction path — drops all cached LiteRT engines. */
    fun evictAllLiteRt(): Int {
        val n = liteRtEngines.size()
        liteRtEngines.evictAll()
        return n
    }

    /** Memory-pressure path — keep at most [keep] cached. */
    fun trimLiteRtTo(keep: Int): Int {
        val before = liteRtEngines.size()
        liteRtEngines.trimToSize(keep)
        return before - liteRtEngines.size()
    }

    companion object {
        /**
         * Total KV-cache budget (prompt + generated tokens) every engine is
         * built with. The pinned LiteRT-LM bundles (Gemma 4 E2B/E4B) support
         * up to 32k context; 32768 is the largest power-of-two window, which
         * is what a desktop client like TypingMind wants and matches
         * [com.nickzam.server.ChatValidation.MAX_TOKENS_CEILING]. Larger
         * budgets cost GPU/CPU memory proportional to the window, so this is
         * the ceiling, not a floor.
         */
        const val KV_BUDGET_TOKENS = 32_768
    }

    /** Drop a specific entry (used when an engine is found to be wedged). */
    fun dropLiteRt(cacheKey: EngineKey) {
        liteRtEngines.remove(cacheKey)
    }

    /** Acquired-engine holder. LiteRT-only in the MVP. */
    sealed class AcquiredEngine {
        data class LiteRt(
            val engine: LiteRtEngine,
            val cacheKey: EngineKey,
            val attempts: List<BackendAttempt>,
        ) : AcquiredEngine()
    }
}
