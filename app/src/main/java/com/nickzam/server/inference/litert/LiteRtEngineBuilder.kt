// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference.litert

import com.google.ai.edge.litertlm.Backend as LiteRtBackend
import com.google.ai.edge.litertlm.Engine as LiteRtNativeEngine
import com.google.ai.edge.litertlm.EngineConfig
import com.nickzam.server.Backend
import com.nickzam.server.LogManager
import java.io.File

/**
 * Constructs LiteRT-LM engines on the catalog-declared backend. No fallback,
 * no chain — if the requested backend can't initialize, the failure
 * propagates verbatim so the user sees a misconfigured model or device
 * instead of a silent fall-through.
 *
 * Tensor-SoC primer: on Google Tensor (Pixel 6+), a direct CPU init can
 * SIGABRT inside `llm_litert_compiled_model_executor.cc` unless the JNI
 * library has first attempted (and gracefully failed) some other backend.
 * The NPU attempt below is the cheapest such warmup — without a vendor
 * delegate it fails fast at init but leaves the JNI lib in a state where the
 * subsequent CPU init succeeds. (Only applies to CPU/GPU requests; the MVP
 * NPU path never triggers it.)
 *
 * MVP scope: text-only, so `visionBackend`/`audioBackend` are null. Enabling
 * multimodal inputs later means passing explicit backends here (Box uses GPU
 * vision / CPU audio for Gemma 4) — never silently.
 */
object LiteRtEngineBuilder {

    fun build(
        modelFile: File,
        maxTokens: Int?,
        backend: Backend,
        nativeLibDir: String,
    ): LiteRtNativeEngine {
        if (TensorSoCDetector.isTensorSoc() &&
            !isTensorG5() &&
            (backend == Backend.LITERT_CPU || backend == Backend.LITERT_GPU)
        ) {
            primeTensorJniState(modelFile, nativeLibDir)
        }
        val native = backend.toLiteRt(nativeLibDir)
        return buildOne(modelFile, maxTokens, native)
    }

    private fun Backend.toLiteRt(nativeLibDir: String): LiteRtBackend = when (this) {
        Backend.LITERT_CPU -> LiteRtBackend.CPU()
        Backend.LITERT_GPU -> LiteRtBackend.GPU()
        Backend.LITERT_NPU -> LiteRtBackend.NPU(nativeLibDir)
    }

    private fun buildOne(
        modelFile: File,
        maxTokens: Int?,
        backend: LiteRtBackend,
    ): LiteRtNativeEngine {
        val cfg = EngineConfig(
            modelFile.absolutePath,
            backend,
            /*visionBackend=*/ null,
            /*audioBackend=*/ null,
            /*maxNumTokens=*/ maxTokens,
            /*maxNumImages=*/ null,
            /*cacheDir=*/ null,
        )
        return LiteRtNativeEngine(cfg).also { it.initialize() }
    }

    /**
     * Tensor G5 (Pixel 10) reports `ro.soc.model` = "Tensor G5". On this
     * SoC the NPU warmup below is NOT a safe no-op: the GoogleTensor
     * dispatch plugin aborts the process (SIGABRT) instead of throwing,
     * so priming via NPU would kill the app. GPU init directly works on
     * G5 (verified via the LiteRT-LM sample path), so priming is skipped
     * there entirely.
     */
    private fun isTensorG5(): Boolean {
        if (!android.os.Build.VERSION.SDK_INT.let { it >= android.os.Build.VERSION_CODES.S }) return false
        return android.os.Build.SOC_MODEL?.lowercase()?.contains("tensor g5") == true
    }

    /**
     * On Google Tensor (pre-G5), calling [LiteRtBackend.CPU] or
     * [LiteRtBackend.GPU] without first touching another backend
     * reproducibly fails. Attempting [LiteRtBackend.NPU] first throws (no
     * vendor delegate present) but leaves the JNI lib in a state where the
     * subsequent real init succeeds. The warmup engine is discarded — only
     * the JNI side effects are wanted.
     */
    private fun primeTensorJniState(modelFile: File, nativeLibDir: String) {
        try {
            val warmup = buildOne(modelFile, /*maxTokens=*/null, LiteRtBackend.NPU(nativeLibDir))
            try { warmup.close() } catch (_: Throwable) {}
            LogManager.i("LiteRtEngineBuilder", "Tensor JNI primer ran (unexpectedly succeeded; engine discarded)")
        } catch (e: Throwable) {
            // Expected — no NPU delegate on stock Tensor for a CPU/GPU model.
            // The primer side effects on the JNI library are what we want.
            LogManager.i("LiteRtEngineBuilder", "Tensor JNI primer ran (expected NPU fail: ${e.message ?: e.javaClass.simpleName})")
        }
    }
}
