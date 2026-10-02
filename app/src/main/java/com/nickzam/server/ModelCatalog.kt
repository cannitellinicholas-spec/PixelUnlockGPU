// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import android.os.Build

/**
 * Which inference engine + delegate a model expects. Declared per-model in
 * the catalog so the server never guesses from a global setting or falls
 * back through a chain — the catalog is the source of truth and the engine
 * layer either honors it or fails loudly.
 *
 * - [LITERT_CPU] → LiteRT-LM with `Backend.CPU()`.
 * - [LITERT_GPU] → LiteRT-LM with `Backend.GPU()`. Strict — no fallback.
 * - [LITERT_NPU] → LiteRT-LM with `Backend.NPU(nativeLibraryDir)`. Requires
 *   the vendor delegate `.so` to be present. On Pixel 10 this is the
 *   Tensor G5 TPU via `libLiteRtDispatch_GoogleTensor.so`.
 *
 * Ported from localLLM's `Backend` minus `AICORE` (the MVP has no AICore
 * engine; see docs/device-model-matrix.md).
 */
enum class Backend { LITERT_CPU, LITERT_GPU, LITERT_NPU }

/** The original stable model ID this server advertises (Tensor G5 NPU bundle). */
const val STABLE_MODEL_ID = "gemma-4-e2b-it-tpu-g5"

/** GPU-delegate bundle ID — the working path on stock Tensor G5 firmware. */
const val GPU_MODEL_ID = "gemma-4-e4b-it-gpu"

/**
 * Metadata for the served model bundle: stable public ID, artifact hash,
 * compatible SoC, backend, context limit, download metadata, license notice.
 */
data class ModelInfo(
    val id: String,
    val name: String,
    val description: String,
    /** Pinned download URL (commit-pinned; never a floating branch). */
    val url: String,
    val filename: String,
    /**
     * The engine + delegate this model must run on. No default — every entry
     * must declare it explicitly so an accidental omission surfaces at
     * compile time.
     */
    val backend: Backend,
    /**
     * Lowercase hex SHA-256 of the file at [url]. Required: hash mismatch
     * (or a missing hash) refuses the load. Never null for served entries.
     */
    val sha256: String,
    /**
     * Lowercase SoC marker required by an NPU-compiled `.litertlm`.
     * The model only runs on a device whose [Build.SOC_MODEL] contains this
     * marker. Used for load gating; does NOT drive backend selection.
     */
    val requiredSocMarker: String? = null,
    /**
     * Verified context-token limit, or null when unverified. The server never
     * advertises an unverified number — null means "metadata claim only".
     */
    val contextTokens: Int? = null,
    /** Expected artifact size in bytes (from the pinned source), if known. */
    val expectedSizeBytes: Long? = null,
    /** Source/license notice surfaced in UI and docs. */
    val licenseNotice: String = "",
)

/**
 * Human-readable SoC label for [ModelInfo.requiredSocMarker], or `null` when
 * the model isn't NPU-gated.
 */
fun ModelInfo.npuSocLabel(): String? = when (requiredSocMarker?.lowercase()) {
    "sm8550" -> "Snapdragon 8 Gen 2 (SM8550)"
    "sm8650" -> "Snapdragon 8 Gen 3 (SM8650)"
    "sm8750" -> "Snapdragon 8 Elite (SM8750)"
    "sm8850" -> "Snapdragon 8 Elite Gen 5 (SM8850)"
    "mt6989" -> "Dimensity 9300 (MT6989)"
    "mt6991" -> "Dimensity 9400 (MT6991)"
    "mt6993" -> "Dimensity 9500 (MT6993)"
    "tensor g5" -> "Google Tensor G5 (Pixel 10)"
    null -> null
    else -> requiredSocMarker.uppercase()
}

/**
 * Pure SoC-marker check over an explicit SoC string, so the rule is unit
 * testable on the JVM. The check is intentionally a case-insensitive
 * substring — manufacturers prefix the marker inconsistently.
 */
fun socMarkerMatches(socModel: String?, marker: String?): Boolean {
    if (marker == null) return true
    if (socModel == null) return false
    return socModel.lowercase().contains(marker.lowercase())
}

/**
 * True when the model isn't NPU-gated, or the current device's SoC string
 * contains [ModelInfo.requiredSocMarker].
 *
 * `Build.SOC_MODEL` is only populated on API 31+; on older devices the
 * check returns false for NPU-gated models, which is the right default
 * (NPU support is API 31+ anyway).
 */
fun ModelInfo.matchesCurrentSoc(): Boolean {
    val marker = requiredSocMarker ?: return true
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return false
    return socMarkerMatches(Build.SOC_MODEL, marker)
}

/** True on Pixel 10 family devices (Box's `isPixel10` rule, generalized). */
fun isPixel10Device(): Boolean {
    return Build.MODEL != null && Build.MODEL.lowercase().contains("pixel 10")
}

/**
 * Built-in model catalog.
 *
 * Provenance (see docs/device-model-matrix.md): artifact coordinates
 * (repo, commit, file, size, SHA-256) come from the pinned-commit bytes
 * (re-verified against the HuggingFace LFS metadata); context figures from
 * upstream metadata stay UNVERIFIED until Gate 5 measures them.
 */
val AVAILABLE_MODELS: List<ModelInfo> = listOf(
    ModelInfo(
        id = STABLE_MODEL_ID,
        name = "Gemma 4 E2B (Tensor G5)",
        description = "Instruction-tuned Gemma 4 E2B in LiteRT-LM format, served on the " +
            "Tensor G5 TPU via the bundled GoogleTensor dispatch library. ~3.1 GB. " +
            "Pixel 10 only. NOTE: the TPU dispatch path aborts on stock " +
            "Android 17 (needs /vendor/lib64/libedgetpu_litert.so, which an " +
            "app linker namespace cannot see) — kept listed for a future " +
            "vendor-access path; use the GPU entry for a working load.",
        url = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm" +
            "/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it_Google_Tensor_G5.litertlm",
        filename = "gemma-4-e2b-it-tpu-g5.litertlm",
        backend = Backend.LITERT_NPU,
        // x-linked-etag at the pinned commit; the app verifies the
        // downloaded bytes against it fail-closed before load.
        sha256 = BuildConfig.MODEL_SHA256,
        // Measured on Pixel 10 Pro XL: ro.soc.model is "Tensor G5".
        // "laguna" lives in ro.board.platform, which has no public
        // Build API — so the gate matches the marketing string.
        requiredSocMarker = "tensor g5",
        // Upstream metadata claims 32K; unverified on-device — stays null.
        contextTokens = null,
        expectedSizeBytes = 3_113_545_589L,
        licenseNotice = "Gemma Terms of Use: https://ai.google.dev/gemma/terms. " +
            "Artifact: litert-community/gemma-4-E2B-it-litert-lm @ b3ca0d2.",
    ),
    ModelInfo(
        id = GPU_MODEL_ID,
        name = "Gemma 4 E4B (GPU)",
        description = "Instruction-tuned Gemma 4 E4B in LiteRT-LM GPU format, served on the " +
            "OpenCL GPU delegate (libLiteRtClGlAccelerator, bundled in the " +
            "AAR — no vendor namespace needed). ~2.97 GB. Works on any " +
            "arm64 Android device with an OpenCL GPU.",
        url = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm" +
            "/resolve/2eee7ac325f20eb8c9ac1d0e972f7c84663062da/gemma-4-E4B-it-gpu.litertlm",
        filename = "gemma-4-e4b-it-gpu.litertlm",
        backend = Backend.LITERT_GPU,
        // LFS sha256 at the pinned commit; verified fail-closed before load.
        sha256 = "4912bb5a9c30993c51a7711f763212077458529312175df0573a78323a2bb7ff",
        // GPU delegate is SoC-agnostic — no gate.
        requiredSocMarker = null,
        contextTokens = null,
        expectedSizeBytes = 2_969_059_328L,
        licenseNotice = "Gemma Terms of Use: https://ai.google.dev/gemma/terms. " +
            "Artifact: litert-community/gemma-4-E4B-it-litert-lm @ 2eee7ac.",
    ),
)

/** The catalog entry served by default when nothing has been selected. */
val DEFAULT_MODEL: ModelInfo get() = findModelInfo(STABLE_MODEL_ID) ?: AVAILABLE_MODELS.first()

/** Strict lookup: unknown IDs return null and the caller must fail explicitly. */
fun findModelInfo(id: String): ModelInfo? = AVAILABLE_MODELS.firstOrNull { it.id == id }
