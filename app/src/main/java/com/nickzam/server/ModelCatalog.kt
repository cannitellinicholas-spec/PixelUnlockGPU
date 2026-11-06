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
 * - [AICORE] → Gemini Nano via the on-device AICore service, reached through
 *   the ML Kit GenAI Prompt API (see NanoEngine). There is no in-process
 *   engine: the weights are owned by AICore, so this lane has no local
 *   `.litertlm` file and no SHA-256 gate (see [ModelInfo.hosted]).
 *
 * Ported from localLLM's `Backend` plus the AICORE lane (v0.1.8).
 */
enum class Backend { LITERT_CPU, LITERT_GPU, LITERT_NPU, AICORE }

/**
 * Catalog ID of the Tensor G5 NPU bundle. The entry stays listed for a
 * future vendor-access path, but it is NOT the served default: its NPU
 * dispatch aborts on stock Android 17, so [DEFAULT_MODEL] resolves to
 * [GPU_MODEL_ID] instead.
 */
const val STABLE_MODEL_ID = "gemma-4-e2b-it-tpu-g5"

/** GPU-delegate bundle ID — the working path on stock Tensor G5 firmware. */
const val GPU_MODEL_ID = "gemma-4-e4b-it-gpu"

/**
 * E4B standard bundle on the GPU backend, with the bundled MTP drafter
 * section: speculative decoding drafts several tokens ahead and the target
 * model verifies them in one pass, so decode no longer pays a full
 * weight-stream per token. Google's own model-card numbers for this exact
 * file on GPU (S26 Ultra): 21.9 → 46–49 tok/s decode.
 */
const val GPU_SPEC_MODEL_ID = "gemma-4-e4b-it-gpu-spec"

/**
 * The one real community int4 `.litertlm` found (UrielSalto's re-quant of
 * Gemma 4 E2B). CPU-format only — no GPU-artisan section, so it runs on
 * the CPU backend. Unofficial artifact: the app pins commit + SHA-256 so
 * the exact bytes audited are the bytes loaded; weights remain Gemma under
 * the Gemma Terms of Use.
 */
const val INT4_CPU_MODEL_ID = "gemma-4-e2b-it-int4-cpu"

/**
 * Gemini Nano served by the on-device AICore service through the ML Kit
 * GenAI Prompt API — FAST preference only (FULL is known-broken on this
 * firmware: AICore's preview Full bundle fails inference even after a
 * clean model re-download, verified 2026-10-05 in the sibling PixelChat
 * integration). No local file: AICore owns the weights, so there is no
 * URL/hash gate; availability is probed live via FeatureStatus.
 */
const val NANO_FAST_MODEL_ID = "gemini-nano-fast"

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
    /**
     * True when this bundle ships a `tf_lite_mtp_drafter` section and the
     * engine layer should attempt speculative decoding. Declared here (not
     * probed from the file) so the catalog stays the single source of truth;
     * the runtime additionally verifies support via Capabilities and only
     * claims it when both agree.
     */
    val speculativeDecoding: Boolean = false,
    /**
     * True when the model is served by an external on-device service (AICore)
     * rather than an in-process LiteRT engine. Hosted entries have no local
     * `.litertlm` file, no download URL, and no SHA-256 to verify — readiness
     * is decided by probing the host service (see NanoEngine), not by file
     * presence. Every filesystem/hash path in the app must branch on this.
     */
    val hosted: Boolean = false,
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
    ModelInfo(
        id = GPU_SPEC_MODEL_ID,
        name = "Gemma 4 E4B + draft (GPU)",
        description = "The standard E4B bundle (3.66 GB) running on the GPU with " +
            "speculative decoding: a multi-token-prediction drafter proposes " +
            "several tokens ahead and the model verifies them in one weight " +
            "pass, so decode speed is no longer capped at one full weight " +
            "read per token. ~1.5-2x decode in Google's own GPU benchmarks " +
            "for this file; actual gain depends on how predictable the text " +
            "is. Needs ~3.7 GB free storage and more RAM while loaded.",
        url = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm" +
            "/resolve/2eee7ac325f20eb8c9ac1d0e972f7c84663062da/gemma-4-E4B-it.litertlm",
        filename = "gemma-4-e4b-it-spec.litertlm",
        backend = Backend.LITERT_GPU,
        // LFS sha256 at the pinned commit; verified fail-closed before load.
        sha256 = "0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0",
        requiredSocMarker = null,
        contextTokens = null,
        expectedSizeBytes = 3_659_530_240L,
        licenseNotice = "Gemma Terms of Use: https://ai.google.dev/gemma/terms. " +
            "Artifact: litert-community/gemma-4-E4B-it-litert-lm @ 2eee7ac.",
        speculativeDecoding = true,
    ),
    ModelInfo(
        id = INT4_CPU_MODEL_ID,
        name = "Gemma 4 E2B int4 (CPU)",
        description = "int4-quantized Gemma 4 E2B in LiteRT-LM CPU format — " +
            "2.56 GB of weights (vs 2.97 GB for the GPU E4B). Smaller AND a " +
            "smaller model, but this artifact is CPU-delegate only (no GPU " +
            "section), so expect lower decode speed than the GPU entries. " +
            "Community re-quant (UrielSalto): unofficial artifact, verified " +
            "byte-for-byte against the pinned hash; output quality vs the " +
            "official quants has not been benchmarked here.",
        url = "https://huggingface.co/UrielSalto/gemma-4-e2b-it-int4-litert-lm" +
            "/resolve/652d52eb3599a29c88156d468de6072d2545cd5b/model.litertlm",
        filename = "gemma-4-e2b-it-int4.litertlm",
        backend = Backend.LITERT_CPU,
        // LFS sha256 at the pinned commit; verified fail-closed before load.
        sha256 = "50db55a621e9319e0ac52bd5ec9b6e9534ddc651946708ff599967c8fc91a939",
        requiredSocMarker = null,
        contextTokens = null,
        expectedSizeBytes = 2_556_483_760L,
        licenseNotice = "Gemma Terms of Use: https://ai.google.dev/gemma/terms. " +
            "Artifact: UrielSalto/gemma-4-e2b-it-int4-litert-lm @ 652d52e.",
    ),
    ModelInfo(
        id = NANO_FAST_MODEL_ID,
        name = "Gemini Nano (Fast, AICore)",
        description = "Google's on-device Gemini Nano, FAST variant only, served " +
            "by the AICore system service through the ML Kit GenAI Prompt API. " +
            "No download here: AICore owns the model and downloads it through " +
            "its own channel. FAST is the only supported preference — the " +
            "FULL preview bundle fails inference on this firmware. AICore's own " +
            "token window applies; long prompts are rejected before generation.",
        url = "",
        filename = "",
        backend = Backend.AICORE,
        // Hosted by AICore — no artifact to pin or hash. The empty string is
        // never verified: hosted entries skip the SHA-256 gate entirely.
        sha256 = "",
        requiredSocMarker = null,
        contextTokens = null,
        expectedSizeBytes = null,
        licenseNotice = "Gemini Nano is provided by the AICore system service on " +
            "this device; subject to Google's AICore terms.",
        hosted = true,
    ),
)

/**
 * The catalog entry served by default when nothing has been selected: the
 * GPU bundle — the working, on-device-verified path on stock Tensor G5
 * firmware. ([STABLE_MODEL_ID]'s NPU bundle can never load there: its
 * dispatch aborts and the engine refuses it.)
 */
val DEFAULT_MODEL: ModelInfo get() = findModelInfo(GPU_MODEL_ID) ?: AVAILABLE_MODELS.first()

/** Strict lookup: unknown IDs return null and the caller must fail explicitly. */
fun findModelInfo(id: String): ModelInfo? = AVAILABLE_MODELS.firstOrNull { it.id == id }
