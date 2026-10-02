# Device–model compatibility matrix (Gate 0 manifest)

PixelUnlockGPU serves a small pinned catalog; each entry names its own backend and
SoC gate. This file is the pinned compatibility set from the Gate 0
source/license audit (2026-09-30). If any dependency or artifact changes,
update this file and re-run the compatibility gate before merging.

Catalog additions after the original Gate 0 audit:

- `gemma-4-e4b-it-gpu` — Gemma 4 E4B GPU bundle
  (`litert-community/gemma-4-E4B-it-litert-lm` @ `2eee7ac`,
  `gemma-4-E4B-it-gpu.litertlm`, 2,969,059,328 B,
  sha256 `4912bb5a9c30993c51a7711f763212077458529312175df0573a78323a2bb7ff`,
  re-verified against the HF LFS metadata 2026-10-01). Backend
  `LITERT_GPU` (OpenCL delegate, bundled in the AAR), SoC gate: none.
  **Requires `<uses-native-library android:name="libOpenCL.so"
  android:required="false"/>` (and the `libOpenCL-pixel.so` variant) inside
  `<application>` in the manifest**: from targetSdk 31+ the app linker
  keeps vendor public libraries out of the app namespace unless declared,
  and the delegate fails `Can not find OpenCL library on this device`
  without it. Verified on-device 2026-10-01 (`GPU_OPENCL` in
  `/health`, chat works). Added because the Tensor G5 NPU dispatch plugin
  SIGABRTs on stock Android 17 (needs
  `/vendor/lib64/libedgetpu_litert.so`, invisible to app linker
  namespaces and undeclarable as a public vendor lib) — see section on the
  NPU crash. The Tensor JNI NPU-warmup primer is skipped on G5 for the
  same reason. `EngineRegistry.acquire()` now refuses NPU builds outright
  on Tensor G5 with a clean `last_load_error` ("… Use the GPU model
  instead.") so selecting the NPU entry and pressing Load cannot abort the
  process.

Evidence labels (used across issues, commits, and test reports):

- **Documented:** stated by a source or README.
- **Implemented:** code exists in this branch.
- **Wired:** the app selects this code path for this device/model.
- **Exercised:** a test or request executed it.
- **Verified:** exact build, artifact, runtime, phone, and observed backend
  passed acceptance.

## 1. Pinned sources (retrieved 2026-09-30)

| Source | Pin | Role |
|---|---|---|
| localLLM (`mlnomadpy/localllm`) | commit `761bc94cba5eb176edfd814531da8edf17b76443` (HEAD; no tags) | Server/lifecycle foundation: Ktor routes, foreground service, engine registry, LiteRT builder |
| Box_with_API (`jmahatpure01/Box_with_API`) | HEAD `fe6485c8aa134fc4f435c00ad1d0fb8ac812d8a5`; latest tag `v1.2.0` (`6a673a77…`) | Tensor G5 model/runtime reference: artifact coordinates, NPU backend setup, dispatch lib |
| Upstream Box (`jegly/Box`) | HEAD `aaa0bb22a89078b827b46a055f61ba306fc19b57`; latest tag `v3.5.5` | Cross-check for G5 claims |
| LiteRT-LM (`google-ai-edge/LiteRT-LM`) | HEAD `615d193f8e322e4e01ab99d5415033f67f42a0f2` | Runtime API reference |
| AI Edge Gallery (`google-ai-edge/gallery`) | HEAD `85c6275eba505c824cc0fb5fc1563c3ef7f45870` | Allowlist/model-distribution reference |

## 2. Toolchain pins

| Component | Version |
|---|---|
| Android Gradle Plugin | 8.7.3 |
| Kotlin | 2.2.21 (jvmTarget 11) |
| Gradle wrapper | 8.10.2 (JDK 17 required) |
| compileSdk / minSdk / targetSdk | 35 / 29 / 34 |
| Ktor (server-core, netty, content-negotiation, gson) | 3.4.3 |
| Compose BOM | 2024.09.02 |
| `com.google.ai.edge.litertlm:litertlm-android` (Google Maven) | **0.12.0** |

Newer `litertlm-android` releases exist (up to 0.17.1 at audit time) but are
**untested with this integration** — never float. Box pins 0.10.0; PixelUnlockGPU
pins **0.12.0** to match localLLM's proven server-side API usage
(`Conversation.sendMessageAsync → Flow`, `SamplerConfig(topK, topP,
temperature, seed)`). If Gate 1 shows the artifact will not init under
0.12.0, the bounded spike is: retry under 0.10.0 as a tested unit.

## 3. Model artifact (the compatibility unit)

| Field | Value |
|---|---|
| Stable model ID | `gemma-4-e2b-it-tpu-g5` |
| Display name | Gemma 4 E2B (Tensor G5) |
| HF repo | `litert-community/gemma-4-E2B-it-litert-lm` |
| Commit pin | `b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1` (HEAD as of 2026-09-30; the G5 bundle is absent from the `7fa1d78` pin) |
| File | `gemma-4-E2B-it_Google_Tensor_G5.litertlm` (the generic `gemma-4-E2B-it.litertlm` has no NPU section — NPU load fails `TF_LITE_AUX not found`) |
| Pinned URL | `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it_Google_Tensor_G5.litertlm` |
| Size | 3,113,545,589 bytes (**confirmed**: `x-linked-size` + on-device bytes) |
| SHA-256 | `af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc` (**confirmed**: `x-linked-etag` at the pin + app verify + independent `sha256sum` on-device, 2026-09-30) |
| On-device filename | `gemma-4-e2b-it-tpu-g5.litertlm` (app external files dir) |
| Backend | `LITERT_NPU` → `Backend.NPU(nativeLibraryDir)`; **no fallback** |
| SoC gate | `Build.SOC_MODEL` contains `tensor g5` (**measured**: Pixel 10 Pro XL reports `Tensor G5`; `laguna` is `ro.board.platform`, not `ro.soc.model`) |
| Context limit | **UNVERIFIED** — upstream metadata claims 32K; the server advertises `null` until Gate 5 measures it |
| License | Gemma Terms of Use (`https://ai.google.dev/gemma/terms`); weights are downloaded, never bundled |

### Hash findings (audit)

- Box records the artifact by HF commit (`7fa1d78…`), size (2,583,085,056),
  and accelerators `gpu,npu,cpu` — but **no SHA-256**. There is **no
  G5-specific Gemma 4 file**: the Tensor G5 path is this generic artifact on
  the NPU backend with the GoogleTensor dispatch library (plus Box's Pixel 10
  NPU→TPU label remap and GPU removal). Status: **Documented**.
- localLLM records SHA-256 `18193810…` for a `main`-branch (unpinned)
  download of the same filename. The pinned-commit bytes hash to
  `ab7838cd…` — **these do not match**, so localLLM's hash is treated as
  stale/superseded and is NOT used. Status: superseded by measurement.
- The `x-linked-etag` header for the pinned commit equaled the computed
  SHA-256 for this file — supporting (not proving) the etag heuristic.

## 4. Native libraries (arm64-v8a, `app/src/main/jniLibs`)

| File | SHA-256 | Origin |
|---|---|---|
| `libLiteRtDispatch_GoogleTensor.so` (409,920 bytes) | `5731f46e9ac6e016113524a7a15fe9feb7bb55e03a0ce1c1b9aee7d5aab46fad` | **Byte-identical** in localLLM (`761bc94`) and Box_with_API (`fe6485c`) — origin unambiguous |

Box additionally ships Qualcomm/MediaTek dispatch + compiler plugins and QNN
libraries; PixelUnlockGPU ports only the GoogleTensor dispatch lib (single-SoC
MVP). The LiteRT-LM JNI itself ships inside the `litertlm-android` AAR.

## 5. Backend mapping (ported path)

| Layer | G5 selection |
|---|---|
| Catalog | `Backend.LITERT_NPU`, SoC marker `tensor g5` |
| Engine build | `Backend.NPU(context.applicationInfo.nativeLibraryDir)` |
| EngineConfig | `(modelPath, NPU, vision=null, audio=null, maxNumTokens, null, null)` — text-only MVP (Box uses GPU vision / CPU audio; localLLM uses CPU vision) |
| Conversation | `ConversationConfig(system, prior, tools=[], SamplerConfig(topK, topP, temp, seed=0), autoTools=false)` |
| Streaming | `sendMessageAsync(prompt, {}) → Flow<Message>` (cumulative text; deltas forwarded) |
| Blocking | `sendMessage(prompt, {})` |
| Cancel/cleanup | `cancelProcess()`; `conversation.close()` before `engine.close()` |

Box's TPU accelerator maps to the same `Backend.NPU(nativeLibraryDir)`
call — TPU is branding; the NPU delegate + dispatch lib do the work.

### Verified-from-AAR API notes (litertlm-android 0.12.0)

Signatures were read from the published AAR (`javap`), not copied from old
fragments:

- `ConversationConfig` has **no stop-sequence field** → `stop` is rejected
  with an explicit 400 (`unsupported_feature`).
- `SamplerConfig(topK: Int, topP: Double, temperature: Double, seed: Int)`
  → `top_p` is exposed (localLLM hardcodes 0.95; PixelUnlockGPU threads it through
  and tracks it in session reuse).
- `Conversation.benchmarkInfo` exposes init/TTFT/prefill/decode rates for
  Gate 5 measurements.

## 6. Corrections applied (from the spec)

1. **Box is not an API server.** Verified: no Ktor/HTTP-server code exists in
   Box_with_API. The endpoint comes from the localLLM port.
2. **localLLM's G5 support is Gemma 3 1B, not Gemma 4.** Its G5 entry is
   `Gemma3-1B-IT_q8_ekv1280_Google_Tensor_G5.litertlm` (SoC marker `laguna`,
   reused here). Gemma 4-on-G5 is the Box-documented path, pending Gate 1.
3. **Compatibility unit** = artifact (pinned commit + SHA-256) + AAR 0.12.0
   + dispatch lib (hash above) + `Backend.NPU` config, recorded as one set.
4. **Nothing floats**: commit pins, Maven pins, commit-pinned model URL,
   enforced SHA-256 (fail-closed, memoized by path+size+mtime).
5. **Single engine**: LiteRT-LM only. No AICore, no llama.cpp, no ONNX,
   no RAG/ObjectBox.
6. **HTTP ≠ NPU proof**: `/health.actual_backend` reports
   `GOOGLE_TENSOR_NPU` only when a healthy NPU engine is cached; NPU support
   is **not** verified until runtime evidence lands on the Pixel 10 Pro XL.
7. **MVP scope**: text chat only. Tools, images, audio, `stop`, CORS,
   embeddings, RAG, NSD, boot autostart are excluded (rejected explicitly
   where they appear on the wire).

## 7. Target device (to be filled at Gate 1)

| Field | Value |
|---|---|
| Phone | Pixel 10 Pro XL (pending — tester to provide) |
| Android build fingerprint | TBD |
| Free storage / RAM | TBD |
| Selected backend + log evidence | TBD |
| Clean-install / model-load result | TBD |

## 8. Gate status

| Gate | Status |
|---|---|
| 0 — Source/license audit | **Pass (code side)** — pins + hashes recorded; **blocked on device fields** (§7) |
| 1 — Reproduce model on phone | Blocked — needs the Pixel 10 Pro XL |
| 2 — Server + engine independently | Server Implemented (untested on device); engine spike = Gate 1 |
| 3 — Integrate one model | Implemented; needs Gate 1 |
| 4 — Client/lifecycle acceptance | Planned — see `docs/test-report.md` |
| 5 — Performance/stability | Planned — targets 6 tok/s min, 15+ preferred |

## 9. License record

See the repository `LICENSE` and `NOTICE` files for the distribution
mechanism (Apache 2.0 with per-file attribution headers marking derived
work and modifications, per Apache 2.0 Section 4(b)).

- localLLM: Apache 2.0 (server/engine code ported with package rename).
  Upstream declares Apache 2.0 in its README but ships no LICENSE file;
  attribution is made in NOTICE in the spirit of Section 4.
- Box_with_API / Box / AI Edge Gallery: Apache 2.0 (artifact coordinates,
  backend mapping, dispatch lib).
- LiteRT-LM AAR + GoogleTensor dispatch lib: per Google AI Edge / LiteRT
  distribution terms; dispatch lib redistributed as build input per Box's
  established distribution.
- Gemma 4 weights: Gemma Terms of Use — user downloads in-app; never
  bundled in the APK or this repo.
