# PixelUnlockGPU — Pixel 10 Pro XL LiteRT-LM OpenAI server

One Android app that serves **Gemma 4 E2B on the Tensor G5 TPU** over an
OpenAI-compatible HTTP API for TypingMind and other clients.

- Server/lifecycle foundation ported from **localLLM** (Ktor HTTP server,
  OpenAI routes, foreground service, engine registry).
- Tensor G5 model/runtime path ported from **Box** (artifact coordinates,
  NPU backend setup, GoogleTensor dispatch library).
- Stable model ID: **`gemma-4-e2b-it-tpu-g5`**.
- Status: Gate 0 (audit) complete; device gates pending a Pixel 10 Pro XL.

See [docs/device-model-matrix.md](docs/device-model-matrix.md) for the pinned
compatibility set, [docs/api.md](docs/api.md) for the HTTP contract, and
[docs/test-report.md](docs/test-report.md) for the acceptance plan.

## Build

Requirements: JDK 17, Android SDK (compileSdk 35, build-tools 34).

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17   # or your JDK 17
export ANDROID_HOME=~/Library/Android/sdk       # or your SDK
./gradlew :app:assembleDebug
./gradlew :app:testDebugUnitTest
```

The APK installs as package `com.nickzam.server` — a separate app from any
existing Mobile LM Server / Off Grid install, so the control setup stays
usable. Stop the old listener while testing to avoid port conflicts (both
default to 8080).

## Use

1. Install the APK on the Pixel 10 Pro XL.
2. Open PixelUnlockGPU → **Download** the model (~3.1 GB; SHA-256 verified before
   load) or **Import** a `.litertlm` file.
3. **Start** the server (loopback `127.0.0.1:8080` by default).
4. Point TypingMind at `http://127.0.0.1:8080/v1` with model
   `gemma-4-e2b-it-tpu-g5`.

Remote access is explicit opt-in: set an API key, enable Remote access,
restart the server, and prefer Tailscale/VPN over LAN.

## Layout

```
app/src/main/java/com/nickzam/server/
  ApiTypes.kt            OpenAI wire types (text chat)
  ChatValidation.kt      strict request validation (pure, unit-tested)
  ModelCatalog.kt        Backend enum + single-model catalog
  Settings.kt            SharedPreferences-backed settings
  RequestTracker.kt      queue/history/stats
  RateLimiter.kt         per-client token bucket
  LogManager.kt          log ring + logcat bridge
  NickZamServerService.kt foreground service (lifecycle owner)
  ServiceLocator.kt      in-process UI handle (load/unload)
  MainActivity.kt        single-screen Compose UI
  inference/             Engine, EngineKey, EngineRegistry, Sha256
  inference/litert/      LiteRtEngineBuilder, LiteRtEngine,
                         LlmMessageConverter, SessionManager
  server/                ServerDeps, ServerEngine (Ktor)
  server/auth/           bearer-token gate
  server/routes/         health+models, chat completions
app/src/main/jniLibs/arm64-v8a/
  libLiteRtDispatch_GoogleTensor.so   Tensor delegate (hash-pinned)
```

## What's excluded from the MVP (by design)

Text chat only: no tools/function calling, images, audio, `stop` sequences
(all rejected explicitly on the wire), no embeddings, RAG, AICore, CORS,
NSD discovery, or boot autostart. See the manifest for the full list and
rationale.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

This app is a derivative work, built with real credit where it's due:

- **Server/lifecycle/inference foundation** is derived from
  [localllm](https://github.com/mlnomadpy/localllm)
  (pin `761bc94`), Apache 2.0 (declared in that project's README). Every
  derived file carries a header marking it and the modifications.
- **Tensor G5 model/runtime path and the prebuilt
  `libLiteRtDispatch_GoogleTensor.so`** come from
  [Box](https://github.com/jegly/Box)
  (Copyright 2026 Jesse Li-Yates), Apache 2.0, cross-checked through
  [Box_with_API](https://github.com/jmahatpure01/Box_with_API) (pin
  `fe6485c`). Full provenance with SHAs in
  [docs/device-model-matrix.md](docs/device-model-matrix.md).
- **Gemma weights are never bundled.** They are downloaded by the user and
  governed by [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).

If you reuse this code, keep the LICENSE, NOTICE, and the per-file
attribution headers, and you're in good standing with all three upstreams.
