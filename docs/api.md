# PixelUnlockGPU HTTP API

Base URL (default): `http://127.0.0.1:8080`. OpenAI path prefix: `/v1`.

Auth: when an API key is set in the app, all three routes require
`Authorization: Bearer <key>` (401 + `WWW-Authenticate: Bearer` otherwise).
With no key set, the server is open (loopback default).

## `GET /health`

Liveness + readiness + backend evidence. Never emits prompts, tokens, full
model paths, or filesystem details.

The top-level `model` object tracks the model **selected in the app**
(picker); the `models` array carries per-catalog-entry evidence.

```json
{
  "status": "ok",
  "service": "pixelunlockgpu-android",
  "version": "0.1.0",
  "server": { "port": 8080, "loopback_only": true },
  "model": {
    "id": "gemma-4-e4b-it-gpu",
    "installed": true,
    "loaded": true,
    "size_bytes": 2969059328,
    "sha256_prefix": "4912bb5a9c30993c",
    "context_tokens": null
  },
  "models": [
    { "id": "gemma-4-e2b-it-tpu-g5", "declared_backend": "LITERT_NPU",
      "installed": true, "loaded": false, "backend_proven": null,
      "size_bytes": 3113545589, "sha256_prefix": "af1082986639ecde",
      "context_tokens": null },
    { "id": "gemma-4-e4b-it-gpu", "declared_backend": "LITERT_GPU",
      "installed": true, "loaded": true, "backend_proven": "GPU_OPENCL",
      "size_bytes": 2969059328, "sha256_prefix": "4912bb5a9c30993c",
      "context_tokens": null }
  ],
  "engine": {
    "actual_backend": "GPU_OPENCL",
    "litert_version": "0.12.0",
    "loaded_count": 1,
    "last_load_error": null
  },
  "engines": [
    {
      "key": "gemma-4-e4b-it-gpu_model_LITERT_GPU",
      "backend": "LITERT_GPU",
      "attempts": [{ "backend": "LITERT_GPU", "result": "ok", "duration_ms": 12345 }]
    }
  ],
  "queue": { "depth": 0, "active": false, "max_depth": 8 },
  "stats": { "total_requests": 12, "total_completed": 12, "total_errors": 0, "total_cancelled": 0 },
  "app": { "version": "0.1.0", "version_code": 1 },
  "device": { "model": "Pixel 10 Pro XL", "soc": "Tensor G5" }
}
```

- `engine.actual_backend` is proven only by a healthy cached engine:
  `GOOGLE_TENSOR_NPU` (NPU), `GPU_OPENCL` (GPU), or `CPU`; otherwise
  `unknown` (or `error` with `last_load_error` set). A valid chat response
  alone never proves backend execution — this field plus runtime logs is
  the proof. Per-model: `models[].backend_proven` names the backend proven
  for that entry's cached engine (or `declared_*` while an un-attempted
  engine sits cached).
- Only one MODEL's engine is resident at a time — acquiring a different
  model evicts the previous model's engines (each bundle is ~3 GB).
- `model.context_tokens` is `null` until the limit is verified on-device.

## `POST /health/warm?model=<id>`

Force-load the engine and run a 1-token generation (60s budget) so scripts
can wait for first-token-ready. `model` defaults to the model selected in
the app; unknown IDs return 400. Response: `{model, status: warm|timeout|error, ms, …}`.
On Tensor G5 devices, NPU-backend models return `status: "error"` with a
refusal message (the delegate is not loadable from an app on stock
firmware); the process stays alive and any loaded GPU engine keeps serving.

## `GET /v1/models`

OpenAI-shaped list of **installed** models only (`owned_by` names each
entry's declared backend):

```json
{ "object": "list", "data": [
  { "id": "gemma-4-e2b-it-tpu-g5", "object": "model", "created": 1710000000, "owned_by": "pixelunlockgpu-litert_npu" },
  { "id": "gemma-4-e4b-it-gpu", "object": "model", "created": 1710000000, "owned_by": "pixelunlockgpu-litert_gpu" }
]}
```

Empty `data` means no bundle is on disk — download/import one in the app.

## `POST /v1/chat/completions`

Text chat. Request fields:

| Field | Required | Notes |
|---|---|---|
| `model` | yes | Must equal a catalog ID exactly (`gemma-4-e2b-it-tpu-g5` or `gemma-4-e4b-it-gpu`); unknown IDs are a 400, never mapped to the selected model |
| `messages` | yes | Non-empty; roles `system`/`user`/`assistant`; string or text-parts content; last non-system message must be `user` |
| `stream` | no | `true` → SSE; default `false` |
| `session_id` | no | In-memory KV-cache reuse across turns (prefix-validated) |
| `temperature` | no | Finite, 0–2 (default 0.8) |
| `top_p` | no | Finite, (0, 1] (default 0.95) |
| `top_k` | no | ≥ 1 (default 40) |
| `max_tokens` | no | 1–32768; `null` lets the bundle decide the KV budget |

`tools` declarations are accepted and **ignored** (like llama.cpp-family
servers): the model can never emit `tool_calls`, so agent clients that always
advertise tools still get plain-text generations. Explicitly rejected with
400 `unsupported_feature` (never silently ignored): `tool_choice` other than
`"none"`/`"auto"` (it would force a call the server can't produce), `stop`,
`image_url` / audio / multimodal parts, `tool_calls`, `tool` roles.

All error bodies are JSON regardless of the client `Accept` header, so
streaming-only clients (OpenAI SDK, agent CLIs) see the real error instead
of a bare 406 from content negotiation.

Response headers (set before the first SSE byte): `X-Request-Id`,
`X-Client-Id`, `X-Queue-Position`, `X-Queue-Depth`, `X-Estimated-Wait-Ms`;
`Retry-After` on 429.

Streaming: `Content-Type: text/event-stream`, an initial `role` chunk,
incremental `content` deltas, a final `finish_reason: "stop"` chunk, then
exactly one `data: [DONE]` terminator. `: ka` heartbeat every 10s. Client
disconnect cancels generation and frees the slot. Mid-stream failures close
with an error chunk + `[DONE]`.

Non-streaming returns the final assistant message with `finish_reason:
"stop"`. No usage block is emitted (counts are omitted rather than
fabricated).

### Status codes

| Code | When |
|---|---|
| 400 | Malformed JSON, unknown model, bad roles/content, unsupported feature, invalid sampling |
| 401 | Missing/invalid bearer key (when a key is set) |
| 408 | Inference timeout |
| 413 | Body or prompt exceeds caps |
| 429 | Per-client rate limit or queue full (`Retry-After`) |
| 503 | Engine not loadable (`LITERT_INIT_FAILED` envelope with next steps) |
| 504 | Warm-up timeout (`/health/warm` only) |
| 500 | Anything else (sanitized message, no stack traces) |
