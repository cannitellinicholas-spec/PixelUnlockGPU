# PixelUnlockGPU HTTP API

Base URL (default): `http://127.0.0.1:8080`. OpenAI path prefix: `/v1`.

Auth: when an API key is set in the app, all three routes require
`Authorization: Bearer <key>` (401 + `WWW-Authenticate: Bearer` otherwise).
With no key set, the server is open (loopback default).

## Access modes

Set in the app; `/health` reports the selected mode in its `access` block.

- **Loopback** (default): binds `127.0.0.1` only.
- **LAN**: binds `0.0.0.0`. Unencrypted; the bearer key is required only
  if one is set — with no key set the server is open on the LAN, so set a
  key before enabling LAN.
- **Tailscale**: binds the phone's `100.x` tailnet IP. Traffic is
  WireGuard-encrypted end-to-end. Takes precedence over LAN when both
  toggles are on.

In Tailscale mode with the tailnet down, the server falls back to
loopback — it never falls back to `0.0.0.0`. `/health` does not expose
that fallback: its `access.mode`, `access.encrypted`, and
`server.loopback_only` derive from the selected mode, not the actual
bound host, so the loopback-bound server still reports mode `tailscale`,
`encrypted: true`, and `loopback_only: false`.

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
  "access": { "mode": "loopback", "tailnet_ip": null, "encrypted": false },
  "throughput": {
    "note": "estimated: ~4 chars/token (no token API in LiteRT-LM)",
    "total_tokens_since_start": 4608,
    "total_output_chars_since_start": 18432,
    "avg_tokens_per_sec": 8.5,
    "avg_decode_tokens_per_sec": 12.3,
    "avg_prefill_tokens_per_sec": 145.0,
    "live_tokens_per_sec": 0.0,
    "live_decode_tokens_per_sec": 0.0,
    "live_prefill_ms": null,
    "live_prefill_tokens_per_sec": 0.0,
    "phase": "idle",
    "queue_depth": 0,
    "generating": false
  },
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

- `access.mode` is the selected mode (`loopback`, `lan`, or
  `tailscale`; see Access modes). `access.tailnet_ip` is the phone's
  `100.x` tailnet IP, or `null` when the tailnet is not up — reported in
  every mode, so in loopback or LAN mode with a `100.x` interface up it
  is that IP, not `null`.
  `access.encrypted` is `true` only in Tailscale mode.
- `throughput` token counts are estimated at ~4 chars/token (LiteRT-LM
  exposes no token API) and are counted since server start.
  `avg_tokens_per_sec` is end-to-end, including cold-engine build and
  prefill; `avg_decode_tokens_per_sec` measures first-token to
  last-token and is the number comparable to published GPU tok/s
  figures. The `live_*` fields describe the in-flight request only
  (zero or `null` when idle): `live_prefill_tokens_per_sec` is measured
  from engine-ready, cold build excluded, and is `0` while queued or
  building. `phase` breaks down what the in-flight request is doing:
  `engine_start`, `prefill`, or `decode` while a request is in flight;
  with no request in flight, `queued` when the queue is non-empty and
  `idle` otherwise. `generating` is `true` whenever a request is in
  flight — including the engine-build phase (`engine_start`), before any
  engine is held, and prefill.
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
