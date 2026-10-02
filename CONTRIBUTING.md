# Contributing to PixelUnlockGPU

Contributions are welcome — bug fixes, docs, new endpoints, backend work.
The fastest path to a merged PR is a small, focused change with a passing
test and a clear reason.

## Build and test

Requirements: JDK 17, Android SDK (compileSdk 35).

```bash
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=~/Library/Android/sdk
./gradlew testDebugUnitTest   # unit tests (no device needed)
./gradlew assembleDebug       # APK at app/build/outputs/apk/debug/
```

Everything that merges should keep `testDebugUnitTest` green. The pure
logic (validation, rate limiting, request tracking, stream reassembly)
is unit-tested and runs without a phone.

## Signing

The release build signs with the `NICKZAM_KEYSTORE_*` properties if you
set them in `~/.gradle/gradle.properties` or the environment; if unset it
falls back to the standard debug key. That fallback is intentional —
you never need the project's upload key to build, test, or open a PR.
**Never commit a keystore or password**; `*.jks` is gitignored.

## Licensing and attribution (please read — this is the one strict rule)

This project is Apache 2.0 and is itself a derivative of
[localllm](https://github.com/mlnomadpy/localllm) and builds on
[Box](https://github.com/jegly/Box). See `NOTICE` for the full record.

- Files derived from upstream carry a header marking the source and the
  modifications. If you edit one of those files, keep the header.
- Your new files: no header required; the repo-level `LICENSE` covers
  them. Do not add a different license or a `SPDX-License-Identifier`
  that conflicts with Apache-2.0.
- By opening a PR you license your contribution under Apache 2.0 as
  well (Apache 2.0 Section 5 — no extra paperwork needed).
- Never commit model weights, keys, or anything from a device's app
  storage. Gemma weights are user-downloaded under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and stay that
  way.

## Pull request checklist

- [ ] `./gradlew testDebugUnitTest` passes
- [ ] New behavior has a test where it's unit-testable
- [ ] Wire-format changes update `docs/api.md` in the same PR
- [ ] No secrets, keystores, device IDs, or personal paths in the diff
- [ ] If you changed pinned versions (LiteRT-LM, model SHA), update
      `docs/device-model-matrix.md` — those pins are load-bearing

## What I'm looking for

Issues labeled [`good first issue`](https://github.com/cannitellinicholas-spec/PixelUnlockGPU/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
are scoped for a first PR and I'll answer questions on them quickly.
The Tensor G5 NPU tracking issue is the big prize and needs a device.

Before building something large, open an issue first so we can agree on
scope — a PR that rewrites the engine lifecycle will likely need to be
broken up regardless of quality.
