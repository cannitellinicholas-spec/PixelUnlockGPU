// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference

import com.nickzam.server.Backend

/**
 * Abstraction over a single, ready-to-use generative engine.
 *
 * The interface is intentionally narrow — what it unifies is the
 * **lifecycle and identity** concerns the [EngineRegistry] cares about:
 * which backend, which model id, and how to release native resources.
 * The chat route talks to the native LiteRT-LM engine for conversation
 * lifecycle (too rich to wedge through this interface).
 */
interface Engine : AutoCloseable {
    /** The model id (the stable catalog id, e.g. `gemma-4-e2b-it-tpu-g5`). */
    val modelId: String

    /** The backend this engine was built for. Drawn from [ModelInfo.backend]. */
    val backend: Backend

    /** Free any native resources. Idempotent — multiple closes are safe. */
    override fun close()
}
