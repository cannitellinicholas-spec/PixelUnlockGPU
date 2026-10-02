// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference.litert

import com.google.ai.edge.litertlm.Engine as LiteRtNativeEngine
import com.nickzam.server.Backend
import com.nickzam.server.LogManager
import com.nickzam.server.inference.Engine
import com.nickzam.server.inference.EngineKey

/**
 * [Engine] wrapper around a LiteRT-LM native engine. Holds the cache key
 * separately so the registry can correlate eviction with conversation
 * cleanup, without exposing the LiteRT-LM SDK type on the [Engine] surface.
 */
class LiteRtEngine(
    override val modelId: String,
    override val backend: Backend,
    val native: LiteRtNativeEngine,
    val cacheKey: EngineKey,
) : Engine {

    @Volatile private var closed = false

    override fun close() {
        if (closed) return
        closed = true
        try { native.close() } catch (e: Exception) {
            LogManager.e("LiteRtEngine", "Error closing native engine ${cacheKey.asString()}", e)
        }
    }
}
