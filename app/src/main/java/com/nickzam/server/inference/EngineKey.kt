// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference

import com.nickzam.server.Backend

/**
 * Typed cache key for a LiteRT-LM engine instance.
 *
 * Kotlin data-class equality is used by [android.util.LruCache] and by
 * [java.util.concurrent.ConcurrentHashMap] so no manual hashCode/equals
 * override is needed.
 *
 * @property modelId   Stable catalog model id.
 * @property maxTokens KV-cache budget passed to `EngineConfig.maxNumTokens`;
 *                     `null` means "let the model's bundle decide".
 * @property backend   Hardware backend declared in the catalog for this model.
 */
data class EngineKey(
    val modelId: String,
    val maxTokens: Int?,   // null means "model default"
    val backend: Backend,
) {
    /** Stable string form used for logs and the `/health` `key` field. */
    fun asString(): String = "${modelId}_${maxTokens ?: "model"}_${backend.name}"
}
