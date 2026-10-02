// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.server

import android.content.Context
import com.nickzam.server.RateLimiter
import com.nickzam.server.inference.EngineRegistry
import com.nickzam.server.inference.litert.SessionManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.sync.Mutex
import java.util.concurrent.atomic.AtomicLong

/**
 * Composition-root dependency bundle for the Ktor server. Destructured once
 * in [ServerEngine] — route extensions take only the deps they actually use
 * so every route file stays independently testable.
 */
data class ServerDeps(
    val appContext: Context,
    val engineRegistry: EngineRegistry,
    val sessionManager: SessionManager,
    val inferenceMutex: Mutex,
    val rateLimiter: RateLimiter,
    val serviceScope: CoroutineScope,
    val lastActivityAt: AtomicLong,
    val acquireWakeLock: suspend (timeoutMs: Long, block: suspend () -> Unit) -> Unit,
)
