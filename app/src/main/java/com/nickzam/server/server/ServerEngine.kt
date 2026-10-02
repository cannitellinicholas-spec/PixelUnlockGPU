// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.server

import com.nickzam.server.server.routes.chatRoute
import com.nickzam.server.server.routes.healthRoute
import com.nickzam.server.server.routes.modelsRoute
import io.ktor.serialization.gson.gson
import io.ktor.server.application.install
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.netty.NettyApplicationEngine
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.routing.routing

/**
 * Builds a Ktor [EmbeddedServer] configured with the OpenAI-compatible MVP
 * API surface: `GET /health`, `GET /v1/models`,
 * `POST /v1/chat/completions` (+ `POST /health/warm` for cold-start probes).
 *
 * No CORS plugin: native clients and TypingMind don't need browser-wide
 * CORS, so a random web page can't drive the API from a user's browser.
 *
 * Route extensions take only the deps they actually use — the [ServerDeps]
 * bundle is destructured here once at the composition root and never
 * crosses a route boundary.
 */
object ServerEngine {

    fun build(
        deps: ServerDeps,
        port: Int,
        host: String,
    ): EmbeddedServer<NettyApplicationEngine, NettyApplicationEngine.Configuration> =
        embeddedServer(Netty, port = port, host = host) {
            install(ContentNegotiation) { gson() }
            routing {
                healthRoute(deps.engineRegistry, deps.appContext, deps.inferenceMutex)
                modelsRoute(deps.appContext, deps.engineRegistry)
                chatRoute(
                    appContext = deps.appContext,
                    engineRegistry = deps.engineRegistry,
                    sessionManager = deps.sessionManager,
                    rateLimiter = deps.rateLimiter,
                    inferenceMutex = deps.inferenceMutex,
                    serviceScope = deps.serviceScope,
                    lastActivityAt = deps.lastActivityAt,
                    acquireWakeLock = deps.acquireWakeLock,
                )
            }
        }
}
