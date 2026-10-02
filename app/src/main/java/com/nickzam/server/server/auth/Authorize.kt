// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.server.auth

import android.content.Context
import com.nickzam.server.ErrorDetails
import com.nickzam.server.ErrorResponse
import com.nickzam.server.Settings
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.header
import io.ktor.server.response.respond

/**
 * Bearer-token gate. Returns `true` when the request is allowed through.
 * On failure, writes a 401 + WWW-Authenticate header and returns `false`.
 *
 * Configuration is read from [Settings.apiKey]: empty string disables auth
 * entirely (default — loopback development; users opt into keyed access via
 * the Security section of the app, which is required before enabling
 * remote/LAN binding).
 */
suspend fun authorize(call: ApplicationCall, context: Context): Boolean {
    val configured = Settings.apiKey(context)
    if (configured.isEmpty()) return true
    val header = call.request.headers["Authorization"]
    val ok = header != null && header.startsWith("Bearer ") &&
        header.substring(7).trim() == configured
    if (!ok) {
        call.response.header("WWW-Authenticate", "Bearer")
        call.respond(
            HttpStatusCode.Unauthorized,
            ErrorResponse(ErrorDetails("Invalid or missing API key", "invalid_api_key", 401))
        )
    }
    return ok
}
