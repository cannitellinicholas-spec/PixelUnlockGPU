// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import android.content.Context
import android.content.SharedPreferences

/**
 * Centralized typed access to user preferences for the NickZam server.
 *
 * Single source of truth — both the UI and the server service read from
 * here so they can never drift apart. Backed directly by [SharedPreferences]
 * (synchronous, no DataStore dependency); the MVP settings surface is small
 * enough that reactivity beyond Compose-local state is unnecessary.
 */
object Settings {
    private const val PREFS = "nickzam_settings"

    const val KEY_SERVER_PORT = "server_port"
    const val KEY_TEMPERATURE = "temperature"
    const val KEY_TOP_K = "top_k"
    const val KEY_TOP_P = "top_p"
    const val KEY_BIND_LAN = "bind_lan"
    const val KEY_REQUEST_TIMEOUT_MS = "request_timeout_ms"
    const val KEY_MAX_QUEUE_DEPTH = "max_queue_depth"
    const val KEY_MAX_PROMPT_CHARS = "max_prompt_chars"
    const val KEY_API_KEY = "api_key"
    const val KEY_KEEP_AWAKE = "keep_awake"
    const val KEY_RATE_LIMIT_PER_SEC = "rate_limit_per_sec"
    const val KEY_RATE_LIMIT_BURST = "rate_limit_burst"
    const val KEY_IDLE_EVICT_MS = "idle_evict_ms"
    const val KEY_IDLE_STOP_MS = "idle_stop_ms"
    const val KEY_SELECTED_MODEL_ID = "selected_model_id"
    const val KEY_CONTEXT_TOKENS = "context_tokens"
    const val KEY_TAILSCALE = "tailscale"

    const val DEFAULT_PORT = 8080
    const val DEFAULT_TEMPERATURE = 0.8f
    const val DEFAULT_TOP_K = 40
    const val DEFAULT_TOP_P = 0.95f
    const val DEFAULT_REQUEST_TIMEOUT_MS = 120_000L
    const val DEFAULT_MAX_QUEUE_DEPTH = 8
    const val DEFAULT_MAX_PROMPT_CHARS = 100_000
    const val DEFAULT_CONTEXT_TOKENS = 32_768
    /** Per-client requests per second (0 disables the limiter entirely). */
    const val DEFAULT_RATE_LIMIT_PER_SEC = 0.0
    /** Burst capacity per client when the limiter is enabled. */
    const val DEFAULT_RATE_LIMIT_BURST = 10.0
    const val DEFAULT_IDLE_EVICT_MS = 5L * 60_000L   // 5 minutes; 0 disables
    const val DEFAULT_IDLE_STOP_MS = 0L               // disabled by default
    const val DEFAULT_KEEP_AWAKE = true
    /** Default selection: the GPU bundle — the NPU path aborts on stock G5 firmware. */
    const val DEFAULT_MODEL_ID = GPU_MODEL_ID

    fun prefs(context: Context): SharedPreferences =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun port(context: Context): Int = prefs(context).getInt(KEY_SERVER_PORT, DEFAULT_PORT)
    fun setPort(context: Context, value: Int) =
        prefs(context).edit().putInt(KEY_SERVER_PORT, value.coerceIn(1, 65535)).apply()

    fun temperature(context: Context): Float =
        prefs(context).getFloat(KEY_TEMPERATURE, DEFAULT_TEMPERATURE)
    fun setTemperature(context: Context, value: Float) =
        prefs(context).edit().putFloat(KEY_TEMPERATURE, value).apply()

    fun topK(context: Context): Int = prefs(context).getInt(KEY_TOP_K, DEFAULT_TOP_K)
    fun setTopK(context: Context, value: Int) =
        prefs(context).edit().putInt(KEY_TOP_K, value).apply()

    fun topP(context: Context): Float = prefs(context).getFloat(KEY_TOP_P, DEFAULT_TOP_P)
    fun setTopP(context: Context, value: Float) =
        prefs(context).edit().putFloat(KEY_TOP_P, value).apply()

    /**
     * How the HTTP server is exposed, derived from two independent
     * preferences so the Tailscale and raw-LAN options stay separate:
     *
     * - [AccessMode.TAILSCALE] when [tailscale] is on — bind the phone's
     *   CGNAT 100.x tailnet IP; traffic is WireGuard-encrypted end-to-end.
     *   Takes precedence over the LAN toggle (raw stays off).
     * - [AccessMode.LAN] when [bindLan] is on and Tailscale is off — bind
     *   0.0.0.0; unencrypted, requires the bearer key.
     * - [AccessMode.LOOPBACK] when both are off (default).
     */
    enum class AccessMode { LOOPBACK, TAILSCALE, LAN }

    /** Raw-LAN toggle (bind 0.0.0.0). Its own preference, independent of Tailscale. */
    fun bindLan(context: Context): Boolean = prefs(context).getBoolean(KEY_BIND_LAN, false)
    fun setBindLan(context: Context, value: Boolean) =
        prefs(context).edit().putBoolean(KEY_BIND_LAN, value).apply()

    /** Tailscale toggle (bind tailnet IP, E2E encrypted). Its own preference. */
    fun tailscale(context: Context): Boolean = prefs(context).getBoolean(KEY_TAILSCALE, false)
    fun setTailscale(context: Context, value: Boolean) =
        prefs(context).edit().putBoolean(KEY_TAILSCALE, value).apply()

    fun accessMode(context: Context): AccessMode = when {
        tailscale(context) -> AccessMode.TAILSCALE
        bindLan(context) -> AccessMode.LAN
        else -> AccessMode.LOOPBACK
    }

    /**
     * Interface the server binds to. Tailscale mode resolves the live tailnet
     * IP; with Tailscale down it returns null and the service falls back to
     * loopback (it never falls back to 0.0.0.0, which would leak).
     */
    fun bindHost(context: Context): String? = when (accessMode(context)) {
        AccessMode.LOOPBACK -> "127.0.0.1"
        AccessMode.LAN -> "0.0.0.0"
        AccessMode.TAILSCALE -> tailscaleIp()
    }

    /**
     * Finds the phone's Tailscale IPv4 (CGNAT range 100.64.0.0/10, which the
     * Android Tailscale app exposes as a `tun` interface). Null when the
     * tailnet isn't up.
     */
    fun tailscaleIp(): String? {
        return try {
            val ifaces = java.net.NetworkInterface.getNetworkInterfaces() ?: return null
            while (ifaces.hasMoreElements()) {
                val iface = ifaces.nextElement()
                if (iface.isLoopback || !iface.isUp) continue
                val addrs = iface.inetAddresses
                while (addrs.hasMoreElements()) {
                    val a = addrs.nextElement()
                    if (a is java.net.Inet4Address && !a.isLoopbackAddress &&
                        a.hostAddress?.startsWith("100.") == true
                    ) {
                        return a.hostAddress
                    }
                }
            }
            null
        } catch (_: Exception) {
            null
        }
    }

    fun requestTimeoutMs(context: Context): Long =
        prefs(context).getLong(KEY_REQUEST_TIMEOUT_MS, DEFAULT_REQUEST_TIMEOUT_MS)
    fun setRequestTimeoutMs(context: Context, value: Long) =
        prefs(context).edit().putLong(KEY_REQUEST_TIMEOUT_MS, value).apply()

    fun maxQueueDepth(context: Context): Int =
        prefs(context).getInt(KEY_MAX_QUEUE_DEPTH, DEFAULT_MAX_QUEUE_DEPTH)
    fun setMaxQueueDepth(context: Context, value: Int) =
        prefs(context).edit().putInt(KEY_MAX_QUEUE_DEPTH, value.coerceAtLeast(1)).apply()

    fun maxPromptChars(context: Context): Int =
        prefs(context).getInt(KEY_MAX_PROMPT_CHARS, DEFAULT_MAX_PROMPT_CHARS)
    fun setMaxPromptChars(context: Context, value: Int) =
        prefs(context).edit().putInt(KEY_MAX_PROMPT_CHARS, value.coerceAtLeast(1)).apply()

    fun apiKey(context: Context): String = prefs(context).getString(KEY_API_KEY, "") ?: ""
    fun setApiKey(context: Context, value: String) =
        prefs(context).edit().putString(KEY_API_KEY, value).apply()

    fun keepAwake(context: Context): Boolean =
        prefs(context).getBoolean(KEY_KEEP_AWAKE, DEFAULT_KEEP_AWAKE)
    fun setKeepAwake(context: Context, value: Boolean) =
        prefs(context).edit().putBoolean(KEY_KEEP_AWAKE, value).apply()

    fun rateLimitPerSec(context: Context): Double =
        prefs(context).getFloat(KEY_RATE_LIMIT_PER_SEC, DEFAULT_RATE_LIMIT_PER_SEC.toFloat()).toDouble()
    fun setRateLimitPerSec(context: Context, value: Double) =
        prefs(context).edit().putFloat(KEY_RATE_LIMIT_PER_SEC, value.toFloat()).apply()

    fun rateLimitBurst(context: Context): Double =
        prefs(context).getFloat(KEY_RATE_LIMIT_BURST, DEFAULT_RATE_LIMIT_BURST.toFloat()).toDouble()
    fun setRateLimitBurst(context: Context, value: Double) =
        prefs(context).edit().putFloat(KEY_RATE_LIMIT_BURST, value.toFloat()).apply()

    fun idleEvictMs(context: Context): Long =
        prefs(context).getLong(KEY_IDLE_EVICT_MS, DEFAULT_IDLE_EVICT_MS)
    fun setIdleEvictMs(context: Context, value: Long) =
        prefs(context).edit().putLong(KEY_IDLE_EVICT_MS, value).apply()

    fun idleStopMs(context: Context): Long =
        prefs(context).getLong(KEY_IDLE_STOP_MS, DEFAULT_IDLE_STOP_MS)
    fun setIdleStopMs(context: Context, value: Long) =
        prefs(context).edit().putLong(KEY_IDLE_STOP_MS, value).apply()

    fun selectedModelId(context: Context): String =
        prefs(context).getString(KEY_SELECTED_MODEL_ID, DEFAULT_MODEL_ID)
            ?.takeIf { it.isNotBlank() } ?: DEFAULT_MODEL_ID
    fun setSelectedModelId(context: Context, value: String) =
        prefs(context).edit().putString(KEY_SELECTED_MODEL_ID, value.trim().ifEmpty { DEFAULT_MODEL_ID }).apply()

    /**
     * Engine KV-cache budget in tokens (the model's context window). Baked
     * into the engine at build time, so changing it requires an engine
     * rebuild — the UI evicts cached engines on change. Clamped to the
     * verified catalog ceiling; smaller windows cost less memory and prefill
     * faster at the price of shorter chats.
     */
    val ALLOWED_CONTEXT_TOKENS = listOf(1_024, 2_048, 4_096, 8_192, 16_384, 32_768)

    fun contextTokens(context: Context): Int =
        prefs(context).getInt(KEY_CONTEXT_TOKENS, DEFAULT_CONTEXT_TOKENS)
            .coerceAtMost(MAX_CONTEXT_TOKENS)
    fun setContextTokens(context: Context, value: Int) =
        prefs(context).edit()
            .putInt(KEY_CONTEXT_TOKENS, value.coerceIn(1_024, MAX_CONTEXT_TOKENS))
            .apply()

    /** Verified ceiling for the pinned Gemma 4 bundles (32k). */
    const val MAX_CONTEXT_TOKENS = 32_768
}
