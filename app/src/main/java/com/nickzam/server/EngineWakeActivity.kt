package com.nickzam.server

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.core.content.ContextCompat

/**
 * Headless wake door for the local engine (added 2026-10-05 for Cadence).
 *
 * The GPU engine used to start only from this app's Start button, so a chat
 * app that routed a long-context turn here found the server STOPPED and
 * stalled ("switched to GPU" in text, nothing running). Cadence launches this
 * activity (explicit action + package, transparent theme, finishes instantly)
 * to bring the server up without stealing the screen.
 *
 * Safe to export: the server binds loopback unless the user explicitly toggled
 * LAN/tailscale exposure, and starting it is idempotent (onStartCommand +
 * START_STICKY already handle re-entry).
 */
class EngineWakeActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            ContextCompat.startForegroundService(
                this, Intent(this, NickZamServerService::class.java),
            )
        } catch (t: Throwable) {
            LogManager.e("EngineWake", "wake failed: ${t.message}")
        }
        finish()
    }
}
