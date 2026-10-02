// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow

data class LogEntry(val timestamp: Long, val level: String, val message: String) {
    val formattedTime: String
        get() {
            val sdf = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.US)
            return sdf.format(java.util.Date(timestamp))
        }
}

object LogManager {
    private val _logs = MutableSharedFlow<LogEntry>(replay = 100, extraBufferCapacity = 100)
    val logs = _logs.asSharedFlow()

    fun d(tag: String, message: String) {
        val entry = LogEntry(System.currentTimeMillis(), "DEBUG", "[$tag] $message")
        _logs.tryEmit(entry)
        android.util.Log.d(tag, message)
    }

    fun i(tag: String, message: String) {
        val entry = LogEntry(System.currentTimeMillis(), "INFO", "[$tag] $message")
        _logs.tryEmit(entry)
        android.util.Log.i(tag, message)
    }

    fun w(tag: String, message: String) {
        val entry = LogEntry(System.currentTimeMillis(), "WARN", "[$tag] $message")
        _logs.tryEmit(entry)
        android.util.Log.w(tag, message)
    }

    fun e(tag: String, message: String, throwable: Throwable? = null) {
        val errMessage = throwable?.let { "$message: ${it.message}" } ?: message
        val entry = LogEntry(System.currentTimeMillis(), "ERROR", "[$tag] $errMessage")
        _logs.tryEmit(entry)
        android.util.Log.e(tag, message, throwable)
    }
}
