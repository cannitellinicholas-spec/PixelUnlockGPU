// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.download

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.produceState
import androidx.compose.ui.platform.LocalContext
import java.io.File
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Immutable snapshot of the on-disk `.litertlm` model directory.
 */
data class ModelDirSnapshot(
    val existingModels: Set<String>,
    val modelSizes: Map<String, Long>,
    val modelMtimes: Map<String, Long>,
) {
    companion object {
        val EMPTY = ModelDirSnapshot(emptySet(), emptyMap(), emptyMap())
    }
}

/**
 * Polls `context.getExternalFilesDir(null)` for `*.litertlm` files and
 * emits a [ModelDirSnapshot] on every tick.
 */
class ModelDirectoryScanner(private val context: Context) {

    /** Emit one snapshot immediately, then re-poll every [intervalMs]. */
    fun snapshots(intervalMs: Long = 2_000L): Flow<ModelDirSnapshot> = flow {
        while (true) {
            emit(scanOnce())
            delay(intervalMs)
        }
    }

    /** Read the directory exactly once. Safe to call off the main thread. */
    fun scanOnce(): ModelDirSnapshot {
        val dir = context.getExternalFilesDir(null)
        val files = dir?.listFiles { file -> file.name.endsWith(".litertlm") } ?: emptyArray<File>()
        val names = files.map { it.name }.toSet()
        val sizes = files.associate { it.name to it.length() }
        val mtimes = files.associate { it.name to it.lastModified() }
        return ModelDirSnapshot(names, sizes, mtimes)
    }
}

/**
 * Composable wrapper: returns a [State] that auto-refreshes while [active]
 * is true. Setting [active] false stops polling but keeps the last value.
 */
@Composable
fun rememberModelDirSnapshot(
    active: Boolean,
    intervalMs: Long = 2_000L,
): State<ModelDirSnapshot> {
    val context = LocalContext.current
    return produceState(initialValue = ModelDirSnapshot.EMPTY, active, context) {
        val scanner = ModelDirectoryScanner(context)
        value = scanner.scanOnce()
        if (active) {
            while (true) {
                delay(intervalMs)
                value = scanner.scanOnce()
            }
        }
    }
}
