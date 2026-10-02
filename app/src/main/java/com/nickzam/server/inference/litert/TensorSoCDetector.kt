// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference.litert

import android.os.Build

/**
 * SoC detection helpers. Tensor-family quirks do not drive backend
 * selection (that's catalog-declared) — they only drive the JNI primer
 * workaround inside [LiteRtEngineBuilder] for stuck LiteRT-LM CPU/GPU
 * initialization on Pixel 6+.
 *
 * `Build.SOC_MODEL` is API 31+. Older devices return `false` from
 * [isTensorSoc]; the Tensor family launched on Android 12 so this is correct
 * by construction.
 */
object TensorSoCDetector {
    fun isTensorSoc(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return false
        val soc = Build.SOC_MODEL?.lowercase() ?: return false
        // ro.soc.model carries the marketing name on newer Pixels
        // (Pixel 10 reports "Tensor G5"); older ones report the codename
        // (GS101/GS201/ZUMA/ZUMA_PRO). Match either shape.
        if (soc.contains("tensor")) return true
        return soc.startsWith("gs10") ||
            soc.startsWith("gs20") ||
            soc == "zuma" ||
            soc == "zuma_pro" ||
            soc == "laguna"
    }
}
