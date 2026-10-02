package com.nickzam.server

import android.content.Context
import android.os.PowerManager
import android.util.Log

/**
 * Best-effort device thermals for the GPU temp gauge.
 *
 * Normal apps cannot read the sysfs thermal nodes (Permission denied), so this
 * goes through the framework:
 *  1. Public API: PowerManager.getCurrentThermalStatus() +
 *     getThermalHeadroom() — always available, coarse.
 *  2. Reflection on IThermalService.getCurrentTemperatures() for the named
 *     "GPU" sensor — hidden API, blocked on recent Android unless the
 *     device grants it. Everything is runCatching; null means unavailable.
 */
object ThermalMonitor {
    private const val TAG = "ThermalMonitor"

    /** Cached reflection handles so the poll loop doesn't re-resolve names. */
    private var reflectTried = false
    private var reflectOk = false

    data class Snapshot(
        val status: Int,           // PowerManager.THERMAL_STATUS_*
        val headroom: Float,       // 0..1, 1 = maximum margin left
        val gpuTempC: Double?,     // named GPU sensor, null if blocked
        val sensorSource: String,  // "IThermalService" | "blocked"
    )

    fun read(context: Context): Snapshot {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val status = runCatching { pm.currentThermalStatus }.getOrDefault(0)
        val headroom = runCatching { pm.getThermalHeadroom(0) }.getOrDefault(-1f)
        val (temp, source) = readGpuTemp()
        return Snapshot(status, headroom, temp, source)
    }

    fun statusName(status: Int): String = when (status) {
        0 -> "NOMINAL"
        1 -> "FAIR"
        2 -> "SERIOUS"
        3 -> "CRITICAL"
        4 -> "EMERGENCY"
        5 -> "SHUTDOWN"
        else -> "STATUS_$status"
    }

    private fun readGpuTemp(): Pair<Double?, String> {
        if (!reflectTried) {
            reflectTried = true
            // Try every documented route; log each failure so the exact wall
            // is on the record (adb: `logcat -s ThermalMonitor`).
            reflectOk = probeHalService() || probeFrameworkService()
        }
        if (!reflectOk) return null to "blocked"
        val temp = runCatching {
            val temps = svcCall!!.invoke(svcObj) as List<*>
            temps.mapNotNull { t ->
                val name = t!!.javaClass.getMethod("getName").invoke(t) as? String
                val value = t.javaClass.getMethod("getValue").invoke(t) as? Float
                if (name == "GPU" && value != null && value > -1e30f) value.toDouble() else null
            }.firstOrNull()
        }.getOrNull()
        return temp to "IThermalService"
    }

    private var svcObj: Any? = null
    private var svcCall: java.lang.reflect.Method? = null

    /** HAL: android.hardware.thermal.IThermalService via ServiceManager "thermal". */
    private fun probeHalService(): Boolean = runCatching {
        val binder = Class.forName("android.os.ServiceManager")
            .getMethod("getService", String::class.java)
            .invoke(null, "thermal")
            ?: error("getService(thermal)=null")
        val stub = Class.forName("android.hardware.thermal.IThermalService\$Stub")
        val obj = stub.getMethod("asInterface", android.os.IBinder::class.java)
            .invoke(null, binder) ?: error("asInterface=null")
        val m = obj.javaClass.getMethod("getCurrentTemperatures")
        m.invoke(obj) // must be callable, not just resolvable
        svcObj = obj; svcCall = m
        Log.i(TAG, "HAL IThermalService reachable")
        true
    }.getOrElse {
        Log.w(TAG, "HAL path failed: ${it.javaClass.simpleName}: ${it.message}")
        false
    }

    /** Framework: android.os.IThermalService via ServiceManager "thermalservice". */
    private fun probeFrameworkService(): Boolean = runCatching {
        val binder = Class.forName("android.os.ServiceManager")
            .getMethod("getService", String::class.java)
            .invoke(null, "thermalservice")
            ?: error("getService(thermalservice)=null")
        val stub = Class.forName("android.os.IThermalService\$Stub")
        val obj = stub.getMethod("asInterface", android.os.IBinder::class.java)
            .invoke(null, binder) ?: error("asInterface=null")
        // getCurrentTemperatures() no-arg returns all sensors.
        val m = obj.javaClass.getMethod("getCurrentTemperatures")
        m.invoke(obj)
        svcObj = obj; svcCall = m
        Log.i(TAG, "framework IThermalService reachable")
        true
    }.getOrElse {
        Log.w(TAG, "framework path failed: ${it.javaClass.simpleName}: ${it.message}")
        false
    }
}
