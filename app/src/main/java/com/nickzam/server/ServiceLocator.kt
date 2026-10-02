package com.nickzam.server

import com.nickzam.server.inference.EngineRegistry
import com.nickzam.server.inference.litert.SessionManager
import kotlinx.coroutines.sync.Mutex

/**
 * In-process handle to the running service's inference stack, so the UI can
 * implement model load/unload without extra HTTP routes. Set in
 * [NickZamServerService.onCreate], cleared in `onDestroy`. Null when the
 * service isn't running.
 */
object ServiceLocator {
    @Volatile var engineRegistry: EngineRegistry? = null
    @Volatile var sessionManager: SessionManager? = null
    @Volatile var inferenceMutex: Mutex? = null
}
