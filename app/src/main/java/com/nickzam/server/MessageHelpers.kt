package com.nickzam.server

/**
 * Pure, dependency-free helpers so that the non-trivial request-hashing
 * logic is independently testable on the JVM (no Android, no LiteRT-LM).
 */

/**
 * Hash of `messages[0 until count]`. Used by
 * [com.nickzam.server.inference.litert.SessionManager] to detect when a
 * client is replaying a different prefix than what was recorded on the
 * cached conversation — in which case the cache entry must be rebuilt.
 *
 * Mixes in role, full content body (JSON-serialized so string and array
 * shapes hash to the same value when the underlying content is equivalent),
 * tool_call_id, and tool_calls so a changed turn invalidates the cache.
 */
fun messagesPrefixHash(messages: List<Message>, count: Int): Long {
    var h = 1L
    val n = minOf(count, messages.size)
    for (i in 0 until n) {
        val m = messages[i]
        h = h * 31L + m.role.hashCode()
        h = h * 31L + (m.content?.toString()?.hashCode() ?: 0)
        h = h * 31L + (m.toolCallId?.hashCode() ?: 0)
        h = h * 31L + (m.toolCalls?.hashCode() ?: 0)
    }
    return h
}
