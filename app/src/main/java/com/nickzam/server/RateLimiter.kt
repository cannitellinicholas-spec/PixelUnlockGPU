// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import java.util.concurrent.ConcurrentHashMap
import kotlin.math.max
import kotlin.math.min

/**
 * Per-client token-bucket rate limiter. One bucket per client identity
 * (typically the request's `User-Agent` header). Each bucket starts full at
 * [burst] tokens and refills at [ratePerSec] tokens per second. A request
 * that finds an empty bucket is rejected with the number of seconds the
 * caller should wait before retrying.
 */
class RateLimiter(
    @Volatile var ratePerSec: Double,
    @Volatile var burst: Double,
) {
    private data class Bucket(
        @Volatile var tokens: Double,
        @Volatile var lastRefillNanos: Long,
    )

    private val buckets = ConcurrentHashMap<String, Bucket>()

    /**
     * Attempts to charge one token to [client]. Returns null on success;
     * returns the number of seconds the caller should wait before
     * retrying when the bucket is empty.
     */
    fun tryAcquire(client: String, now: Long = System.nanoTime()): Long? {
        val bucket = buckets.computeIfAbsent(client) {
            Bucket(tokens = burst, lastRefillNanos = now)
        }
        synchronized(bucket) {
            val elapsedSec = (now - bucket.lastRefillNanos) / 1_000_000_000.0
            bucket.tokens = min(burst, bucket.tokens + elapsedSec * ratePerSec)
            bucket.lastRefillNanos = now
            if (bucket.tokens >= 1.0) {
                bucket.tokens -= 1.0
                return null
            }
            // Not enough tokens — compute when one full token will be available.
            val deficit = 1.0 - bucket.tokens
            val secondsToWait = if (ratePerSec > 0) deficit / ratePerSec else Double.POSITIVE_INFINITY
            // Round up to at least 1 second so a 429 sets a meaningful Retry-After.
            return max(1L, kotlin.math.ceil(secondsToWait).toLong())
        }
    }

    /** Drop accumulated buckets (e.g. on settings change so new limits take effect). */
    fun reset() {
        buckets.clear()
    }
}
