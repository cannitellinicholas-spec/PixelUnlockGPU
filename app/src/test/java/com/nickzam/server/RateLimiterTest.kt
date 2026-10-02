// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class RateLimiterTest {

    @Test fun `burst allows then rejects with retry-after`() {
        val rl = RateLimiter(ratePerSec = 1.0, burst = 2.0)
        val t0 = 1_000_000_000L
        assertNull(rl.tryAcquire("c", t0))
        assertNull(rl.tryAcquire("c", t0))
        val wait = rl.tryAcquire("c", t0)
        assertNotNull(wait)
        assertEquals(1L, wait)
    }

    @Test fun `refill over time admits again`() {
        val rl = RateLimiter(ratePerSec = 1.0, burst = 1.0)
        val t0 = 1_000_000_000L
        assertNull(rl.tryAcquire("c", t0))
        assertNotNull(rl.tryAcquire("c", t0))
        assertNull(rl.tryAcquire("c", t0 + 1_100_000_000L))
    }

    @Test fun `clients have independent buckets`() {
        val rl = RateLimiter(ratePerSec = 1.0, burst = 1.0)
        val t0 = 1_000_000_000L
        assertNull(rl.tryAcquire("a", t0))
        assertNull(rl.tryAcquire("b", t0))
        assertNotNull(rl.tryAcquire("a", t0))
    }
}
