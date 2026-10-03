package com.nickzam.server.inference

import com.nickzam.server.Backend
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class EngineKeyTest {

    @Test fun `spec flag changes the flat cache key`() {
        val k = EngineKey("gemma-4-e4b-it-gpu-spec", 4096, Backend.LITERT_GPU)
        val ks = EngineKey("gemma-4-e4b-it-gpu-spec", 4096, Backend.LITERT_GPU, spec = true)
        assertEquals("gemma-4-e4b-it-gpu-spec_4096_LITERT_GPU", k.asString())
        assertEquals("gemma-4-e4b-it-gpu-spec_4096_LITERT_GPU_spec", ks.asString())
        assertNotEquals(k, ks)
    }

    @Test fun `spec engines are distinct keys without maxTokens`() {
        val k = EngineKey("m", null, Backend.LITERT_CPU)
        val ks = EngineKey("m", null, Backend.LITERT_CPU, spec = true)
        assertEquals("m_model_LITERT_CPU", k.asString())
        assertEquals("m_model_LITERT_CPU_spec", ks.asString())
        // Equal model-id prefixes: the flat-key prefix check in /health must
        // use endsWith("_spec") for spec state, never startsWith alone.
        assertEquals(k.modelId, ks.modelId)
    }
}
