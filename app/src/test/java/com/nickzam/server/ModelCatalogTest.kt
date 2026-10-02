// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCatalogTest {

    @Test fun `catalog serves the stable NPU model and the GPU model`() {
        assertEquals(2, AVAILABLE_MODELS.size)
        val npu = findModelInfo(STABLE_MODEL_ID)!!
        assertEquals("gemma-4-e2b-it-tpu-g5", npu.id)
        assertEquals(Backend.LITERT_NPU, npu.backend)
        val gpu = findModelInfo(GPU_MODEL_ID)!!
        assertEquals("gemma-4-e4b-it-gpu", gpu.id)
        assertEquals(Backend.LITERT_GPU, gpu.backend)
    }

    @Test fun `gpu entry has pinned commit URL, hash, and no SoC gate`() {
        val info = findModelInfo(GPU_MODEL_ID)!!
        assertTrue(
            "URL must pin the audited commit, got ${info.url}",
            info.url.contains("/resolve/2eee7ac325f20eb8c9ac1d0e972f7c84663062da/"),
        )
        assertTrue(info.url.endsWith("/gemma-4-E4B-it-gpu.litertlm"))
        assertEquals("gemma-4-e4b-it-gpu.litertlm", info.filename)
        assertTrue("SHA-256 must be 64 lowercase hex", info.sha256.matches(Regex("[0-9a-f]{64}")))
        assertEquals(
            "4912bb5a9c30993c51a7711f763212077458529312175df0573a78323a2bb7ff",
            info.sha256,
        )
        assertNull(info.requiredSocMarker)
        assertEquals(2_969_059_328L, info.expectedSizeBytes)
        assertTrue(info.licenseNotice.isNotBlank())
    }

    @Test fun `stable entry has pinned commit URL and hash`() {
        val info = findModelInfo(STABLE_MODEL_ID)!!
        assertTrue(
            "URL must pin the audited commit, got ${info.url}",
            info.url.contains("/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/"),
        )
        assertTrue(info.url.endsWith("/gemma-4-E2B-it_Google_Tensor_G5.litertlm"))
        assertTrue("SHA-256 must be 64 lowercase hex", info.sha256.matches(Regex("[0-9a-f]{64}")))
        assertEquals("tensor g5", info.requiredSocMarker)
        assertEquals(3_113_545_589L, info.expectedSizeBytes)
        assertTrue(info.licenseNotice.isNotBlank())
    }

    @Test fun `context limit stays unverified until measured`() {
        assertNull(findModelInfo(STABLE_MODEL_ID)!!.contextTokens)
    }

    @Test fun `unknown model lookup returns null for explicit failure`() {
        assertNull(findModelInfo("gemma-4-e2b"))
        assertNull(findModelInfo(""))
        assertNotNull(findModelInfo(STABLE_MODEL_ID))
    }

    @Test fun `soc marker matching is case-insensitive substring`() {
        assertTrue(socMarkerMatches("Tensor G5", "tensor g5"))
        assertTrue(socMarkerMatches("Qualcomm SM8750", "sm8750"))
        assertFalse(socMarkerMatches("Tensor G4", "tensor g5"))
        assertFalse(socMarkerMatches("SM8750", "tensor g5"))
        assertFalse(socMarkerMatches(null, "tensor g5"))
        assertTrue(socMarkerMatches(null, null))
        assertTrue(socMarkerMatches("anything", null))
    }

    @Test fun `npu soc label covers tensor g5`() {
        assertEquals(
            "Google Tensor G5 (Pixel 10)",
            findModelInfo(STABLE_MODEL_ID)!!.npuSocLabel(),
        )
    }

    @Test fun `default selection is the GPU bundle`() {
        // The NPU path aborts on stock G5 firmware, so the app defaults to GPU.
        assertEquals(GPU_MODEL_ID, Settings.DEFAULT_MODEL_ID)
    }
}
