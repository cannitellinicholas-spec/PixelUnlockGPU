package com.nickzam.server.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class Sha256Test {

    @Rule @JvmField val tmp = TemporaryFolder()

    @Test fun `ofFile matches known vector`() {
        val f = tmp.newFile("a.bin").apply { writeBytes("abc".toByteArray()) }
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            Sha256.ofFile(f),
        )
    }

    @Test fun `verify passes on match case-insensitively`() {
        val f = tmp.newFile("b.bin").apply { writeBytes("abc".toByteArray()) }
        assertNull(Sha256.verifyOrRefusal(f, "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"))
    }

    @Test fun `verify refuses mismatch without leaking full path`() {
        val f = tmp.newFile("c.bin").apply { writeBytes("abc".toByteArray()) }
        val refusal = Sha256.verifyOrRefusal(f, "0".repeat(64))!!
        assertTrue(refusal.contains("mismatch"))
        assertTrue(refusal.contains("c.bin"))
        // Sanitized: hash prefixes only, no full private path.
        assertTrue(!refusal.contains(f.absolutePath))
        assertTrue(!refusal.contains("0".repeat(64)))
    }

    @Test fun `verify refuses missing file`() {
        val missing = File(tmp.root, "nope.litertlm")
        val refusal = Sha256.verifyOrRefusal(missing, "0".repeat(64))!!
        assertNotNull(refusal)
        assertTrue(refusal.contains("nope.litertlm"))
    }
}
