package com.nickzam.server.inference

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * SHA-256 artifact verification. Fail-closed: any mismatch, missing file, or
 * read error refuses the load — the server never runs an unverified bundle
 * under a catalog model ID.
 */
object Sha256 {

    /** Lowercase hex SHA-256 of [file]. Streams — safe for multi-GB bundles. */
    fun ofFile(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /**
     * Returns null when [file]'s hash equals [expectedHex] (case-insensitive),
     * otherwise a sanitized refusal reason naming only the file name — never
     * a full private path.
     */
    fun verifyOrRefusal(file: File, expectedHex: String): String? {
        if (!file.exists()) return "Model file not found: ${file.name}. Download or import it first."
        val actual = try {
            ofFile(file)
        } catch (e: Exception) {
            return "Could not hash ${file.name}: ${e.javaClass.simpleName}. Refusing to load."
        }
        if (!actual.equals(expectedHex, ignoreCase = true)) {
            return "SHA-256 mismatch for ${file.name} " +
                "(expected ${expectedHex.take(16)}…, got ${actual.take(16)}…). " +
                "Delete and re-download the model."
        }
        return null
    }
}
