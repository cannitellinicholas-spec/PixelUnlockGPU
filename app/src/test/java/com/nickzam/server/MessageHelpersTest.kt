package com.nickzam.server

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MessageHelpersTest {

    @Test fun `string content parses to one text part`() {
        val parts = stringContent("hello").toContentParts()
        assertEquals(listOf(ContentPart.TextPart("hello")), parts)
    }

    @Test fun `text array parses to text parts`() {
        val el = JsonParser.parseString("""[{"type":"text","text":"a"},{"type":"text","text":"b"}]""")
        assertEquals(listOf(ContentPart.TextPart("a"), ContentPart.TextPart("b")), el.toContentParts())
    }

    @Test fun `image_url object and string shapes parse`() {
        val obj = JsonParser.parseString("""[{"type":"image_url","image_url":{"url":"http://x"}}]""")
        assertEquals(listOf(ContentPart.ImagePart("http://x")), obj.toContentParts())
        val str = JsonParser.parseString("""[{"type":"image_url","image_url":"http://y"}]""")
        assertEquals(listOf(ContentPart.ImagePart("http://y")), str.toContentParts())
    }

    @Test fun `unknown part types are preserved for error reporting`() {
        val el = JsonParser.parseString("""[{"type":"input_audio"}]""")
        assertEquals(listOf(ContentPart.OtherPart("input_audio")), el.toContentParts())
    }

    @Test fun `non-array non-string content yields no parts`() {
        assertTrue(JsonParser.parseString("42").toContentParts().isEmpty())
    }

    @Test fun `textChars counts string and text parts`() {
        assertEquals(5, Message("user", stringContent("hello")).textChars())
        val arr = JsonParser.parseString("""[{"type":"text","text":"ab"},{"type":"image_url","image_url":"http://x"}]""")
        assertEquals(2, Message("user", arr).textChars())
        assertEquals(0, Message("assistant", null).textChars())
    }

    @Test fun `prefix hash is stable and prefix-sensitive`() {
        val a = listOf(
            Message("user", stringContent("hi")),
            Message("assistant", stringContent("hello")),
            Message("user", stringContent("more")),
        )
        val b = listOf(
            Message("user", stringContent("hi")),
            Message("assistant", stringContent("hello")),
            Message("user", stringContent("different")),
        )
        assertEquals(messagesPrefixHash(a, 2), messagesPrefixHash(b, 2))
        assertNotEquals(messagesPrefixHash(a, 3), messagesPrefixHash(b, 3))
        assertEquals(messagesPrefixHash(a, 3), messagesPrefixHash(a, 3))
    }

    @Test fun `prefix hash mixes in role`() {
        val a = listOf(Message("user", stringContent("x")))
        val b = listOf(Message("assistant", stringContent("x")))
        assertNotEquals(messagesPrefixHash(a, 1), messagesPrefixHash(b, 1))
    }
}
