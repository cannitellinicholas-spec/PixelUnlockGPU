package com.nickzam.server

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatValidationTest {

    private fun msg(role: String, content: String) =
        Message(role = role, content = stringContent(content))

    private fun base() = ChatRequest(
        model = STABLE_MODEL_ID,
        messages = listOf(msg("user", "Hi")),
    )

    @Test fun `valid minimal request passes`() {
        assertNull(ChatValidation.validate(base()))
    }

    @Test fun `valid request with system prompt and sampling passes`() {
        val req = base().copy(
            messages = listOf(msg("system", "Be brief"), msg("user", "Hi")),
            stream = true,
            sessionId = "abc",
            temperature = 0.7f,
            topK = 40,
            topP = 0.9f,
            maxTokens = 512,
        )
        assertNull(ChatValidation.validate(req))
    }

    @Test fun `blank model is 400`() {
        val f = ChatValidation.validate(base().copy(model = ""))!!
        assertEquals(400, f.httpStatus)
    }

    @Test fun `any catalog model is accepted`() {
        for (info in AVAILABLE_MODELS) {
            assertNull("catalog id ${info.id} must validate", ChatValidation.validate(base().copy(model = info.id)))
        }
    }

    @Test fun `unknown model is explicit 400 never mapped`() {
        val f = ChatValidation.validate(base().copy(model = "gemma-4-e2b"))!!
        assertEquals(400, f.httpStatus)
        assertTrue(f.message.contains("gemma-4-e2b"))
        assertTrue(f.message.contains(STABLE_MODEL_ID))
        assertTrue(f.message.contains(GPU_MODEL_ID))
    }

    @Test fun `empty messages is 400`() {
        val f = ChatValidation.validate(base().copy(messages = emptyList()))!!
        assertEquals(400, f.httpStatus)
    }

    @Test fun `unsupported role is 400`() {
        val f = ChatValidation.validate(base().copy(messages = listOf(msg("tool", "x"))))!!
        assertEquals(400, f.httpStatus)
        assertTrue(f.message.contains("tool"))
    }

    @Test fun `developer role is 400`() {
        val f = ChatValidation.validate(base().copy(messages = listOf(msg("developer", "x"))))!!
        assertEquals(400, f.httpStatus)
    }

    @Test fun `last non-system message must be user`() {
        val f = ChatValidation.validate(base().copy(
            messages = listOf(msg("user", "Hi"), msg("assistant", "Hello")),
        ))!!
        assertEquals(400, f.httpStatus)
    }

    @Test fun `system-only request is 400`() {
        val f = ChatValidation.validate(base().copy(messages = listOf(msg("system", "x"))))!!
        assertEquals(400, f.httpStatus)
    }

    @Test fun `tools are rejected explicitly`() {
        val tools = JsonParser.parseString("""[{"type":"function"}]""")
        val f = ChatValidation.validate(base().copy(tools = tools))!!
        assertEquals(400, f.httpStatus)
        assertEquals("unsupported_feature", f.type)
    }

    @Test fun `empty tools array is accepted`() {
        val req = base().copy(tools = JsonArray())
        assertNull(ChatValidation.validate(req))
    }

    @Test fun `tool_choice auto is rejected`() {
        val req = base().copy(toolChoice = JsonParser.parseString(""""auto""""))
        val f = ChatValidation.validate(req)!!
        assertEquals("unsupported_feature", f.type)
    }

    @Test fun `tool_choice none is accepted`() {
        val req = base().copy(toolChoice = JsonParser.parseString(""""none""""))
        assertNull(ChatValidation.validate(req))
    }

    @Test fun `stop string is rejected`() {
        val req = base().copy(stop = JsonParser.parseString(""""END""""))
        val f = ChatValidation.validate(req)!!
        assertEquals("unsupported_feature", f.type)
    }

    @Test fun `stop list is rejected`() {
        val req = base().copy(stop = JsonParser.parseString("""["a"]"""))
        val f = ChatValidation.validate(req)!!
        assertEquals("unsupported_feature", f.type)
    }

    @Test fun `image_url part is rejected explicitly`() {
        val content = JsonParser.parseString(
            """[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]"""
        )
        val req = base().copy(messages = listOf(Message("user", content)))
        val f = ChatValidation.validate(req)!!
        assertEquals("unsupported_feature", f.type)
        assertTrue(f.message.contains("image_url"))
    }

    @Test fun `audio part is rejected explicitly`() {
        val content = JsonParser.parseString("""[{"type":"input_audio","input_audio":{}}]""")
        val req = base().copy(messages = listOf(Message("user", content)))
        val f = ChatValidation.validate(req)!!
        assertEquals("unsupported_feature", f.type)
        assertTrue(f.message.contains("input_audio"))
    }

    @Test fun `text parts array is accepted`() {
        val content = partsContent(listOf(ContentPart.TextPart("a"), ContentPart.TextPart("b")))
        val req = base().copy(messages = listOf(Message("user", content)))
        assertNull(ChatValidation.validate(req))
    }

    @Test fun `null content is 400`() {
        val req = base().copy(messages = listOf(Message("user", null)))
        assertEquals(400, ChatValidation.validate(req)!!.httpStatus)
    }

    @Test fun `assistant tool_calls are rejected`() {
        val m = Message("assistant", stringContent("x"),
            toolCalls = JsonParser.parseString("""[{"id":"1"}]"""))
        val req = ChatRequest(STABLE_MODEL_ID, listOf(msg("user", "Hi"), m, msg("user", "go")))
        assertEquals("unsupported_feature", ChatValidation.validate(req)!!.type)
    }

    @Test fun `temperature out of range is 400`() {
        assertNotNull(ChatValidation.validate(base().copy(temperature = -0.1f)))
        assertNotNull(ChatValidation.validate(base().copy(temperature = 2.1f)))
        assertNotNull(ChatValidation.validate(base().copy(temperature = Float.NaN)))
        assertNull(ChatValidation.validate(base().copy(temperature = 0f)))
        assertNull(ChatValidation.validate(base().copy(temperature = 2f)))
    }

    @Test fun `top_p out of range is 400`() {
        assertNotNull(ChatValidation.validate(base().copy(topP = 0f)))
        assertNotNull(ChatValidation.validate(base().copy(topP = 1.5f)))
        assertNull(ChatValidation.validate(base().copy(topP = 1f)))
    }

    @Test fun `top_k below 1 is 400`() {
        assertNotNull(ChatValidation.validate(base().copy(topK = 0)))
        assertNull(ChatValidation.validate(base().copy(topK = 1)))
    }

    @Test fun `max_tokens bounds enforced`() {
        assertNotNull(ChatValidation.validate(base().copy(maxTokens = 0)))
        assertNotNull(ChatValidation.validate(base().copy(maxTokens = -5)))
        assertNotNull(ChatValidation.validate(base().copy(maxTokens = ChatValidation.MAX_TOKENS_CEILING + 1)))
        assertNull(ChatValidation.validate(base().copy(maxTokens = 1)))
        assertNull(ChatValidation.validate(base().copy(maxTokens = ChatValidation.MAX_TOKENS_CEILING)))
    }
}
