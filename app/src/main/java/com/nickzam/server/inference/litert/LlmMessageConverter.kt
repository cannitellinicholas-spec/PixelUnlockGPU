// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server.inference.litert

import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine as LiteRtNativeEngine
import com.google.ai.edge.litertlm.Message as LlmMessage
import com.google.ai.edge.litertlm.SamplerConfig
import com.nickzam.server.ContentPart
import com.nickzam.server.Message
import com.nickzam.server.contentParts

/**
 * Conversion helpers between OpenAI-shaped API types and LiteRT-LM's native
 * message model. Nothing here touches Ktor or SharedPreferences.
 *
 * MVP scope is text-only: the chat route rejects image/audio/tool content
 * before it reaches this converter, and [buildContents]/[apiToLlmMessage]
 * throw defensively if any ever arrives.
 */
object LlmMessageConverter {

    /** Plain-text view of a [LlmMessage] — non-text parts are dropped. */
    fun messageText(msg: LlmMessage): String {
        val parts = msg.contents.contents
        if (parts.isEmpty()) return ""
        val sb = StringBuilder()
        for (p in parts) {
            if (p is Content.Text) sb.append(p.text)
        }
        return sb.toString()
    }

    fun apiToLlmMessage(m: Message): LlmMessage {
        return when (m.role) {
            "assistant" -> LlmMessage.model(Contents.of(buildContents(m)))
            "system" -> LlmMessage.system(Contents.of(buildContents(m)))
            "tool" -> throw IllegalArgumentException("tool messages are not supported")
            else -> LlmMessage.user(Contents.of(buildContents(m)))
        }
    }

    fun buildContents(message: Message): List<Content> {
        val parts = message.contentParts()
        if (parts.isEmpty()) return emptyList()
        val out = mutableListOf<Content>()
        for (p in parts) {
            when (p) {
                is ContentPart.TextPart -> if (p.text.isNotEmpty()) out += Content.Text(p.text)
                is ContentPart.ImagePart ->
                    throw IllegalArgumentException("image_url content is not supported")
                is ContentPart.OtherPart ->
                    throw IllegalArgumentException("content type '${p.type}' is not supported")
            }
        }
        return out
    }

    /**
     * Build a pre-loaded [Conversation]. The caller is responsible for
     * closing it (or registering it as the active conversation for the
     * engine and letting [EngineRegistry] clean it up on eviction).
     *
     * `automaticToolCalling` is false and no tools are attached: the MVP
     * serves text only.
     */
    fun createConversation(
        engine: LiteRtNativeEngine,
        temperature: Float,
        topK: Int,
        topP: Float,
        systemText: String?,
        initial: List<Message>,
    ): Conversation {
        val systemInstruction = systemText?.takeIf { it.isNotBlank() }?.let { Contents.of(it) }
        val priorMessages = initial.map { apiToLlmMessage(it) }
        val cfg = ConversationConfig(
            systemInstruction,
            priorMessages,
            /*tools=*/ emptyList(),
            SamplerConfig(topK, topP.toDouble(), temperature.toDouble(), /*seed=*/ 0),
            /*automaticToolCalling=*/ false,
        )
        return engine.createConversation(cfg)
    }
}
