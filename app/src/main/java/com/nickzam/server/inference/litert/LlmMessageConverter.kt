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
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool
import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.nickzam.server.ContentPart
import com.nickzam.server.Message
import com.nickzam.server.contentParts
import com.nickzam.server.contentString
import com.nickzam.server.toolCallsList

/**
 * Conversion helpers between OpenAI-shaped API types and LiteRT-LM's native
 * message model. Nothing here touches Ktor or SharedPreferences.
 *
 * Text and native tool-calling are supported (tools path): a client's
 * declared `tools` array becomes [DelegatingToolProvider]s with
 * automaticToolCalling=false, so generation halts on tool_calls, the HTTP
 * route relays them to the client, and follow-up `role:"tool"` turns map
 * back to native TOOL messages. Images/audio remain out of MVP scope and
 * throw defensively.
 */
object LlmMessageConverter {

    private val GSON = Gson()

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

    fun apiToLlmMessage(m: Message, toolNames: Map<String, String> = emptyMap()): LlmMessage {
        return when (m.role) {
            "assistant" -> {
                val tcs = m.toolCallsList()
                if (tcs.isEmpty()) {
                    LlmMessage.model(Contents.of(buildContents(m)))
                } else {
                    // Replay of a prior tool-call turn: OpenAI's
                    // tool_calls[i].function.{name, arguments(json string)}
                    // maps onto LiteRT-LM ToolCall(name, argsMap).
                    val calls = tcs.mapNotNull { tc ->
                        val fn = tc.getAsJsonObject()?.get("function")?.asJsonObject ?: return@mapNotNull null
                        val name = fn.get("name")?.asString ?: return@mapNotNull null
                        val args = runCatching {
                            @Suppress("UNCHECKED_CAST")
                            GSON.fromJson(fn.get("arguments")?.asString ?: "{}", Map::class.java)
                                as? Map<String, Any?> ?: emptyMap()
                        }.getOrDefault(emptyMap())
                        ToolCall(name, args)
                    }
                    LlmMessage.model(Contents.of(buildContents(m)), calls, emptyMap())
                }
            }
            "system" -> LlmMessage.system(Contents.of(buildContents(m)))
            "tool" -> {
                // OpenAI tool results carry only tool_call_id; the tool NAME
                // comes from the preceding assistant tool_calls entry.
                val name = m.toolCallId?.let { toolNames[it] } ?: m.toolCallId ?: "unknown_tool"
                val text = m.contentString()
                    ?: buildContents(m).joinToString(" ") { (it as? Content.Text)?.text ?: "" }
                LlmMessage.tool(Contents.of(Content.ToolResponse(name, text)))
            }
            else -> LlmMessage.user(Contents.of(buildContents(m)))
        }
    }

    /** id -> tool-name map scraped from assistant `tool_calls` entries. */
    fun toolNameMap(messages: List<Message>): Map<String, String> {
        val out = mutableMapOf<String, String>()
        for (m in messages) {
            if (m.role != "assistant") continue
            for (tc in m.toolCallsList()) {
                val o = tc.getAsJsonObject() ?: continue
                val id = o.get("id")?.asString ?: continue
                val name = o.get("function")?.asJsonObject?.get("name")?.asString ?: continue
                out[id] = name
            }
        }
        return out
    }

    /**
     * Relay direction: LiteRT-LM ToolCall -> OpenAI tool_calls entry. Ids are
     * synthesized as "call_<index>" per response; clients echo them back in
     * the tool result turn and [toolNameMap] recovers the correlation.
     */
    fun toOpenAiToolCalls(calls: List<ToolCall>): JsonArray {
        val arr = JsonArray()
        calls.forEachIndexed { i, tc ->
            val fn = JsonObject().apply {
                addProperty("name", tc.name)
                addProperty("arguments", GSON.toJson(tc.arguments))
            }
            arr.add(JsonObject().apply {
                addProperty("id", "call_$i")
                addProperty("type", "function")
                add("function", fn)
            })        }
        return arr
    }

    /**
     * Wrap one OpenAI `tools[i]` element as a LiteRT-LM ToolProvider. The
     * element is either `{"type":"function","function":{name,description,
     * parameters}}` or the bare function object. The model's tool_calls are
     * never executed server-side (automaticToolCalling=false): execute()
     * exists only to satisfy the interface and is never reached, mirroring
     * the OpenAI relay contract where the client runs the tool.
     */
    fun toolProviderFrom(toolsEl: JsonElement): List<ToolProvider> {
        val arr = when {
            toolsEl.isJsonArray -> toolsEl.asJsonArray
            toolsEl.isJsonObject -> JsonArray().apply { add(toolsEl) }
            else -> return emptyList()
        }
        return arr.mapNotNull { el ->
            val o = el.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            val fn = if (o.has("function") && o.get("function").isJsonObject)
                o.getAsJsonObject("function") else o
            val name = fn.get("name")?.asString?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
            val desc = JsonParser.parseString(fn.toString()).asJsonObject
            val openApi = object : OpenApiTool {
                override fun getToolDescriptionJsonString(): String = desc.toString()
                override fun execute(argsJson: String): String =
                    throw UnsupportedOperationException(
                        "tools are relayed to the client; the server never executes them")
            }
            tool(openApi)
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
     * With [tools] non-empty, `automaticToolCalling` stays false: the model
     * halts on tool_calls and the HTTP route relays them to the client,
     * which executes the tool and replays the conversation with a tool
     * result. Without tools the behavior is the classic text-only path.
     */
    fun createConversation(
        engine: LiteRtNativeEngine,
        temperature: Float,
        topK: Int,
        topP: Float,
        systemText: String?,
        initial: List<Message>,
        toolProviders: List<ToolProvider> = emptyList(),
    ): Conversation {
        val systemInstruction = systemText?.takeIf { it.isNotBlank() }?.let { Contents.of(it) }
        val toolNames = toolNameMap(initial)
        val priorMessages = initial.map { apiToLlmMessage(it, toolNames) }
        val cfg = ConversationConfig(
            systemInstruction,
            priorMessages,
            /*tools=*/ toolProviders,
            SamplerConfig(topK, topP.toDouble(), temperature.toDouble(), /*seed=*/ 0),
            /*automaticToolCalling=*/ false,
        )
        return engine.createConversation(cfg)
    }
}
