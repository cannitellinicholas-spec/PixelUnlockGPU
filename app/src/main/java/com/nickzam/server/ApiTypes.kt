// Portions of this file are derived from localllm
// (https://github.com/mlnomadpy/localllm), Copyright the localllm
// contributors, licensed under the Apache License, Version 2.0.
// Modifications by Nicholas Cannitelli, 2026.

package com.nickzam.server

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import com.google.gson.annotations.SerializedName

/**
 * OpenAI-compatible wire types for the NickZam MVP surface:
 * `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.
 *
 * Ported from localLLM's ApiTypes.kt (see docs/device-model-matrix.md for the
 * pinned source revision) with the MVP scope applied:
 * - text chat only — no tools, images, audio, or embeddings types;
 * - `tools` / `tool_choice` / `stop` are held as raw [JsonElement] so the
 *   validator can reject them explicitly instead of silently ignoring them;
 * - adds `top_p`, which the pinned LiteRT-LM `SamplerConfig` supports.
 *
 * Field names follow the OpenAI Chat Completions contract; @SerializedName
 * maps the snake_case JSON keys to camelCase Kotlin properties.
 */

data class Message(
    val role: String,
    /**
     * Polymorphic per OpenAI: string, array of parts, or null. The MVP only
     * accepts string content and arrays of `{type: "text"}` parts — anything
     * else is rejected by [validateChatRequest] before it reaches the engine.
     */
    val content: JsonElement? = null,
    /** Held raw so any presence can be rejected explicitly (no tool support). */
    @SerializedName("tool_calls") val toolCalls: JsonElement? = null,
    @SerializedName("tool_call_id") val toolCallId: String? = null,
)

/** Convenience: returns the string form of [Message.content] iff it's a primitive. */
fun Message.contentString(): String? {
    val c = content ?: return null
    return if (c.isJsonPrimitive && c.asJsonPrimitive.isString) c.asString else null
}

/** Convenience: returns parsed content parts, handling all three shapes. */
fun Message.contentParts(): List<ContentPart> = content?.toContentParts() ?: emptyList()

/** Aggregate character count across textual parts (used for prompt-size capping). */
fun Message.textChars(): Int {
    val c = content ?: return 0
    return when {
        c.isJsonNull -> 0
        c.isJsonPrimitive -> if (c.asJsonPrimitive.isString) c.asString.length else c.toString().length
        c.isJsonArray -> c.asJsonArray.sumOf { el ->
            if (el.isJsonObject) {
                val o = el.asJsonObject
                when {
                    o.has("text") && o["text"].isJsonPrimitive -> o["text"].asString.length
                    else -> 0
                }
            } else 0
        }
        else -> 0
    }
}

/**
 * A typed content fragment after parsing [Message.content]. Mirrors the
 * `{type, ...}` discriminated union OpenAI uses. Image/audio parts parse
 * successfully here so validation can report them precisely; the engine
 * path never receives them.
 */
sealed class ContentPart {
    data class TextPart(val text: String) : ContentPart()
    data class ImagePart(val url: String) : ContentPart()
    /** Any other `type` value (e.g. `input_audio`), kept for error reporting. */
    data class OtherPart(val type: String) : ContentPart()
}

/**
 * Decode any of the three `content` shapes into a uniform list of parts.
 * - `JsonPrimitive(string)` → one [ContentPart.TextPart].
 * - `JsonArray` → each element's `type` discriminator selects the variant.
 * - anything else → empty list.
 */
fun JsonElement.toContentParts(): List<ContentPart> {
    if (isJsonPrimitive && asJsonPrimitive.isString) {
        return listOf(ContentPart.TextPart(asString))
    }
    if (!isJsonArray) return emptyList()
    val out = mutableListOf<ContentPart>()
    for (el in asJsonArray) {
        if (!el.isJsonObject) continue
        val o = el.asJsonObject
        val type = o["type"]?.takeIf { it.isJsonPrimitive }?.asString ?: continue
        when (type) {
            "text" -> {
                val t = o["text"]?.takeIf { it.isJsonPrimitive }?.asString ?: continue
                out += ContentPart.TextPart(t)
            }
            "image_url" -> {
                val urlElem = o["image_url"] ?: continue
                val url = when {
                    urlElem.isJsonPrimitive -> urlElem.asString
                    urlElem.isJsonObject -> urlElem.asJsonObject["url"]
                        ?.takeIf { it.isJsonPrimitive }?.asString ?: continue
                    else -> continue
                }
                out += ContentPart.ImagePart(url)
            }
            else -> out += ContentPart.OtherPart(type)
        }
    }
    return out
}

/** Helper to construct a string-shaped content quickly (used by tests + responses). */
fun stringContent(s: String): JsonElement = JsonPrimitive(s)

/** Helper to construct a parts-array content (used by tests). */
fun partsContent(parts: List<ContentPart>): JsonElement {
    val arr = JsonArray()
    for (p in parts) {
        val o = JsonObject()
        when (p) {
            is ContentPart.TextPart -> {
                o.addProperty("type", "text")
                o.addProperty("text", p.text)
            }
            is ContentPart.ImagePart -> {
                o.addProperty("type", "image_url")
                val urlObj = JsonObject()
                urlObj.addProperty("url", p.url)
                o.add("image_url", urlObj)
            }
            is ContentPart.OtherPart -> {
                o.addProperty("type", p.type)
            }
        }
        arr.add(o)
    }
    return arr
}

data class ChatRequest(
    val model: String,
    val messages: List<Message>,
    val stream: Boolean = false,
    /**
     * Opaque conversation ID for KV-cache reuse across turns. When empty
     * (the default) every request runs in a fresh session. When non-empty
     * the server caches a LiteRT-LM `Conversation` for this ID and only
     * feeds the new user turn into it on follow-up requests — the KV cache
     * for the prior turns is preserved on the engine side. In-memory only;
     * nothing is persisted across restarts.
     */
    @SerializedName("session_id") val sessionId: String? = null,
    val temperature: Float? = null,
    @SerializedName("top_k") val topK: Int? = null,
    @SerializedName("top_p") val topP: Float? = null,
    @SerializedName("max_tokens") val maxTokens: Int? = null,
    /**
     * Polymorphic on the wire; held raw so any non-empty presence is
     * rejected with an explicit unsupported-feature error (MVP: no tools).
     */
    val tools: JsonElement? = null,
    @SerializedName("tool_choice") val toolChoice: JsonElement? = null,
    /**
     * The pinned LiteRT-LM 0.12.0 `ConversationConfig` has no stop-sequence
     * support (verified against the AAR), so any presence is rejected.
     */
    val stop: JsonElement? = null,
)

data class ChatResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>
)

data class Choice(
    val index: Int,
    val message: Message,
    @SerializedName("finish_reason") val finishReason: String
)

data class StreamResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<StreamChoice>
)

data class StreamChoice(
    val index: Int,
    val delta: StreamDelta,
    @SerializedName("finish_reason") val finishReason: String?
)

data class StreamDelta(
    val role: String? = null,
    val content: String? = null,
)

data class ErrorResponse(val error: ErrorDetails)

data class ErrorDetails(
    val message: String,
    val type: String,
    val code: Int
)

/**
 * Richer error envelope for the LiteRT-LM failure paths in
 * `/v1/chat/completions`. Extension to the baseline OpenAI error shape —
 * clients that only know OpenAI still get a readable `message` / `type`.
 */
data class RichErrorResponse(val error: RichErrorDetails)

data class RichErrorDetails(
    val message: String,
    val type: String,
    val code: String,
    val actionable: Boolean = false,
    @SerializedName("next_steps") val nextSteps: List<String> = emptyList(),
)

data class ModelListResponse(
    val `object`: String = "list",
    val data: List<ModelData>
)

data class ModelData(
    val id: String,
    val `object`: String = "model",
    val created: Long = 0,
    @SerializedName("owned_by") val ownedBy: String = "nickzam"
)
