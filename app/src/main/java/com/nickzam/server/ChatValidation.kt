package com.nickzam.server

import com.google.gson.JsonElement

/**
 * Pure (JVM-testable, no Android) validation for `POST /v1/chat/completions`
 * implementing the spec's HTTP API contract:
 *
 * - `model` must equal a catalog ID exactly; unknown IDs fail explicitly
 *   (never mapped to "the selected model").
 * - Only system/user/assistant text messages, in order, ending in a user turn.
 * - tools/function calling, images, audio, multimodal parts, and `stop` are
 *   rejected with explicit unsupported-feature errors — never silently
 *   ignored.
 * - Sampling knobs are range-checked where the pinned engine implements them
 *   (temperature, top_p, top_k, max_tokens).
 */
object ChatValidation {

    /** Hard ceiling for `max_tokens` until Gate 5 measures the real limit. */
    const val MAX_TOKENS_CEILING = 32_768

    data class Failure(
        val httpStatus: Int,
        val type: String,
        val message: String,
    )

    fun validate(req: ChatRequest): Failure? {
        // Model: required, must be an exact catalog ID (any served entry).
        if (req.model.isBlank()) {
            return Failure(400, "invalid_request_error",
                "model is required; this server serves " +
                    AVAILABLE_MODELS.joinToString(", ") { "\"${it.id}\"" })
        }
        if (findModelInfo(req.model) == null) {
            return Failure(400, "invalid_request_error",
                "Unknown model '${req.model}'. This server only serves " +
                    AVAILABLE_MODELS.joinToString(", ") { "\"${it.id}\"" } + ".")
        }

        if (req.messages.isEmpty()) {
            return Failure(400, "invalid_request_error", "messages must not be empty")
        }

        // Explicitly unsupported request features (MVP is text-only).
        if (isNonEmpty(req.tools)) {
            return Failure(400, "unsupported_feature",
                "tools/function calling is not supported by this server")
        }
        if (isPresent(req.toolChoice) && !isNoneChoice(req.toolChoice)) {
            return Failure(400, "unsupported_feature",
                "tool_choice is not supported by this server")
        }
        if (isNonEmpty(req.stop)) {
            return Failure(400, "unsupported_feature",
                "stop sequences are not supported by the pinned inference engine")
        }

        // Roles and content, in order.
        for ((i, m) in req.messages.withIndex()) {
            if (m.role != "system" && m.role != "user" && m.role != "assistant") {
                return Failure(400, "invalid_request_error",
                    "messages[$i]: unsupported role '${m.role}' (supported: system, user, assistant)")
            }
            if (isPresent(m.toolCalls)) {
                return Failure(400, "unsupported_feature",
                    "messages[$i]: tool_calls are not supported by this server")
            }
            if (m.toolCallId != null) {
                return Failure(400, "unsupported_feature",
                    "messages[$i]: tool follow-up turns are not supported by this server")
            }
            val content = m.content
            if (content == null || content.isJsonNull) {
                return Failure(400, "invalid_request_error",
                    "messages[$i]: content must be a non-empty string or text parts")
            }
            for (part in content.toContentParts()) {
                when (part) {
                    is ContentPart.TextPart -> Unit
                    is ContentPart.ImagePart -> return Failure(400, "unsupported_feature",
                        "messages[$i]: image_url content is not supported by this server")
                    is ContentPart.OtherPart -> return Failure(400, "unsupported_feature",
                        "messages[$i]: content type '${part.type}' is not supported by this server")
                }
            }
            if (content.isJsonArray && content.asJsonArray.size() > 0 && m.textChars() == 0 &&
                content.toContentParts().isEmpty()
            ) {
                return Failure(400, "invalid_request_error",
                    "messages[$i]: content array has no usable text parts")
            }
        }

        // The last non-system message must be a user turn (what the engine sends).
        val nonSystem = req.messages.filter { it.role != "system" }
        if (nonSystem.isEmpty()) {
            return Failure(400, "invalid_request_error", "Request has no non-system messages")
        }
        if (nonSystem.last().role != "user") {
            return Failure(400, "invalid_request_error",
                "Last message must have role=user (got '${nonSystem.last().role}')")
        }

        // Sampling ranges (LiteRT-LM SamplerConfig: topK Int, topP/temperature Double).
        req.temperature?.let { t ->
            if (!t.isFinite() || t < 0f || t > 2f) {
                return Failure(400, "invalid_request_error",
                    "temperature must be a finite number in [0, 2] (got $t)")
            }
        }
        req.topP?.let { p ->
            if (!p.isFinite() || p <= 0f || p > 1f) {
                return Failure(400, "invalid_request_error",
                    "top_p must be a finite number in (0, 1] (got $p)")
            }
        }
        req.topK?.let { k ->
            if (k < 1) {
                return Failure(400, "invalid_request_error",
                    "top_k must be >= 1 (got $k)")
            }
        }
        req.maxTokens?.let { n ->
            if (n < 1) {
                return Failure(400, "invalid_request_error",
                    "max_tokens must be >= 1 (got $n)")
            }
            if (n > MAX_TOKENS_CEILING) {
                return Failure(400, "invalid_request_error",
                    "max_tokens must be <= $MAX_TOKENS_CEILING until the device limit is verified (got $n)")
            }
        }

        return null
    }

    private fun isPresent(el: JsonElement?): Boolean {
        if (el == null || el.isJsonNull) return false
        return true
    }

    /** `tool_choice: "none"` explicitly disables tools, so it is accepted. */
    private fun isNoneChoice(el: JsonElement?): Boolean {
        return el != null && el.isJsonPrimitive && el.asJsonPrimitive.isString &&
            el.asString.equals("none", ignoreCase = true)
    }

    private fun isNonEmpty(el: JsonElement?): Boolean {
        if (el == null || el.isJsonNull) return false
        if (el.isJsonArray && el.asJsonArray.size() == 0) return false
        if (el.isJsonPrimitive && el.asJsonPrimitive.isString && el.asString.isEmpty()) return false
        return true
    }
}
