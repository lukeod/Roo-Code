import { Anthropic } from "@anthropic-ai/sdk"
import { BetaThinkingConfigParam } from "@anthropic-ai/sdk/resources/beta"
import axios, { AxiosRequestConfig } from "axios"
import OpenAI from "openai"
import delay from "delay"

import { ApiHandlerOptions, ModelInfo, openRouterDefaultModelId, openRouterDefaultModelInfo } from "../../shared/api"
import { parseApiPrice } from "../../utils/cost"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStreamChunk, ApiStreamUsageChunk } from "../transform/stream"
import { convertToR1Format } from "../transform/r1-format"

import { DEEP_SEEK_DEFAULT_TEMPERATURE } from "./constants"
import { getModelParams, SingleCompletionHandler } from ".."
import { BaseProvider } from "./base-provider"
import { defaultHeaders } from "./openai"

const OPENROUTER_DEFAULT_PROVIDER_NAME = "[default]"

// Add custom interface for OpenRouter params.
type OpenRouterChatCompletionParams = OpenAI.Chat.ChatCompletionCreateParams & {
	transforms?: string[]
	include_reasoning?: boolean
	thinking?: BetaThinkingConfigParam
}

export class OpenRouterHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: OpenAI
	private requestCount: number = 0
	private lastRequestTime: number = 0
	private consecutiveErrorCount: number = 0

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options

		const baseURL = this.options.openRouterBaseUrl || "https://openrouter.ai/api/v1"
		const apiKey = this.options.openRouterApiKey ?? "not-provided"

		console.log(`[OpenRouterHandler] Initializing with baseURL: ${baseURL}`)
		this.client = new OpenAI({ baseURL, apiKey, defaultHeaders })
	}

	override async *createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
	): AsyncGenerator<ApiStreamChunk> {
		this.requestCount++
		const currentTime = Date.now()
		const timeSinceLastRequest = currentTime - this.lastRequestTime
		this.lastRequestTime = currentTime

		console.log(
			`[OpenRouterHandler] Request #${this.requestCount} | Time since last request: ${timeSinceLastRequest}ms`,
		)

		let { id: modelId, maxTokens, thinking, temperature, topP } = this.getModel()

		// Enforce model-specific max_tokens limits
		const safeMaxTokens = this.getSafeMaxTokens(modelId, maxTokens || 4096)
		if (safeMaxTokens !== maxTokens) {
			console.log(
				`[OpenRouterHandler] Limiting maxTokens from ${maxTokens} to ${safeMaxTokens} for model ${modelId}`,
			)
			maxTokens = safeMaxTokens
		}

		console.log(
			`[OpenRouterHandler] Using model: ${modelId} | maxTokens: ${maxTokens} | temperature: ${temperature}`,
		)

		// Convert Anthropic messages to OpenAI format.
		let openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]

		// DeepSeek highly recommends using user instead of system role.
		if (modelId.startsWith("deepseek/deepseek-r1") || modelId === "perplexity/sonar-reasoning") {
			openAiMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
		}

		// prompt caching: https://openrouter.ai/docs/prompt-caching
		// this is specifically for claude models (some models may 'support prompt caching' automatically without this)
		switch (true) {
			case modelId.startsWith("anthropic/"):
				openAiMessages[0] = {
					role: "system",
					content: [
						{
							type: "text",
							text: systemPrompt,
							// @ts-ignore-next-line
							cache_control: { type: "ephemeral" },
						},
					],
				}
				// Add cache_control to the last two user messages
				// (note: this works because we only ever add one user message at a time, but if we added multiple we'd need to mark the user message before the last assistant message)
				const lastTwoUserMessages = openAiMessages.filter((msg) => msg.role === "user").slice(-2)
				lastTwoUserMessages.forEach((msg) => {
					if (typeof msg.content === "string") {
						msg.content = [{ type: "text", text: msg.content }]
					}
					if (Array.isArray(msg.content)) {
						// NOTE: this is fine since env details will always be added at the end. but if it weren't there, and the user added a image_url type message, it would pop a text part before it and then move it after to the end.
						let lastTextPart = msg.content.filter((part) => part.type === "text").pop()

						if (!lastTextPart) {
							lastTextPart = { type: "text", text: "..." }
							msg.content.push(lastTextPart)
						}
						// @ts-ignore-next-line
						lastTextPart["cache_control"] = { type: "ephemeral" }
					}
				})
				break
			default:
				break
		}

		// https://openrouter.ai/docs/transforms
		let fullResponseText = ""

		const completionParams: OpenRouterChatCompletionParams = {
			model: modelId,
			max_tokens: maxTokens,
			temperature,
			thinking, // OpenRouter is temporarily supporting this.
			top_p: topP,
			messages: openAiMessages,
			stream: true,
			stream_options: { include_usage: true },
			// Only include provider if openRouterSpecificProvider is not "[default]".
			...(this.options.openRouterSpecificProvider &&
				this.options.openRouterSpecificProvider !== OPENROUTER_DEFAULT_PROVIDER_NAME && {
					provider: { order: [this.options.openRouterSpecificProvider] },
				}),
			// This way, the transforms field will only be included in the parameters when openRouterUseMiddleOutTransform is true.
			...((this.options.openRouterUseMiddleOutTransform ?? true) && { transforms: ["middle-out"] }),
		}

		// Log prompt size and message count for diagnostics
		const promptCharCount = openAiMessages.reduce((acc, msg) => acc + (msg.content?.length || 0), 0)

		// Estimate token count (rough approximation: ~4 chars per token)
		const estimatedTokens = Math.ceil(promptCharCount / 4)
		const modelContextWindow = this.getModelContextWindow(modelId)

		console.log(`[OpenRouterHandler] Sending request to OpenRouter with params:`, {
			model: modelId,
			max_tokens: maxTokens,
			temperature,
			thinking: thinking ? "enabled" : "disabled",
			messages_count: openAiMessages.length,
			prompt_char_count: promptCharCount,
			estimated_tokens: estimatedTokens,
			context_window: modelContextWindow,
			stream: true,
		})

		// Warn if prompt might be too large
		if (modelContextWindow > 0 && estimatedTokens + maxTokens > modelContextWindow) {
			console.warn(
				`[OpenRouterHandler] WARNING: Estimated tokens (${estimatedTokens}) + max_tokens (${maxTokens}) may exceed model context window (${modelContextWindow})`,
			)
		}

		try {
			const stream = await this.client.chat.completions.create(completionParams)
			console.log(`[OpenRouterHandler] Stream created successfully`)
			this.consecutiveErrorCount = 0

			let lastUsage

			for await (const chunk of stream as unknown as AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>) {
				// OpenRouter returns an error object instead of the OpenAI SDK throwing an error.
				if ("error" in chunk) {
					const error = chunk.error as { message?: string; code?: number }
					this.consecutiveErrorCount++
					console.error(
						`[OpenRouterHandler] Error in stream chunk #${this.consecutiveErrorCount}: ${error?.code} - ${error?.message}`,
					)
					console.error(`[OpenRouterHandler] Full error object:`, JSON.stringify(chunk, null, 2))
					throw new Error(`OpenRouter API Error ${error?.code}: ${error?.message}`)
				}

				const delta = chunk.choices[0]?.delta

				if ("reasoning" in delta && delta.reasoning) {
					yield { type: "reasoning", text: delta.reasoning } as ApiStreamChunk
				}

				if (delta?.content) {
					fullResponseText += delta.content
					yield { type: "text", text: delta.content } as ApiStreamChunk
				}

				if (chunk.usage) {
					lastUsage = chunk.usage
				}
			}

			if (lastUsage) {
				console.log(`[OpenRouterHandler] Request completed with usage:`, lastUsage)
				yield this.processUsageMetrics(lastUsage)
			}
		} catch (error) {
			this.consecutiveErrorCount++
			console.error(
				`[OpenRouterHandler] Error during stream creation (attempt #${this.consecutiveErrorCount}):`,
				error,
			)
			console.error(
				`[OpenRouterHandler] Error details:`,
				JSON.stringify(error, Object.getOwnPropertyNames(error), 2),
			)
			// Surface provider error details to the user
			let userErrorMsg = "An error occurred while communicating with the model provider."
			if (error && typeof error === "object" && "message" in error) {
				userErrorMsg += ` Provider message: ${(error as any).message}`
			}
			if (error && typeof error === "object" && "code" in error) {
				userErrorMsg += ` (code ${(error as any).code})`
			}
			throw new Error(userErrorMsg)
		}
	}

	processUsageMetrics(usage: any): ApiStreamUsageChunk {
		return {
			type: "usage",
			inputTokens: usage?.prompt_tokens || 0,
			outputTokens: usage?.completion_tokens || 0,
			totalCost: usage?.cost || 0,
		}
	}

	override getModel() {
		const modelId = this.options.openRouterModelId
		const modelInfo = this.options.openRouterModelInfo

		let id = modelId ?? openRouterDefaultModelId
		const info = modelInfo ?? openRouterDefaultModelInfo

		const isDeepSeekR1 = id.startsWith("deepseek/deepseek-r1") || modelId === "perplexity/sonar-reasoning"
		const defaultTemperature = isDeepSeekR1 ? DEEP_SEEK_DEFAULT_TEMPERATURE : 0
		const topP = isDeepSeekR1 ? 0.95 : undefined

		return {
			id,
			info,
			...getModelParams({ options: this.options, model: info, defaultTemperature }),
			topP,
		}
	}

	/**
	 * Get a safe max_tokens value for the given model
	 * @param modelId The model ID
	 * @param requestedMaxTokens The requested max_tokens value
	 * @returns A safe max_tokens value that won't exceed the model's capabilities
	 */
	getSafeMaxTokens(modelId: string, requestedMaxTokens: number): number {
		// Default limits for common models
		const modelLimits: Record<string, number> = {
			// Claude models
			"anthropic/claude-3-opus": 4096,
			"anthropic/claude-3-sonnet": 4096,
			"anthropic/claude-3-haiku": 4096,
			"anthropic/claude-3.5-sonnet": 4096,
			"anthropic/claude-3.7-sonnet": 4096,
			"anthropic/claude-3.7-sonnet:thinking": 128000,

			// OpenAI models
			"openai/gpt-4": 4096,
			"openai/gpt-4-turbo": 4096,
			"openai/gpt-4o": 4096,
			"openai/gpt-3.5-turbo": 4096,

			// Anthropic models
			"anthropic/claude-instant-1.2": 4096,
			"anthropic/claude-2.0": 4096,
			"anthropic/claude-2.1": 4096,

			// Other models
			"google/gemini-pro": 8192,
			"google/gemini-1.5-pro": 8192,
			"meta/llama-3-70b": 4096,
			"meta/llama-3-8b": 4096,
			"mistral/mistral-7b": 4096,
			"mistral/mistral-large": 4096,
		}

		// Find the most specific model limit
		let limit = 4096 // Default safe limit

		// Try exact match first
		if (modelLimits[modelId]) {
			limit = modelLimits[modelId]
		} else {
			// Try prefix match
			for (const [prefix, prefixLimit] of Object.entries(modelLimits)) {
				if (modelId.startsWith(prefix)) {
					limit = prefixLimit
					break
				}
			}
		}

		// Return the minimum of the requested max_tokens and the model limit
		return Math.min(requestedMaxTokens, limit)
	}

	/**
	 * Get the context window size for the given model
	 * @param modelId The model ID
	 * @returns The context window size, or 0 if unknown
	 */
	getModelContextWindow(modelId: string): number {
		// Default context windows for common models
		const contextWindows: Record<string, number> = {
			// Claude models
			"anthropic/claude-3-opus": 200000,
			"anthropic/claude-3-sonnet": 200000,
			"anthropic/claude-3-haiku": 200000,
			"anthropic/claude-3.5-sonnet": 200000,
			"anthropic/claude-3.7-sonnet": 200000,
			"anthropic/claude-3.7-sonnet:thinking": 200000,

			// OpenAI models
			"openai/gpt-4": 8192,
			"openai/gpt-4-turbo": 128000,
			"openai/gpt-4o": 128000,
			"openai/gpt-3.5-turbo": 16385,

			// Anthropic models
			"anthropic/claude-instant-1.2": 100000,
			"anthropic/claude-2.0": 100000,
			"anthropic/claude-2.1": 200000,

			// Other models
			"google/gemini-pro": 32768,
			"google/gemini-1.5-pro": 1000000,
			"meta/llama-3-70b": 8192,
			"meta/llama-3-8b": 8192,
			"mistral/mistral-7b": 8192,
			"mistral/mistral-large": 32768,
		}

		// Find the most specific context window
		let contextWindow = 8192 // Default safe context window

		// Try exact match first
		if (contextWindows[modelId]) {
			contextWindow = contextWindows[modelId]
		} else {
			// Try prefix match
			for (const [prefix, prefixWindow] of Object.entries(contextWindows)) {
				if (modelId.startsWith(prefix)) {
					contextWindow = prefixWindow
					break
				}
			}
		}

		return contextWindow
	}

	async completePrompt(prompt: string) {
		this.requestCount++
		const currentTime = Date.now()
		const timeSinceLastRequest = currentTime - this.lastRequestTime
		this.lastRequestTime = currentTime

		console.log(
			`[OpenRouterHandler] completePrompt #${this.requestCount} | Time since last request: ${timeSinceLastRequest}ms`,
		)

		let { id: modelId, maxTokens, thinking, temperature } = this.getModel()

		// Enforce model-specific max_tokens limits for completePrompt too
		const safeMaxTokens = this.getSafeMaxTokens(modelId, maxTokens || 4096)
		if (safeMaxTokens !== maxTokens) {
			console.log(
				`[OpenRouterHandler] completePrompt: Limiting maxTokens from ${maxTokens} to ${safeMaxTokens} for model ${modelId}`,
			)
			maxTokens = safeMaxTokens
		}

		console.log(`[OpenRouterHandler] completePrompt using model: ${modelId} | maxTokens: ${maxTokens}`)

		const completionParams: OpenRouterChatCompletionParams = {
			model: modelId,
			max_tokens: maxTokens,
			thinking,
			temperature,
			messages: [{ role: "user", content: prompt }],
			stream: false,
		}

		console.log(`[OpenRouterHandler] Sending completePrompt request to OpenRouter`)

		try {
			const response = await this.client.chat.completions.create(completionParams)
			console.log(`[OpenRouterHandler] completePrompt response received successfully`)

			if ("error" in response) {
				const error = response.error as { message?: string; code?: number }
				this.consecutiveErrorCount++
				console.error(
					`[OpenRouterHandler] completePrompt error #${this.consecutiveErrorCount}: ${error?.code} - ${error?.message}`,
				)
				throw new Error(`OpenRouter API Error ${error?.code}: ${error?.message}`)
			}

			this.consecutiveErrorCount = 0
			const completion = response as OpenAI.Chat.ChatCompletion
			console.log(
				`[OpenRouterHandler] completePrompt completed successfully with ${completion.choices[0]?.message?.content?.length || 0} chars`,
			)
			return completion.choices[0]?.message?.content || ""
		} catch (error) {
			this.consecutiveErrorCount++
			console.error(`[OpenRouterHandler] completePrompt error (attempt #${this.consecutiveErrorCount}):`, error)
			console.error(
				`[OpenRouterHandler] completePrompt error details:`,
				JSON.stringify(error, Object.getOwnPropertyNames(error), 2),
			)
			throw error
		}
	}
}

export async function getOpenRouterModels(options?: ApiHandlerOptions) {
	console.log(`[OpenRouter] Fetching available models from OpenRouter`)
	const models: Record<string, ModelInfo> = {}

	const baseURL = options?.openRouterBaseUrl || "https://openrouter.ai/api/v1"
	console.log(`[OpenRouter] Using baseURL: ${baseURL}`)

	try {
		console.log(`[OpenRouter] Making request to ${baseURL}/models`)
		const response = await axios.get(`${baseURL}/models`)
		console.log(`[OpenRouter] Successfully fetched ${response.data.data.length} models`)
		const rawModels = response.data.data

		for (const rawModel of rawModels) {
			const modelInfo: ModelInfo = {
				maxTokens: rawModel.top_provider?.max_completion_tokens,
				contextWindow: rawModel.context_length,
				supportsImages: rawModel.architecture?.modality?.includes("image"),
				supportsPromptCache: false,
				inputPrice: parseApiPrice(rawModel.pricing?.prompt),
				outputPrice: parseApiPrice(rawModel.pricing?.completion),
				description: rawModel.description,
				thinking: rawModel.id === "anthropic/claude-3.7-sonnet:thinking",
			}

			// NOTE: this needs to be synced with api.ts/openrouter default model info.
			switch (true) {
				case rawModel.id.startsWith("anthropic/claude-3.7-sonnet"):
					modelInfo.supportsComputerUse = true
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 3.75
					modelInfo.cacheReadsPrice = 0.3
					modelInfo.maxTokens = rawModel.id === "anthropic/claude-3.7-sonnet:thinking" ? 128_000 : 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3.5-sonnet-20240620"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 3.75
					modelInfo.cacheReadsPrice = 0.3
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3.5-sonnet"):
					modelInfo.supportsComputerUse = true
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 3.75
					modelInfo.cacheReadsPrice = 0.3
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3-5-haiku"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 1.25
					modelInfo.cacheReadsPrice = 0.1
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3-opus"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 18.75
					modelInfo.cacheReadsPrice = 1.5
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3-haiku"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 0.3
					modelInfo.cacheReadsPrice = 0.03
					modelInfo.maxTokens = 8192
					break
				default:
					break
			}

			models[rawModel.id] = modelInfo
		}
	} catch (error) {
		console.error(`[OpenRouter] Error fetching models:`)
		console.error(`[OpenRouter] ${JSON.stringify(error, Object.getOwnPropertyNames(error), 2)}`)

		// Log specific error details for common issues
		if (error.response) {
			console.error(`[OpenRouter] Response status: ${error.response.status}`)
			console.error(`[OpenRouter] Response data: ${JSON.stringify(error.response.data)}`)
		} else if (error.request) {
			console.error(`[OpenRouter] No response received from server`)
		} else {
			console.error(`[OpenRouter] Error setting up request: ${error.message}`)
		}
	}

	return models
}
