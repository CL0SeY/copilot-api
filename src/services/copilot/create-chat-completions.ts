import consola from "consola"
import { events } from "fetch-event-stream"

import type { CompactType } from "~/lib/compact"
import type { SubagentMarker } from "~/lib/subagent"

import {
  copilotBaseUrl,
  copilotHeaders,
  prepareForCompact,
  prepareInteractionHeaders,
} from "~/lib/api-config"
import { logCopilotRateLimits } from "~/lib/copilot-rate-limit"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"

const CHAT_COMPLETIONS_MAX_TOOLS = 128
const TOOL_SCHEMA_NUMERIC_BOUNDARY_KEYS = [
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
] as const

export const createChatCompletions = async (
  payload: ChatCompletionsPayload,
  options: {
    subagentMarker?: SubagentMarker | null
    requestId: string
    sessionId?: string
    compactType?: CompactType
  },
) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  sanitizeToolsForChatCompletions(payload)

  const requestPayload = payload

  const enableVision = requestPayload.messages.some(
    (x) =>
      typeof x.content !== "string"
      && x.content?.some((x) => x.type === "image_url"),
  )

  // Agent/user check for x-initiator header
  // Determine if any message is from an agent ("assistant" or "tool")
  // Refactor `isAgentCall` logic to check only the last message in the history rather than any message. This prevents valid user messages from being incorrectly flagged as agent calls due to previous assistant history, ensuring proper credit consumption for multi-turn conversations.
  let isAgentCall = false
  if (requestPayload.messages.length > 0) {
    const lastMessage = requestPayload.messages.at(-1)
    if (lastMessage) {
      isAgentCall = ["assistant", "tool"].includes(lastMessage.role)
    }
  }

  // Build headers and add x-initiator
  const headers: Record<string, string> = {
    ...copilotHeaders(state, options.requestId, enableVision),
    "x-initiator": isAgentCall ? "agent" : "user",
  }

  prepareInteractionHeaders(
    options.sessionId,
    Boolean(options.subagentMarker),
    headers,
  )

  prepareForCompact(headers, options.compactType)

  consola.log(`<-- model: ${payload.model}`)

  const body = JSON.stringify(requestPayload)

  const response = await fetch(`${copilotBaseUrl(state)}/chat/completions`, {
    method: "POST",
    headers,
    body,
  })

  logCopilotRateLimits(response.headers)

  if (shouldRetryWithToolCompaction(response, requestPayload, false)) {
    const compactedPayload = compactPayloadToolsForRetry(payload)
    const compactedBody = JSON.stringify(compactedPayload)

    const retriedResponse = await fetch(
      `${copilotBaseUrl(state)}/chat/completions`,
      {
        method: "POST",
        headers,
        body: compactedBody,
      },
    )

    logCopilotRateLimits(retriedResponse.headers)

    if (!retriedResponse.ok) {
      consola.error("Chat completions retry request diagnostics", {
        ...createChatCompletionsDiagnostics(compactedPayload, compactedBody),
        compactType: options.compactType ?? 0,
        requestId: options.requestId,
        sessionIdPresent: Boolean(options.sessionId),
        status: retriedResponse.status,
        statusText: retriedResponse.statusText,
        upstreamRequestId:
          retriedResponse.headers.get("x-request-id") ?? undefined,
      })
      consola.error("Failed to create chat completions", retriedResponse)
      throw new HTTPError("Failed to create chat completions", retriedResponse)
    }

    if (payload.stream) {
      return events(retriedResponse)
    }

    return (await retriedResponse.json()) as ChatCompletionResponse
  }

  if (!response.ok) {
    consola.error("Chat completions request diagnostics", {
      ...createChatCompletionsDiagnostics(payload, body),
      compactType: options.compactType ?? 0,
      requestId: options.requestId,
      sessionIdPresent: Boolean(options.sessionId),
      status: response.status,
      statusText: response.statusText,
      upstreamRequestId: response.headers.get("x-request-id") ?? undefined,
    })
    consola.error("Failed to create chat completions", response)
    throw new HTTPError("Failed to create chat completions", response)
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as ChatCompletionResponse
}

const shouldRetryWithToolCompaction = (
  response: Response,
  payload: ChatCompletionsPayload,
  alreadyCompactedTools: boolean,
): boolean => {
  if (alreadyCompactedTools) {
    return false
  }

  if (
    response.status !== 400
    || !Array.isArray(payload.tools)
    || payload.tools.length === 0
  ) {
    return false
  }

  return true
}

const compactPayloadToolsForRetry = (
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload => {
  if (!Array.isArray(payload.tools) || payload.tools.length === 0) {
    return payload
  }

  return {
    ...payload,
    tools: payload.tools.map((tool) => ({
      ...tool,
      function: {
        name: tool.function.name,
        parameters: compactToolSchema(tool.function.parameters),
      },
    })),
  }
}

const compactToolSchema = (
  schema: Record<string, unknown>,
): Record<string, unknown> => {
  const type = typeof schema.type === "string" ? schema.type : "object"

  if (type === "object") {
    return {
      type: "object",
      properties: {},
    }
  }

  return { type }
}

const sanitizeToolsForChatCompletions = (
  payload: ChatCompletionsPayload,
): void => {
  if (
    !Array.isArray(payload.tools)
    || payload.tools.length <= CHAT_COMPLETIONS_MAX_TOOLS
  ) {
    return
  }

  const explicitToolChoiceName =
    (
      typeof payload.tool_choice === "object"
      && payload.tool_choice?.type === "function"
    ) ?
      payload.tool_choice.function?.name
    : undefined

  if (!explicitToolChoiceName) {
    payload.tools = payload.tools.slice(0, CHAT_COMPLETIONS_MAX_TOOLS)
    return
  }

  const explicitTool = payload.tools.find(
    (tool) => tool.function.name === explicitToolChoiceName,
  )
  const fallbackTools = payload.tools.filter(
    (tool) => tool.function.name !== explicitToolChoiceName,
  )

  if (!explicitTool) {
    payload.tools = payload.tools.slice(0, CHAT_COMPLETIONS_MAX_TOOLS)
    return
  }

  payload.tools = [
    explicitTool,
    ...fallbackTools.slice(0, CHAT_COMPLETIONS_MAX_TOOLS - 1),
  ]
}

const createChatCompletionsDiagnostics = (
  payload: ChatCompletionsPayload,
  body: string,
): Record<string, unknown> => {
  const messages = Array.isArray(payload.messages) ? payload.messages : []
  const tools = Array.isArray(payload.tools) ? payload.tools : []
  const explicitToolChoiceName =
    (
      typeof payload.tool_choice === "object"
      && payload.tool_choice?.type === "function"
    ) ?
      payload.tool_choice.function?.name
    : undefined

  const messageRoleCounts: Record<string, number> = {}
  const contentPartTypeCounts: Record<string, number> = {}
  let stringContentMessages = 0
  let nullContentMessages = 0
  let arrayContentMessages = 0

  for (const message of messages) {
    messageRoleCounts[message.role] = (messageRoleCounts[message.role] ?? 0) + 1

    const content = message.content
    if (typeof content === "string") {
      stringContentMessages += 1
      continue
    }

    if (content === null) {
      nullContentMessages += 1
      continue
    }

    if (!Array.isArray(content)) {
      continue
    }

    arrayContentMessages += 1
    for (const part of content) {
      const partType = part?.type ?? "unknown"
      contentPartTypeCounts[partType] =
        (contentPartTypeCounts[partType] ?? 0) + 1
    }
  }

  const toolNames = tools.map((tool) => tool.function?.name ?? "")
  const uniqueToolNames = new Set(toolNames.filter((name) => name.length > 0))
  const duplicateToolNames = [
    ...new Set(
      toolNames.filter(
        (name, index) => name.length > 0 && toolNames.indexOf(name) !== index,
      ),
    ),
  ]

  const boundaryKeyCounts = countRemainingBoundaryKeys(tools)

  return {
    payloadByteLength: new TextEncoder().encode(body).byteLength,
    topLevelKeys: Object.keys(payload).sort(),
    model: payload.model,
    stream: Boolean(payload.stream),
    hasReasoningEffort: typeof payload.reasoning_effort === "string",
    hasThinkingBudget: typeof payload.thinking_budget === "number",
    messageCount: messages.length,
    messageRoleCounts,
    stringContentMessages,
    nullContentMessages,
    arrayContentMessages,
    contentPartTypeCounts,
    toolsCount: tools.length,
    uniqueToolNames: uniqueToolNames.size,
    duplicateToolNames,
    toolChoice: payload.tool_choice ?? null,
    explicitToolChoiceName,
    hasExplicitToolChoice: Boolean(explicitToolChoiceName),
    explicitToolChoiceFound:
      explicitToolChoiceName ?
        uniqueToolNames.has(explicitToolChoiceName)
      : false,
    toolNameLengthMax: Math.max(0, ...toolNames.map((name) => name.length)),
    toolSchemaBoundaryKeyCounts: boundaryKeyCounts,
  }
}

const countRemainingBoundaryKeys = (
  tools: Array<Tool>,
): Record<string, number> => {
  const counts = Object.fromEntries(
    TOOL_SCHEMA_NUMERIC_BOUNDARY_KEYS.map((key) => [key, 0]),
  ) as Record<string, number>

  for (const tool of tools) {
    collectBoundaryKeyCounts(tool.function.parameters, counts)
  }

  return counts
}

const collectBoundaryKeyCounts = (
  value: unknown,
  counts: Record<string, number>,
): void => {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectBoundaryKeyCounts(item, counts)
    }
    return
  }

  if (typeof value !== "object" || value === null) {
    return
  }

  for (const [key, nestedValue] of Object.entries(value)) {
    if (Object.hasOwn(counts, key)) {
      counts[key] += 1
    }
    collectBoundaryKeyCounts(nestedValue, counts)
  }
}

// Streaming types

export interface ChatCompletionChunk {
  id: string
  object: "chat.completion.chunk"
  created: number
  model: string
  choices: Array<Choice>
  copilot_usage?: CopilotUsage | null
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_cache_hit_tokens?: number
    prompt_cache_miss_tokens?: number
    prompt_tokens_details?: {
      cache_creation_input_tokens?: number
      cached_tokens?: number
    }
    completion_tokens_details?: {
      accepted_prediction_tokens: number
      rejected_prediction_tokens: number
    }
  }
}

export interface Delta {
  content?: string | null
  role?: "user" | "assistant" | "system" | "tool"
  tool_calls?: Array<{
    index: number
    id?: string
    type?: "function"
    function?: {
      name?: string
      arguments?: string | Record<string, unknown>
    }
  }>
  reasoning_text?: string | null
  reasoning_content?: string | null
  reasoning_opaque?: string | null
}

export interface Choice {
  index: number
  delta: Delta
  finish_reason: string | null
  logprobs: object | null
}

// Non-streaming types

export interface ChatCompletionResponse {
  id: string
  object: "chat.completion"
  created: number
  model: string
  choices: Array<ChoiceNonStreaming>
  copilot_usage?: CopilotUsage | null
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_cache_hit_tokens?: number
    prompt_cache_miss_tokens?: number
    prompt_tokens_details?: {
      cache_creation_input_tokens?: number
      cached_tokens?: number
    }
  }
}

export interface CopilotUsage {
  total_nano_aiu?: number | null
}

interface ResponseMessage {
  role: "assistant"
  content: string | null
  reasoning_text?: string | null
  reasoning_content?: string | null
  reasoning_opaque?: string | null
  tool_calls?: Array<ToolCall>
}

interface ChoiceNonStreaming {
  index: number
  message: ResponseMessage
  logprobs: object | null
  finish_reason: string
}

// Payload types

export interface ChatCompletionsPayload {
  [key: string]: unknown

  messages: Array<Message>
  model: string
  temperature?: number | null
  top_p?: number | null
  max_tokens?: number | null
  max_completion_tokens?: number | null
  stop?: string | Array<string> | null
  n?: number | null
  stream?: boolean | null

  frequency_penalty?: number | null
  presence_penalty?: number | null
  logit_bias?: Record<string, number> | null
  logprobs?: boolean | null
  response_format?: { type: "json_object" } | null
  seed?: number | null
  tools?: Array<Tool> | null
  tool_choice?:
    | "none"
    | "auto"
    | "required"
    | { type: "function"; function: { name: string } }
    | null
  user?: string | null
  stream_options?: {
    include_usage?: boolean | null
  } | null
  thinking_budget?: number
  reasoning_effort?: string
  top_k?: number | null
  parallel_tool_calls?: boolean | null
}

export interface Tool {
  type: "function"
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

export interface Message {
  role: "user" | "assistant" | "system" | "tool" | "developer"
  content: string | Array<ContentPart> | null

  name?: string
  tool_calls?: Array<ToolCall>
  tool_call_id?: string
  reasoning_content?: string | null
  reasoning_text?: string | null
  reasoning_opaque?: string | null
  copilot_cache_control?: CopilotCacheControl
}

export interface ToolCall {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string | Record<string, unknown>
  }
}

export type ContentPart = TextPart | ImagePart | FilePart

export interface CacheControl {
  type: "ephemeral"
}

export interface CopilotCacheControl {
  type: "ephemeral"
}

export interface TextPart {
  type: "text"
  text: string
  cache_control?: CacheControl
}

export interface ImagePart {
  type: "image_url"
  image_url: {
    url: string
    detail?: "low" | "high" | "auto"
  }
  cache_control?: CacheControl
}

export interface FilePart {
  type: "file"
  file: {
    file_data: string
    filename?: string
  }
  cache_control?: CacheControl
}
