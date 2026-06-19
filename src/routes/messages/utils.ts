import consola from "consola"

import { type AnthropicResponse } from "./anthropic-types"

export function mapOpenAIStopReasonToAnthropic(
  finishReason: string | null,
): AnthropicResponse["stop_reason"] {
  if (finishReason === null) {
    return null
  }
  const stopReasonMap = {
    stop: "end_turn",
    length: "max_tokens",
    tool_calls: "tool_use",
    function_call: "tool_use",
    content_filter: "end_turn",
  } as const

  return stopReasonMap[finishReason as keyof typeof stopReasonMap] ?? "end_turn"
}

const FLATTENED_KEY_PATTERN = /\.|\[[0-9]+\]/
const ARRAY_INDEX_PATTERN = /^\d+$/

const parseFlattenedPath = (key: string): Array<string> => {
  const matches = key.match(/[^.[\]]+/g)
  return matches ?? []
}

const setPathValue = (
  target: Record<string, unknown>,
  path: Array<string>,
  value: unknown,
): void => {
  if (path.length === 0) {
    return
  }

  let cursor: Record<string, unknown> | Array<unknown> = target

  for (const [index, token] of path.entries()) {
    const isLast = index === path.length - 1
    const tokenIsIndex = ARRAY_INDEX_PATTERN.test(token)

    if (Array.isArray(cursor)) {
      if (!tokenIsIndex) {
        return
      }

      const numericToken = Number(token)
      if (isLast) {
        cursor[numericToken] = value
        return
      }

      const nextToken = path[index + 1] ?? ""
      const nextShouldBeArray = ARRAY_INDEX_PATTERN.test(nextToken)
      const nextValue = cursor[numericToken]
      if (!nextValue || typeof nextValue !== "object") {
        cursor[numericToken] = nextShouldBeArray ? [] : {}
      }
      cursor = cursor[numericToken]
      continue
    }

    if (!cursor || typeof cursor !== "object") {
      return
    }

    if (Array.isArray(cursor)) {
      return
    }

    if (isLast) {
      cursor[token] = value
      return
    }

    const nextToken = path[index + 1] ?? ""
    const nextShouldBeArray = ARRAY_INDEX_PATTERN.test(nextToken)
    const nextValue = cursor[token]

    if (!nextValue || typeof nextValue !== "object") {
      cursor[token] = nextShouldBeArray ? [] : {}
    }

    cursor = cursor[token] as Record<string, unknown> | Array<unknown>
  }
}

export const normalizePotentiallyFlattenedObject = <T extends object>(
  value: T,
): T | Record<string, unknown> => {
  const entries = Object.entries(value as Record<string, unknown>)
  const hasFlattenedKeys = entries.some(([key]) =>
    FLATTENED_KEY_PATTERN.test(key),
  )
  if (!hasFlattenedKeys) {
    return value
  }

  consola.info(
    "[compat] normalized flattened tool arguments into nested schema",
  )

  const normalized: Record<string, unknown> = {}
  for (const [key, entryValue] of entries) {
    if (!FLATTENED_KEY_PATTERN.test(key)) {
      normalized[key] = entryValue
      continue
    }

    const path = parseFlattenedPath(key)
    if (path.length === 0) {
      normalized[key] = entryValue
      continue
    }

    setPathValue(normalized, path, entryValue)
  }

  return normalized
}
