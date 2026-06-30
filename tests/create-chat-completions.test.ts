import { afterEach, beforeEach, expect, mock, test } from "bun:test"

import type { ChatCompletionsPayload } from "../src/services/copilot/create-chat-completions"

import { state } from "../src/lib/state"
import { createChatCompletions } from "../src/services/copilot/create-chat-completions"

const originalFetch = globalThis.fetch
const originalState = {
  accountType: state.accountType,
  copilotToken: state.copilotToken,
  vsCodeVersion: state.vsCodeVersion,
}

// Helper to mock fetch
const fetchMock = mock(
  (_url: string, opts: { headers: Record<string, string> }) => {
    return {
      ok: true,
      json: () => ({ id: "123", object: "chat.completion", choices: [] }),
      headers: opts.headers,
    }
  },
)

const getPostedPayload = (): ChatCompletionsPayload => {
  const requestOptions = fetchMock.mock.calls[0]?.[1] as
    | { body?: string }
    | undefined

  return JSON.parse(requestOptions?.body ?? "{}") as ChatCompletionsPayload
}

const getPostedPayloadAt = (index: number): ChatCompletionsPayload => {
  const requestOptions = fetchMock.mock.calls[index]?.[1] as
    | { body?: string }
    | undefined

  return JSON.parse(requestOptions?.body ?? "{}") as ChatCompletionsPayload
}
beforeEach(() => {
  state.copilotToken = "test-token"
  state.vsCodeVersion = "1.0.0"
  state.accountType = "individual"
  fetchMock.mockClear()
  ;(globalThis as unknown as { fetch: typeof fetch }).fetch =
    fetchMock as unknown as typeof fetch
})

afterEach(() => {
  state.copilotToken = originalState.copilotToken
  state.vsCodeVersion = originalState.vsCodeVersion
  state.accountType = originalState.accountType
  ;(globalThis as unknown as { fetch: typeof fetch }).fetch = originalFetch
})

test("sets x-initiator to agent if tool/assistant present", async () => {
  const payload: ChatCompletionsPayload = {
    messages: [
      { role: "user", content: "hi" },
      { role: "tool", content: "tool call" },
    ],
    model: "gpt-test",
  }
  await createChatCompletions(payload, { requestId: "1" })
  expect(fetchMock).toHaveBeenCalledTimes(1)
  const headers = (
    fetchMock.mock.calls[0][1] as { headers: Record<string, string> }
  ).headers
  expect(headers["x-initiator"]).toBe("agent")
})

test("sets x-initiator to user if only user present", async () => {
  const payload: ChatCompletionsPayload = {
    messages: [
      { role: "user", content: "hi" },
      { role: "user", content: "hello again" },
    ],
    model: "gpt-test",
  }
  await createChatCompletions(payload, { requestId: "1" })
  expect(fetchMock).toHaveBeenCalledTimes(1)
  const headers = (
    fetchMock.mock.calls[0][1] as { headers: Record<string, string> }
  ).headers
  expect(headers["x-initiator"]).toBe("user")
})

test("caps tools to 128 when request includes more tools", async () => {
  const payload: ChatCompletionsPayload = {
    model: "gpt-test",
    messages: [{ role: "user", content: "hello" }],
    tools: Array.from({ length: 129 }, (_, index) => ({
      type: "function",
      function: {
        name: `tool_${index}`,
        parameters: { type: "object" },
      },
    })),
  }

  await createChatCompletions(payload, { requestId: "1" })

  const postedPayload = getPostedPayload()
  expect(postedPayload.tools).toHaveLength(128)
  expect(postedPayload.tools?.[127]?.function.name).toBe("tool_127")
})

test("keeps explicit tool_choice function when tools exceed limit", async () => {
  const payload: ChatCompletionsPayload = {
    model: "gpt-test",
    messages: [{ role: "user", content: "hello" }],
    tools: Array.from({ length: 129 }, (_, index) => ({
      type: "function",
      function: {
        name: `tool_${index}`,
        parameters: { type: "object" },
      },
    })),
    tool_choice: {
      type: "function",
      function: { name: "tool_128" },
    },
  }

  await createChatCompletions(payload, { requestId: "1" })

  const postedPayload = getPostedPayload()
  const toolNames = postedPayload.tools?.map((tool) => tool.function.name)
  expect(toolNames).toContain("tool_128")
  expect(postedPayload.tools).toHaveLength(128)
})

test("retries with compacted tools for invalid_request_body", async () => {
  fetchMock.mockReset()
  fetchMock
    .mockImplementationOnce(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            error: {
              message: "invalid request body",
              code: "invalid_request_body",
            },
          }),
          {
            status: 400,
            headers: {
              "content-type": "application/json",
            },
          },
        ),
      ),
    )
    .mockImplementationOnce(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            id: "123",
            object: "chat.completion",
            choices: [],
          }),
          {
            status: 200,
            headers: {
              "content-type": "application/json",
            },
          },
        ),
      ),
    )

  const payload: ChatCompletionsPayload = {
    model: "gpt-test",
    messages: [{ role: "user", content: "hello" }],
    tools: Array.from({ length: 20 }, (_, index) => ({
      type: "function",
      function: {
        name: `tool_${index}`,
        description: `description ${index}`,
        parameters: {
          type: "object",
          properties: {
            timeout: {
              type: "integer",
              minimum: -9007199254740991,
              maximum: 9007199254740991,
            },
          },
        },
      },
    })),
  }

  await createChatCompletions(payload, { requestId: "1" })

  expect(fetchMock).toHaveBeenCalledTimes(2)
  const retriedPayload = getPostedPayloadAt(1)
  expect(retriedPayload.tools).toHaveLength(20)
  for (const tool of retriedPayload.tools ?? []) {
    expect(tool.function).not.toHaveProperty("description")
    expect(tool.function.parameters).toEqual({
      type: "object",
      properties: {},
    })
  }
})

test("retries with compacted tools when invalid_request_body omits content-type", async () => {
  fetchMock.mockReset()
  fetchMock
    .mockImplementationOnce(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            error: {
              message: "invalid request body",
              code: "invalid_request_body",
            },
          }),
          {
            status: 400,
          },
        ),
      ),
    )
    .mockImplementationOnce(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            id: "123",
            object: "chat.completion",
            choices: [],
          }),
          {
            status: 200,
          },
        ),
      ),
    )

  const payload: ChatCompletionsPayload = {
    model: "gpt-test",
    messages: [{ role: "user", content: "hello" }],
    tools: [
      {
        type: "function",
        function: {
          name: "tool_1",
          description: "description",
          parameters: {
            type: "object",
            properties: {
              timeout: {
                type: "integer",
                minimum: -9007199254740991,
                maximum: 9007199254740991,
              },
            },
          },
        },
      },
    ],
  }

  await createChatCompletions(payload, { requestId: "1" })

  expect(fetchMock).toHaveBeenCalledTimes(2)
  const retriedPayload = getPostedPayloadAt(1)
  expect(retriedPayload.tools?.[0]?.function).not.toHaveProperty("description")
  expect(retriedPayload.tools?.[0]?.function.parameters).toEqual({
    type: "object",
    properties: {},
  })
})

test("keeps Gemini tool schemas on the initial request", async () => {
  fetchMock.mockReset()
  fetchMock.mockImplementationOnce(() =>
    Promise.resolve(
      new Response(
        JSON.stringify({
          id: "123",
          object: "chat.completion",
          choices: [],
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
          },
        },
      ),
    ),
  )

  const payload: ChatCompletionsPayload = {
    model: "gemini-3-flash-preview",
    messages: [{ role: "user", content: "hello" }],
    tools: [
      {
        type: "function",
        function: {
          name: "tool_1",
          description: "description",
          parameters: {
            type: "object",
            properties: {
              timeout: {
                type: "integer",
                minimum: -9007199254740991,
                maximum: 9007199254740991,
              },
            },
          },
        },
      },
    ],
  }

  await createChatCompletions(payload, { requestId: "1" })

  expect(fetchMock).toHaveBeenCalledTimes(1)
  const postedPayload = getPostedPayloadAt(0)
  expect(postedPayload.tools?.[0]?.function.description).toBe("description")
  expect(postedPayload.tools?.[0]?.function.parameters).toEqual({
    type: "object",
    properties: {
      timeout: {
        type: "integer",
        minimum: -9007199254740991,
        maximum: 9007199254740991,
      },
    },
  })
})
