"""Optional HTTP providers. None of these are used unless selected and configured."""

from __future__ import annotations

import json
from typing import Any

import httpx

from arl.llm.types import ChatMessage, LLMResult, ToolCall, ToolSchema, estimate_tokens


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema],
        seed: int,
    ) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [_ollama_message(message) for message in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            "stream": False,
            "options": {"temperature": 0, "seed": seed},
        }
        data = await _post(self._client, f"{self.base_url}/api/chat", payload, headers=None)
        message = data.get("message", {})
        calls = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function", {})
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(
                ToolCall(
                    id=f"ollama-{index}",
                    name=str(function.get("name", "")),
                    arguments=dict(arguments),
                )
            )
        content = str(message.get("content") or "")
        prompt_tokens = int(data.get("prompt_eval_count") or estimate_tokens(content))
        output_tokens = int(data.get("eval_count") or estimate_tokens(content))
        return LLMResult(content, calls, prompt_tokens, output_tokens)


class OpenAILLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema],
        seed: int,
    ) -> LLMResult:
        del seed
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [_openai_message(message) for message in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
        }
        data = await _post(
            self._client,
            "https://api.openai.com/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        choice = data["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw in choice.get("tool_calls") or []:
            function = raw["function"]
            arguments = json.loads(function["arguments"])
            calls.append(ToolCall(id=str(raw["id"]), name=str(function["name"]), arguments=arguments))
        usage = data.get("usage") or {}
        return LLMResult(
            str(choice.get("content") or ""),
            calls,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )


class AnthropicLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema],
        seed: int,
    ) -> LLMResult:
        del seed
        system = "\n".join(message.content for message in messages if message.role == "system")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": system,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ],
            "messages": [_anthropic_message(message) for message in messages if message.role != "system"],
        }
        data = await _post(
            self._client,
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block["id"]),
                        name=str(block["name"]),
                        arguments=dict(block.get("input") or {}),
                    )
                )
        usage = data.get("usage") or {}
        return LLMResult(
            "\n".join(text_parts),
            calls,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )


async def _post(
    client: httpx.AsyncClient | None,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None,
) -> dict[str, Any]:
    async def _send(http: httpx.AsyncClient) -> dict[str, Any]:
        response = await http.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body

    if client is not None:
        return await _send(client)
    async with httpx.AsyncClient(timeout=120.0) as http:
        return await _send(http)


def _ollama_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}} for call in message.tool_calls
        ]
    return payload


def _openai_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


def _anthropic_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
            ],
        }
    if message.tool_calls:
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return {"role": "assistant", "content": blocks}
    return {"role": message.role, "content": message.content}
