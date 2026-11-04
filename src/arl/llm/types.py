"""Provider-neutral chat types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema],
        seed: int,
    ) -> LLMResult:
        """Return either tool calls or a final message."""


def estimate_tokens(text: str) -> int:
    """Rough token count used when a provider does not report usage.

    The mock path always uses this estimator. README numbers from the mock are
    estimates (about 4 characters per token), not billed tokenizer counts.
    """
    return max(1, len(text) // 4)
