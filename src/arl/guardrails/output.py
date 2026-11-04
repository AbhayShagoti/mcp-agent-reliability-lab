"""Force the final answer into the AgentResponse schema."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from arl.domain.models import AgentResponse


def parse_agent_response(content: str) -> tuple[AgentResponse, bool]:
    """Return (response, repaired).

    `repaired` is true when the model text was not already a valid object.
    A fenced or surrounding JSON object is accepted so a small local model can
    still be graded on the answer, not on markdown fences.
    """
    payload, extracted = _load_object(content)
    if payload is None:
        return _wrap(content), True
    try:
        return AgentResponse.model_validate(payload), extracted
    except ValidationError:
        answer = payload.get("answer")
        text = answer if isinstance(answer, str) else content
        return _wrap(text), True


def _load_object(content: str) -> tuple[dict[str, Any] | None, bool]:
    text = content.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None, True
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None, True
        if isinstance(payload, dict):
            return payload, True
        return None, True
    if not isinstance(payload, dict):
        return None, True
    return payload, text != content.strip()


def _wrap(text: str) -> AgentResponse:
    cleaned = text.strip() or "I don't have a response."
    return AgentResponse(answer=cleaned, refused=False, refusal_reason=None)


def dump_response(response: AgentResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")
