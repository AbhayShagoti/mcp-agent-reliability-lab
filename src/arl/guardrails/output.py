"""Force the final answer into the AgentResponse schema."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from arl.domain.models import AgentResponse


def parse_agent_response(content: str) -> tuple[AgentResponse, bool]:
    """Return (response, repaired).

    `repaired` is true when the model text was not already a valid object.
    """
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _wrap(content), True
    if not isinstance(payload, dict):
        return _wrap(content), True
    try:
        return AgentResponse.model_validate(payload), False
    except ValidationError:
        answer = payload.get("answer")
        text = answer if isinstance(answer, str) else content
        return _wrap(text), True


def _wrap(text: str) -> AgentResponse:
    cleaned = text.strip() or "I don't have a response."
    return AgentResponse(answer=cleaned, refused=False, refusal_reason=None)


def dump_response(response: AgentResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")
