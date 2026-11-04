"""Optional tone judge. Not part of the merge gate.

The default implementation is a keyword heuristic so it can run with no model.
``llm_judge`` calls a provider when a nightly run asks for one.
"""

from __future__ import annotations

from arl.llm.types import ChatMessage, LLMClient


def heuristic_tone(answer: str) -> dict[str, object]:
    lowered = answer.lower()
    polite = any(token in lowered for token in ("i ", "order", "refund", "policy", "can't"))
    return {"provider": "heuristic", "nonempty": bool(answer.strip()), "has_substance": polite}


async def llm_tone(answer: str, llm: LLMClient) -> dict[str, object]:
    """Ask the configured model to score tone. Kept out of the gate on purpose."""
    prompt = 'Score the customer-support reply as JSON {"polite": true} or {"polite": false}. Reply only:\n' + answer
    result = await llm.complete(
        [ChatMessage(role="user", content=prompt)],
        tools=[],
        seed=0,
    )
    polite = "true" in result.content.lower()
    return {"provider": "llm", "polite": polite, "raw": result.content}
