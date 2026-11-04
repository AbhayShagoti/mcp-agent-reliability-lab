"""Redact high-risk identifiers. Emails and phones are not stripped from prompts.

The agent is allowed to talk about the authenticated customer's own contact data,
which it loads from the database. Social security numbers and payment card numbers
are never needed, so those block the turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD = re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b")


@dataclass(frozen=True)
class ScrubResult:
    text: str
    blocked: bool
    kinds: tuple[str, ...]


def scrub_pii(text: str) -> ScrubResult:
    kinds: list[str] = []
    redacted = text
    if SSN.search(redacted):
        kinds.append("ssn")
        redacted = SSN.sub("[SSN]", redacted)
    if CARD.search(redacted):
        kinds.append("card")
        redacted = CARD.sub("[CARD]", redacted)
    return ScrubResult(text=redacted, blocked=bool(kinds), kinds=tuple(kinds))
