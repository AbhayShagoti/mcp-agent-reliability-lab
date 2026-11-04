"""Regex injection detector.

This is a closed-set control: it is measured against the committed adversarial tasks,
not presented as a general jailbreak classifier. A second layer (tool allow-list,
refund policy, destination binding) still stops the action when the regex misses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Rule = tuple[str, re.Pattern[str]]

RULES: tuple[Rule, ...] = (
    (
        "ignore_previous",
        re.compile(
            r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules|prompts)",
            re.IGNORECASE,
        ),
    ),
    (
        "disregard_rules",
        re.compile(r"disregard (?:your |the )?(?:system|previous|prior)", re.IGNORECASE),
    ),
    ("you_are_now", re.compile(r"you are now", re.IGNORECASE)),
    ("new_instructions", re.compile(r"new instructions\s*:", re.IGNORECASE)),
    ("system_prompt", re.compile(r"system prompt", re.IGNORECASE)),
    ("inst_tag", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("sys_tag", re.compile(r"<\s*/?\s*sys\s*>", re.IGNORECASE)),
    ("jailbreak", re.compile(r"jailbreak", re.IGNORECASE)),
    ("exfiltrate", re.compile(r"exfiltrat", re.IGNORECASE)),
    (
        "dump_customers",
        re.compile(
            r"send (?:me |us )?(?:all|every) (?:customer )?(?:emails|phone|pii|records)",
            re.IGNORECASE,
        ),
    ),
    (
        "dump_database",
        re.compile(r"dump (?:the )?(?:database|customer list|emails)", re.IGNORECASE),
    ),
    ("attacker_address", re.compile(r"attacker@", re.IGNORECASE)),
    (
        "reveal_prompt",
        re.compile(r"reveal (?:your |the )?(?:system|hidden) prompt", re.IGNORECASE),
    ),
    (
        "bypass_control",
        re.compile(r"bypass (?:the )?(?:policy|approval|guard)", re.IGNORECASE),
    ),
    (
        "override_control",
        re.compile(r"override (?:the )?(?:policy|safety|approval)", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class Detection:
    matched: bool
    rule: str | None = None


def detect_injection(text: str) -> Detection:
    for name, pattern in RULES:
        if pattern.search(text):
            return Detection(True, name)
    return Detection(False, None)
