"""Intent labels shared by the allow-list and the deterministic mock."""

from __future__ import annotations

import re

REFUND_POLICY_QUERY = "refund window final sale human approval partial refund amount"

_REFUND = re.compile(r"refund order\b", re.IGNORECASE)
_NOTIFY = re.compile(r"\b(?:email me|text me)\b", re.IGNORECASE)
_SHIPMENT = re.compile(r"where is shipment|where is order", re.IGNORECASE)
_LIST = re.compile(r"list (?:my |the )?(?:delivered )?orders", re.IGNORECASE)
_POLICY = re.compile(
    r"policy|refund window|how many days|final sale|human approval|partial refund|"
    r"privacy|shipping status|email or sms|exceed the order total|clearance|"
    r"thirty-day|tracking numbers|customer list",
    re.IGNORECASE,
)


def classify_intent(text: str) -> str:
    """Classify a user utterance. Order matters: actions beat general policy words."""
    if _REFUND.search(text):
        return "refund"
    if _NOTIFY.search(text):
        return "notify"
    if _SHIPMENT.search(text):
        return "shipment"
    if _LIST.search(text):
        return "list"
    if _POLICY.search(text):
        return "policy"
    return "lookup"
