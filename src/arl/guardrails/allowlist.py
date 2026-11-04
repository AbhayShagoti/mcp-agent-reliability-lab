"""Tool allow-lists keyed by the intent of the user text, not the model's claim."""

from __future__ import annotations

READ_TOOLS = frozenset(
    {
        "get_order",
        "list_orders",
        "get_shipment",
        "get_customer",
        "search_policies",
        "get_policy",
    }
)
SIDE_EFFECTS = frozenset({"create_refund", "send_email", "send_sms"})

INTENT_TOOLS: dict[str, frozenset[str]] = {
    "lookup": READ_TOOLS,
    "shipment": READ_TOOLS,
    "list": READ_TOOLS,
    "policy": frozenset({"search_policies", "get_policy"}),
    "refund": READ_TOOLS | frozenset({"create_refund"}),
    "notify": READ_TOOLS | SIDE_EFFECTS,
    "attack": frozenset(),
}


def tools_for_intent(intent: str) -> frozenset[str]:
    return INTENT_TOOLS.get(intent, READ_TOOLS)


def downgrade_for_untrusted_output(allowed: set[str]) -> set[str]:
    """Drop side effects after a tool result contains an injection payload."""
    return {name for name in allowed if name not in SIDE_EFFECTS}
