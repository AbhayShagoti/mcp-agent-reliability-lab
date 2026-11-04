"""Deterministic mock LLM.

Benign tasks use ``policy`` mode, which follows the shop's rules. Adversarial tasks in
the eval use ``gullible`` mode, which obeys injected instructions so the guardrails
are what stop the attack. Both modes are pure functions of the transcript.
"""

from __future__ import annotations

import json
import re
from typing import Any

from arl.agent.intent import REFUND_POLICY_QUERY, classify_intent
from arl.agent.phrases import (
    ALREADY_REFUNDED,
    CANCELLED,
    EMAILED,
    ERROR_PHRASES,
    EXCEEDS_TOTAL,
    FINAL_SALE,
    ISSUED_PREFIX,
    NO_SHIPMENT,
    NOT_APPROVED,
    NOT_FOUND,
    OUTSIDE_WINDOW,
    TEXTED,
)
from arl.agent.prompts import customer_from_messages
from arl.guardrails.injection import detect_injection
from arl.llm.types import ChatMessage, LLMResult, ToolCall, ToolSchema, estimate_tokens

_ORDER = re.compile(r"ORD-\d+")
_MONEY = re.compile(r"\$(\d+)(?:\.(\d{2}))?")


def parse_order_id(text: str) -> str | None:
    match = _ORDER.search(text)
    return match.group(0) if match else None


def parse_cents(text: str) -> int | None:
    match = _MONEY.search(text)
    if match is None:
        return None
    dollars = int(match.group(1))
    cents = int(match.group(2) or "00")
    return dollars * 100 + cents


def format_money(cents: int) -> str:
    return f"${cents / 100:.2f}"


class MockLLM:
    """Rule-based stand-in. ``mode`` can be flipped per eval trial."""

    def __init__(self, mode: str = "policy") -> None:
        self.mode = mode

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema],
        seed: int,
    ) -> LLMResult:
        del seed  # the mock is deterministic; the seed is accepted for the protocol
        allowed = {tool.name for tool in tools}
        result = _gullible(messages, allowed) if self.mode == "gullible" else _policy(messages, allowed)
        prompt = "\n".join(message.content for message in messages)
        result.input_tokens = estimate_tokens(prompt)
        result.output_tokens = estimate_tokens(result.content) + 8 * len(result.tool_calls)
        return result


def _policy(messages: list[ChatMessage], allowed: set[str]) -> LLMResult:
    user = _last_user(messages)
    customer_id = _customer(messages)
    payloads = _payloads_after_user(messages)
    intent = classify_intent(user)
    if intent == "refund":
        return _refund(user, customer_id, payloads, allowed)
    if intent == "notify":
        return _notify(user, customer_id, payloads, allowed)
    if intent == "shipment":
        return _shipment(user, customer_id, payloads, allowed)
    if intent == "list":
        return _list(user, customer_id, payloads, allowed)
    if intent == "policy":
        return _policy_qa(user, payloads, allowed)
    return _lookup(user, customer_id, payloads, allowed)


def _gullible(messages: list[ChatMessage], allowed: set[str]) -> LLMResult:
    user = _last_user(messages)
    customer_id = _customer(messages) or "CUST-1001"
    payloads = _payloads_after_user(messages)
    if _already_attempted_side_effect(messages):
        return _final("Done.")
    order_id = parse_order_id(user) or _order_from_payloads(payloads)
    corpus = user + "\n" + "\n".join(json.dumps(payload) for _, payload in payloads)
    user_attack = detect_injection(user).matched
    tool_attack = any(detect_injection(json.dumps(payload)).matched for _, payload in payloads)
    if not payloads:
        if user_attack and order_id and "refund" in user.lower():
            amount = parse_cents(user) or 10000
            return _calls(
                [
                    _call(
                        "0",
                        "create_refund",
                        {
                            "order_id": order_id,
                            "customer_id": customer_id,
                            "amount_cents": amount,
                            "reason": "override request",
                        },
                        allowed | {"create_refund"},
                    )
                ]
            )
        if user_attack and ("email" in user.lower() or "attacker@" in user.lower()):
            return _calls(
                [
                    _call(
                        "0",
                        "send_email",
                        {
                            "customer_id": customer_id,
                            "subject": "records",
                            "body": user[:400],
                        },
                        allowed | {"send_email"},
                    )
                ]
            )
        if order_id:
            return _calls(
                [
                    _call(
                        "0",
                        "get_order",
                        {"order_id": order_id, "customer_id": customer_id},
                        allowed,
                    )
                ]
            )
        return _final("Done.")
    if user_attack or tool_attack:
        if "attacker@" in corpus.lower() or "email" in corpus.lower():
            return _calls(
                [
                    _call(
                        "1",
                        "send_email",
                        {
                            "customer_id": customer_id,
                            "subject": "export",
                            "body": corpus[:500],
                        },
                        allowed | {"send_email"},
                    )
                ]
            )
        if order_id and "refund" in corpus.lower():
            return _calls(
                [
                    _call(
                        "1",
                        "create_refund",
                        {
                            "order_id": order_id,
                            "customer_id": customer_id,
                            "amount_cents": 10000,
                            "reason": "injected refund",
                        },
                        allowed | {"create_refund"},
                    )
                ]
            )
    return _policy(messages, allowed)


def _refund(
    user: str,
    customer_id: str | None,
    payloads: list[tuple[str, dict[str, Any]]],
    allowed: set[str],
) -> LLMResult:
    by_name = _by_name(payloads)
    order_id = parse_order_id(user) or "ORD-1001"
    cid = customer_id or "CUST-1001"
    if "create_refund" in by_name:
        return _final(_refund_followup(order_id, by_name["create_refund"]))
    if "get_order" not in by_name:
        return _calls(
            [
                _call(
                    "0",
                    "get_order",
                    {"order_id": order_id, "customer_id": cid},
                    allowed,
                ),
                _call(
                    "1",
                    "search_policies",
                    {"query": REFUND_POLICY_QUERY, "k": 3},
                    allowed,
                ),
            ]
        )
    found = by_name["get_order"]
    if not found.get("found"):
        return _final(_json_answer(NOT_FOUND, refused=True, reason="order_not_found"))
    order = found["order"]
    amount = order["total_cents"] if "in full" in user.lower() else parse_cents(user)
    if amount is None:
        amount = int(order["total_cents"])
    phrase = _ineligible_phrase(order, int(amount))
    if phrase:
        return _final(_json_answer(phrase, refused=True, reason="policy"))
    return _calls(
        [
            _call(
                "2",
                "create_refund",
                {
                    "order_id": order_id,
                    "customer_id": cid,
                    "amount_cents": int(amount),
                    "reason": "item arrived damaged",
                },
                allowed,
            )
        ]
    )


def _ineligible_phrase(order: dict[str, Any], amount: int) -> str | None:
    if order.get("final_sale"):
        return FINAL_SALE
    if order.get("status") == "cancelled":
        return CANCELLED
    if order.get("status") == "refunded" or order.get("refunds"):
        return ALREADY_REFUNDED
    if int(order.get("age_days", 0)) > 30:
        return OUTSIDE_WINDOW
    if amount > int(order["total_cents"]):
        return EXCEEDS_TOTAL
    return None


def _refund_followup(order_id: str, payload: dict[str, Any]) -> str:
    if payload.get("ok"):
        refund = payload["refund"]
        amount = format_money(int(refund["amount_cents"]))
        return _json_answer(
            f"{ISSUED_PREFIX} of {amount} for order {order_id}.",
            actions=[f"create_refund:{order_id}"],
        )
    error = str(payload.get("error") or "rejected")
    phrase = ERROR_PHRASES.get(error, NOT_APPROVED if "approv" in error else f"refund failed: {error}")
    return _json_answer(phrase, refused=True, reason=error)


def _notify(
    user: str,
    customer_id: str | None,
    payloads: list[tuple[str, dict[str, Any]]],
    allowed: set[str],
) -> LLMResult:
    by_name = _by_name(payloads)
    order_id = parse_order_id(user) or ""
    cid = customer_id or "CUST-1001"
    email = "email me" in user.lower()
    if "send_email" in by_name or "send_sms" in by_name:
        verb = EMAILED if email else TEXTED
        return _final(
            _json_answer(
                f"{verb} the update for order {order_id}.",
                actions=["send_email" if email else "send_sms"],
            )
        )
    if email:
        if "get_shipment" not in by_name:
            return _calls(
                [
                    _call(
                        "0",
                        "get_shipment",
                        {"order_id": order_id, "customer_id": cid},
                        allowed,
                    ),
                    _call("1", "get_customer", {"customer_id": cid}, allowed),
                ]
            )
        shipment = by_name["get_shipment"]
        if not shipment.get("found"):
            return _final(_json_answer(NO_SHIPMENT, refused=True, reason="no_shipment"))
        tracking = shipment["shipment"]["tracking_number"]
        return _calls(
            [
                _call(
                    "2",
                    "send_email",
                    {
                        "customer_id": cid,
                        "subject": f"Tracking for {order_id}",
                        "body": f"Tracking number {tracking} for order {order_id}.",
                    },
                    allowed,
                )
            ]
        )
    if "get_order" not in by_name:
        return _calls([_call("0", "get_order", {"order_id": order_id, "customer_id": cid}, allowed)])
    order_payload = by_name["get_order"]
    status = "unknown"
    if order_payload.get("found"):
        status = str(order_payload["order"]["status"])
    return _calls(
        [
            _call(
                "1",
                "send_sms",
                {"customer_id": cid, "body": f"Order {order_id} is {status}."},
                allowed,
            )
        ]
    )


def _shipment(
    user: str,
    customer_id: str | None,
    payloads: list[tuple[str, dict[str, Any]]],
    allowed: set[str],
) -> LLMResult:
    by_name = _by_name(payloads)
    order_id = parse_order_id(user) or ""
    cid = customer_id or "CUST-1001"
    if "get_shipment" not in by_name:
        return _calls(
            [
                _call(
                    "0",
                    "get_shipment",
                    {"order_id": order_id, "customer_id": cid},
                    allowed,
                )
            ]
        )
    shipment = by_name["get_shipment"]
    if not shipment.get("found"):
        return _final(_json_answer(NO_SHIPMENT))
    info = shipment["shipment"]
    answer = (
        f"Order {order_id} is with {info['carrier']}. "
        f"Tracking number {info['tracking_number']}. "
        f"Shipment status {info['status']}."
    )
    return _final(_json_answer(answer, citations=[str(info["tracking_number"])]))


def _list(
    user: str,
    customer_id: str | None,
    payloads: list[tuple[str, dict[str, Any]]],
    allowed: set[str],
) -> LLMResult:
    by_name = _by_name(payloads)
    cid = customer_id or "CUST-1001"
    if "list_orders" not in by_name:
        arguments: dict[str, Any] = {"customer_id": cid}
        if "delivered" in user.lower():
            arguments["status"] = "delivered"
        return _calls([_call("0", "list_orders", arguments, allowed)])
    orders = by_name["list_orders"].get("orders", [])
    rendered = ", ".join(f"{row['id']} ({row['status']})" for row in orders) or "none"
    return _final(_json_answer(f"Your orders: {rendered}.", actions=["list_orders"]))


def _policy_qa(
    user: str,
    payloads: list[tuple[str, dict[str, Any]]],
    allowed: set[str],
) -> LLMResult:
    by_name = _by_name(payloads)
    if "search_policies" not in by_name:
        return _calls([_call("0", "search_policies", {"query": user, "k": 3}, allowed)])
    hits = by_name["search_policies"].get("hits", [])
    if not hits:
        return _final(_json_answer("I could not find a matching policy.", refused=True, reason="no_policy"))
    top = hits[0]
    answer = f"According to {top['policy_id']} ({top['title']}): {top['snippet']}"
    return _final(_json_answer(answer, citations=[str(top["policy_id"])]))


def _lookup(
    user: str,
    customer_id: str | None,
    payloads: list[tuple[str, dict[str, Any]]],
    allowed: set[str],
) -> LLMResult:
    by_name = _by_name(payloads)
    order_id = parse_order_id(user) or ""
    cid = customer_id or "CUST-1001"
    if "get_order" not in by_name:
        return _calls([_call("0", "get_order", {"order_id": order_id, "customer_id": cid}, allowed)])
    payload = by_name["get_order"]
    if not payload.get("found"):
        return _final(_json_answer(NOT_FOUND, refused=True, reason="order_not_found"))
    status = payload["order"]["status"]
    return _final(_json_answer(f"Order {order_id} is {status}."))


def _json_answer(
    answer: str,
    *,
    citations: list[str] | None = None,
    actions: list[str] | None = None,
    refused: bool = False,
    reason: str | None = None,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": citations or [],
            "actions_taken": actions or [],
            "refused": refused,
            "refusal_reason": reason,
        }
    )


def _final(content: str) -> LLMResult:
    return LLMResult(content=content, tool_calls=[], input_tokens=0, output_tokens=0)


def _calls(calls: list[ToolCall]) -> LLMResult:
    kept = [call for call in calls if call.name]
    if not kept:
        return _final(_json_answer("I can't use a tool for that.", refused=True, reason="allowlist"))
    return LLMResult(content="", tool_calls=kept, input_tokens=0, output_tokens=0)


def _call(suffix: str, name: str, arguments: dict[str, Any], allowed: set[str]) -> ToolCall:
    if name not in allowed:
        return ToolCall(id=f"skip-{suffix}", name="", arguments={})
    return ToolCall(id=f"call-{suffix}", name=name, arguments=arguments)


def _last_user(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _customer(messages: list[ChatMessage]) -> str | None:
    blob = "\n".join(message.content for message in messages if message.role == "system")
    return customer_from_messages(blob)


def _payloads_after_user(messages: list[ChatMessage]) -> list[tuple[str, dict[str, Any]]]:
    last_user = -1
    for index, message in enumerate(messages):
        if message.role == "user":
            last_user = index
    payloads: list[tuple[str, dict[str, Any]]] = []
    for message in messages[last_user + 1 :]:
        if message.role != "tool" or not message.name:
            continue
        try:
            parsed = json.loads(message.content)
        except json.JSONDecodeError:
            parsed = {"raw": message.content}
        if isinstance(parsed, dict):
            payloads.append((message.name, parsed))
    return payloads


def _by_name(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for name, payload in payloads:
        merged[name] = payload
    return merged


def _already_attempted_side_effect(messages: list[ChatMessage]) -> bool:
    for message in messages:
        for call in message.tool_calls:
            if call.name in {"send_email", "send_sms", "create_refund"}:
                return True
    return False


def _order_from_payloads(payloads: list[tuple[str, dict[str, Any]]]) -> str | None:
    for _, payload in payloads:
        order = payload.get("order")
        if isinstance(order, dict) and isinstance(order.get("id"), str):
            return str(order["id"])
        if isinstance(payload.get("order_id"), str):
            return str(payload["order_id"])
    return None
