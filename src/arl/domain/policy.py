"""Refund rules shared by the verifier and the orders MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from arl.domain.models import Order


@dataclass(frozen=True)
class RefundDecision:
    """`reason` is None when the refund may be written."""

    reason: str | None

    @property
    def allowed(self) -> bool:
        return self.reason is None


def evaluate_refund(
    order: Order | None,
    amount_cents: int,
    today: date,
    *,
    has_existing_refund: bool,
    threshold_cents: int,
    has_approval: bool,
    window_days: int = 30,
) -> RefundDecision:
    """Return the first policy reason that blocks a refund, if any."""
    if order is None:
        return RefundDecision("order_not_found")
    if amount_cents <= 0:
        return RefundDecision("amount_not_positive")
    if order.final_sale:
        return RefundDecision("final_sale")
    if order.status == "cancelled":
        return RefundDecision("cancelled")
    if order.status == "refunded" or has_existing_refund:
        return RefundDecision("already_refunded")
    age = (today - order.created_at).days
    if age > window_days:
        return RefundDecision("outside_window")
    if amount_cents > order.total_cents:
        return RefundDecision("amount_exceeds_total")
    if amount_cents > threshold_cents and not has_approval:
        return RefundDecision("approval_required")
    return RefundDecision(None)
