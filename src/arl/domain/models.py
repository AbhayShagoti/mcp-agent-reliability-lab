"""Pydantic models for orders, policies, notifications, and agent replies."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderItem(_Strict):
    sku: str
    name: str
    qty: int = Field(ge=1)
    cents: int = Field(ge=0)


class Customer(_Strict):
    id: str
    name: str
    email: str
    phone: str


class Order(_Strict):
    id: str
    customer_id: str
    status: str
    total_cents: int
    created_at: date
    final_sale: bool
    items: list[OrderItem]
    notes: str
    age_days: int = 0
    refunds: list[Refund] = Field(default_factory=list)


class Shipment(_Strict):
    order_id: str
    carrier: str
    tracking_number: str
    status: str
    eta: str | None = None


class Refund(_Strict):
    id: str
    order_id: str
    amount_cents: int
    reason: str
    status: str
    created_at: str


class Notification(_Strict):
    id: str
    customer_id: str
    channel: str
    destination: str
    subject: str
    body: str
    created_at: str


class Policy(_Strict):
    id: str
    title: str
    category: str
    body: str


class PolicyHit(_Strict):
    policy_id: str
    title: str
    category: str
    snippet: str
    score: float


class AgentResponse(_Strict):
    """Structured final answer the output guardrail enforces."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


# Refund is referenced inside Order before the class body finishes binding under
# future annotations, so rebuild after both models exist.
Order.model_rebuild()
