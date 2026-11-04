"""Eval task schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequiredTool(_Strict):
    name: str
    arguments: dict[str, str | int] = Field(default_factory=dict)


class ExpectedRefund(_Strict):
    order_id: str
    amount_cents: int


class ExpectedNotification(_Strict):
    channel: Literal["email", "sms"]
    destination: str
    body_contains: str


class EvalTask(_Strict):
    id: str
    split: Literal["benign", "adversarial"]
    category: str
    attack_type: str | None = None
    user: str
    customer_id: str
    approve: bool | None = None
    expected_interrupt: bool = False
    required_tools: list[RequiredTool] = Field(default_factory=list)
    response_contains: list[str] = Field(default_factory=list)
    expect_refused: bool | None = None
    expect_guardrail_types: list[str] = Field(default_factory=list)
    expected_new_refunds: list[ExpectedRefund] = Field(default_factory=list)
    expected_order_status: dict[str, str] = Field(default_factory=dict)
    expected_notifications: list[ExpectedNotification] = Field(default_factory=list)
