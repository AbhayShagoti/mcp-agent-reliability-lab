"""HMAC approval tokens so a model cannot authorize its own large refund."""

from __future__ import annotations

import hashlib
import hmac


def approval_token(secret: str, order_id: str, customer_id: str, amount_cents: int) -> str:
    message = f"{order_id}|{customer_id}|{amount_cents}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def token_matches(
    secret: str,
    order_id: str,
    customer_id: str,
    amount_cents: int,
    presented: str | None,
) -> bool:
    if not presented:
        return False
    expected = approval_token(secret, order_id, customer_id, amount_cents)
    return hmac.compare_digest(expected, presented)
