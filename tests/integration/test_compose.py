"""Smoke the compose stack when ARL_INTEGRATION=1."""

from __future__ import annotations

import os

import httpx
import pytest


@pytest.mark.integration
def test_compose_chat() -> None:
    if os.environ.get("ARL_INTEGRATION") != "1":
        pytest.skip("set ARL_INTEGRATION=1 when docker compose is up")
    base = os.environ.get("ARL_BASE_URL", "http://127.0.0.1:8000")
    health = httpx.get(f"{base}/health", timeout=10)
    assert health.status_code == 200
    response = httpx.post(
        f"{base}/v1/chat",
        json={"message": "What is the status of order ORD-1001?", "customer_id": "CUST-1001"},
        headers={"Authorization": "Bearer dev-token"},
        timeout=30,
    )
    assert response.status_code == 200
    assert "delivered" in response.text.lower()
