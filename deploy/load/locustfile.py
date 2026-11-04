"""Locust file for the mock API. The committed p50/p95 come from scripts/loadtest.py."""

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def lookup(self) -> None:
        self.client.post(
            "/v1/chat",
            json={
                "message": "What is the status of order ORD-1001?",
                "customer_id": "CUST-1001",
            },
            headers={"Authorization": "Bearer dev-token"},
        )
