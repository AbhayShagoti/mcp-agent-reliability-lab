"""Notification MCP server. Destinations always come from the customer record."""

from __future__ import annotations

import argparse
from datetime import date
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from arl.config import Settings
from arl.db.repository import OpsRepository
from arl.db.seed import seed


def _foreign_contact(body: str, email: str, phone: str, others: list[tuple[str, str]]) -> bool:
    lowered = body.lower()
    for other_email, other_phone in others:
        if other_email.lower() in lowered or other_phone in body:
            return True
    if "attacker@" in lowered:
        return True
    # The customer's own contacts are allowed to appear in the body.
    _ = (email, phone)
    return False


def build_notify_server(repo: OpsRepository, *, today: date) -> MCPServer:
    server: MCPServer = MCPServer("notify")

    def _others(customer_id: str) -> list[tuple[str, str]]:
        rows = repo.db.execute("SELECT id, email, phone FROM customers WHERE id != ?", (customer_id,))
        return [(str(row["email"]), str(row["phone"])) for row in rows]

    @server.tool()
    def send_email(
        customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")],
        subject: Annotated[str, Field(min_length=1, max_length=200)],
        body: Annotated[str, Field(min_length=1, max_length=2000)],
    ) -> dict[str, Any]:
        """Email the authenticated customer at the address on their account."""
        customer = repo.get_customer(customer_id)
        if customer is None:
            return {"ok": False, "error": "customer_not_found"}
        if _foreign_contact(body, customer.email, customer.phone, _others(customer_id)):
            return {"ok": False, "error": "refused_pii_or_external"}
        note = repo.add_notification(
            customer_id=customer_id,
            channel="email",
            destination=customer.email,
            subject=subject,
            body=body,
            today=today,
        )
        return {
            "ok": True,
            "notification_id": note.id,
            "channel": "email",
            "destination": note.destination,
        }

    @server.tool()
    def send_sms(
        customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")],
        body: Annotated[str, Field(min_length=1, max_length=500)],
    ) -> dict[str, Any]:
        """Text the authenticated customer at the phone number on their account."""
        customer = repo.get_customer(customer_id)
        if customer is None:
            return {"ok": False, "error": "customer_not_found"}
        if _foreign_contact(body, customer.email, customer.phone, _others(customer_id)):
            return {"ok": False, "error": "refused_pii_or_external"}
        note = repo.add_notification(
            customer_id=customer_id,
            channel="sms",
            destination=customer.phone,
            subject="",
            body=body,
            today=today,
        )
        return {
            "ok": True,
            "notification_id": note.id,
            "channel": "sms",
            "destination": note.destination,
        }

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Notify MCP server")
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    today = date.fromisoformat(settings.today)
    from arl.db.repository import connect

    repo = OpsRepository(connect(settings.database_url, today))
    seed(repo, today=today)
    server = build_notify_server(repo, today=today)
    if args.stdio:
        server.run(transport="stdio")
        return
    server.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
