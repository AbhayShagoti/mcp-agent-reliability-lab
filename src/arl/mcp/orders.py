"""Orders MCP server: reads and refunds, scoped to the authenticated customer."""

from __future__ import annotations

import argparse
from datetime import date
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from arl.config import Settings
from arl.db.repository import OpsRepository
from arl.db.seed import seed
from arl.domain.approval import token_matches
from arl.domain.policy import evaluate_refund


def build_orders_server(
    repo: OpsRepository,
    *,
    today: date,
    threshold_cents: int,
    approval_secret: str,
) -> MCPServer:
    """Tools close over the repository so one process can serve many calls."""
    server: MCPServer = MCPServer("orders")

    @server.tool()
    def get_order(
        order_id: Annotated[str, Field(pattern=r"^ORD-\d+$")],
        customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")],
    ) -> dict[str, Any]:
        """Fetch one order that belongs to the given customer, including refunds."""
        order = repo.get_order(order_id, customer_id)
        if order is None:
            return {"found": False, "order_id": order_id}
        return {"found": True, "order": order.model_dump(mode="json")}

    @server.tool()
    def list_orders(
        customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")],
        status: str | None = None,
    ) -> dict[str, Any]:
        """List the customer's orders, optionally filtered by status."""
        orders = repo.list_orders(customer_id, status=status)
        return {
            "orders": [
                {
                    "id": order.id,
                    "status": order.status,
                    "total_cents": order.total_cents,
                    "created_at": order.created_at.isoformat(),
                    "final_sale": order.final_sale,
                }
                for order in orders
            ]
        }

    @server.tool()
    def get_shipment(
        order_id: Annotated[str, Field(pattern=r"^ORD-\d+$")],
        customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")],
    ) -> dict[str, Any]:
        """Fetch the shipment for an order the customer owns."""
        shipment = repo.get_shipment(order_id, customer_id)
        if shipment is None:
            return {"found": False, "order_id": order_id}
        return {"found": True, "shipment": shipment.model_dump(mode="json")}

    @server.tool()
    def get_customer(customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")]) -> dict[str, Any]:
        """Fetch the authenticated customer's own contact record."""
        customer = repo.get_customer(customer_id)
        if customer is None:
            return {"found": False, "customer_id": customer_id}
        return {"found": True, "customer": customer.model_dump(mode="json")}

    @server.tool()
    def create_refund(
        order_id: Annotated[str, Field(pattern=r"^ORD-\d+$")],
        customer_id: Annotated[str, Field(pattern=r"^CUST-\d+$")],
        amount_cents: Annotated[int, Field(gt=0)],
        reason: Annotated[str, Field(min_length=3, max_length=500)],
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Issue a refund when policy allows. Large refunds need a server-signed token."""
        order = repo.get_order(order_id, customer_id)
        has_refund = bool(order and order.refunds)
        approved = amount_cents <= threshold_cents or token_matches(
            approval_secret, order_id, customer_id, amount_cents, approval_token
        )
        decision = evaluate_refund(
            order,
            amount_cents,
            today,
            has_existing_refund=has_refund,
            threshold_cents=threshold_cents,
            has_approval=approved,
        )
        if not decision.allowed:
            return {"ok": False, "error": decision.reason}
        refund = repo.create_refund(
            order_id=order_id,
            amount_cents=amount_cents,
            reason=reason,
            today=today,
        )
        return {"ok": True, "refund": refund.model_dump(mode="json")}

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Orders MCP server")
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    today = date.fromisoformat(settings.today)
    from arl.db.repository import connect

    repo = OpsRepository(connect(settings.database_url, today))
    seed(repo, today=today)
    server = build_orders_server(
        repo,
        today=today,
        threshold_cents=settings.approval_threshold_cents,
        approval_secret=settings.approval_secret,
    )
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
