"""Policy MCP server: hybrid BM25 + hashing-trick vectors, pgvector when Postgres is up."""

from __future__ import annotations

import argparse
from datetime import date
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from arl.config import Settings
from arl.db.repository import OpsRepository
from arl.db.seed import seed


def build_policy_server(repo: OpsRepository) -> MCPServer:
    server: MCPServer = MCPServer("policy")

    @server.tool()
    def search_policies(
        query: Annotated[str, Field(min_length=2, max_length=500)],
        k: Annotated[int, Field(ge=1, le=8)] = 3,
    ) -> dict[str, Any]:
        """Hybrid search over customer-ops policy documents."""
        hits = repo.search_policies(query, k=k)
        return {"hits": [hit.model_dump(mode="json") for hit in hits]}

    @server.tool()
    def get_policy(policy_id: Annotated[str, Field(pattern=r"^POL-[A-Z0-9-]+$")]) -> dict[str, Any]:
        """Fetch one policy document by id."""
        row = repo.get_policy(policy_id)
        if row is None:
            return {"found": False, "policy_id": policy_id}
        return {"found": True, "policy": row}

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy MCP server")
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    today = date.fromisoformat(settings.today)
    from arl.db.repository import connect

    repo = OpsRepository(connect(settings.database_url, today))
    seed(repo, today=today)
    server = build_policy_server(repo)
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
