"""MCP client used by the agent. In-process for tests, HTTP URL in Docker."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from mcp import Client
from opentelemetry.trace import Tracer

from arl.config import Settings
from arl.db.repository import OpsRepository
from arl.mcp.notify import build_notify_server
from arl.mcp.orders import build_orders_server
from arl.mcp.policy import build_policy_server


def decode_tool_result(result: Any) -> dict[str, Any]:
    """Normalize an MCP CallToolResult into a dict."""
    if bool(getattr(result, "is_error", False)):
        return {"ok": False, "error": _text(result)}
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    text = _text(result)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    if isinstance(parsed, dict):
        return parsed
    return {"raw": parsed}


def _text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolService:
    """Routes a tool name to the MCP server that advertised it."""

    clients: dict[str, Any]
    tracer: Tracer
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    routing: dict[str, str] = field(default_factory=dict)

    async def load(self) -> None:
        self.specs.clear()
        self.routing.clear()
        for server_name, client in self.clients.items():
            listed = await client.list_tools()
            for tool in listed.tools:
                schema = dict(tool.input_schema)
                self.specs[tool.name] = ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=schema,
                )
                self.routing[tool.name] = server_name

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from arl.observability.metrics import TOOL_CALLS, TOOL_ERRORS

        server_name = self.routing[name]
        with self.tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("tool.name", name)
            span.set_attribute("mcp.server", server_name)
            try:
                result = await self.clients[server_name].call_tool(name, arguments)
            except Exception as exc:
                TOOL_ERRORS.labels(tool=name).inc()
                TOOL_CALLS.labels(tool=name, status="error").inc()
                span.set_attribute("tool.status", "error")
                span.record_exception(exc)
                return {"ok": False, "error": str(exc)}
            payload = decode_tool_result(result)
            status = "error" if payload.get("ok") is False or payload.get("error") else "ok"
            if status == "error":
                TOOL_ERRORS.labels(tool=name).inc()
            TOOL_CALLS.labels(tool=name, status=status).inc()
            span.set_attribute("tool.status", status)
            return payload


async def open_tool_service(
    stack: AsyncExitStack,
    repo: OpsRepository,
    settings: Settings,
    tracer: Tracer,
) -> ToolService:
    """Connect three MCP clients, preferring HTTP URLs when configured."""
    today = date.fromisoformat(settings.today)
    targets: dict[str, Any] = {
        "orders": settings.orders_mcp_url
        or build_orders_server(
            repo,
            today=today,
            threshold_cents=settings.approval_threshold_cents,
            approval_secret=settings.approval_secret,
        ),
        "policy": settings.policy_mcp_url or build_policy_server(repo),
        "notify": settings.notify_mcp_url or build_notify_server(repo, today=today),
    }
    clients: dict[str, Any] = {}
    for name, target in targets.items():
        client = Client(target)
        await stack.enter_async_context(client)
        clients[name] = client
    service = ToolService(clients=clients, tracer=tracer)
    await service.load()
    return service
