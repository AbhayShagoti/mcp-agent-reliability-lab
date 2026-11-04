"""HTTP gateway: auth, rate limit, idempotency, and SSE chat."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from arl.agent.graph import build_graph, run_turn
from arl.api.limit import IdempotencyStore, RateLimiter
from arl.config import Settings
from arl.db.repository import OpsRepository, connect
from arl.db.seed import seed
from arl.llm.factory import build_llm
from arl.logging import configure_logging
from arl.mcp.client import open_tool_service
from arl.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS, render_metrics
from arl.observability.tracing import build_tracer


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    customer_id: str = Field(pattern=r"^CUST-\d+$")
    thread_id: str | None = None
    seed: int = 0
    approve: bool | None = None


def create_app(
    settings: Settings | None = None,
    *,
    span_exporter: Any | None = None,
) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(cfg.log_level)
        tracer, provider = build_tracer(cfg, span_exporter)
        app.state.tracer_provider = provider
        today = date.fromisoformat(cfg.today)
        repo = OpsRepository(connect(cfg.database_url, today))
        seed(repo, today=today)
        app.state.repo = repo
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            tools = await open_tool_service(stack, repo, cfg, tracer)
            llm = build_llm(cfg)
            app.state.graph = build_graph(llm, tools, cfg, tracer)
            app.state.tools = tools
            app.state.limiter = _limiter(cfg)
            app.state.idempotency = _idempotency(cfg)
            yield
        finally:
            await stack.aclose()
            provider.force_flush()
            repo.db.close()

    app = FastAPI(title="Agent Reliability Lab", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - started
        endpoint = request.url.path
        HTTP_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        HTTP_REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(render_metrics(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        _check_auth(request, cfg)
        token = _bearer(request) or "anonymous"
        if not app.state.limiter.allow(token):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        fingerprint = _fingerprint(body)
        cached = _read_idempotency(request, app.state.idempotency, fingerprint)
        if cached is not None:
            return StreamingResponse(_replay(cached), media_type="text/event-stream")
        thread_id = body.thread_id or str(uuid.uuid4())
        outcome = await run_turn(
            app.state.graph,
            customer_id=body.customer_id,
            message=body.message,
            thread_id=thread_id,
            today=cfg.today,
            seed=body.seed,
            approve=body.approve,
        )
        events = _events_from_outcome(outcome, thread_id)
        _store_idempotency(request, app.state.idempotency, fingerprint, events)
        return StreamingResponse(_replay(events), media_type="text/event-stream")

    return app


def _limiter(settings: Settings) -> RateLimiter:
    client = _redis(settings.redis_url)
    return RateLimiter(settings.rate_limit_per_minute, client)


def _idempotency(settings: Settings) -> IdempotencyStore:
    return IdempotencyStore(_redis(settings.redis_url))


def _redis(url: str | None) -> Any | None:
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url)


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()
    return None


def _check_auth(request: Request, settings: Settings) -> None:
    if _bearer(request) != settings.api_token:
        raise HTTPException(status_code=401, detail="unauthorized")


def _fingerprint(body: ChatRequest) -> str:
    raw = json.dumps(body.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _idem_key(request: Request) -> str | None:
    key = request.headers.get("idempotency-key")
    if key is None:
        return None
    cleaned = key.strip()
    return cleaned or None


def _read_idempotency(
    request: Request,
    store: IdempotencyStore,
    fingerprint: str,
) -> list[dict[str, Any]] | None:
    key = _idem_key(request)
    if key is None:
        return None
    existing = store.get(key)
    if existing is None:
        return None
    payload = json.loads(existing.decode())
    if payload["fingerprint"] != fingerprint:
        raise HTTPException(status_code=409, detail="idempotency key reused with a different body")
    events = payload["events"]
    if not isinstance(events, list):
        raise HTTPException(status_code=409, detail="corrupt idempotency record")
    return events


def _store_idempotency(
    request: Request,
    store: IdempotencyStore,
    fingerprint: str,
    events: list[dict[str, Any]],
) -> None:
    key = _idem_key(request)
    if key is None:
        return
    blob = json.dumps({"fingerprint": fingerprint, "events": events}).encode()
    store.put(key, blob)


def _events_from_outcome(outcome: dict[str, Any], thread_id: str) -> list[dict[str, Any]]:
    values = outcome["values"]
    events: list[dict[str, Any]] = []
    for event in values.get("guardrail_events", []):
        events.append({"event": "guardrail", "data": event})
    for call in values.get("tool_trace", []):
        events.append(
            {
                "event": "tool",
                "data": {"name": call["name"], "status": call["status"], "detail": call.get("detail", "")},
            }
        )
    if outcome["interrupts"]:
        events.append(
            {
                "event": "approval_required",
                "data": {
                    "thread_id": thread_id,
                    "interrupt": _interrupt_payload(outcome["interrupts"][0]),
                },
            }
        )
    events.append({"event": "final", "data": {"thread_id": thread_id, "response": values.get("response")}})
    return events


def _interrupt_payload(item: Any) -> Any:
    value = getattr(item, "value", None)
    return value if value is not None else str(item)


async def _replay(events: list[dict[str, Any]]) -> AsyncIterator[str]:
    for event in events:
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"


app = create_app()
