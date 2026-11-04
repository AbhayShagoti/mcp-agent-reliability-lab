"""Prometheus metrics for the gateway, tools, and guardrails."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "arl_http_requests_total",
    "HTTP requests",
    ["endpoint", "status"],
)
HTTP_LATENCY = Histogram(
    "arl_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
TOOL_CALLS = Counter(
    "arl_tool_calls_total",
    "MCP tool calls",
    ["tool", "status"],
)
TOOL_ERRORS = Counter(
    "arl_tool_errors_total",
    "MCP tool errors",
    ["tool"],
)
GUARDRAIL_BLOCKS = Counter(
    "arl_guardrail_blocks_total",
    "Guardrail blocks by rule",
    ["rule"],
)
LLM_TOKENS = Counter(
    "arl_llm_tokens_total",
    "Tokens reported by the provider, or estimated for the mock",
    ["direction"],
)


def render_metrics() -> bytes:
    return generate_latest()
