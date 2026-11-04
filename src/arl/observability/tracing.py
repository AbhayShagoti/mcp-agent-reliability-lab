"""Tracer construction. Spans are recorded on this provider, not a process global."""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Tracer

from arl.config import Settings


def build_tracer(settings: Settings, exporter: Any | None = None) -> tuple[Tracer, TracerProvider]:
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    if settings.otel_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True))
        )
    # Best-effort global provider so auto-instrumentation can see the same spans.
    # The SDK ignores a second assignment; our returned tracer still exports.
    current = trace.get_tracer_provider()
    if isinstance(current, trace.ProxyTracerProvider):
        trace.set_tracer_provider(provider)
    return provider.get_tracer("arl"), provider
