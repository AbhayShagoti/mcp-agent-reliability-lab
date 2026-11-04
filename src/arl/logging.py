"""Structured JSON logs for the API and MCP servers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra fields on the record are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_ATTRS:
                continue
            if key in payload:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


_STANDARD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}


def configure_logging(level: str = "INFO") -> None:
    """Replace the root handler with a JSON stream handler."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
