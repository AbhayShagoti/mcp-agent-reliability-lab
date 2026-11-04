"""Uvicorn entrypoint."""

from __future__ import annotations

import uvicorn

from arl.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run("arl.api.app:app", host=settings.host, port=settings.port, factory=False)
