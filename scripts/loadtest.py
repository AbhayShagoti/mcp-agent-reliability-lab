"""Measure API overhead with the mock LLM.

Writes reports/loadtest.json. This is the source of the p50/p95 and RPS numbers
in the README. It does not measure a local neural model.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REQUESTS = 200
CONCURRENCY = 10
PORT = 8099


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


async def _once(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> tuple[float, int]:
    started = time.perf_counter()
    response = await client.post(
        url,
        json={
            "message": "What is the status of order ORD-1001?",
            "customer_id": "CUST-1001",
        },
        headers=headers,
    )
    elapsed = time.perf_counter() - started
    return elapsed, response.status_code


async def _drive(url: str) -> dict[str, object]:
    headers = {"Authorization": "Bearer dev-token"}
    timeout = httpx.Timeout(30.0)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for _ in range(5):
            _, status = await _once(client, url, headers)
            if status != 200:
                raise RuntimeError(f"warmup failed with {status}")
        semaphore = asyncio.Semaphore(CONCURRENCY)
        started = time.perf_counter()

        async def wrapped() -> tuple[float, int]:
            async with semaphore:
                return await _once(client, url, headers)

        results = await asyncio.gather(*(wrapped() for _ in range(REQUESTS)))
        elapsed = time.perf_counter() - started
    latencies = [item[0] * 1000 for item in results]
    errors = sum(1 for _, status in results if status != 200)
    return {
        "requests": REQUESTS,
        "concurrency": CONCURRENCY,
        "errors": errors,
        "elapsed_s": round(elapsed, 4),
        "rps": round(REQUESTS / elapsed, 3),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "mean": round(sum(latencies) / len(latencies), 3),
        },
        "provider": "mock",
        "note": (
            "API overhead with the deterministic mock LLM, not a local model. "
            "The per-token rate limit was raised to 100000/min so the sample is not 429s."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db = root / "data" / "loadtest.db"
    if db.exists():
        db.unlink()
    env = os.environ.copy()
    env.update(
        {
            "ARL_LLM_PROVIDER": "mock",
            "ARL_DATABASE_URL": f"sqlite:///{db}",
            "ARL_API_TOKEN": "dev-token",
            "ARL_TODAY": "2026-11-01",
            "ARL_RATE_LIMIT_PER_MINUTE": "100000",
        }
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "arl.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=root,
        env=env,
    )
    url = f"http://127.0.0.1:{PORT}/v1/chat"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1).raise_for_status()
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("API did not become healthy")
        report = asyncio.run(_drive(url))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if db.exists():
            db.unlink()
    out = root / "reports" / "loadtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if int(report["errors"]) != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
