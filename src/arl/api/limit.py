"""In-memory rate limit and idempotency, with an optional Redis client."""

from __future__ import annotations

import time
from typing import Protocol


class CounterStore(Protocol):
    def incr(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> None: ...


class CacheStore(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def setex(self, key: str, seconds: int, value: bytes) -> None: ...


class RateLimiter:
    """Fixed window. ``redis_client`` is anything with ``incr`` and ``expire``."""

    def __init__(self, limit: int, redis_client: CounterStore | None = None, window_s: int = 60) -> None:
        self.limit = limit
        self.redis = redis_client
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.redis is not None:
            redis_key = f"arl:rate:{key}"
            count = int(self.redis.incr(redis_key))
            if count == 1:
                self.redis.expire(redis_key, self.window_s)
            return count <= self.limit
        moment = time.monotonic() if now is None else now
        recent = [hit for hit in self._hits.get(key, []) if moment - hit < self.window_s]
        if len(recent) >= self.limit:
            self._hits[key] = recent
            return False
        recent.append(moment)
        self._hits[key] = recent
        return True


class IdempotencyStore:
    """Remember a response body for an idempotency key."""

    def __init__(self, redis_client: CacheStore | None = None, ttl_s: int = 600) -> None:
        self.redis = redis_client
        self.ttl_s = ttl_s
        self._items: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        if self.redis is not None:
            return self.redis.get(f"arl:idem:{key}")
        return self._items.get(key)

    def put(self, key: str, payload: bytes) -> None:
        if self.redis is not None:
            self.redis.setex(f"arl:idem:{key}", self.ttl_s, payload)
            return
        self._items[key] = payload
