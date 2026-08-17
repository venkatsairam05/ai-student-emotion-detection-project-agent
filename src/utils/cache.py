"""Small caching helpers (in-memory TTL cache)."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class TTLCache:
    """A minimal thread-safe-ish TTL cache with an LRU eviction policy.

    Falls back gracefully to computing the value when the cache is full or a
    computation raises, so callers never receive stale/None results for
    non-cacheable inputs.
    """

    def __init__(self, max_size: int = 512, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() > expires_at:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.monotonic() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def cached(
    key_fn: Callable[..., str],
    ttl_seconds: float = 3600.0,
    max_size: int = 512,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that memoizes a function through a shared :class:`TTLCache`.

    On a cache hit the wrapped function is not called. If the wrapped function
    raises an exception the result is NOT cached.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: TTLCache = TTLCache(max_size=max_size, ttl_seconds=ttl_seconds)

        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = key_fn(*args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator
