# app/cache.py
# Purpose: Simple in-memory cache for live quotes.
# Why: Reduce API calls to Yahoo, protect against rate limits.
# Pitfalls: Not persistent; resets if container restarts.

import time
from typing import Any

# store: (symbol, interval) -> (expiry_epoch, data)
_cache: dict[tuple[str, str], tuple[float, Any]] = {}


def get_cache(symbol: str, interval: str) -> Any:
    """Return cached data if valid, else None."""
    key = (symbol.upper(), interval)
    entry = _cache.get(key)
    if not entry:
        return None
    expiry, data = entry
    if time.time() > expiry:
        # expired
        _cache.pop(key, None)
        return None
    return data


def set_cache(symbol: str, interval: str, data: Any, ttl_s: int = 15) -> None:
    """Store data with expiry."""
    key = (symbol.upper(), interval)
    expiry = time.time() + ttl_s
    _cache[key] = (expiry, data)
