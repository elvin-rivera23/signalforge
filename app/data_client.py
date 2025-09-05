"""
Lightweight market data client with a tiny in-process cache.

What it returns:
- A normalized dict with symbol, interval, as_of, and a list of validated OHLCV bars.

Pitfalls:
- Yahoo's endpoints are unofficial and can rate-limit or change; we handle basic errors and keep TTL short.
- Symbols use Yahoo tickers (e.g., "AAPL", "MSFT"). Some exotic tickers may differ.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from app.utils import utc_now_iso
from app.validator import validate_bar

# very small cache: {(key): (expires_at, value)}
_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SEC = float(os.getenv("SF_DATA_TTL_SEC", "15"))
MAX_CACHE_SIZE = 16

# Supported intervals for Yahoo Chart
_ALLOWED_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"}

PROVIDER = os.getenv("SF_PROVIDER", "YAHOO_CHART").upper()


def _cache_get(key: tuple[str, str]) -> dict[str, Any] | None:
    now = time.time()
    hit = _CACHE.get(key)
    if not hit:
        return None
    exp, val = hit
    if now > exp:
        # stale
        _CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: tuple[str, str], val: dict[str, Any]) -> None:
    # simple cap
    if len(_CACHE) >= MAX_CACHE_SIZE:
        # drop arbitrary (first) item
        try:
            _CACHE.pop(next(iter(_CACHE)))
        except StopIteration:
            pass
    _CACHE[key] = (time.time() + CACHE_TTL_SEC, val)


def _normalize_interval(interval: str) -> str:
    interval = interval.lower().strip()
    if interval == "5":
        interval = "5m"
    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(f"unsupported interval: {interval}")
    return interval


async def fetch_candles(symbol: str, interval: str) -> dict[str, Any]:
    """
    Returns a dict:
    {
      "symbol": "AAPL",
      "interval": "1m",
      "as_of": "<UTC ISO>",
      "candles": [{"ts": <epoch_sec>, "open":..., "high":..., "low":..., "close":..., "volume":...}, ...]
    }
    Raises httpx.RequestError for network issues, or ValueError for invalid symbol/data.
    """
    norm_interval = _normalize_interval(interval)
    key = (symbol.upper(), norm_interval)

    cached = _cache_get(key)
    if cached:
        return cached

    if PROVIDER == "YAHOO_CHART":
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval={norm_interval}&range=1d&includePrePost=false&.tsrc=finance"
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=10.0)) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        # Normalize and validate
        candles = normalize_yahoo_chart(data, symbol)
        if not candles:
            raise ValueError("no candles parsed")
        as_of = utc_now_iso()
        payload = {
            "symbol": symbol.upper(),
            "interval": norm_interval,
            "as_of": as_of,
            "candles": candles,
        }
        _cache_set(key, payload)
        return payload

    raise ValueError(f"Unsupported provider: {PROVIDER}")


def normalize_yahoo_chart(resp: dict[str, Any], symbol: str) -> list[dict[str, Any]]:
    """
    Convert Yahoo Chart API response into validated OHLCV bars.
    We expect:
    resp["chart"]["result"][0]["timestamp"] -> list of epoch seconds
    resp["chart"]["result"][0]["indicators"]["quote"][0] -> dict with open/high/low/close/volume arrays
    """
    try:
        result = resp["chart"]["result"][0]
    except Exception:
        return []

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes_list = indicators.get("quote") or []
    if not timestamps or not quotes_list:
        return []
    quotes = quotes_list[0] or {}

    opens = quotes.get("open", [])
    highs = quotes.get("high", [])
    lows = quotes.get("low", [])
    closes = quotes.get("close", [])
    volumes = quotes.get("volume", [])

    bars: list[dict[str, Any]] = []
    n = min(len(timestamps), len(opens), len(highs), len(lows), len(closes))
    for i in range(n):
        ts = timestamps[i]
        o = opens[i]
        h = highs[i]
        lo = lows[i]
        c = closes[i]
        v = volumes[i] if i < len(volumes) else None

        # drop empty/Nan lines
        if any(
            val is None or (isinstance(val, float) and math.isnan(val)) for val in (o, h, lo, c)
        ):
            continue

        candidate = {
            "ts": int(ts),
            "open": float(o),
            "high": float(h),
            "low": float(lo),
            "close": float(c),
            "volume": int(v or 0),
        }
        good = validate_bar(candidate, symbol)
        if good:
            bars.append(good)

    return bars
