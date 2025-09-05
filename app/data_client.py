"""
Lightweight market data client with a tiny in-process cache.

Returns:
  {
    "symbol": "AAPL",
    "interval": "5m",
    "as_of": "2025-09-05T16:38:08Z",
    "candles": [ {ts, open, high, low, close, volume}, ... ]
  }

Notes / Pitfalls:
- Yahoo endpoints are unofficial -> can rate-limit (429) or change.
- We keep a short TTL cache and gracefully fall back to synthetic data.
- Intervals must be in _ALLOWED_INTERVALS.
"""

from __future__ import annotations

import math
import os
import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app.utils import utc_now_iso
from app.validator import validate_bar

# --------------------------------------------------------------------------------------
# Cache + config
# --------------------------------------------------------------------------------------
_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SEC = float(os.getenv("SF_DATA_TTL_SEC", "15"))
MAX_CACHE_SIZE = 16

# Supported intervals for Yahoo Chart
_ALLOWED_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"}

PROVIDER = os.getenv("SF_PROVIDER", "YAHOO_CHART").upper()

_INTERVAL_TO_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90, "1h": 60}


def _cache_get(key: tuple[str, str, str]) -> dict[str, Any] | None:
    now = time.time()
    hit = _CACHE.get(key)
    if not hit:
        return None
    exp, val = hit
    if now > exp:
        _CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: tuple[str, str, str], val: dict[str, Any]) -> None:
    if len(_CACHE) >= MAX_CACHE_SIZE:
        try:
            _CACHE.pop(next(iter(_CACHE)))
        except StopIteration:
            pass
    _CACHE[key] = (time.time() + CACHE_TTL_SEC, val)


def _normalize_interval(interval: str) -> str:
    interval = (interval or "").lower().strip()
    if interval == "5":
        interval = "5m"
    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(f"unsupported interval: {interval}")
    return interval


# --------------------------------------------------------------------------------------
# Synthetic fallback
# --------------------------------------------------------------------------------------
def _synthetic_candles_list(
    limit: int = 120, interval: str = "5m", mode: str = "flat"
) -> list[dict[str, Any]]:
    """
    Generate a list of OHLCV bars:
      {ts (epoch seconds), open, high, low, close, volume}
    mode: 'up' (drift up), 'down' (drift down), 'flat' (no drift)
    """
    minutes = _INTERVAL_TO_MIN.get(interval, 5)
    now = datetime.now(UTC)

    # simple random-walk with small drift
    drift = {"up": 0.04, "down": -0.04, "flat": 0.0}.get(mode, 0.0)
    price = 100.0
    bars: list[dict[str, Any]] = []

    for i in range(limit):
        # pseudo-random step
        step = random.gauss(drift, 0.5)
        new_close = max(1.0, price + step)
        open_ = price
        # small high/low range around open/close
        hi = max(open_, new_close) + random.random() * 0.3
        lo = min(open_, new_close) - random.random() * 0.3
        vol = random.randint(1000, 5000)
        ts = int((now - timedelta(minutes=minutes * (limit - 1 - i))).timestamp())

        candidate = {
            "ts": ts,
            "open": float(open_),
            "high": float(hi),
            "low": float(lo),
            "close": float(new_close),
            "volume": int(vol),
        }
        good = validate_bar(candidate, "SYNTH")
        if good:
            bars.append(good)
        price = new_close

    return bars


# --------------------------------------------------------------------------------------
# Yahoo parsing
# --------------------------------------------------------------------------------------
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

        # drop empty/NaN lines
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


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
async def fetch_market_data(
    symbol: str,
    interval: str,
    limit: int = 120,
    *,
    synthetic: int | bool = 0,
    synthetic_mode: str = "flat",
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Fetch candles and return normalized payload:
      {symbol, interval, as_of, candles: [bars]}
    Behavior:
      - If synthetic=1: return synthetic bars immediately (no network).
      - Else: try Yahoo; on HTTP 429 or network error, fall back to synthetic.
      - Tiny TTL cache to reduce calls.
    """
    sym = (symbol or "").upper()
    norm_interval = _normalize_interval(interval)
    as_of = utc_now_iso()

    cache_key = (sym, norm_interval, f"{limit}:{int(bool(synthetic))}:{synthetic_mode}")
    if use_cache:
        hit = _cache_get(cache_key)
        if hit:
            return hit

    # Synthetic requested explicitly
    if synthetic:
        candles = _synthetic_candles_list(limit=limit, interval=norm_interval, mode=synthetic_mode)
        payload = {"symbol": sym, "interval": norm_interval, "as_of": as_of, "candles": candles}
        _cache_set(cache_key, payload)
        return payload

    # Only provider we support right now
    if PROVIDER == "YAHOO_CHART":
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
            f"?interval={norm_interval}&range=1d&includePrePost=false&.tsrc=finance"
        )
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                candles = normalize_yahoo_chart(r.json(), sym)
        except httpx.HTTPStatusError as e:
            # 429 -> fallback to synthetic
            if e.response is not None and e.response.status_code == 429:
                candles = _synthetic_candles_list(
                    limit=limit, interval=norm_interval, mode=synthetic_mode
                )
            else:
                raise
        except httpx.RequestError:
            # network failure -> fallback
            candles = _synthetic_candles_list(
                limit=limit, interval=norm_interval, mode=synthetic_mode
            )

        payload = {"symbol": sym, "interval": norm_interval, "as_of": as_of, "candles": candles}
        _cache_set(cache_key, payload)
        return payload

    raise ValueError(f"Unsupported provider: {PROVIDER}")


# --------------------------------------------------------------------------------------
# Back-compat exports so routers/serve can resolve a fetcher by name.
# It may look for any of these; make them all point to the same thing.
# All are async functions; callers should `await` them.
get_series = fetch_market_data
fetch_series = fetch_market_data
get_market_data = fetch_market_data
