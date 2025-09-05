from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# Return ISO8601 UTC or None


def _parse_ts(ts: Any) -> str | None:
    """Convert epoch seconds or ISO string into ISO8601 UTC string."""
    if ts is None:
        return None
    # Epoch timestamp
    if isinstance(ts, int | float):
        return datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
    # ISO string
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC).isoformat()
        except Exception:
            return None
    return None


def validate_bar(bar: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    """Check a single OHLCV bar and return normalized dict or None if invalid."""
    required = ("open", "high", "low", "close")
    for k in required:
        if k not in bar:
            return None
    try:
        ts = _parse_ts(bar.get("ts")) if "ts" in bar else None
    except Exception:
        ts = None

    try:
        o, h, low, c = map(float, (bar["open"], bar["high"], bar["low"], bar["close"]))
    except Exception:
        return None
    if not (low <= o <= h and low <= c <= h):
        return None

    out = {
        "symbol": symbol.upper(),
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": int(bar.get("volume") or 0),
    }
    if ts is not None:
        # keep both numeric and iso if caller prefers
        out["ts"] = bar.get("ts")
        out["time"] = ts
    return out
