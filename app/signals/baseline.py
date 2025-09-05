# app/signals/baseline.py
from __future__ import annotations

from typing import Literal

# Expected candle dict:
# {"t": "2025-09-04T16:05:00Z", "o": 228.12, "h": 229.10, "l": 227.98, "c": 228.77, "v": 123456}


def _sma(values: list[float], period: int) -> float | None:
    """Simple moving average over the last `period` values. None if insufficient."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _rsi14(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI. None if insufficient."""
    if len(closes) <= period:
        return None
    gains, losses = 0.0, 0.0
    for i in range(len(closes) - period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change  # add absolute of negative change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features(candles: list[dict], sma_fast_p: int = 20, sma_slow_p: int = 50) -> dict:
    """
    Compute baseline indicators and decision from OHLCV candles.
    Returns a dict with 'ok', indicators, and 'decision'.
    """
    if not candles:
        return {"ok": False, "reason": "no_candles"}

    # Extract closes; be forgiving if some candles miss 'c'
    closes = [c["c"] for c in candles if "c" in c]
    if len(closes) == 0:
        return {"ok": False, "reason": "no_closes"}

    sma_fast = _sma(closes, sma_fast_p)
    sma_slow = _sma(closes, sma_slow_p)
    rsi = _rsi14(closes, 14)

    # Crossover logic
    if sma_fast is not None and sma_slow is not None:
        if sma_fast > sma_slow:
            crossover = "bullish"
        elif sma_fast < sma_slow:
            crossover = "bearish"
        else:
            crossover = "none"
    else:
        crossover = "none"

    # RSI state
    if rsi is None:
        rsi_state = "unknown"
    elif rsi >= 70:
        rsi_state = "overbought"
    elif rsi <= 30:
        rsi_state = "oversold"
    else:
        rsi_state = "neutral"

    # Decision
    decision: Literal["BUY", "SELL", "HOLD"]
    if sma_fast is not None and sma_slow is not None and rsi is not None:
        if (sma_fast > sma_slow) and (rsi < 65):
            decision = "BUY"
        elif (sma_fast < sma_slow) and (rsi > 35):
            decision = "SELL"
        else:
            decision = "HOLD"
    else:
        decision = "HOLD"

    return {
        "ok": True,
        "latest_close": closes[-1],
        "sma_fast_p": sma_fast_p,
        "sma_slow_p": sma_slow_p,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "rsi14": rsi,
        "crossover": crossover,
        "rsi_state": rsi_state,
        "decision": decision,
        "num_candles": len(closes),
    }
