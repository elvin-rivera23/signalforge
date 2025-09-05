# app/dev/test_baseline.py
from app.signals.baseline import compute_features

# Build synthetic candles rising gradually, then cooling off.
# We only need the 'c' field for our indicators.
closes = [100 + i * 0.3 for i in range(60)]  # 60 bars, mild uptrend
candles = [{"c": c} for c in closes]

features = compute_features(candles, sma_fast_p=20, sma_slow_p=50)
print(
    {
        "ok": features["ok"],
        "num_candles": features["num_candles"],
        "sma_fast": round(features["sma_fast"], 4) if features["sma_fast"] else None,
        "sma_slow": round(features["sma_slow"], 4) if features["sma_slow"] else None,
        "rsi14": round(features["rsi14"], 2) if features["rsi14"] is not None else None,
        "crossover": features["crossover"],
        "rsi_state": features["rsi_state"],
        "decision": features["decision"],
    }
)
