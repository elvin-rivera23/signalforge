# app/routers/signal.py

from fastapi import APIRouter, HTTPException, Query

from app.signals.baseline import compute_features

router = APIRouter(prefix="/signal", tags=["signal"])

# Try to import a real data client if you have one.
_get_recent_candles = None
try:
    from app.data_client import get_recent_candles as _real_get_recent_candles

    _get_recent_candles = _real_get_recent_candles
except Exception:
    try:
        from app.cache import get_recent_candles as _cache_get_recent_candles

        _get_recent_candles = _cache_get_recent_candles
    except Exception:
        _get_recent_candles = None


def _gen_synthetic_candles(limit: int = 200, mode: str = "up") -> list[dict[str, float]]:
    """Generate simple synthetic candles with only 'c' (close)."""
    if mode == "up":
        closes = [100 + i * 0.3 for i in range(limit)]
    elif mode == "down":
        closes = [100 - i * 0.3 for i in range(limit)]
    else:
        closes = [100.0 for _ in range(limit)]
    return [{"c": c} for c in closes]


@router.get("/")
def signal(
    symbol: str = Query(..., description="Ticker (e.g., AAPL)"),
    interval: str = Query("5m", description="Bar interval (e.g., 1m, 5m, 15m)"),
    limit: int = Query(200, ge=20, le=1000, description="Bars to load"),
    sma_fast: int = Query(20, ge=2, le=200),
    sma_slow: int = Query(50, ge=3, le=400),
    synthetic: int = Query(0, description="1 = use synthetic candles"),
    synthetic_mode: str = Query("up", description="up|down|flat (when synthetic=1)"),
):
    """Return baseline features + decision for a symbol/interval."""
    if synthetic == 1:
        candles = _gen_synthetic_candles(limit=limit, mode=synthetic_mode)
        source = f"synthetic:{synthetic_mode}"
    else:
        if _get_recent_candles is None:
            raise HTTPException(
                status_code=500,
                detail="No data client available. Use ?synthetic=1 or implement get_recent_candles.",
            )
        try:
            candles = _get_recent_candles(symbol=symbol, interval=interval, limit=limit)
            source = "live_client"
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"data_client_error: {e}")

    features = compute_features(candles, sma_fast_p=sma_fast, sma_slow_p=sma_slow)
    if not features.get("ok"):
        raise HTTPException(status_code=422, detail=features.get("reason", "feature_error"))

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "features": features,
        "meta": {
            "source": "baseline_v1",
            "model_version": "0.0.1",
            "data_source": source,
        },
    }
