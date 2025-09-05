# app/routers/serve.py
from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.infer import Predictor
from app.signals.basic import build_features

router = APIRouter(prefix="/api/v1", tags=["serve"])


# --------- flexible fetcher resolution & calling ---------


def _resolve_fetcher():
    dc = import_module("app.data_client")
    for name in [
        "fetch_candles",
        "get_series",
        "fetch_series",
        "fetch_ohlcv",
        "load_series",
        "load_bars",
    ]:
        fn = getattr(dc, name, None)
        if callable(fn):
            return fn
    raise RuntimeError("No series fetcher found in app.data_client")


GET_SERIES = _resolve_fetcher()


def _call_fetcher_flex(
    symbol: str,
    interval: str,
    limit: int,
    synthetic: bool,
    synthetic_mode: str,
):
    """
    Try several kwarg shapes so we don't have to know the exact signature.
    Falls back to positional calls. Awaits coroutine automatically.
    """
    tried = []

    # Try richest signatures first, then progressively fewer kwargs
    kwarg_attempts = [
        dict(
            symbol=symbol,
            interval=interval,
            limit=limit,
            synthetic=synthetic,
            synthetic_mode=synthetic_mode,
        ),
        dict(symbol=symbol, interval=interval, synthetic=synthetic, synthetic_mode=synthetic_mode),
        dict(symbol=symbol, interval=interval, limit=limit),
        dict(symbol=symbol, interval=interval),
        dict(symbol=symbol),
    ]

    for kwargs in kwarg_attempts:
        try:
            res = GET_SERIES(**kwargs)
            return asyncio.run(res) if asyncio.iscoroutine(res) else res
        except TypeError as e:
            tried.append(f"{kwargs} -> {e}")

    # Positional fallbacks
    for args in [
        (symbol, interval, limit),
        (symbol, interval),
        (symbol,),
    ]:
        try:
            res = GET_SERIES(*args)
            return asyncio.run(res) if asyncio.iscoroutine(res) else res
        except TypeError as e:
            tried.append(f"{args} -> {e}")

    raise TypeError(
        "Could not call series fetcher with any common signatures.\n"
        "Tried:\n- " + "\n- ".join(tried)
    )


# --------- request/response models ---------


class ScoreRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    interval: str = Field(..., description="Bar interval, e.g., 5m")
    limit: int = Field(300, ge=1, le=5000)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    synthetic: int = Field(0, description="1 to use synthetic; 0 otherwise")
    synthetic_mode: str = Field("up", description="up or down")


@router.get("/version")
def version() -> dict[str, Any]:
    pred = Predictor()
    svc_ver = getattr(import_module("app.version"), "SERVICE_VERSION", "0.1.0")
    return {
        "service_version": svc_ver,
        "model_version": pred.meta.get("model_version"),
        "dataset_version": pred.meta.get("dataset_version"),
        "artifact_hashes": pred.artifact_hashes,
        "feature_count": (
            len(pred.feature_names)
            if pred.feature_names
            else getattr(pred._scaler, "n_features_in_", None)
        ),
    }


@router.post("/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    try:
        # 1) Fetch raw series with flexible call (handles async internally)
        series = _call_fetcher_flex(
            symbol=req.symbol,
            interval=req.interval,
            limit=req.limit,
            synthetic=bool(req.synthetic),
            synthetic_mode=req.synthetic_mode,
        )

        # 2) Build features and score
        df_feats: pd.DataFrame = build_features(series)

        predictor = Predictor()
        proba = predictor.predict_proba(df_feats)
        pred = predictor.predict(df_feats, threshold=req.threshold)

        # 3) Last timestamp (best effort)
        last_time = None
        try:
            idx = df_feats.index
            last_time = idx[-1].isoformat() if hasattr(idx, "isoformat") else str(idx[-1])
        except Exception:
            pass

        return {
            "symbol": req.symbol,
            "interval": req.interval,
            "n_rows_scored": int(len(df_feats)),
            "threshold": float(
                req.threshold if req.threshold is not None else predictor.meta.get("threshold", 0.5)
            ),
            "last": {
                "time": (
                    last_time
                    if last_time is not None
                    else (str(len(df_feats) - 1) if len(df_feats) else None)
                ),
                "proba": float(proba[-1]),
                "pred": int(pred[-1]),
            },
            "counts": {
                "pred_0": int((pred == 0).sum()),
                "pred_1": int((pred == 1).sum()),
            },
            "model_version": predictor.meta.get("model_version"),
            "dataset_version": predictor.meta.get("dataset_version"),
            "artifact_hashes": predictor.artifact_hashes,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
