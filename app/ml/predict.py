# app/ml/predict.py
# CLI scorer that reuses the shared Predictor.
# - Awaits async fetchers (e.g., data_client.fetch_candles coroutine)
# - Normalizes raw candles into a DataFrame (tries data_client.normalize_yahoo_chart if available)
# - Searches app.signals/* for a feature builder; if none, falls back to:
#     - if model_meta.json has "feature_names": select those columns
#     - else: pass normalized DataFrame through and rely on Predictor to validate n_features

import argparse
import asyncio
import importlib
import json
import pkgutil
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from app.ml.infer import Predictor

# ---------- small reflection helpers ----------


def _resolve_func(modname: str, candidates: list[str]) -> Callable[..., Any] | None:
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _resolve_any_callable_names(modname: str) -> list[str]:
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return []
    return [k for k, v in vars(mod).items() if callable(v)]


def _resolve_feature_builder_from_package() -> Callable[..., pd.DataFrame] | None:
    """Search app.signals (and submodules) for a function that builds features."""
    try:
        pkg = importlib.import_module("app.signals")
    except Exception:
        return None

    candidates = [
        "build_features",
        "make_features",
        "featurize",
        "build_feature_frame",
        "features_from_series",
        "build",
    ]

    # Check package root first
    for name in candidates:
        fn = getattr(pkg, name, None)
        if callable(fn):
            return fn

    # Walk submodules under app/signals/*
    if hasattr(pkg, "__path__"):
        prefix = pkg.__name__ + "."
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix):
            try:
                submod = importlib.import_module(modname)
            except Exception:
                continue
            for name in candidates:
                fn = getattr(submod, name, None)
                if callable(fn):
                    return fn
    return None


# ---------- data normalization ----------


def _await_if_needed(obj: Any) -> Any:
    if asyncio.iscoroutine(obj):
        return asyncio.run(obj)
    return obj


def _normalize_to_dataframe(raw: Any) -> pd.DataFrame:
    """
    Convert raw candles into a DataFrame.
    Tries project-specific normalizer first, then generic patterns.
    """
    # 1) Project normalizer if available
    norm = _resolve_func("app.data_client", ["normalize_yahoo_chart"])
    if norm is not None:
        try:
            df = norm(raw)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass  # fall back to generic

    # 2) Generic conversions
    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], dict)):
        return pd.DataFrame(raw)

    if isinstance(raw, dict):
        # Common container keys
        for key in ("candles", "bars", "items", "data", "result"):
            if key in raw and isinstance(raw[key], list | tuple):
                return pd.DataFrame(raw[key])
        # dict-of-arrays
        try:
            return pd.DataFrame(raw)
        except Exception:
            pass

    if isinstance(raw, pd.DataFrame):
        return raw.copy()

    raise TypeError(
        f"Unable to convert fetched data into a DataFrame (type={type(raw)}). "
        "Consider adding a normalizer in app.data_client."
    )


def _try_set_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: set datetime index from common columns."""
    for col in ("time", "timestamp", "t", "datetime"):
        if col in df.columns:
            s = df[col]
            try:
                if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
                    unit = "ms" if (s.dropna().iloc[-1] > 10**12) else "s"
                    idx = pd.to_datetime(s, unit=unit, utc=True)
                else:
                    idx = pd.to_datetime(s, utc=True, errors="coerce")
                out = df.copy()
                out.index = idx
                return out
            except Exception:
                continue
    return df


def _ensure_features_df(
    series: Any,
    predictor: Predictor,
    feature_builder: Callable[..., pd.DataFrame] | None,
) -> pd.DataFrame:
    """
    Use feature_builder if provided; else normalize to DataFrame and:
      - if predictor.feature_names exists: select those columns
      - else: pass the entire normalized DataFrame (Predictor will validate n_features)
    """
    if feature_builder is not None:
        df = feature_builder(series)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Feature builder did not return a pandas DataFrame.")
        return df

    df_all = _normalize_to_dataframe(series)
    df_all = _try_set_time_index(df_all)

    feats = predictor.feature_names  # may be None in fallback mode
    if isinstance(feats, list) and len(feats) > 0:
        missing = [c for c in feats if c not in df_all.columns]
        if missing:
            raise ValueError(
                "Raw data is missing required feature columns from model_meta.json.\n"
                f"Missing: {missing}\n"
                "Either implement app/signals/.../build_features(series)->DataFrame "
                "to create these columns, or retrain and include correct 'feature_names'."
            )
        return df_all[feats]

    # No feature_names in meta -> pass through; Predictor will validate dimensionality.
    return df_all


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(description="Score latest data using the trained model.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--synthetic_mode", choices=["up", "down"], default="up")
    args = ap.parse_args()

    # Resolve fetcher (your repo exposes fetch_candles)
    GET_SERIES = _resolve_func(
        "app.data_client",
        ["fetch_candles", "get_series", "fetch_series", "fetch_ohlcv", "load_series", "load_bars"],
    )
    if GET_SERIES is None:
        available = _resolve_any_callable_names("app.data_client")
        raise ImportError(
            "Could not resolve a series fetcher from app.data_client. "
            f"Available callables: {sorted(available)}"
        )

    # Call it with flexible kwargs; handle async automatically
    tried_kwargs = [
        {
            "symbol": args.symbol,
            "interval": args.interval,
            "limit": args.limit,
            "synthetic": bool(args.synthetic),
            "synthetic_mode": args.synthetic_mode,
        },
        {
            "symbol": args.symbol,
            "interval": args.interval,
            "limit": args.limit,
            "synthetic": bool(args.synthetic),
        },
        {"symbol": args.symbol, "interval": args.interval, "limit": args.limit},
        {"symbol": args.symbol, "interval": args.interval},
        {"symbol": args.symbol},
    ]
    series = None
    for kw in tried_kwargs:
        try:
            series = GET_SERIES(**kw)
            break
        except TypeError:
            continue
    if series is None:
        # positional fallbacks
        try:
            series = GET_SERIES(args.symbol, args.interval, args.limit)
        except TypeError:
            series = GET_SERIES(args.symbol, args.interval)

    series = _await_if_needed(series)

    # Resolve feature builder if present anywhere under app.signals/*
    feature_builder = _resolve_feature_builder_from_package()

    # Build features or pass-through
    predictor = Predictor()
    df_feats = _ensure_features_df(series, predictor, feature_builder)

    # Score
    proba = predictor.predict_proba(df_feats)
    pred = predictor.predict(df_feats, threshold=args.threshold)

    # Robust last timestamp
    try:
        idx = df_feats.index
        last_time = idx[-1].isoformat() if hasattr(idx, "isoformat") else str(idx[-1])
    except Exception:
        last_time = datetime.utcnow().isoformat()

    out: dict[str, Any] = {
        "symbol": args.symbol,
        "interval": args.interval,
        "n_rows_scored": int(len(df_feats)),
        "threshold": float(
            args.threshold if args.threshold is not None else predictor.meta.get("threshold", 0.5)
        ),
        "last": {
            "time": last_time,
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
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
