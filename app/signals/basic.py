# app/signals/basic.py
from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from app.ml.infer import Predictor


def _resolve_func(modname: str, names: list[str]):
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return None
    for nm in names:
        fn = getattr(mod, nm, None)
        if callable(fn):
            return fn
    return None


def _to_dataframe(raw: Any) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
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
    if isinstance(raw, list | tuple):
        return pd.DataFrame(list(raw))
    raise TypeError("cannot convert input to DataFrame")


def build_features(raw: Any) -> pd.DataFrame:
    """
    Normalize an OHLCV-like structure into a feature frame suitable for Predictor().
    - Renames short columns (o/h/l/c/v) to open/high/low/close/volume
    - Ensures numeric dtypes
    - Keeps ordering to match Predictor().feature_names when available
    """
    df = _to_dataframe(raw)

    # Rename common short forms to canonical names
    rename_map: dict[str, str] = {}
    if "o" in df.columns and "open" not in df.columns:
        rename_map["o"] = "open"
    if "h" in df.columns and "high" not in df.columns:
        rename_map["h"] = "high"
    if "l" in df.columns and "low" not in df.columns:
        rename_map["l"] = "low"
    if "c" in df.columns and "close" not in df.columns:
        rename_map["c"] = "close"
    if "v" in df.columns and "volume" not in df.columns:
        rename_map["v"] = "volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce to numeric where present
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic derived features (safe/no lookahead)
    if {"close", "open"}.issubset(df.columns):
        df["ret_1"] = df["close"].pct_change().fillna(0.0)
    else:
        df["ret_1"] = 0.0

    # SMA examples if close exists
    if "close" in df.columns:
        df["sma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["sma_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
    else:
        df["sma_5"] = 0.0
        df["sma_10"] = 0.0
        df["sma_20"] = 0.0

    # Reorder columns to match model if available
    try:
        pred = Predictor()
        feature_names = getattr(pred, "feature_names", None)
        if feature_names:
            # keep only known features in correct order, append extras at the end
            known = [c for c in feature_names if c in df.columns]
            extras = [c for c in df.columns if c not in known]
            df = df[known + extras]
    except Exception:
        pass

    return df.fillna(0.0)
