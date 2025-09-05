# app/signals/basic.py

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _ensure_df(raw: Any) -> pd.DataFrame:
    """
    Accepts:
      - dict payload like {"candles": [{ts, open, high, low, close, volume}, ...]}
      - list[dict] of candles
      - DataFrame with OHLCV
    Returns a DataFrame with columns: open, high, low, close, volume (float)
    """
    if isinstance(raw, dict) and "candles" in raw:
        data = raw["candles"]
    elif isinstance(raw, Iterable) and not isinstance(raw, str | bytes):
        data = raw
    elif isinstance(raw, pd.DataFrame):
        df = raw.copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["open", "high", "low", "close", "volume"]]

    df = pd.DataFrame(list(data))
    # some sources use 'ts' instead of 'time' — keep as index if present
    if "ts" in df and "time" not in df:
        df = df.sort_values("ts").reset_index(drop=True)
    elif "time" in df:
        df = df.sort_values("time").reset_index(drop=True)

    # coerce numeric
    for col in ("open", "high", "low", "close", "volume"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df[["open", "high", "low", "close", "volume"]]


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0.0).astype(float)
    down = (-delta.clip(upper=0.0)).astype(float)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ret(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def build_features(raw: Any) -> pd.DataFrame:
    """
    Normalize OHLCV into the features expected by the trained model.
    Guaranteed outputs when enough rows exist:
      ret_1, ret_3, sma_5, sma_10, sma_20, rsi_14, vol_10, typ_price, tp_sma_10

    Returns a DataFrame whose columns will be trimmed/reordered later in the router
    to match Predictor().feature_names if available.
    """
    df = _ensure_df(raw)

    # bases
    df_feat = pd.DataFrame(index=df.index)
    df_feat["open"] = df["open"].astype(float)
    df_feat["high"] = df["high"].astype(float)
    df_feat["low"] = df["low"].astype(float)
    df_feat["close"] = df["close"].astype(float)
    df_feat["volume"] = df["volume"].astype(float)

    # derived bases
    df_feat["typ_price"] = (df_feat["high"] + df_feat["low"] + df_feat["close"]) / 3.0

    # required features
    df_feat["ret_1"] = _ret(df_feat["close"], 1)
    df_feat["ret_3"] = _ret(df_feat["close"], 3)
    df_feat["sma_5"] = _sma(df_feat["close"], 5)
    df_feat["sma_10"] = _sma(df_feat["close"], 10)
    df_feat["sma_20"] = _sma(df_feat["close"], 20)
    df_feat["rsi_14"] = _rsi(df_feat["close"], 14)
    df_feat["vol_10"] = df_feat["volume"].rolling(10, min_periods=10).mean()
    df_feat["tp_sma_10"] = _sma(df_feat["typ_price"], 10)

    # drop warmup NaNs so model sees only complete rows
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Return ALL engineered columns; the router will select/align to model.feature_names
    return df_feat
