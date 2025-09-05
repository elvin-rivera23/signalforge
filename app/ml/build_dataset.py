import json
import os
from datetime import datetime

import pandas as pd
import requests
from pydantic import ValidationError
from ta.momentum import RSIIndicator

from .schema import DatasetConfig


def fetch_candles(cfg: DatasetConfig) -> pd.DataFrame:
    base = "http://localhost:8001/series"
    params = {
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "limit": 1000,
        "synthetic": 1 if cfg.synthetic else 0,
    }
    if cfg.synthetic and cfg.synthetic_mode:
        params["synthetic_mode"] = cfg.synthetic_mode

    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    candles = payload.get("candles", [])
    if not candles:
        raise RuntimeError("No candles returned from /series")
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    for win in (5, 10, 20):
        out[f"sma_{win}"] = out["close"].rolling(win).mean()
    rsi = RSIIndicator(close=out["close"], window=14)
    out["rsi_14"] = rsi.rsi()
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["typ_price"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["tp_sma_10"] = out["typ_price"].rolling(10).mean()
    return out


def add_label(df: pd.DataFrame, lookahead_n: int, up_threshold: float) -> pd.DataFrame:
    out = df.copy()
    future_close = out["close"].shift(-lookahead_n)
    out["target_up_next_N"] = (future_close >= out["close"] * (1.0 + up_threshold)).astype(float)
    return out


def clean_for_training(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)


def save_parquet(df: pd.DataFrame, cfg: DatasetConfig, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"dataset_{cfg.symbol}_{cfg.interval}_{cfg.dataset_version}_{stamp}.parquet"
    fpath = os.path.join(out_dir, fname)
    df.to_parquet(fpath, index=False)
    meta = {
        "created_utc": stamp,
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "dataset_version": cfg.dataset_version,
        "lookahead_n": cfg.lookahead_n,
        "up_threshold": cfg.up_threshold,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    with open(fpath.replace(".parquet", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return fpath


def build(cfg: DatasetConfig) -> str:
    candles = fetch_candles(cfg)
    feats = add_features(candles)
    labeled = add_label(feats, cfg.lookahead_n, cfg.up_threshold)
    final = clean_for_training(labeled)
    return save_parquet(final, cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build offline training dataset from /series")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookahead_n", type=int, default=3)
    parser.add_argument("--up_threshold", type=float, default=0.001)
    parser.add_argument("--dataset_version", default="v0")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--synthetic_mode", default="up")
    args = parser.parse_args()

    try:
        cfg = DatasetConfig(
            symbol=args.symbol,
            interval=args.interval,
            lookahead_n=args.lookahead_n,
            up_threshold=args.up_threshold,
            dataset_version=args.dataset_version,
            synthetic=args.synthetic,
            synthetic_mode=args.synthetic_mode,
        )
    except ValidationError as e:
        print("Invalid config:", e)
        raise SystemExit(2)

    out_path = build(cfg)
    print(f"Wrote dataset → {out_path}")
