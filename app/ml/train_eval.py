"""
Time-aware train/val/test split + baseline model (Logistic Regression) with optional artifact saving.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


FEATURES: list[str] = [
    "ret_1",
    "ret_3",
    "sma_5",
    "sma_10",
    "sma_20",
    "rsi_14",
    "vol_10",
    "typ_price",
    "tp_sma_10",
]
TARGET = "target_up_next_N"


def load_latest_dataset(pattern: str = "data/dataset_*_*.parquet") -> tuple[pd.DataFrame, str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No dataset parquet found in data/. Run the builder first.")
    path = files[-1]
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df, path


def time_split_idx(n: int, ratios: SplitRatios) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(ratios.train + ratios.val + ratios.test - 1.0) < 1e-6
    i_train_end = int(n * ratios.train)
    i_val_end = i_train_end + int(n * ratios.val)
    return np.arange(0, i_train_end), np.arange(i_train_end, i_val_end), np.arange(i_val_end, n)


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def simple_profit_metric(y_true: np.ndarray, y_pred: np.ndarray, ret_next: np.ndarray) -> float:
    return float(np.sum(ret_next * (y_pred == 1)))


def main():
    import argparse

    p = argparse.ArgumentParser(description="Time-aware split + baseline LR")
    p.add_argument("--dataset_glob", default="data/dataset_*_*.parquet")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold for class 1")
    p.add_argument("--out", default="data/eval_report.json")
    p.add_argument(
        "--save_artifacts", action="store_true", help="save model/scaler/meta into data/"
    )
    args = p.parse_args()

    df, path = load_latest_dataset(args.dataset_glob)

    # Ensure required columns exist
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Toy next return for simple profit metric
    df["ret_next"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURES].values
    y = df[TARGET].astype(int).values

    n = len(df)
    tr_idx, va_idx, te_idx = time_split_idx(n, SplitRatios())

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])
    X_te = scaler.transform(X[te_idx])

    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
    ret_next_tr = df["ret_next"].iloc[tr_idx].values
    ret_next_va = df["ret_next"].iloc[va_idx].values
    ret_next_te = df["ret_next"].iloc[te_idx].values

    clf = LogisticRegression(
        C=args.C, max_iter=args.max_iter, class_weight="balanced", solver="lbfgs"
    )
    clf.fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    thr = args.threshold
    yhat_tr = (p_tr >= thr).astype(int)
    yhat_va = (p_va >= thr).astype(int)
    yhat_te = (p_te >= thr).astype(int)

    m_tr = evaluate(y_tr, p_tr, yhat_tr)
    m_va = evaluate(y_va, p_va, yhat_va)
    m_te = evaluate(y_te, p_te, yhat_te)

    pnl_tr = simple_profit_metric(y_tr, yhat_tr, ret_next_tr)
    pnl_va = simple_profit_metric(y_va, yhat_va, ret_next_va)
    pnl_te = simple_profit_metric(y_te, yhat_te, ret_next_te)

    report = {
        "dataset_path": path,
        "n_rows": n,
        "split": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
        "features": FEATURES,
        "target": TARGET,
        "model": "LogisticRegression",
        "params": {
            "C": args.C,
            "max_iter": args.max_iter,
            "class_weight": "balanced",
            "threshold": thr,
        },
        "metrics": {"train": m_tr, "val": m_va, "test": m_te},
        "toy_profit_sum": {"train": pnl_tr, "val": pnl_va, "test": pnl_te},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote eval report → {args.out}")

    if args.save_artifacts:
        os.makedirs("data", exist_ok=True)
        joblib.dump(clf, "data/model.pkl")
        joblib.dump(scaler, "data/scaler.pkl")
        with open("data/model_meta.json", "w") as f:
            json.dump(
                {
                    "features": FEATURES,
                    "target": TARGET,
                    "threshold": thr,
                    "model": "LogisticRegression",
                    "params": {"C": args.C, "max_iter": args.max_iter, "class_weight": "balanced"},
                    "dataset_path": path,
                },
                f,
                indent=2,
            )
        print("Saved artifacts → data/model.pkl, data/scaler.pkl, data/model_meta.json")


if __name__ == "__main__":
    main()
