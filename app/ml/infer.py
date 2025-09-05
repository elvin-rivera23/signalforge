# app/ml/infer.py
# Single source of truth for inference (CLI + API).
# What it does:
# - Loads scaler, model, and metadata from data/*
# - Enforces training-time feature order via model_meta.json["feature_names"] when available
# - Falls back gracefully if feature_names are missing (validates dimensionality only)
# - Exposes predict_proba() and predict(threshold=...)
# Why needed:
# - Prevent serving drift; CLI and API call the same code.
# Pitfalls:
# - Best practice is to include "feature_names" in model_meta.json; fallback is a temporary convenience.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: str = "data/model.pkl"
    scaler_path: str = "data/scaler.pkl"
    meta_path: str = "data/model_meta.json"


class Predictor:
    def __init__(self, paths: ArtifactPaths | None = None):
        self.paths = paths or ArtifactPaths()
        self._model = None
        self._scaler = None
        self._meta: dict[str, Any] = {}
        self._artifact_hashes: dict[str, str] = {}
        self._load()

    # ---------- Public API ----------

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return self._artifact_hashes

    @property
    def feature_names(self):
        """
        Returns list of feature names if present in meta, else None.
        Fallback mode (None) means we won't reorder DataFrame columns; we only
        validate dimensionality against the scaler/model.
        """
        feats = self._meta.get("feature_names")
        if isinstance(feats, list) and len(feats) > 0:
            return feats
        return None  # fallback mode

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = self._prepare_features(X)
        X_scaled = self._scaler.transform(X_arr)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X_scaled)[:, 1]
        if hasattr(self._model, "decision_function"):
            s = self._model.decision_function(X_scaled)
            return 1.0 / (1.0 + np.exp(-s))
        preds = self._model.predict(X_scaled)
        return preds.astype(float)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        p = self.predict_proba(X)
        if threshold is None:
            threshold = float(self._meta.get("threshold", 0.5))
        return (p >= threshold).astype(int)

    # ---------- Internals ----------

    def _load(self) -> None:
        self._scaler = joblib.load(self.paths.scaler_path)
        self._model = joblib.load(self.paths.model_path)
        with open(self.paths.meta_path) as f:
            self._meta = json.load(f)
        self._artifact_hashes = {
            "model.pkl": self._sha256(self.paths.model_path),
            "scaler.pkl": self._sha256(self.paths.scaler_path),
            "model_meta.json": self._sha256(self.paths.meta_path),
        }

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]  # short for logs

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        If feature_names exist in meta, reorder DataFrame accordingly.
        If not, accept DataFrame/ndarray as-is but validate shape using the scaler/model.
        """
        # Infer expected dimension from scaler/model
        expected = None
        # Prefer scaler.n_features_in_ if present
        if hasattr(self._scaler, "n_features_in_"):
            try:
                expected = int(getattr(self._scaler, "n_features_in_"))
            except Exception:
                expected = None
        # Fallback to model.n_features_in_
        if expected is None and hasattr(self._model, "n_features_in_"):
            try:
                expected = int(getattr(self._model, "n_features_in_"))
            except Exception:
                expected = None

        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                missing = [c for c in self.feature_names if c not in X.columns]
                if missing:
                    raise ValueError(f"Missing required feature columns: {missing}")
                X_arr = X[self.feature_names].to_numpy(dtype=float)
            else:
                # Fallback: use DataFrame as-is
                X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")
        if expected is not None and X_arr.shape[1] != expected:
            raise ValueError(
                f"Feature dimension mismatch. Expected {expected} features, got {X_arr.shape[1]}.\n"
                "Tip: add 'feature_names' to model_meta.json to lock column order,"
                " or ensure your feature builder returns the same columns used during training."
            )
        return X_arr
