# app/main.py
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from app.logging_conf import setup_logging

# --- Observability ---
from app.observability import metrics_endpoint, timing_middleware

# --- Routers ---
from app.routers import serve  # /api/v1/score etc.
from app.routes_backtest import router as backtest_router
from app.routes_stream import router as stream_router
from app.version import version_payload  # central version info

# --- App ---
setup_logging()
app = FastAPI(title="SignalForge", version="0.1.0")

# --- Include routers ---
app.include_router(serve.router)
app.include_router(stream_router)
app.include_router(backtest_router)

# --- Observability ---
app.middleware("http")(timing_middleware)


# --- Schemas for utility endpoints ---
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    as_of: str
    service: Literal["signalforge"] = "signalforge"
    model_file: bool
    scaler_file: bool


class VersionResponse(BaseModel):
    service_version: str
    model_version: str | None = None
    featureset: str | None = None


# --- Utility endpoints ---


@app.get("/health", response_model=HealthResponse)
def health():
    model_ok = Path("data/model.pkl").exists()
    scaler_ok = Path("data/scaler.pkl").exists()
    status = "ok" if (model_ok and scaler_ok) else "degraded"
    return {
        "status": status,
        "as_of": datetime.now(UTC).isoformat(),
        "service": "signalforge",
        "model_file": model_ok,
        "scaler_file": scaler_ok,
    }


@app.get("/version", response_model=VersionResponse)
def version():
    p = version_payload()  # may include: service, model_version, featureset, commit, build_time
    # Derive service_version robustly
    service_version = p.get("service_version")
    if not service_version:
        svc = p.get("service")
        if isinstance(svc, str) and ":" in svc:
            # e.g., "signalforge:0.1.0" -> "0.1.0"
            service_version = svc.split(":", 1)[1]
        else:
            service_version = "unknown"

    return VersionResponse(
        service_version=service_version,
        model_version=p.get("model_version"),
        featureset=p.get("featureset"),
    )


@app.get("/metrics")
def metrics():
    return metrics_endpoint()
