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

# --- Versioning ---
from app.version import service_version_payload, set_model_version

# --- App ---
setup_logging()
app = FastAPI(title="SignalForge", version="0.2.0")  # bumped for release

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
    service: str
    model_version: str | None = None


# --- Startup hook ---
@app.on_event("startup")
async def on_startup():
    # load model version from data/model_meta.json
    set_model_version()


# --- Utility endpoints ---
@app.get("/health", response_model=HealthResponse)
def health():
    model_ok = Path("data/model.pkl").exists()
    scaler_ok = Path("data/scaler.pkl").exists()
    status = "ok" if (model_ok and scaler_ok) else "degraded"
    return {
        "status": status,
        "as_of": datetime.now(tz=UTC).isoformat(),
        "model_file": model_ok,
        "scaler_file": scaler_ok,
    }


@app.get("/metrics")
def metrics():
    return metrics_endpoint()


@app.get("/version", response_model=VersionResponse)
async def version():
    return service_version_payload()
