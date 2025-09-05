from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# --- Signals (future use for /predict) ---
class SignalLabel(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    HOLD = "HOLD"


# --- Health payload ---
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok"] = "ok"
    as_of: str
    service: str = "signalforge"


# --- Version payload ---
class VersionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    service: str  # "signalforge:0.1.0"
    model_version: str  # "unloaded" in M1
    featureset: str  # "none" in M1
    commit: str  # "local" in M1
    build_time: str  # UTC ISO


# --- Error taxonomy (we’ll wire real cases later) ---
class ErrorCode(str, Enum):
    INVALID_SYMBOL = "INVALID_SYMBOL"
    STALE_DATA = "STALE_DATA"
    RATE_LIMIT = "RATE_LIMIT"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    hint: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# --- Predict payload (placeholder for later milestones) ---
class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str
    interval: str
    lookback: int
    signal: SignalLabel
    confidence: float = Field(ge=0.0, le=1.0)
    as_of: str
    model_version: str
    featureset: str
    request_id: str
