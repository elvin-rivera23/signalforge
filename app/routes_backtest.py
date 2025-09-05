# app/routes_backtest.py
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()
_SCORE_PATH = "/api/v1/score"


class BacktestRequest(BaseModel):
    # Core data controls
    symbol: str = Field(default="AAPL", description="Ticker symbol")
    interval: str = Field(default="5m", description="Bar interval, e.g. 1m/5m/1h")
    threshold: float = Field(default=0.20, ge=0.0, le=1.0)

    # Replay controls
    window: int = Field(default=300, ge=1, description="Initial lookback length for the first step")
    steps: int = Field(default=50, ge=1, description="How many prediction steps to replay")
    step_size: int = Field(
        default=1, ge=1, description="How many rows to advance the window each step"
    )

    # Synthetic knobs (optional)
    synthetic: bool = Field(
        default=False, description="Use synthetic series instead of live/historical"
    )
    synthetic_mode: str = Field(default="flat", description="up/down/flat")

    # Future: start/end timestamps for real historical ranges (not used in this first pass)


@router.post("/api/v1/backtest")
async def backtest(request: Request, body: BacktestRequest):
    """
    Replay the inference core by repeatedly calling /api/v1/score with increasing limits.
    This avoids duplicating ML logic and produces a deterministic sequence of predictions.

    Strategy:
      - For i in [0..steps-1]:
          limit_i = window + i*step_size
          POST /score with that limit
          collect the response['last'] (time/proba/pred) and a snapshot count
      - Return a JSON report containing all steps and simple counts.
    """
    port = request.url.port or 8010
    base_url = f"http://127.0.0.1:{port}"

    started_at = datetime.now(UTC).isoformat()

    records: list[dict[str, Any]] = []
    last_artifacts: dict[str, Any] | None = None

    timeout = httpx.Timeout(20.0, read=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(body.steps):
            if await request.is_disconnected():
                break

            limit_i = body.window + i * body.step_size

            payload = {
                "symbol": body.symbol,
                "interval": body.interval,
                "limit": limit_i,
                "threshold": body.threshold,
                "synthetic": body.synthetic,
                "synthetic_mode": body.synthetic_mode,
            }

            try:
                resp = await client.post(f"{base_url}{_SCORE_PATH}", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                data = {"error": str(e)}

            # Extract the "last" prediction snapshot if present
            last = data.get("last", None)
            counts = data.get("counts", {})
            rec = {
                "step": i,
                "limit": limit_i,
                "last": last,  # {"time", "proba", "pred"} if provided
                "counts": counts,  # {"pred_0": x, "pred_1": y} if provided
                "error": data.get("error"),
            }
            records.append(rec)

            # Capture artifacts/model metadata if present
            last_artifacts = data.get("artifact_hashes", last_artifacts)

            # Yield the event loop briefly so we don't monopolize it
            await asyncio.sleep(0)

    # Summarize counts across steps (prediction-of-last only)
    pred_0 = 0
    pred_1 = 0
    for r in records:
        last = r.get("last")
        if isinstance(last, dict):
            p = last.get("pred")
            if p == 0:
                pred_0 += 1
            elif p == 1:
                pred_1 += 1

    ended_at = datetime.now(UTC).isoformat()

    report: dict[str, Any] = {
        "params": body.model_dump(),
        "runtime": {"started_at": started_at, "ended_at": ended_at},
        "n_steps": len(records),
        "summary": {
            "last_pred_0": pred_0,
            "last_pred_1": pred_1,
        },
        "artifact_hashes": last_artifacts,
        "records": records,  # full timeline
    }
    return report
