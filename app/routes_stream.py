# app/routes_stream.py
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

# Keep this path aligned with your existing score route
_SCORE_PATH = "/api/v1/score"


def _bool_param(v: str | int | bool | None, default: bool = False) -> bool:
    """
    Normalize truthy params from query strings (e.g., "1", "true", 1, True).
    Returns default if None.
    """
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


@router.get("/api/v1/stream")
async def stream_inference(
    request: Request,
    symbol: str = "AAPL",
    interval: str = "5m",
    limit: int = 300,
    threshold: float = 0.20,
    synthetic: str | int | bool = 0,
    synthetic_mode: str = "flat",
    refresh_sec: float = 2.0,
) -> StreamingResponse:
    """
    Server-Sent Events stream that repeatedly calls your /api/v1/score endpoint
    and emits each result as an SSE message: `data: {...}\\n\\n`.

    Why this way?
    - Reuses your validated scoring path (no duplicate ML logic).
    - Simple to test with curl and browsers.
    """

    # Use loopback to avoid hairpin/NAT issues even if app binds 0.0.0.0
    port = request.url.port or 8010
    base_url = f"http://127.0.0.1:{port}"

    # Build payload for /score; add fields if your /score expects more.
    score_payload: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "threshold": threshold,
        "synthetic": _bool_param(synthetic, default=False),
        "synthetic_mode": synthetic_mode,
    }

    async def event_gen():
        # One shared client to reduce overhead
        timeout = httpx.Timeout(10.0, read=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                # Stop streaming if client disconnects
                if await request.is_disconnected():
                    break

                try:
                    resp = await client.post(f"{base_url}{_SCORE_PATH}", json=score_payload)
                    resp.raise_for_status()
                    data = resp.json()  # should match your /score JSON
                except Exception as e:
                    data = {"error": str(e)}

                # Emit SSE frame (each frame ends with a blank line)
                yield f"data: {json.dumps(data)}\n\n"

                # Pace the stream
                await asyncio.sleep(refresh_sec)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
