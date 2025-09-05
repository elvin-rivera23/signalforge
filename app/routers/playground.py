# app/routers/playground.py
import json
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")
router = APIRouter(tags=["playground"])


@router.get("/playground", response_class=HTMLResponse)
def playground(
    request: Request,
    symbol: str = Query("AAPL"),
    interval: str = Query("5m"),
    sma_fast: int = Query(20),
    sma_slow: int = Query(50),
    synthetic: int = Query(0),
    synthetic_mode: str = Query("up"),
):
    """
    Simple HTML playground that calls our own /signal/ endpoint internally.
    Uses stdlib urllib to avoid extra dependencies.
    NOTE: Inside the container, the app listens on port 8000.
    """
    qs = {
        "symbol": symbol,
        "interval": interval,
        "sma_fast": str(sma_fast),
        "sma_slow": str(sma_slow),
        "limit": "200",
        "synthetic": str(synthetic),
        "synthetic_mode": synthetic_mode,
    }
    url = f"http://127.0.0.1:8000/signal/?{urlencode(qs)}"
    try:
        with urlopen(url, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"signal_fetch_error: {e}")

    return templates.TemplateResponse(
        "playground.html",
        {
            "request": request,
            "symbol": symbol.upper(),
            "interval": interval,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "synthetic": synthetic,
            "synthetic_mode": synthetic_mode,
            "data": data,
        },
    )
