# app/observability.py
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ---- Prometheus metrics (low-cardinality labels) ----
REQUEST_COUNT = Counter(
    "sf_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "sf_http_request_duration_seconds",
    "HTTP request latency (seconds)",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0),
)


def metrics_endpoint():
    """Return Prometheus exposition format."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---- Per-request timing + JSON request log ----
async def timing_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    path = request.url.path
    status = str(response.status_code)

    REQUEST_COUNT.labels(method=request.method, path=path, status=status).inc()
    REQUEST_LATENCY.observe(elapsed)

    logging.getLogger("request").info(
        json.dumps(
            {
                "method": request.method,
                "path": path,
                "status": status,
                "duration_s": round(elapsed, 6),
                "client": request.client.host if request.client else None,
            }
        )
    )
    return response
