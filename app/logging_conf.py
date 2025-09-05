# app/logging_conf.py
from __future__ import annotations

import json
import logging
import os
from logging.config import dictConfig
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Include standard extras if present
        for extra_key in ("module", "funcName"):
            val = getattr(record, extra_key, None)
            if val:
                payload[extra_key] = val
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure JSON logging for app + uvicorn, suppress duplicate access logs."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    dict_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
            },
            "plain": {
                "format": "%(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
            "console_plain": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "stream": "ext://sys.stdout",
            },
        },
        # Root logger uses JSON
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
        # Make sure FastAPI/uvicorn loggers don’t print duplicate or plaintext lines
        "loggers": {
            # Uvicorn internals → JSON
            "uvicorn": {"level": log_level, "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": log_level, "handlers": ["console"], "propagate": False},
            # Suppress default uvicorn access logs (we emit our own request JSON in middleware)
            "uvicorn.access": {"level": "WARNING", "handlers": ["console"], "propagate": False},
            # Starlette/FastAPI
            "fastapi": {"level": log_level, "handlers": ["console"], "propagate": False},
            "starlette": {"level": log_level, "handlers": ["console"], "propagate": False},
            # Our app namespace (you can log with logging.getLogger("app.something"))
            "app": {"level": log_level, "handlers": ["console"], "propagate": False},
            # Our per-request JSON logs from the timing middleware use logger name "request"
            "request": {"level": log_level, "handlers": ["console"], "propagate": False},
        },
    }

    dictConfig(dict_config)
