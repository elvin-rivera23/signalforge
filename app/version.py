# app/version.py

import json
from pathlib import Path

SERVICE_NAME = "signalforge-api"
SERVICE_VERSION = "0.2.0"  # bump for first real Pi release
MODEL_VERSION = "unloaded"  # set at startup from model_meta.json


def set_model_version(meta_path: str = "data/model_meta.json") -> None:
    """
    Reads the model version from model_meta.json and sets MODEL_VERSION.
    Safe if the file is missing or malformed.
    """
    global MODEL_VERSION
    try:
        p = Path(meta_path)
        if p.exists():
            meta = json.loads(p.read_text())
            # Accept common keys like "model_version" or "version"
            MODEL_VERSION = str(meta.get("model_version") or meta.get("version") or MODEL_VERSION)
    except Exception:
        # Leave MODEL_VERSION as-is if anything goes wrong
        pass


def service_version_payload() -> dict:
    """Used by /version endpoints."""
    return {
        "service": f"{SERVICE_NAME}:{SERVICE_VERSION}",
        "model_version": MODEL_VERSION,
    }
