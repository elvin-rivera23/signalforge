# Minimal version payload for Milestone 1 (Part A)
from datetime import UTC, datetime

SERVICE_NAME = "signalforge"
SERVICE_VERSION = "0.1.0"  # semantic version of the service
MODEL_VERSION = "unloaded"  # filled in later milestones
FEATURESET = "none"  # filled in later milestones
COMMIT = "local"  # can be overridden by env/CI later
BUILD_TIME = datetime.now(tz=UTC).isoformat()


def version_payload() -> dict:
    return {
        "service": f"{SERVICE_NAME}:{SERVICE_VERSION}",
        "model_version": MODEL_VERSION,
        "featureset": FEATURESET,
        "commit": COMMIT,
        "build_time": BUILD_TIME,
    }
