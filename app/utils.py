import time
import uuid
from contextlib import contextmanager
from datetime import UTC


def utc_now_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat()


@contextmanager
def timer_ms():
    start = time.perf_counter()
    yield lambda: int((time.perf_counter() - start) * 1000)


def new_request_id() -> str:
    return str(uuid.uuid4())
