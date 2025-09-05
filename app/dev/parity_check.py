# app/dev/parity_check.py
# Simple parity check: compares CLI vs API outputs for the same inputs.

import json
import math
import subprocess
import sys
import time

import httpx


def run_cli(
    symbol: str, interval: str, limit: int, threshold: float, synthetic: int, synthetic_mode: str
):
    cmd = [
        sys.executable,
        "-m",
        "app.ml.predict",
        "--symbol",
        symbol,
        "--interval",
        interval,
        "--limit",
        str(limit),
        "--threshold",
        str(threshold),
        "--synthetic",
        str(synthetic),
        "--synthetic_mode",
        synthetic_mode,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def call_api(
    base: str,
    symbol: str,
    interval: str,
    limit: int,
    threshold: float,
    synthetic: int,
    synthetic_mode: str,
):
    payload = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "threshold": threshold,
        "synthetic": synthetic,
        "synthetic_mode": synthetic_mode,
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{base}/api/v1/score", json=payload)
        r.raise_for_status()
        return r.json()


def approx(a: float, b: float, eps: float = 1e-9) -> bool:
    if any(math.isnan(x) or math.isinf(x) for x in (a, b)):
        return False
    return abs(a - b) <= eps


def main():
    base = "http://localhost:8010"
    symbol = "AAPL"
    interval = "5m"
    limit = 300
    threshold = 0.2
    synthetic = 1
    synthetic_mode = "down"

    # 1) CLI
    cli = run_cli(symbol, interval, limit, threshold, synthetic, synthetic_mode)

    # 2) API (retry a couple times in case server just started)
    last_err = None
    for _ in range(3):
        try:
            api = call_api(base, symbol, interval, limit, threshold, synthetic, synthetic_mode)
            break
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    else:
        raise SystemExit(f"API call failed: {last_err}")

    # 3) Compare key fields
    keys = ["symbol", "interval", "n_rows_scored", "threshold"]
    for k in keys:
        assert cli[k] == api[k], f"Mismatch on {k}: cli={cli[k]} api={api[k]}"

    # last.proba and last.pred should match approximately
    cli_p = float(cli["last"]["proba"])
    api_p = float(api["last"]["proba"])
    assert approx(cli_p, api_p, 1e-9), f"last.proba mismatch: cli={cli_p} api={api_p}"
    assert int(cli["last"]["pred"]) == int(api["last"]["pred"]), "last.pred mismatch"

    # Optional: artifact hashes must match
    assert cli["artifact_hashes"] == api["artifact_hashes"], "artifact_hashes mismatch"

    print("✅ Parity OK — CLI and API outputs match.")


if __name__ == "__main__":
    main()
