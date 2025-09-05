# signalforge

AI-powered trading signal service with FastAPI, Docker, and Raspberry Pi deployment.

---

## Features

- REST API with health and version endpoints  
- `/api/v1/score` → model scoring (FastAPI)  
- `/api/v1/backtest` → simple replay/backtest  
- `/api/v1/stream` → Server-Sent Events (demo stream)  
- Prometheus metrics at `/metrics`  
- Linting and formatting with Ruff + Black  
- Testing with Pytest  
- Optional pre-commit hook for style and checks  

---

## Quick start (dev)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API locally (auto-reload in dev mode):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

The API will be available at: [http://localhost:8010](http://localhost:8010)

---

## Endpoints

- **Health check** → `GET /health`  
- **Version info** → `GET /version`  
- **Prometheus metrics** → `GET /metrics`  
- **Trading signals** → `POST /api/v1/score`  
- **Backtesting** → `POST /api/v1/backtest`  
- **Streaming** → `GET /api/v1/stream` (Server-Sent Events)  

---

## Docker (local)

Build and run the container manually:

```bash
docker build -t signalforge:dev .
docker run --rm -p 8010:8000 signalforge:dev
```

Or with Docker Compose (dev profile):

```bash
docker compose --profile dev up -d
docker compose --profile dev down
```

---

## Tests & Lint

Run unit tests:

```bash
pytest -q
```

Lint with Ruff:

```bash
ruff check .
```

Check formatting with Black:

```bash
black --check .
```

---

## Observability

Metrics are exposed at `GET /metrics` (Prometheus exposition format).  

Example metrics:  
- `sf_http_requests_total`  
- `sf_http_request_duration_seconds`  

---

## Configuration

Optional environment variables (see `.env.example`):  
- `LOG_LEVEL` → default `INFO`  
- `SF_PROVIDER` → data source (default `YAHOO_CHART`)  

---

## Project layout

```
app/
  main.py                # FastAPI app
  observability.py       # Prometheus metrics & middleware
  routers/               # API routes
  ml/                    # Model utils (train/infer)
  signals/               # Feature builders
  ...

data/                    # Model artifacts (meta, eval report)

Dockerfile
docker-compose.yml
requirements.txt
pyproject.toml           # Ruff / Black / Pytest config
tests/                   # Unit tests
```

---

## 🚀 Release & Raspberry Pi Deploy (v0.2.0)

This service is packaged for Raspberry Pi via Docker Compose profiles.

### Prereqs
- Docker + Docker Compose installed on the Pi
- Repo cloned on the Pi: `~/signalforge`
- Model artifacts present in `./data`:
  - `data/model.pkl`, `data/scaler.pkl`, `data/model_meta.json`, `data/eval_report.json`

### Quick Start (Pi)
```bash
cd ~/signalforge

# (Optional) review env knobs (safe template)
cat .env.example

# Build & run the Pi profile
docker compose --profile pi build api-pi
docker compose --profile pi up -d api-pi

# Wait for healthcheck to pass
docker ps --filter "name=api-pi"
```

You should see: `Up ... (healthy)` and port mapping `0.0.0.0:8010->8000/tcp`.

### Verify
```bash
# Health & version
curl -s http://localhost:8010/health | jq
curl -s http://localhost:8010/version | jq

# Inference (synthetic path to avoid external rate limits)
curl -s -X POST "http://localhost:8010/api/v1/score"   -H "Content-Type: application/json"   -d '{"symbol":"AAPL","interval":"5m","limit":180,"synthetic":1,"synthetic_mode":"up"}' | jq
```

Expected fields: `n_rows_scored`, `last.proba`, `last.pred`, and `model_version`.

### Ports
- Host **8010** → container **8000** (FastAPI/Uvicorn)

### Logs & Troubleshooting
```bash
# Tail logs
docker logs -f signalforge-api-pi

# Port in use?
docker ps --format "table {{.Names}}	{{.Ports}}" | grep 8010
# If a dev container holds 8010, stop it:
docker stop signalforge-api-dev
```

### Notes
- `/version` reports `service` (e.g., `signalforge-api:0.2.0`) and the loaded `model_version`.
- Live data uses Yahoo and may rate-limit (HTTP 429). The API auto-falls back to **synthetic** when `synthetic=1` or on 429/network errors.
- Env knobs are documented in `.env.example`. Don’t commit a real `.env`.

---

## Contributing

We use pre-commit hooks to enforce style and linting:

```bash
pip install pre-commit
pre-commit install
```

This runs Ruff, Black, and other checks automatically on commit.  

Conventional commit prefixes are encouraged (e.g., `feat:`, `fix:`, `chore:`).  

---

## License

MIT License © 2025 Elvin Rivera  

---

## Credits

Built with:  
- FastAPI  
- Uvicorn  
- Pandas  
- NumPy  
- Scikit-learn  
- Prometheus client  

---

## Roadmap

- Model artifact registry & versioning  
- More robust data providers  
- CI workflow (lint/test/build) on GitHub Actions  
