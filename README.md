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
