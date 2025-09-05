# syntax=docker/dockerfile:1.7
# --- Base image (runtime) ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=1 \
    LOG_LEVEL=INFO

# Minimal OS deps (curl for healthcheck, ca-certificates for HTTPS)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -u 1001 -m appuser

WORKDIR /app

# Install Python deps first for better layer cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only necessary source
COPY app /app/app
COPY data /app/data

# Tighten perms: appuser owns code + data
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Default command: uvicorn (port 8000 inside the container)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
