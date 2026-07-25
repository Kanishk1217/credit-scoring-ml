# Credit Scoring API — container image
FROM python:3.12-slim

# OpenMP: keep xgboost + torch from clashing in one process
ENV OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# keep caches inside /app (writable on Hugging Face Spaces)
ENV HOME=/app UV_CACHE_DIR=/app/.uv-cache

# install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# app code + model artifacts
COPY api ./api
COPY src ./src
COPY models ./models

EXPOSE 8000
# run uvicorn straight from the built venv (no runtime `uv` work); fixed port for Hugging Face
CMD [".venv/bin/uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
