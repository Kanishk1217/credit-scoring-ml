# Credit Scoring API — slim container (no PyTorch; LSTM runs in NumPy). Fits a free 512MB host.
FROM python:3.12-slim

ENV OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app

WORKDIR /app

# install only the runtime deps (cached layer)
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# app code + model artifacts (no torch model needed; NumPy weights .npz is used)
COPY api ./api
COPY model_registry.json ./
COPY models/hybrid_xgb.joblib models/hybrid_config.json models/hybrid_fusion.npz ./models/
COPY models_real/hybrid_xgb.joblib models_real/hybrid_config.json models_real/hybrid_fusion.npz ./models_real/
COPY models_real_rich/hybrid_xgb.joblib models_real_rich/hybrid_config.json models_real_rich/hybrid_fusion.npz ./models_real_rich/

EXPOSE 8000
# bind to the host's $PORT if provided (Render), else 8000 (local / fixed-port hosts)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
