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
COPY models/hybrid_xgb.joblib models/hybrid_config.json models/hybrid_fusion.npz ./models/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
