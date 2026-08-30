# syntax=docker/dockerfile:1

# Stage 1: Dependency builder & model pre-caching
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into /root/.local
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-cache SentenceTransformer weights into dedicated directory
ENV HF_HOME=/app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy installed wheels and binaries
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy pre-cached transformer weights
ENV HF_HOME=/app/model_cache
COPY --from=builder /app/model_cache /app/model_cache

# Copy runtime data and application code
COPY songs_with_mood.parquet .
COPY embeddings.npy .
COPY *.py .

ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

ENV PORT=10000
EXPOSE $PORT

# Single worker to respect Render's memory constraints
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker api:app -b 0.0.0.0:${PORT:-10000} --workers 1 --timeout 120"]