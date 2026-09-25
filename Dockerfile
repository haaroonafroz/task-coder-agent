# syntax=docker/dockerfile:1
# App-only image. Does not start llama.cpp / Ollama.
# Attach host folders via compose bind mounts; use native sandbox (no bwrap).

FROM node:20-bookworm AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TASK_CODER_HOME=/data \
    MISSIONS_CONTAINER=true \
    SANDBOX_EXECUTOR=native \
    MISSIONS_QDRANT_MODE=embedded \
    MISSIONS_EMBEDDINGS=auto \
    MISSIONS_API_TELEMETRY=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY --from=frontend /src/frontend/dist ./frontend/dist

RUN pip install --no-cache-dir -e .

VOLUME ["/data"]
EXPOSE 8088
CMD ["missions", "serve", "--host", "0.0.0.0", "--port", "8088"]
