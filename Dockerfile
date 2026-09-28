# syntax=docker/dockerfile:1

# --- Build/runtime image -----------------------------------------------
# One stage is enough here: every dependency is a pure-Python wheel (no
# compiled extensions beyond scikit-learn's own manylinux wheels), so
# there's nothing a multi-stage build would usefully throw away.
FROM python:3.12-slim

# Prevents Python from writing .pyc files / buffering stdout — makes
# container logs show up immediately instead of being buffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first, separately from the app code, so Docker's
# layer cache can skip this (slow) step on every rebuild that only changes
# Python source.
COPY requirements.txt requirements-embeddings.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional: build with `--build-arg INSTALL_EMBEDDINGS=true` to bake in the
# semantic-matching upgrade (sentence-transformers + torch — adds several
# hundred MB to the image and downloads ~90MB of model weights during this
# build step, so it needs network access at build time). Left off by
# default so the base image stays small and builds without requiring
# network access beyond PyPI — the service runs correctly either way (see
# app/embeddings.py's module docstring for the automatic fallback).
ARG INSTALL_EMBEDDINGS=false
RUN if [ "$INSTALL_EMBEDDINGS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-embeddings.txt && \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

# Optional: build with `--build-arg INSTALL_LLM_SDKS=true` to bake in the
# native Anthropic/Gemini SDKs (only relevant if you also set
# LLM_PROVIDER=anthropic|gemini at runtime — see app/llm_client.py).
ARG INSTALL_LLM_SDKS=false
RUN if [ "$INSTALL_LLM_SDKS" = "true" ]; then pip install --no-cache-dir -r requirements-llm.txt; fi

# Now copy the actual application code.
COPY app ./app
COPY samples ./samples

# Not strictly required (the port is also declared to the platform via
# `docker run -p`), but documents the service's contract for anyone
# reading this file.
EXPOSE 8000

# Basic container-level health check, hitting the liveness endpoint
# defined in app/main.py.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
