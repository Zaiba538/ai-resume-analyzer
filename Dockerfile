# Production image for the Resume Intelligence Platform (FastAPI + LangGraph + Groq).
# One image for standalone Docker and Hugging Face Spaces (Docker SDK).
# See docs/deployment.md for configuration, secrets and limitations.
#
# Build:  docker build -t resume-intelligence .
# Run:    docker run --rm -p 7860:7860 -e GROQ_API_KEY resume-intelligence
#         (GROQ_API_KEY is passed from your shell at run time; it is never part of the image.)

# The hashed lock (requirements.txt) is resolved only for Linux x86_64 on CPython
# 3.12, so the image is always built for linux/amd64, also on ARM build machines.
# The base image is pinned by digest (multi-platform index of python:3.12-slim-bookworm,
# Python 3.12.16, verified on Docker Hub 2026-11-10); update it deliberately.
FROM --platform=linux/amd64 python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# Python: unbuffered JSON logs on stdout, no .pyc writes at run time (the code is
# read-only for the runtime user). Hugging Face cache in the runtime user's home.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# Non-root runtime user with UID/GID 1000 (the UID Hugging Face Spaces runs as).
RUN groupadd --gid 1000 user \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/user \
        --shell /usr/sbin/nologin user

WORKDIR /app

# Runtime dependencies from the hashed lock only: every package pinned and
# hash-checked, nothing resolved outside the lock, and wheels only (a package
# without a wheel fails the build instead of compiling from source).
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --require-hashes --no-deps --only-binary=:all: -r requirements.txt \
    && rm requirements.txt

# Settings module only (and its single local import), so that the model download
# below reads the model name and pinned revision from the application settings
# and is re-run only when these files change, not on every code change.
COPY backend/__init__.py backend/
COPY backend/core/__init__.py backend/core/config.py backend/core/rate_limit.py backend/core/

# Download the embedding model at build time, as the runtime user, into HF_HOME.
# Same arguments as backend.scoring.embeddings.load_sentence_transformer; only
# the files the model needs are fetched. No secret is used or needed.
USER 1000:1000
RUN python -c "from backend.core.config import Settings; \
from sentence_transformers import SentenceTransformer; \
s = Settings(_env_file=None); \
SentenceTransformer(s.embedding_model_name, revision=s.embedding_model_revision, device='cpu', trust_remote_code=False); \
print('embedding model cached:', s.embedding_model_name, s.embedding_model_revision)"

# Application code (root-owned, read-only for the runtime user). Only the FastAPI
# backend and the static frontend; the legacy Streamlit files, tests and docs are
# not copied.
COPY backend/ backend/
COPY frontend/ frontend/

# Verify that the application's own loader finds the cached model without network
# access (fails the build if the cache and the settings ever disagree).
RUN HF_HUB_OFFLINE=1 python -c "from backend.core.config import Settings; \
from backend.scoring.embeddings import load_sentence_transformer; \
embedder = load_sentence_transformer(Settings(_env_file=None)); \
assert embedder.encode(['offline check']).shape[0] == 1; \
print('embedding model loads offline')"

# Run time: never contact the Hugging Face Hub; the model is in the image.
# Overriding EMBEDDING_MODEL_* at run time therefore needs HF_HUB_OFFLINE=0 and a
# writable cache, otherwise /ready reports the model as not loaded.
ENV HF_HUB_OFFLINE=1

EXPOSE 7860

# Liveness only: /health touches no dependency. /ready is not used here because a
# missing GROQ_API_KEY makes it 503 permanently, which a restart cannot fix.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"]

# One process, one worker: reports, rate limits and the analysis concurrency gate
# are held in memory. Do not add --workers or set WEB_CONCURRENCY.
CMD ["uvicorn", "--factory", "backend.main:create_app", "--no-access-log", "--host", "0.0.0.0", "--port", "7860"]
