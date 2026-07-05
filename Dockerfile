# syntax=docker/dockerfile:1

# ---- builder: install all dependencies into an isolated venv ------------------
# The venv is copied wholesale into the runtime stage, so the final image carries
# only installed packages — none of pip's build isolation residue or caches.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install dependencies from pyproject alone first, so this layer is cached across
# app-code changes. The local package resolves the full dependency set; the app
# source itself is copied into the runtime image and imported from /app (this
# builds a metadata-only wheel — the same behavior as the previous single stage).
COPY pyproject.toml ./
RUN pip install .

# Drop bytecode caches the wheels shipped with; regenerated lazily at runtime.
RUN find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# ---- runtime: slim image with just the venv + application source --------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Persist model2vec / Hugging Face model downloads on the data volume so they
    # survive redeploys instead of re-downloading on every container start. Mount
    # a volume at /app/data to persist this alongside the memory store and the
    # voice / model pointers (all of which the app writes under ./data).
    HF_HOME=/app/data/hf-cache

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY frontend ./frontend

EXPOSE 8000

CMD ["python", "-m", "app.server"]
