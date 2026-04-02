# ─── Stage 1: Builder ────────────────────────────────────────────────────────
# Full Python image installs all heavy ML dependencies via uv.
FROM python:3.11 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifest first — cached as long as deps don't change
COPY pyproject.toml uv.lock ./

# Install all production dependencies (skip project install for cache efficiency)
RUN uv sync --no-dev --frozen --no-install-project

# Copy source and install the project itself
COPY src/ ./src/
RUN uv sync --no-dev --frozen


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
# Slim image — only the installed venv and project source.
FROM python:3.11-slim AS runtime

# System libraries required by audio processing packages:
#   libsndfile1  — soundfile / librosa
#   libgomp1     — PyTorch OpenMP threading
#   ffmpeg       — torchaudio backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the full virtualenv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy project source and Hydra configs
COPY src/ ./src/
COPY configs/ ./configs/

# Activate the venv by prepending it to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    HF_HUB_CACHE="/app/checkpoints/hf_cache"

CMD ["python", "-m", "seed_vc.train.train", "--help"]
