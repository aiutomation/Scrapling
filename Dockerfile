FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Railway injects PORT at runtime; default to 8000 for local dev
ENV HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Copy dependency file first for better layer caching
COPY pyproject.toml ./

# Install dependencies only
RUN --mount=type=cache,id=scrapling-uv-cache,target=/root/.cache/uv \
    uv sync --no-install-project --all-extras --compile-bytecode

# Copy source code
COPY . .

# Install browsers and project in one optimized layer
RUN --mount=type=cache,id=scrapling-uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=scrapling-apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=scrapling-apt-lib,target=/var/lib/apt \
    apt-get update && \
    uv run playwright install-deps chromium && \
    uv run playwright install chromium && \
    uv sync --all-extras --compile-bytecode && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE ${PORT}

# Set entrypoint to run scrapling
ENTRYPOINT ["uv", "run", "scrapling"]

# Default: start the REST API server on $HOST:$PORT
CMD ["api"]