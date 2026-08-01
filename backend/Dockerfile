# CodeLens_AI backend — production image
#
# Build:  docker build -t codelens-backend ./backend
# Run:    docker run --env-file .env -p 8000:8000 codelens-backend
#
# Notes:
#   * Multi-stage: wheels built in a fat builder layer, runtime stays slim.
#   * Runs as a non-root user.
#   * HEALTHCHECK hits the dependency-probing endpoint so orchestrators see
#     real readiness, not just process liveness.
#   * STARTUP_TESTS_ENABLED defaults off in production (see tests/runner.py).

FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Build deps for psycopg2 / bcrypt / numpy wheels that lack binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY app ./app

USER appuser

EXPOSE 8000

# Readiness: dependency-probing health endpoint (H-3)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health/detailed || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
