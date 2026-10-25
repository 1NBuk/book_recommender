# syntax=docker/dockerfile:1.7

# ---------- builder: resolve and install dependencies into a venv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# ---------- runtime: only the venv, the API code and the serving artifacts ----------
FROM python:3.12-slim AS runtime

ARG VERSION=dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="book-recommender" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BOOKREC_ARTIFACTS_DIR=/app/artifacts

RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/api ./src/api
COPY artifacts ./artifacts

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).status == 200 else 1)"

# One worker per container on purpose: Prometheus metrics are per-process.
# Scale horizontally with replicas (docker compose / HPA), not with --workers.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
