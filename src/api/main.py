from __future__ import annotations

import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query, Request, Response, Security
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import __version__, metrics
from src.api.config import Settings
from src.api.schemas import (
    Book,
    HealthResponse,
    ModelInfo,
    PopularResponse,
    RecommendationResponse,
)
from src.api.store import ArtifactStore

log = logging.getLogger("bookrec.api")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# user ids end up in logs: restrict charset and length (no log injection)
USER_ID = Path(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:@\-]+$")
ITEM_ID = Path(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:@\-]+$")


def require_api_key(request: Request, key: Optional[str] = Security(_api_key_header)) -> None:
    expected = request.app.state.settings.api_key
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_store(request: Request) -> ArtifactStore:
    store = request.app.state.store
    if store is None:
        raise HTTPException(status_code=503, detail="Artifacts are not loaded")
    return store


def create_app(settings: Optional[Settings] = None, store: Optional[ArtifactStore] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or ArtifactStore.load(settings.artifacts_dir)
        s = app.state.store
        metrics.ARTIFACT_USERS.set(s.n_users)
        metrics.ARTIFACT_ITEMS.set(s.n_items)
        metrics.ARTIFACTS_LOADED_AT.set(time.time())
        created = s.info.get("created_at_unix")
        if created:
            metrics.MODEL_CREATED_AT.set(float(created))
        yield

    app = FastAPI(
        title="Book Recommender API",
        version=__version__,
        description="Serves precomputed hybrid recommendations with a popularity fallback.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = None

    # ---- observability middleware ----------------------------------------
    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        status = 500
        metrics.INFLIGHT.inc()
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            metrics.INFLIGHT.dec()
            route = request.scope.get("route")
            path = route.path if route is not None else "unmatched"  # low-cardinality label
            if path != "/metrics":
                metrics.REQUESTS.labels(request.method, path, str(status)).inc()
                metrics.LATENCY.labels(request.method, path).observe(elapsed)
                log.info(
                    "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                    request_id,
                    request.method,
                    request.url.path,
                    status,
                    elapsed * 1000,
                )

    # ---- infra endpoints ---------------------------------------------------
    @app.get("/health/live", response_model=HealthResponse, tags=["infra"])
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", response_model=HealthResponse, tags=["infra"])
    def ready(request: Request):
        if request.app.state.store is None:
            raise HTTPException(status_code=503, detail="Artifacts are not loaded")
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---- business endpoints ------------------------------------------------
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)], tags=["v1"])

    @v1.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
    def recommendations(
        request: Request,
        user_id: str = USER_ID,
        n: int = Query(None, ge=1, description="Number of books (default/max set by server)"),
        store: ArtifactStore = Depends(get_store),
    ):
        cfg = request.app.state.settings
        n = n or cfg.default_n
        if n > cfg.max_n:
            raise HTTPException(status_code=422, detail=f"n must be <= {cfg.max_n}")
        source, items = store.recommend(user_id, n)
        metrics.RECOMMENDATIONS.labels(source).inc()
        return {
            "user_id": user_id,
            "source": source,
            "model": store.info.get("model", "unknown"),
            "count": len(items),
            "items": items,
        }

    @v1.get("/popular", response_model=PopularResponse)
    def popular(
        request: Request,
        n: int = Query(None, ge=1),
        store: ArtifactStore = Depends(get_store),
    ):
        cfg = request.app.state.settings
        n = n or cfg.default_n
        if n > cfg.max_n:
            raise HTTPException(status_code=422, detail=f"n must be <= {cfg.max_n}")
        items = store.popular(n)
        return {"count": len(items), "items": items}

    @v1.get("/books/{item_id}", response_model=Book)
    def book(item_id: str = ITEM_ID, store: ArtifactStore = Depends(get_store)):
        found = store.book(item_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Book not found")
        return found

    @v1.get("/model", response_model=ModelInfo)
    def model_info(store: ArtifactStore = Depends(get_store)):
        return {
            "model": store.info.get("model", "unknown"),
            "created_at": store.info.get("created_at"),
            "n_users": store.n_users,
            "n_items": store.n_items,
            "n_popular": store.n_popular,
            "metrics": store.info.get("metrics", {}),
        }

    app.include_router(v1)
    return app


app = create_app()
