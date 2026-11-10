"""The application factory (PLAN.md Phase 5 step 2).

    uv run uvicorn --factory app.main:create_app --port 8000 --no-access-log

(run from ``backend/`` or with ``--app-dir backend``; settings come from the environment, see
``.env.example``). Interactive documentation is served at ``/docs``.

The factory takes the settings and, for tests, a model loader and an inspector; by default it uses
``defect_core.artifacts.load_artifact`` and ``defect_core.infer.inspect_image``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from defect_core.artifacts import LoadedArtifact, load_artifact
from defect_core.infer import InspectionResult, inspect_image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from slowapi import Limiter

from app.core.clientip import key_function
from app.core.errors import install_error_handlers
from app.core.logging import configure_logging, get_logger
from app.core.settings import Settings, get_settings
from app.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.routers import catalog, health, inspections
from app.services.inference import InferenceService

API_PREFIX = "/api/v1"
log = get_logger("app.main")


def create_app(
    settings: Settings | None = None,
    *,
    loader: Callable[[Path], LoadedArtifact] = load_artifact,
    inspector: Callable[..., InspectionResult] = inspect_image,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels  # Pillow's own decompression-bomb guard
    service = InferenceService(settings, loader, inspector)
    limiter = Limiter(key_func=key_function(settings.trusted_proxy_hops))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # load in the background so /health and /ready (503) answer while the model loads
        task = asyncio.create_task(service.load())
        app.state.load_task = task
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(
        title="DefectLens API",
        version="0.1.0",
        description="Educational anomaly-detection demo on MVTec AD. Results are demo "
        "decisions, not quality certifications; scores are not probabilities.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service
    app.state.limiter = limiter
    install_error_handlers(app)

    # added innermost first: the request ID wraps everything, then CORS, then the size guard
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
        allow_credentials=False,
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(catalog.router, prefix=API_PREFIX)
    app.include_router(inspections.build_router(settings, limiter), prefix=API_PREFIX)
    log.info("app_created", env=settings.app_env, rate_limit=settings.rate_limit)
    return app
