"""Liveness and readiness. Both are quick flag checks; neither triggers loading."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.inference import READY

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: 200 as soon as the process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """200 only after the model is loaded, verified and warmed up; otherwise 503."""
    service = request.app.state.service
    if service.status == READY:
        return JSONResponse(
            {"status": "ready", "model_version": service.version, "categories": [service.category]}
        )
    return JSONResponse({"status": service.status}, status_code=503, headers={"Retry-After": "10"})
