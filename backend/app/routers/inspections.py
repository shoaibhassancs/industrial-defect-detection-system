"""``POST /api/v1/inspections``: the stateless inspection (Release 1)."""

from __future__ import annotations

import base64
import io
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from PIL import Image
from slowapi import Limiter

from app.core.errors import FileTooLarge, ModelNotReady, UnsupportedCategory
from app.core.settings import Settings
from app.schemas import InspectionResponse, error_responses
from app.services.inference import READY, InferenceService
from app.services.validation import reencode_png, validate_upload

_CHUNK = 1024 * 1024


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _png_of(array) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def build_router(settings: Settings, limiter: Limiter) -> APIRouter:
    router = APIRouter(tags=["inspections"])

    @router.post(
        "/inspections",
        response_model=InspectionResponse,
        responses=error_responses(413, 415, 422, 429, 500, 503),
    )
    @limiter.limit(settings.rate_limit)
    async def create_inspection(
        request: Request,
        category: Annotated[str, Form(description="Category id, e.g. 'bottle'")],
        file: Annotated[UploadFile, File(description="A JPEG or PNG image")],
        sample_id: Annotated[str | None, Form(description="Reserved; ignored in Release 1")] = None,
        save: Annotated[bool | None, Form(description="Reserved; nothing is stored")] = None,
    ) -> dict:
        started = time.perf_counter()
        service: InferenceService = request.app.state.service
        if service.status != READY:
            raise ModelNotReady(
                "The model is not ready yet. Please try again shortly.",
                headers={"Retry-After": "10"},
            )
        if category != service.category:
            raise UnsupportedCategory(
                "That category is not supported.", {"supported": [service.category]}
            )

        # read at most one byte more than the limit, so a huge file is never read to the end
        data = bytearray()
        while chunk := await file.read(_CHUNK):
            data += chunk
            if len(data) > settings.max_upload_bytes:
                raise FileTooLarge(
                    f"Image must be {settings.max_upload_mb} MB or smaller.",
                    {"max_mb": settings.max_upload_mb},
                )
        validated = validate_upload(bytes(data), settings)

        result = await service.inspect(validated.image)
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "category": result.category,
            "model": {"name": result.model_name, "version": result.model_version},
            "prediction": result.verdict,
            "decision_rule": result.decision_rule,
            "anomaly_score": result.score,
            "threshold": result.threshold,
            "calibration_rank": {
                "score_higher_than": result.rank_m,
                "of_calibration_normals": result.rank_n,
            },
            "highlighted_area_fraction": result.highlighted_area_fraction,
            "region_bbox": list(result.region_bbox) if result.region_bbox else None,
            "images": {
                "original": {
                    "url": _data_url(reencode_png(validated.image)),
                    "width": validated.width,
                    "height": validated.height,
                },
                "heatmap": {"url": _data_url(_png_of(result.rendering.heatmap))},
                "overlay": {"url": _data_url(_png_of(result.rendering.overlay))},
            },
            "timing_ms": {"inference": round(result.inference_ms, 1), "total": round(total_ms, 1)},
            "persisted": False,
            "warnings": [],
            "limitation": result.limitation,
            "disclaimer": "Demo decision based only on this model and category. "
            "Not a quality certification.",
        }

    return router
