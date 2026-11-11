"""Response shapes (shown in the generated OpenAPI documentation at /docs)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelRef(BaseModel):
    name: str
    version: str


class CalibrationRank(BaseModel):
    score_higher_than: int
    of_calibration_normals: int


class ImageRef(BaseModel):
    url: str  # a data: URL in stateless mode
    width: int | None = None
    height: int | None = None


class Images(BaseModel):
    original: ImageRef
    heatmap: ImageRef
    overlay: ImageRef


class TimingMs(BaseModel):
    inference: float
    total: float


class InspectionResponse(BaseModel):
    id: str
    created_at: str
    category: str
    model: ModelRef
    prediction: str  # "normal" or "anomalous": a demo decision
    decision_rule: str
    anomaly_score: float  # an anomaly score, not a probability or a confidence
    threshold: float
    calibration_rank: CalibrationRank
    highlighted_area_fraction: float
    region_bbox: list[int] | None
    images: Images
    timing_ms: TimingMs
    persisted: bool
    warnings: list[str]
    limitation: str
    disclaimer: str


class CategoryInfo(BaseModel):
    """One supported category (``GET /api/v1/categories``)."""

    id: str
    display_name: str
    kind: str  # "object" or "texture"
    threshold: float
    alpha: float
    calib_n: int
    metrics_summary: dict[str, Any] | None  # sealed-test summary; see the evaluation document


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


def error_responses(*statuses: int) -> dict[int | str, dict]:
    """OpenAPI ``responses`` entries: each listed status is documented as the error envelope.

    Without this, FastAPI documents only FastAPI's own default 422 shape, which this API does not
    return. The generated frontend types are built from this description.
    """
    descriptions = {
        404: "Not found",
        413: "Upload too large (FILE_TOO_LARGE)",
        415: "Not a JPEG or PNG (UNSUPPORTED_MEDIA_TYPE)",
        422: "Invalid request or image (VALIDATION_ERROR, UNSUPPORTED_CATEGORY, INVALID_IMAGE, "
        "IMAGE_DIMENSIONS)",
        429: "Rate limited (RATE_LIMITED); see the Retry-After header",
        500: "Server error (INFERENCE_FAILED, INTERNAL_ERROR)",
        503: "Not ready or busy (MODEL_NOT_READY, BUSY); see the Retry-After header",
    }
    return {s: {"model": ErrorEnvelope, "description": descriptions[s]} for s in statuses}
