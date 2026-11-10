"""Response shapes (shown in the generated OpenAPI documentation at /docs)."""

from __future__ import annotations

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


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody
