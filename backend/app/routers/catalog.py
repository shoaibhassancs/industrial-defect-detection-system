"""Read-only information about the loaded model: categories, model description, evaluation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.core.errors import ModelNotReady, NotFound
from app.schemas import CategoryInfo, error_responses
from app.services.inference import READY, InferenceService

router = APIRouter(tags=["model"])

# MVTec AD's two kinds of category; only `bottle` is fitted so far.
_TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def _service(request: Request) -> InferenceService:
    service: InferenceService = request.app.state.service
    if service.status != READY or service.loaded is None:
        raise ModelNotReady(
            "The model is not ready yet. Please try again shortly.", headers={"Retry-After": "10"}
        )
    return service


def _results(service: InferenceService) -> dict[str, Any]:
    package = (service.loaded.manifest.get("package") or {}) if service.loaded else {}
    return (package.get("evaluation") or {}).get("results") or {}


def _summary(service: InferenceService) -> dict[str, Any] | None:
    results = _results(service)
    image, pixel = results.get("image"), results.get("pixel")
    if not image:
        return None
    at = image.get("at_calibrated_threshold", {})
    return {
        "image_auroc": (image.get("auroc") or {}).get("estimate"),
        "image_aupr": (image.get("aupr") or {}).get("estimate"),
        "tpr": at.get("tpr"),
        "fpr_on_normal_test_images": at.get("fpr"),
        "precision": at.get("precision"),
        "f1": at.get("f1"),
        "n_normal_test_images": image.get("n_normal"),
        "n_anomalous_test_images": image.get("n_anomalous"),
        "pixel_auroc": (pixel or {}).get("auroc"),
        "note": "Sealed MVTec AD test split, one category; see the evaluation document.",
    }


@router.get("/categories", response_model=list[CategoryInfo], responses=error_responses(503))
async def categories(request: Request) -> list[dict[str, Any]]:
    """Supported categories with their calibration and a short evaluation summary."""
    service = _service(request)
    calibration = service.loaded.calibration
    name = service.category
    return [
        {
            "id": name,
            "display_name": name.replace("_", " ").title(),
            "kind": "texture" if name in _TEXTURES else "object",
            "threshold": calibration.threshold,
            "alpha": calibration.alpha,
            "calib_n": calibration.n,
            "metrics_summary": _summary(service),
        }
    ]


@router.get("/model", responses=error_responses(503))
async def model(request: Request) -> dict[str, Any]:
    """What the model is, how it was calibrated, and what it can not do."""
    service = _service(request)
    loaded, calibration = service.loaded, service.loaded.calibration
    package = loaded.manifest.get("package") or {}
    return {
        "algorithm": "PatchCore (memory bank of normal patch features)",
        "name": package.get("model_name") or loaded.model_config.backbone,
        "backbone": loaded.model_config.backbone,
        "layers": list(loaded.model_config.layers),
        "preprocessing": loaded.preprocess_config.to_dict(),
        "calibration": {
            "method": "split-conformal on held-out normal training images",
            "alpha": calibration.alpha,
            "n": calibration.n,
            "k": calibration.k,
            "threshold": calibration.threshold,
            "false_alarm_bound": calibration.false_alarm_bound,
            "meaning": "Chance that a new normal image is flagged, on average over calibration "
            "sets, if it is exchangeable with the calibration images. Says nothing about defects.",
        },
        "version": service.version,
        "headline_model": (package.get("evaluation") or {}).get("headline"),
        "limitations": [
            "Educational demo trained on MVTec AD; not a certified inspection system.",
            f"Only meaningful for {service.category} images captured like the MVTec AD dataset.",
            "It can not reliably tell that an image is out of scope.",
            "The score is an anomaly score, not a probability or a confidence.",
            "Metrics describe one category, one split and a small test set.",
        ],
    }


@router.get("/metrics/{category}", responses=error_responses(404, 422, 503))
async def metrics(category: str, request: Request) -> dict[str, Any]:
    """Sealed-evaluation results for a category (from files checked against the package)."""
    service = _service(request)
    if category != service.category:
        raise NotFound("No evaluation is available for that category.")
    package = service.loaded.manifest.get("package") or {}
    evaluation = package.get("evaluation") or {}
    body: dict[str, Any] = {
        "category": category,
        "version": service.version,
        "status": evaluation.get("status"),
        "results": evaluation.get("results"),
        "curves_available": False,
        "limitations": "docs/evaluation.md describes the protocol and its limits.",
    }
    if service.evaluation:
        body["curves_available"] = "curves" in service.evaluation
        body["curves"] = service.evaluation.get("curves")
        body["histograms"] = service.evaluation.get("histograms")
        body["per_defect_recall"] = (
            (service.evaluation["metrics"].get("results") or {})
            .get("image", {})
            .get("per_defect_recall")
        )
    return body
