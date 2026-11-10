"""Loading the model and running inspections (PLAN.md section 4.5, Phase 5 step 5).

* The artifact is loaded, **verified** (``defect_core.artifacts.load_artifact``: hashes, shapes,
  stored calibration, package fields, notice) and warmed up with one inference in a background
  task, so the process can answer ``/health`` and ``/ready`` (503) while it loads. ``/ready`` only
  reads a flag; it never triggers loading.
* One inspection runs at a time (a semaphore). A request that waits longer than
  ``INFERENCE_QUEUE_TIMEOUT_S`` gets 503 ``BUSY`` with ``Retry-After``.
* The work runs in a worker thread under ``torch.inference_mode`` and calls
  ``defect_core.infer.inspect_image``, **the same function the command line uses**, so the API and
  the CLI can not differ in preprocessing, scoring, calibration or rendering.
* Any unexpected error inside the model becomes 500 ``INFERENCE_FAILED``; its stack trace is logged,
  never returned.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import torch
from defect_core.artifacts import LoadedArtifact, load_artifact
from defect_core.infer import InspectionResult, inspect_image
from PIL import Image

from app.core.errors import ApiError, Busy, InferenceFailed, ModelNotReady
from app.core.logging import get_logger
from app.core.settings import Settings

log = get_logger("app.inference")

LOADING, READY, FAILED = "loading", "ready", "failed"


class ModelLoadError(Exception):
    """The artifact is missing, damaged, or not usable for serving."""


class InferenceService:
    def __init__(
        self,
        settings: Settings,
        loader: Callable[[Path], LoadedArtifact] = load_artifact,
        inspector: Callable[..., InspectionResult] = inspect_image,
    ) -> None:
        self.settings = settings
        self._loader, self._inspector = loader, inspector
        self._semaphore = asyncio.Semaphore(1)
        self.status = LOADING
        self.loaded: LoadedArtifact | None = None
        self.evaluation: dict | None = None  # served at /metrics; only if it matches the package

    # --- loading --------------------------------------------------------------------------
    async def load(self) -> None:
        """Load in a worker thread. Never raises: a failure leaves the service not ready."""
        try:
            await asyncio.to_thread(self._load_blocking)
            self.status = READY
            log.info("model_ready", version=self.version, category=self.category)
        except Exception as error:  # noqa: BLE001 - logged, and reported only as "failed"
            self.status = FAILED
            log.error("model_load_failed", reason=str(error)[:300], exc_info=error)

    def _load_blocking(self) -> None:
        settings = self.settings
        if settings.model_dir is None:
            raise ModelLoadError("MODEL_DIR is not set")
        torch.set_num_threads(settings.torch_num_threads)
        loaded = self._loader(settings.model_dir)
        stage = loaded.manifest.get("stage")
        if stage not in ("calibrated", "packaged") or loaded.calibration is None:
            raise ModelLoadError(f"the artifact is {stage or 'fitted'}, not calibrated or packaged")
        version = str((loaded.manifest.get("freeze") or {}).get("version"))
        if settings.model_version and settings.model_version != version:
            raise ModelLoadError(f"artifact version {version} differs from MODEL_VERSION")
        self._check_evaluation(loaded)
        with torch.inference_mode():  # one warm-up, so the first real request is not slow
            self._inspector(loaded, Image.new("RGB", (256, 256), (128, 128, 128)))
        self.loaded = loaded

    def _check_evaluation(self, loaded: LoadedArtifact) -> None:
        """Serve the evaluation files only if they are the ones the package was made with."""
        folder = self.settings.evaluation_dir
        if folder is None:
            return
        package = loaded.manifest.get("package") or {}
        expected = (package.get("evaluation") or {}).get("metrics_sha256")
        path = Path(folder) / "metrics.json"
        if not expected or not path.is_file():
            raise ModelLoadError("EVALUATION_DIR is set but its metrics.json is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ModelLoadError("the evaluation files are not the ones the package was made with")
        bundle = {}
        for name in ("metrics", "curves", "histograms"):
            file = Path(folder) / f"{name}.json"
            if file.is_file():
                bundle[name] = json.loads(file.read_text())
        self.evaluation = bundle

    # --- facts about the loaded model -----------------------------------------------------------
    @property
    def version(self) -> str:
        loaded = self.loaded
        return str((loaded.manifest.get("freeze") or {}).get("version")) if loaded else ""

    @property
    def category(self) -> str:
        loaded = self.loaded
        if not loaded:
            return ""
        package = loaded.manifest.get("package") or {}
        provenance = loaded.manifest.get("provenance") or {}
        return str(package.get("category") or provenance.get("category") or "unknown")

    # --- inspecting -------------------------------------------------------------------------
    async def inspect(self, image: Image.Image) -> InspectionResult:
        if self.status != READY or self.loaded is None:
            raise ModelNotReady(
                "The model is not ready yet. Please try again shortly.",
                headers={"Retry-After": "10"},
            )
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.settings.inference_queue_timeout_s
            )
        except TimeoutError:
            raise Busy(
                "The service is busy with another inspection. Please try again shortly.",
                headers={"Retry-After": "5"},
            ) from None
        try:
            return await asyncio.to_thread(self._run, image)
        except ApiError:
            raise
        except Exception as error:  # noqa: BLE001
            log.error("inference_failed", exc_info=error)
            raise InferenceFailed("The inspection could not be completed.") from None
        finally:
            self._semaphore.release()

    def _run(self, image: Image.Image) -> InspectionResult:
        assert self.loaded is not None
        with torch.inference_mode():
            return self._inspector(self.loaded, image)
