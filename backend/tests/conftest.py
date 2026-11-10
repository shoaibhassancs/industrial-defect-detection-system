"""Shared fixtures: a small real (random-weight) packaged artifact, settings and clients.

The artifact is built once per session with the same functions the project pipeline uses (fit,
freeze, calibrate, package), from synthetic images. Nothing here reads the dataset or the sealed
test set, and nothing needs network access.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from app.core.settings import Settings
from app.main import create_app
from backend_imaging import make_image
from defect_core.artifacts import (
    calibrate_artifact,
    freeze_artifact,
    load_artifact,
    package_artifact,
    save_artifact,
)
from defect_core.calibration import build_calibration
from defect_core.model import ModelConfig, build_model, fit_memory_bank, infer
from defect_core.preprocessing import DEFAULT_CONFIG, to_tensor
from fastapi.testclient import TestClient

CONFIG = ModelConfig(backbone="resnet18")
ALPHA = 0.1
N_CALIB = 20
FROZEN_SETTINGS = {
    "alpha": ALPHA, "lower_percentile": 1.0, "highlight_percentile": 99.5, "upper_multiple": 1.5,
    "upper_multiple_status": "unvalidated display choice",
}  # fmt: skip


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach the network (the in-process test client does not use sockets)."""

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


class Built:
    """What the session fixture builds."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.package = root / "package"
        self.calibrated = root / "calibrated"
        self.frozen = root / "frozen"
        self.fitted = root / "fitted"
        self.evaluation = root / "evaluation"
        self.calib_images = [make_image(1000 + i) for i in range(N_CALIB)]
        self.calibration_scores: list[float] = []


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    b = Built(tmp_path_factory.mktemp("api_artifact"))
    torch.manual_seed(5)
    model = build_model(CONFIG, pre_trained=False)
    fit_images = [make_image(i) for i in range(8)]
    batches = [
        torch.stack([to_tensor(img) for img in fit_images[i : i + 4]]) for i in range(0, 8, 4)
    ]
    fit_memory_bank(model, batches, CONFIG.coreset_ratio, seed=0)
    save_artifact(
        model, CONFIG, DEFAULT_CONFIG, b.fitted,
        provenance={"category": "bottle", "run_name": "test_fit-v0"},
    )  # fmt: skip
    freeze_artifact(b.fitted, b.frozen, "1.0.0", FROZEN_SETTINGS)
    frozen = load_artifact(b.frozen)
    scores, maps = [], []
    for image in b.calib_images:
        score, anomaly_map = infer(frozen.model, to_tensor(image)[None])
        scores.append(float(score[0]))
        maps.append(anomaly_map[0, 0].numpy())
    calibration = build_calibration(
        scores, maps, alpha=ALPHA, lower_percentile=1.0, highlight_percentile=99.5,
        upper_multiple=1.5,
    )  # fmt: skip
    b.calibration_scores = list(calibration.calibration_scores)
    inputs = {"group": "calib", "n": N_CALIB, "preprocessing_version": DEFAULT_CONFIG.version}
    calibrate_artifact(b.frozen, b.calibrated, calibration, inputs)

    # evaluation files like the real ones (shape only), and the package that names their hash
    b.evaluation.mkdir()
    weights = json.loads((b.calibrated / "manifest.json").read_text())["files"]["model.safetensors"]
    results = {
        "image": {
            "n_normal": 20, "n_anomalous": 63,
            "auroc": {"estimate": 1.0, "low": 1.0, "high": 1.0},
            "aupr": {"estimate": 1.0},
            "at_calibrated_threshold": {"tpr": 1.0, "fpr": 0.15, "precision": 0.95, "f1": 0.97},
            "per_defect_recall": {"crack": {"n": 5, "detected": 5, "recall": 1.0}},
        },
        "pixel": {"auroc": 0.98},
    }  # fmt: skip
    metrics = {"weights_sha256": weights["sha256"], "results": results}
    (b.evaluation / "metrics.json").write_text(json.dumps(metrics))
    curves = {"image_roc": {"fpr": [0, 1], "tpr": [0, 1]}}
    (b.evaluation / "curves.json").write_text(json.dumps(curves))
    (b.evaluation / "histograms.json").write_text(json.dumps({"bin_edges": [0, 1]}))
    evaluation = {
        "metrics_sha256": hashlib.sha256((b.evaluation / "metrics.json").read_bytes()).hexdigest(),
        "weights_sha256": weights["sha256"],
        "calibrated_manifest_sha256": hashlib.sha256(
            (b.calibrated / "manifest.json").read_bytes()
        ).hexdigest(),
        "headline": False, "status": "test fixture", "results": results,
    }  # fmt: skip
    package_artifact(
        b.calibrated,
        b.package,
        evaluation,
        "# Notice\ntest fixture\n",
        {"model_name": "test_model", "category": "bottle", "created_utc": "2026-01-01T00:00:00Z"},
    )
    return b


@pytest.fixture
def make_settings(built: Built) -> Callable[..., Settings]:
    def factory(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "app_env": "test", "model_dir": built.package, "rate_limit": "1000/minute",
            "inference_queue_timeout_s": 5, "log_level": "WARNING",
        }  # fmt: skip
        values.update(overrides)
        return Settings(_env_file=None, **values)  # type: ignore[arg-type]

    return factory


def wait_ready(client: TestClient, seconds: float = 60) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if client.get("/api/v1/ready").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError("the service did not become ready")


@pytest.fixture
def client(make_settings: Callable[..., Settings]):
    """A ready service with the real artifact and the real inspection function."""
    with TestClient(create_app(make_settings())) as test_client:
        wait_ready(test_client)
        yield test_client
