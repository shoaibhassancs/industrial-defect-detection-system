"""The numbers the web app shows come from the model package; the saved evaluation files are the
source those numbers were taken from. These checks read JSON and CSV files only (no image, no
mask, no model, no dataset) and change nothing.

1. Always: the saved metrics are consistent with themselves (the counts add up).
2. When the package is present on this machine (`artifacts/` is git-ignored, so it is absent in CI):
   the package's copy of the results equals `metrics.json` exactly, and the recorded SHA-256 values
   match the files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "evaluation"
ARTIFACTS = ROOT / "artifacts"

MODELS = {
    "wide_resnet50_2_tv_in1k": "bottle_wide_resnet50_2_tv_in1k_v1.0.0_release",
    "resnet18_tv_in1k": "bottle_resnet18_tv_in1k_v1.0.0_release",
}


def bottle_dir(model: str) -> Path:
    return EVALUATION / model / "v1.0.0" / "bottle"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("model", MODELS)
def test_saved_metrics_are_consistent_with_themselves(model: str) -> None:
    results = json.loads((bottle_dir(model) / "metrics.json").read_text())["results"]
    image = results["image"]
    at = image["at_calibrated_threshold"]
    alarms = image["false_alarm_on_normal_test_images"]
    assert at["tp"] + at["fn"] == image["n_anomalous"]
    assert at["fp"] + at["tn"] == image["n_normal"]
    assert alarms["false_alarms"] == at["fp"]
    assert alarms["n_normal"] == image["n_normal"]
    assert alarms["fpr"] == at["fpr"]
    per_type = image["per_defect_recall"].values()
    assert sum(t["n"] for t in per_type) == image["n_anomalous"]
    assert sum(t["detected"] for t in per_type) == at["tp"]
    assert results["pixel"]["n_images"] == image["n_normal"] + image["n_anomalous"]


def test_the_headline_numbers_the_model_page_states(model: str = "wide_resnet50_2_tv_in1k") -> None:
    """63 of 63 defective images flagged and 3 of 20 normal images flagged (15%), as saved."""
    image = json.loads((bottle_dir(model) / "metrics.json").read_text())["results"]["image"]
    assert (image["at_calibrated_threshold"]["tp"], image["n_anomalous"]) == (63, 63)
    alarms = image["false_alarm_on_normal_test_images"]
    assert (alarms["false_alarms"], alarms["n_normal"], alarms["fpr"]) == (3, 20, 0.15)


@pytest.mark.parametrize("model", MODELS)
def test_the_package_copy_of_the_results_equals_the_saved_file(model: str) -> None:
    manifest_path = ARTIFACTS / MODELS[model] / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("the model package is not on this machine (artifacts/ is not in git)")
    evaluation = json.loads(manifest_path.read_text())["package"]["evaluation"]
    folder = bottle_dir(model)
    assert sha256(folder / "metrics.json") == evaluation["metrics_sha256"]
    assert sha256(folder / "predictions.csv") == evaluation["predictions_sha256"]
    saved = json.loads((folder / "metrics.json").read_text())
    assert evaluation["results"] == saved["results"]
