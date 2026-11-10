"""Package a calibrated, evaluated artifact for publishing (Phase 4, steps 1, 3 and 4).

    uv run python -m defect_ml.package ml/configs/bottle_wrn50.yaml ml/configs/bottle_resnet18.yaml

For each settings file it copies the **calibrated** artifact, byte for byte, to a new folder
``<artifact_dir>/<base>_v<version>_release`` with ``stage: "packaged"``. Nothing about the model,
the threshold or the calibration changes. The package adds:

* a ``package`` block in the manifest: the version, the evaluation summary copied from the
  finished evaluation's ``metrics.json`` together with that file's SHA-256, the data and split
  hashes, the git state and the creation time;
* ``NOTICE.md``: the dataset attribution and terms, the weights' recorded terms and the project's
  limits, with its SHA-256 recorded in the manifest, so the notice travels with the weights.

Refused before anything is written: a missing or mismatched evaluation (it must be for exactly this
calibrated artifact and these weights), an existing release folder, a tight disk.

**Parity check without the test set.** After packaging, the new folder is loaded fresh and the 42
`calib` images are scored again; their sorted scores must **equal** the scores stored in the
calibration block, exactly. It opens only the listed `calib` files (an audit hook checks this) and
never the test set. This replaces the plan's parity test on test images, which would have reopened
the sealed test set.

Uploading to an artifact store is a separate step (open decision O3) and is not done here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from defect_core.artifacts import ArtifactError, load_artifact, package_artifact
from defect_core.model import infer

from defect_ml import runinfo
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import TrainingImages, sha256_of
from defect_ml.evaluate import EVAL_ROOT, EvaluationError, _paths
from defect_ml.fit import FitConfig, FitError, git_state
from defect_ml.freeze import DEFAULT_VERSION, base_name

NOTICE = """\
# Notice for this model package

This folder holds a fitted anomaly-detection model (`model.safetensors`) and its `manifest.json`.
It was made by the DefectLens project, a **non-commercial educational demo**. It is not a certified
inspection system and must not be used for real quality-control or safety decisions.

## Dataset: MVTec AD

The model was fitted and calibrated on normal training images of the **MVTec AD** dataset
(category: {category}) and evaluated on its official test split. The dataset is not included here.

- Creators and copyright holder: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger,
  MVTec Software GmbH ("Copyright 2019 MVTec Software GmbH", from the dataset's own `readme.txt`).
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0), <https://creativecommons.org/licenses/by-nc-sa/4.0/>.
- Official page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>.
- Please cite: Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. (2019), "A Comprehensive
  Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and
  Pattern Recognition; and Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C. (2021),
  "The MVTec Anomaly Detection Dataset", International Journal of Computer Vision 129, 1038-1059,
  <https://doi.org/10.1007/s11263-020-01400-4>.
- The local copy used by the project came from Kaggle; the Kaggle page, its terms and a comparison
  with the official archive are **unconfirmed**.

## How the project treats the model file

The memory bank in `model.safetensors` is made of feature vectors computed from MVTec AD images, so
the project treats the package as **material derived from the dataset** and applies the same terms
to it as a precaution: attribution as above, non-commercial use only, and the same license terms for
the dataset-derived part. The model was modified from the original images (resized, passed through a
network, subsampled). This is the project's cautious policy, **not a legal finding**, and it is not
legal advice.

## Pretrained backbone weights

The backbone started from ImageNet-pretrained weights pinned by tag: `{backbone}`. The model card of
the weight file declares `license: bsd-3-clause` (read on 2026-11-10; not checked against another
source). The terms of the ImageNet data those weights were trained on are **not verified** and are
not inferred from that statement. Exact file hashes are in `manifest.json` (`provenance.weights`).

## No endorsement

Nothing here implies that MVTec Software GmbH, the dataset authors or the weight authors support or
endorse this project. Results and limits: `docs/evaluation.md` and `docs/model-card.md` in the
project repository. The project's own code license is planned (MIT) but not yet in force.
"""


def release_path(config: FitConfig, version: str) -> Path:
    return Path(config.artifact_dir) / f"{base_name(config.run_name)}_v{version}_release"


def run(
    config: FitConfig,
    version: str = DEFAULT_VERSION,
    eval_root: Path = EVAL_ROOT,
    repo: Path | None = None,
) -> int:
    paths = _paths(config, version, eval_root)
    out = release_path(config, version)
    record_path = Path(config.record_dir) / f"package_{base_name(config.run_name)}_v{version}.json"
    if out.exists() and any(out.iterdir()):
        print(f"REFUSING: {out} already exists; artifacts are never overwritten.")
        return 1
    if record_path.exists():
        print(f"REFUSING: {record_path} already exists.")
        return 1
    reason = runinfo.conditions_ok(config.max_swap_used_mb, config.min_free_disk_gib)
    if reason:
        print(f"NOT STARTING: {reason}", file=sys.stderr)
        return 1
    result = paths["result"]
    try:
        metrics_record = json.loads((result / "metrics.json").read_text())
    except (OSError, ValueError) as error:
        raise EvaluationError(f"no finished evaluation at {result}: {error}") from error

    before = runinfo.system_state()
    split_file = Path(config.split_dir) / f"{config.category}.json"
    evaluation = {
        "metrics_sha256": sha256_of(result / "metrics.json"),
        "predictions_sha256": sha256_of(result / "predictions.csv"),
        "weights_sha256": metrics_record["weights_sha256"],
        "calibrated_manifest_sha256": metrics_record["calibrated_manifest_sha256"],
        "headline": metrics_record["headline"],
        "status": metrics_record["status"],
        "run_id": metrics_record["run_id"],
        "results": metrics_record["results"],
        "documents": ["docs/evaluation.md", "docs/model-card.md"],
        "note": "Evaluated once on the official test split of one category; see the documents.",
    }
    details = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": base_name(config.run_name),
        "category": config.category,
        "data": {
            "data_manifest_sha256": sha256_of(Path(config.manifest)),
            "split_file_sha256": sha256_of(split_file),
            "dataset": "MVTec AD (CC BY-NC-SA 4.0); not included",
        },
        "git": git_state(repo or Path.cwd()),
        "license_notice": "NOTICE.md in this folder: dataset attribution and terms; "
        "non-commercial; not legal advice",
    }
    notice = NOTICE.format(category=config.category, backbone=config.model.backbone)
    manifest = package_artifact(paths["calibrated"], out, evaluation, notice, details)

    # parity without the test set: fresh load, re-score the calib images, compare exactly
    loaded = load_artifact(out)
    calibration = loaded.calibration
    torch.set_num_threads(config.torch_threads)
    data_dir = Path(config.data_dir).resolve()
    images = TrainingImages.open(
        data_dir, config.category, "calib", split_dir=config.split_dir,
        manifest_path=config.manifest, config=loaded.preprocess_config,
    )  # fmt: skip
    scores: list[float] = []
    with OpenAudit() as audit:
        for batch in images.batches(1):
            scores.append(float(infer(loaded.model, batch)[0][0]))
        opened, listed = audit.inside(data_dir)
    expected = sorted(f"{config.category}/{n}" for n in images.names)
    opened_rel = sorted(str(p.resolve().relative_to(data_dir)) for p in opened)
    audit_ok = opened_rel == expected and not listed
    parity = calibration is not None and sorted(scores) == list(calibration.calibration_scores)
    only_calib = all("/train/good/" in p for p in opened_rel)

    record = {
        "step": "Phase 4: package a calibrated, evaluated artifact (no upload)",
        "status": "packaged locally; NOT uploaded; no test image or mask opened",
        "run_name": config.run_name,
        "version": version,
        "source_artifact": str(paths["calibrated"]),
        "package": str(out),
        "weights_sha256": evaluation["weights_sha256"],
        "weights_identical_to_calibrated": sha256_of(paths["calibrated"] / "model.safetensors")
        == sha256_of(out / "model.safetensors"),
        "package_manifest_sha256": sha256_of(out / "manifest.json"),
        "notice_sha256": manifest["package"]["notice"]["sha256"],  # type: ignore[index]
        "files_bytes": {p.name: p.stat().st_size for p in sorted(out.iterdir())},
        "evaluation_record_sha256": evaluation["metrics_sha256"],
        "parity_on_calib_images": {
            "n": len(scores),
            "sorted_scores_equal_stored_calibration_scores": parity,
            "only_listed_calib_files_opened": audit_ok and only_calib,
            "files_opened_inside_dataset_folder": len(opened),
            "test_set_opened": False,
        },
        "git": details["git"],
        "system_before": before,
        "system_after": runinfo.system_state(),
        "limitations": [
            "Not uploaded: the artifact store is an open owner decision (O3).",
            "Parity is on the 42 calibration images, not on the test images (those would "
            "reopen the sealed test set).",
            "The notice describes the project's cautious policy; it is not a legal finding.",
        ],
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{base_name(config.run_name)}: packaged; weights identical: "
        f"{record['weights_identical_to_calibrated']}; calib parity exact: {parity}; "
        f"data access as listed: {audit_ok and only_calib}"
    )
    print(f"package: {out}\nrecord:  {record_path}")
    return 0 if (parity and audit_ok and only_calib) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    code = 0
    try:
        for config_path in args.configs:
            code |= run(FitConfig.from_yaml(config_path), args.version)
    except (EvaluationError, FitError, ArtifactError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()
