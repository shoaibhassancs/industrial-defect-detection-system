"""Calibrate a frozen artifact on the held-out `calib` images (Phase 3 step 3).

    uv run python -m defect_ml.calibrate ml/configs/bottle_wrn50.yaml

In order, it:

1. checks the machine (swap, free disk) and refuses to start if it is already too tight;
2. loads the **frozen** artifact (``load_artifact`` verifies it; a fitted or already calibrated
   artifact is refused) and checks that the YAML's settings equal the ones fixed at freeze time;
3. scores the 42 `calib` images one at a time (as the serving path will), through
   ``TrainingImages`` (so only the listed `calib` files are opened, each checked against the
   manifest) and ``defect_core``;
4. computes the conformal threshold, the rank table and the pixel levels with
   ``defect_core.calibration.build_calibration``;
5. writes a **new** folder ``<base>_v<version>`` (the calibrated artifact) and a separate record
   ``experiments/calibration_<base>_v<version>.json``.

It opens no `fit` image, no official test image and no mask. An audit hook checks this and the
record says so. The fitted and frozen folders and the fit and freeze records are not changed.
Nothing here depends on test results; the settings were fixed before this stage.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from defect_core.artifacts import ArtifactError, calibrate_artifact, load_artifact
from defect_core.calibration import CalibrationError, build_calibration
from defect_core.model import infer

from defect_ml import runinfo
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import TrainingImages, sha256_of
from defect_ml.data.splits import list_sha256, read_split
from defect_ml.fit import FitConfig, FitError, git_state
from defect_ml.freeze import DEFAULT_VERSION, stage_paths

BATCH_SIZE = 1  # one image at a time, like serving


def run(config: FitConfig, config_path: str | Path, version: str = DEFAULT_VERSION) -> int:
    paths = stage_paths(config, version)
    if paths["calibrated"].exists() and any(paths["calibrated"].iterdir()):
        print(f"REFUSING: {paths['calibrated']} already exists; artifacts are never overwritten.")
        return 1
    if paths["calibration_record"].exists():
        print(f"REFUSING: {paths['calibration_record']} already exists.")
        return 1
    reason = runinfo.conditions_ok(config.max_swap_used_mb, config.min_free_disk_gib)
    if reason:
        print(f"NOT STARTING: {reason}", file=sys.stderr)
        return 1

    frozen = load_artifact(paths["frozen"])
    if frozen.manifest.get("stage") != "frozen":
        raise FitError(f"{paths['frozen']} is not a frozen artifact")
    settings = frozen.manifest["freeze"]["frozen_settings"]  # type: ignore[index]
    yaml_settings = config.raw["calibration"]
    keys = ("alpha", "lower_percentile", "highlight_percentile", "upper_multiple")
    if any(float(settings[k]) != float(yaml_settings[k]) for k in keys):
        raise FitError("the YAML's calibration settings differ from the ones fixed at freeze time")
    if frozen.model_config != config.model:
        raise FitError("the YAML's model settings differ from the frozen artifact's")

    torch.set_num_threads(config.torch_threads)
    before = runinfo.system_state()
    watchdog = runinfo.Watchdog(config.max_swap_used_mb, config.min_free_disk_gib)
    watchdog.start()
    data_dir = Path(config.data_dir).resolve()
    try:
        images = TrainingImages.open(
            data_dir, config.category, "calib",
            split_dir=config.split_dir, manifest_path=config.manifest,
            config=frozen.preprocess_config,
        )  # fmt: skip
        scores: list[float] = []
        maps: list[np.ndarray] = []
        with OpenAudit() as audit:
            for batch in images.batches(BATCH_SIZE):
                score, anomaly_map = infer(frozen.model, batch)
                scores.extend(float(x) for x in score)
                maps.extend(anomaly_map[:, 0].numpy())
            opened, listed = audit.inside(data_dir)
    finally:
        watchdog.stop()

    expected = sorted(f"{config.category}/{n}" for n in images.names)
    opened_rel = sorted(str(p.resolve().relative_to(data_dir)) for p in opened)
    audit_ok = opened_rel == expected and not listed and sorted(images.opened) == expected

    calibration = build_calibration(
        scores, maps,
        alpha=float(settings["alpha"]),
        lower_percentile=float(settings["lower_percentile"]),
        highlight_percentile=float(settings["highlight_percentile"]),
        upper_multiple=float(settings["upper_multiple"]),
    )  # fmt: skip
    split = read_split(Path(config.split_dir) / f"{config.category}.json")
    inputs = {
        "group": "calib",
        "n": len(images),
        "image_list_sha256": list_sha256([f"{config.category}/{n}" for n in images.names]),
        "split_lists_sha256": split["train_files_sha256"],
        "split_seed": split.get("seed"),
        "data_manifest_sha256": sha256_of(Path(config.manifest)),
        "preprocessing_version": frozen.preprocess_config.version,
        "batch_size": BATCH_SIZE,
        "torch_threads": torch.get_num_threads(),
    }
    manifest = calibrate_artifact(paths["frozen"], paths["calibrated"], calibration, inputs)
    loaded = load_artifact(paths["calibrated"])
    round_trip = loaded.calibration is not None and loaded.calibration == calibration

    ordered = np.array(calibration.calibration_scores)
    flagged = int(sum(calibration.is_anomalous(s) for s in scores))
    pixel = calibration.pixel
    record = {
        "step": "Phase 3 step 3: calibrate a frozen artifact on the `calib` split",
        "status": "calibrated on held-out training normals; NOT evaluated; no test image or mask",
        "version": version,
        "run_name": config.run_name,
        "frozen_artifact": str(paths["frozen"]),
        "calibrated_artifact": str(paths["calibrated"]),
        "weights_sha256": manifest["files"]["model.safetensors"]["sha256"],  # type: ignore[index]
        "calibrated_manifest_sha256": sha256_of(paths["calibrated"] / "manifest.json"),
        "settings": {k: settings[k] for k in keys},
        "calibration": {
            "n": calibration.n,
            "k": calibration.k,
            "threshold": calibration.threshold,
            "false_alarm_bound": calibration.false_alarm_bound,
            "score_min": float(ordered[0]),
            "score_median": float(np.median(ordered)),
            "score_max": float(ordered[-1]),
            "calib_images_above_threshold": flagged,
            "pixel_levels": {
                "lower": pixel.lower,
                "highlight": pixel.highlight,
                "upper": pixel.upper,
            },  # fmt: skip
            "block_sha256": manifest["calibration"]["block_sha256"],  # type: ignore[index]
        },
        "calibrated_manifest_reloads_and_matches": round_trip,
        "inputs": inputs,
        "data_access": {
            "group_read": "calib",
            "images_loaded": len(images),
            "method": "sys.addaudithook: open, os.listdir, os.scandir during scoring",
            "files_opened_inside_dataset_folder": len(opened),
            "exactly_the_listed_calib_files_each_once": audit_ok,
            "directory_listings_inside_dataset_folder": len(listed),
        },
        "git": git_state(Path.cwd()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "peak_rss_mb": runinfo.rss_peak_mb(),
        "system_before": before,
        "system_after": runinfo.system_state(),
        "watchdog": {
            "max_swap_used_mb_seen": round(watchdog.max_swap_seen, 1),
            "min_free_disk_gib_seen": round(watchdog.min_free_seen, 2),
        },
        "limitations": [
            "These numbers describe normal training images only. They say nothing about how well "
            "defects are found (that needs the sealed test set, which has not been opened).",
            "The false-alarm bound is for new normal images drawn like the calibration images; "
            "it is not a statement about a different camera, lighting or product.",
            "With 42 images the threshold is the 41st smallest score; one calibration image lies "
            "above it by construction.",
            "The heatmap multiple 1.5 is an unvalidated display choice (decisions.md entry 060).",
        ],
    }
    paths["calibration_record"].parent.mkdir(parents=True, exist_ok=True)
    paths["calibration_record"].write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{config.run_name}: n={calibration.n} k={calibration.k} "
        f"threshold={calibration.threshold:.6f} bound={calibration.false_alarm_bound:.4f}; "
        f"reload matches: {round_trip}; data access as listed: {audit_ok}"
    )
    print(f"artifact: {paths['calibrated']}\nrecord:   {paths['calibration_record']}")
    return 0 if (audit_ok and round_trip) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="YAML settings file (ml/configs/bottle_resnet18.yaml)")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    try:
        sys.exit(run(FitConfig.from_yaml(args.config), args.config, args.version))
    except (FitError, ArtifactError, CalibrationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
