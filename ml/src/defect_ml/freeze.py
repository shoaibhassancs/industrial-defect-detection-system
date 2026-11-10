"""Freeze a fitted artifact (Phase 3 step 2).

    uv run python -m defect_ml.freeze ml/configs/bottle_wrn50.yaml

Freezing gives a fitted artifact a version and locks it. It does not fit again, does not score
anything, and opens **no** file in the dataset folder (an audit hook checks this and the record
says so). In order, it:

1. reads the YAML settings (including the ``calibration`` settings fixed in advance);
2. finds the fitted artifact ``<artifact_dir>/<run_name>`` and its fit record
   ``<record_dir>/fit_<run_name>.json``, and checks that the record is the one for this artifact
   (manifest hash), says "fitted; NOT frozen", read only the ``fit`` group exactly as listed, and
   that its model settings equal the YAML's;
3. verifies the artifact completely (``load_artifact``) and copies it, byte for byte, to a **new**
   folder ``<artifact_dir>/<base>_frozen-v<version>`` with ``stage: "frozen"`` and a ``freeze``
   block that names the settings fixed now, the weights hash, the YAML hash and the fit record hash;
4. writes ``experiments/freeze_<base>_v<version>.json``.

The fitted folder and its fit record are never changed. A frozen folder is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from defect_core.artifacts import SETTING_KEYS, ArtifactError, freeze_artifact, load_artifact

from defect_ml import runinfo
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import sha256_of
from defect_ml.fit import FitConfig, FitError, git_state

DEFAULT_VERSION = "1.0.0"
FIT_SUFFIX = "_fit-v0"


def base_name(run_name: str) -> str:
    """``bottle_resnet18_tv_in1k_fit-v0`` -> ``bottle_resnet18_tv_in1k``."""
    if not run_name.endswith(FIT_SUFFIX):
        raise FitError(f"run_name {run_name!r} must end with {FIT_SUFFIX!r} to be frozen")
    return run_name[: -len(FIT_SUFFIX)]


def stage_paths(config: FitConfig, version: str) -> dict[str, Path]:
    base = base_name(config.run_name)
    root, records = Path(config.artifact_dir), Path(config.record_dir)
    return {
        "fitted": root / config.run_name,
        "fit_record": records / f"fit_{config.run_name}.json",
        "frozen": root / f"{base}_frozen-v{version}",
        "calibrated": root / f"{base}_v{version}",
        "freeze_record": records / f"freeze_{base}_v{version}.json",
        "calibration_record": records / f"calibration_{base}_v{version}.json",
    }


def frozen_settings(config: FitConfig, config_path: str | Path) -> dict[str, object]:
    raw = config.raw["calibration"]
    return {key: raw[key] for key in (*SETTING_KEYS, "upper_multiple_status")} | {
        "config_file": str(config_path),
        "config_sha256": sha256_of(Path(config_path)),
    }


def check_fit_record(config: FitConfig, paths: dict[str, Path]) -> dict[str, object]:
    """Return the fit record after checking that it belongs to this artifact and run."""
    record_path, artifact = paths["fit_record"], paths["fitted"]
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as error:
        raise FitError(f"cannot read the fit record {record_path}: {error}") from error
    problems = []
    if record.get("run_name") != config.run_name:
        problems.append("the record is for another run")
    if record.get("manifest_sha256") != sha256_of(artifact / "manifest.json"):
        problems.append("the artifact's manifest differs from the one the fit record names")
    if "NOT frozen" not in str(record.get("status")):
        problems.append("the record does not say 'fitted; NOT frozen'")
    access = record.get("data_access", {})
    listed_only = access.get("exactly_the_listed_fit_files_each_once")
    if access.get("group_read") != "fit" or not listed_only:
        problems.append("the record does not show a fit-only, exactly-as-listed data access")
    if record.get("settings", {}).get("model") != config.raw["model"]:
        problems.append("the model settings of the fit differ from the YAML now")
    if problems:
        raise FitError("; ".join(problems))
    return record


def run(config: FitConfig, config_path: str | Path, version: str = DEFAULT_VERSION) -> int:
    paths = stage_paths(config, version)
    if paths["frozen"].exists() and any(paths["frozen"].iterdir()):
        print(f"REFUSING: {paths['frozen']} already exists; artifacts are never overwritten.")
        return 1
    if paths["freeze_record"].exists():
        print(f"REFUSING: {paths['freeze_record']} already exists.")
        return 1
    reason = runinfo.conditions_ok(config.max_swap_used_mb, config.min_free_disk_gib)
    if reason:
        print(f"NOT STARTING: {reason}", file=sys.stderr)
        return 1
    before = runinfo.system_state()
    check_fit_record(config, paths)
    settings = frozen_settings(config, config_path)
    details = {
        "frozen_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": config.run_name,
        "source_fit_record_sha256": sha256_of(paths["fit_record"]),
        "git": git_state(Path.cwd()),
    }
    data_dir = Path(config.data_dir).resolve()
    with OpenAudit() as audit:
        manifest = freeze_artifact(paths["fitted"], paths["frozen"], version, settings, details)
        opened, listed = audit.inside(data_dir)
    reloaded = load_artifact(paths["frozen"])
    weights = manifest["files"]["model.safetensors"]  # type: ignore[index]
    record = {
        "step": "Phase 3 step 2: freeze a fitted artifact",
        "status": "frozen; NOT calibrated; no test image or mask involved",
        "version": version,
        "run_name": config.run_name,
        "source_artifact": str(paths["fitted"]),
        "frozen_artifact": str(paths["frozen"]),
        "weights_sha256": weights["sha256"],
        "weights_bytes": weights["bytes"],
        "weights_identical_to_fitted": sha256_of(paths["fitted"] / "model.safetensors")
        == sha256_of(paths["frozen"] / "model.safetensors"),
        "reloads_cleanly_as_stage": reloaded.manifest["stage"],
        "frozen_settings": settings,
        "fit_record": str(paths["fit_record"]),
        "fit_record_sha256": details["source_fit_record_sha256"],
        "manifest_sha256": sha256_of(paths["frozen"] / "manifest.json"),
        "data_access": {
            "dataset_files_opened": len(opened),
            "directory_listings_inside_dataset_folder": len(listed),
            "method": "sys.addaudithook: open, os.listdir, os.scandir during the freeze",
        },
        "git": details["git"],
        "system_before": before,
        "system_after": runinfo.system_state(),
        "limitations": [
            "Freezing copies and labels; it does not measure quality and does not calibrate.",
            "The fit record was written before the 'calibration' section existed in the YAML; "
            "the frozen settings come from the YAML as it is now and were fixed before any "
            "calibration or test access (decisions.md entry 060).",
        ],
    }
    paths["freeze_record"].parent.mkdir(parents=True, exist_ok=True)
    paths["freeze_record"].write_text(json.dumps(record, indent=1) + "\n")
    ok = not opened and not listed
    print(f"frozen: {paths['frozen']}  (dataset files opened: {len(opened)})")
    print(f"record: {paths['freeze_record']}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="YAML settings file (ml/configs/bottle_resnet18.yaml)")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    try:
        sys.exit(run(FitConfig.from_yaml(args.config), args.config, args.version))
    except (FitError, ArtifactError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
