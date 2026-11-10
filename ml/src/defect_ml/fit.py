"""Fit the PatchCore memory bank on the `fit` images of one category (Phase 2 step 5).

    uv run python -m defect_ml.fit ml/configs/bottle_resnet18.yaml

What it does, in order:

1. reads a YAML settings file (strict: unknown or missing keys are errors);
2. checks the machine (swap, free disk) and refuses to start if it is already too tight;
3. loads the `fit` group (and only that group) through ``TrainingImages``;
4. builds the model with the pinned pretrained weights and records exactly which weight file was
   used (source, tag, cache revision, SHA-256), after checking that file's tensors equal the ones in
   the model;
5. builds the memory bank with ``defect_core`` and saves the artifact;
6. reloads the artifact and checks it gives identical outputs on two of the fit images;
7. writes a run record to ``experiments/fit_<run_name>.json``.

It never reads the `calib` group, the official test images or the masks. It does no calibration and
no scoring of held-out images: the artifact it writes is fitted but **not frozen or calibrated**
(that is Phase 3). A Python audit hook records the files the process opens inside the dataset folder
and the run record states whether they were exactly the listed `fit` files.

Pretrained weights are pinned by tag in the backbone name (for example ``wide_resnet50_2.tv_in1k``).
An untagged name is refused when pretrained weights are requested, so a run can never silently fall
back to a library default. Random weights are possible only with ``--allow-random-weights`` (for
tests) and are labelled in the record and in the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from defect_core.artifacts import load_artifact, save_artifact
from defect_core.model import FitStats, ModelConfig, build_model, fit_memory_bank, infer
from defect_core.preprocessing import DEFAULT_CONFIG
from safetensors.torch import load_file

from defect_ml import runinfo
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import CATEGORIES, TrainingImages, sha256_of

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")  # timm prefers safetensors
BACKBONE_PREFIX = "feature_extractor.feature_extractor."  # in the PatchCore model's state dict

_SECTIONS = {
    "run_name": str,
    "category": str,
    "model": dict,
    "weights": dict,
    "fit": dict,
    "paths": dict,
    "guard": dict,
    "calibration": dict,
}
_KEYS = {
    "model": {"backbone", "layers", "num_neighbors", "coreset_ratio"},
    "weights": {"pretrained"},
    "fit": {"seed", "batch_size", "torch_threads"},
    "paths": {"data_dir", "split_dir", "manifest", "artifact_dir", "record_dir"},
    "guard": {"max_swap_used_mb", "min_free_disk_gib"},
    "calibration": {
        "alpha", "lower_percentile", "highlight_percentile", "upper_multiple",
        "upper_multiple_status",
    },
}  # fmt: skip


class FitError(Exception):
    """The settings or the run are not acceptable."""


@dataclass(frozen=True)
class FitConfig:
    run_name: str
    category: str
    model: ModelConfig
    pretrained: bool
    seed: int
    batch_size: int
    torch_threads: int
    data_dir: str
    split_dir: str
    manifest: str
    artifact_dir: str
    record_dir: str
    max_swap_used_mb: float
    min_free_disk_gib: float
    alpha: float  # frozen Phase 3 settings: carried in the file, not used by the fit itself
    lower_percentile: float
    highlight_percentile: float
    upper_multiple: float
    raw: dict  # the settings exactly as read, kept for the run record

    @classmethod
    def from_dict(cls, data: dict) -> FitConfig:
        if not isinstance(data, dict) or set(data) != set(_SECTIONS):
            raise FitError(f"settings must have exactly these sections: {sorted(_SECTIONS)}")
        for name, kind in _SECTIONS.items():
            if not isinstance(data[name], kind):
                raise FitError(f"'{name}' must be a {kind.__name__}")
        for name, keys in _KEYS.items():
            if set(data[name]) != keys:
                raise FitError(f"'{name}' must have exactly the keys {sorted(keys)}")
        if data["category"] not in CATEGORIES:
            raise FitError(f"unknown category {data['category']!r}")
        m, w, f, p, g = (data[k] for k in ("model", "weights", "fit", "paths", "guard"))
        if not isinstance(w["pretrained"], bool):
            raise FitError("weights.pretrained must be true or false")
        for key, minimum in (("seed", 0), ("batch_size", 1), ("torch_threads", 1)):
            value = f[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise FitError(f"fit.{key} must be a whole number >= {minimum}")
        if not all(isinstance(p[k], str) and p[k] for k in _KEYS["paths"]):
            raise FitError("every paths.* entry must be a non-empty string")
        layers = m["layers"]
        if not (isinstance(layers, list) and layers and all(isinstance(x, str) for x in layers)):
            raise FitError("model.layers must be a non-empty list of names")
        if not isinstance(m["backbone"], str) or not m["backbone"]:
            raise FitError("model.backbone must be a name")
        if w["pretrained"] and "." not in m["backbone"]:
            raise FitError(
                f"model.backbone {m['backbone']!r} has no weight tag: with pretrained weights the "
                "tag must be pinned (for example 'wide_resnet50_2.tv_in1k'), so the run cannot "
                "fall back to a library default"
            )
        c = data["calibration"]
        numbers = [c[k] for k in ("alpha", "lower_percentile", "highlight_percentile")]
        numbers.append(c["upper_multiple"])
        if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in numbers):
            raise FitError("calibration alpha, percentiles and multiple must be numbers")
        if not 0 < c["alpha"] < 1:
            raise FitError("calibration.alpha must be strictly between 0 and 1")
        if not 0 <= c["lower_percentile"] < c["highlight_percentile"] <= 100:
            raise FitError("need 0 <= calibration.lower_percentile < highlight_percentile <= 100")
        if not c["upper_multiple"] > 0:
            raise FitError("calibration.upper_multiple must be positive")
        if not isinstance(c["upper_multiple_status"], str) or not c["upper_multiple_status"]:
            raise FitError("calibration.upper_multiple_status must say how the multiple was chosen")
        try:
            model = ModelConfig(
                backbone=m["backbone"],
                layers=tuple(m["layers"]),
                num_neighbors=int(m["num_neighbors"]),
                coreset_ratio=float(m["coreset_ratio"]),
            )
            limits = float(g["max_swap_used_mb"]), float(g["min_free_disk_gib"])
        except (TypeError, ValueError) as error:
            raise FitError(f"bad number in the settings: {error}") from error
        return cls(
            data["run_name"], data["category"], model, w["pretrained"], f["seed"],
            f["batch_size"], f["torch_threads"], p["data_dir"], p["split_dir"], p["manifest"],
            p["artifact_dir"], p["record_dir"], limits[0], limits[1],
            float(c["alpha"]), float(c["lower_percentile"]), float(c["highlight_percentile"]),
            float(c["upper_multiple"]), data,
        )  # fmt: skip

    @classmethod
    def from_yaml(cls, path: str | Path) -> FitConfig:
        try:
            return cls.from_dict(yaml.safe_load(Path(path).read_text()))
        except (OSError, yaml.YAMLError) as error:
            raise FitError(f"cannot read settings {path}: {error}") from error


# --- pretrained weight records -------------------------------------------------------------


def weights_record(backbone: str, model: torch.nn.Module) -> dict[str, object]:
    """Say exactly which pretrained file is in ``model``, and prove it.

    Finds the cached checkpoint of the timm repository for ``backbone``, hashes it, and checks that
    every tensor in it that the model also has is equal to the model's tensor.
    """
    import timm
    from huggingface_hub import hf_hub_download

    cfg = timm.models.get_pretrained_cfg(backbone)
    if cfg is None or not cfg.hf_hub_id:
        raise FitError(f"timm knows no pretrained weights for {backbone!r}")
    last_error: Exception | None = None
    for filename in WEIGHT_FILES:
        try:
            path = Path(hf_hub_download(cfg.hf_hub_id, filename, local_files_only=True))
        except Exception as error:  # noqa: BLE001 - not cached under that name; try the next
            last_error = error
            continue
        tensors = (
            load_file(str(path))
            if filename.endswith(".safetensors")
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        matched, mismatched = compare_with_model(model, tensors)
        if matched and not mismatched:
            return {
                "timm_name": backbone,
                "hf_hub_id": cfg.hf_hub_id,
                "file": filename,
                # the snapshot folder name is the repository revision; the file in it is a link
                # into a shared blob store, so read the name from the link's own location
                "cache_revision": path.parent.name,
                "sha256": sha256_of(path.resolve()),
                "bytes": path.resolve().stat().st_size,
                "original_source_url": cfg.url or None,
                "tensors_compared_with_model": matched,
                "tensors_differing": 0,
                "terms": "see docs/data.md; the license field in timm's metadata is "
                f"{getattr(cfg, 'license', None)!r} (a statement by the model card, not checked "
                "here, and separate from the terms of the ImageNet training data)",
            }
    raise FitError(
        f"could not find a cached checkpoint matching the model for {cfg.hf_hub_id} "
        f"({last_error!r})"
    )


def compare_with_model(model: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Return (tensors that match the model's, tensors that differ) among those the model has."""
    state = model.state_dict()
    matched = mismatched = 0
    for name, tensor in tensors.items():
        mine = state.get(BACKBONE_PREFIX + name)
        if mine is None:
            continue  # the model keeps only the layers PatchCore uses; other tensors are unused
        if mine.shape == tensor.shape and torch.equal(mine.cpu(), tensor.cpu()):
            matched += 1
        else:
            mismatched += 1
    return matched, mismatched


# --- the run -------------------------------------------------------------------------------


def git_state(repo: Path) -> dict[str, object]:
    """Read-only: the commit the code was at, and whether files had uncommitted changes."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "uncommitted_changes": None if status is None else bool(status)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: FitConfig, *, allow_random_weights: bool = False, repo: Path | None = None) -> int:
    if not config.pretrained and not allow_random_weights:
        print("REFUSING: random weights need --allow-random-weights (tests only).", file=sys.stderr)
        return 1
    artifact_path = Path(config.artifact_dir) / config.run_name
    record_path = Path(config.record_dir) / f"fit_{config.run_name}.json"
    if artifact_path.exists() and any(artifact_path.iterdir()):
        print(f"REFUSING: {artifact_path} already exists; artifacts are never overwritten.")
        return 1

    reason = runinfo.conditions_ok(config.max_swap_used_mb, config.min_free_disk_gib)
    if reason:
        print(f"NOT STARTING: {reason}", file=sys.stderr)
        return 1
    torch.set_num_threads(config.torch_threads)
    before = runinfo.system_state()
    watchdog = runinfo.Watchdog(config.max_swap_used_mb, config.min_free_disk_gib)
    watchdog.start()

    try:
        data_dir = Path(config.data_dir).resolve()
        images = TrainingImages.open(
            data_dir, config.category, "fit",
            split_dir=config.split_dir, manifest_path=config.manifest,
        )  # fmt: skip
        probe: list[torch.Tensor] = []

        def batches():
            for batch in images.batches(config.batch_size):
                if not probe:
                    probe.append(batch[:2].clone())  # two fit images, to check the saved artifact
                yield batch

        with OpenAudit() as audit:
            model = build_model(config.model, pre_trained=config.pretrained)
            weights = weights_record(config.model.backbone, model) if config.pretrained else None
            ratio, seed = config.model.coreset_ratio, config.seed
            stats: FitStats = fit_memory_bank(model, batches(), ratio, seed)
            opened, listed = audit.inside(data_dir)

    finally:
        watchdog.stop()

    expected = sorted(f"{config.category}/{n}" for n in images.names)
    opened_rel = sorted(str(p.resolve().relative_to(data_dir)) for p in opened)
    audit_ok = opened_rel == expected and not listed and sorted(images.opened) == expected

    split_doc = json.loads((Path(config.split_dir) / f"{config.category}.json").read_text())
    random_note = {"pretrained": False, "note": "RANDOM WEIGHTS (test only)"}
    provenance = {
        "run_name": config.run_name,
        "category": config.category,
        "group": "fit",
        "fit_images": len(images),
        "seed": config.seed,
        "batch_size": config.batch_size,
        "split_lists_sha256": split_doc["train_files_sha256"],
        "data_manifest_sha256": file_sha256(Path(config.manifest)),
        "weights": weights if weights else random_note,
        "git": git_state(repo or Path.cwd()),
        "status": "fitted; NOT frozen and NOT calibrated",
    }
    manifest = save_artifact(model, config.model, DEFAULT_CONFIG, artifact_path, provenance)

    loaded = load_artifact(artifact_path)
    reference = infer(model, probe[0])
    reloaded = infer(loaded.model, probe[0])
    reload_identical = bool(
        torch.equal(reference[0], reloaded[0]) and torch.equal(reference[1], reloaded[1])
    )
    files = {p.name: p.stat().st_size for p in sorted(artifact_path.iterdir())}
    bank = model.memory_bank

    record = {
        "step": "Phase 2 step 5: fit on the `fit` split (real images)",
        "run_name": config.run_name,
        "status": "fitted; NOT frozen and NOT calibrated; no test image or mask involved",
        "settings": config.raw,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "timings_seconds": {
            "features": round(stats.features_seconds, 2),
            "coreset": round(stats.coreset_seconds, 2),
            "fit_total": round(stats.features_seconds + stats.coreset_seconds, 2),
        },
        "peak_rss_mb": runinfo.rss_peak_mb(),
        "patch_features": stats.patch_features,
        "memory_bank": {
            "shape": list(stats.memory_bank_shape),
            "dtype": str(bank.dtype),
            "mb": round(bank.numel() * bank.element_size() / 2**20, 1),
        },
        "artifact": {"path": str(artifact_path), "files_bytes": files, "git_ignored": True},
        "reload_check": {
            "images": int(probe[0].shape[0]),
            "outputs_bit_identical": reload_identical,
        },
        "weights": provenance["weights"],
        "data_access": {
            "group_read": "fit",
            "images_loaded": len(images),
            "method": "sys.addaudithook: open, os.listdir, os.scandir during model build and fit",
            "files_opened_inside_dataset_folder": len(opened),
            "exactly_the_listed_fit_files_each_once": audit_ok,
            "directory_listings_inside_dataset_folder": len(listed),
        },
        "provenance": provenance,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": importlib.metadata.version("timm"),
            "anomalib": importlib.metadata.version("anomalib"),
            "platform": platform.platform(),
        },
        "system_before": before,
        "system_after": runinfo.system_state(),
        "watchdog": {
            "max_swap_used_mb_seen": round(watchdog.max_swap_seen, 1),
            "min_free_disk_gib_seen": round(watchdog.min_free_seen, 2),
            "limits": {
                "swap_mb": config.max_swap_used_mb,
                "free_disk_gib": config.min_free_disk_gib,
            },
        },
        "manifest_sha256": file_sha256(artifact_path / "manifest.json"),
        "limitations": [
            "Timings include other apps and any swapping on this 8 GB Mac; see system_before.",
            "Peak resident memory excludes memory the OS compressed or swapped out.",
            "No calibration, threshold or held-out scoring was done: that is Phase 3.",
            "The weight terms recorded here are what the model card states; they were not "
            "independently verified, and are separate from the ImageNet dataset terms.",
        ],
    }
    del manifest
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    ok = audit_ok and reload_identical
    print(
        f"{config.run_name}: fit {record['timings_seconds']['fit_total']} s, bank "
        f"{stats.memory_bank_shape}, peak RSS {record['peak_rss_mb']} MB, "
        f"reload identical: {reload_identical}, data access as listed: {audit_ok}"
    )
    print(f"artifact: {artifact_path}\nrecord:   {record_path}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="YAML settings file (ml/configs/bottle_resnet18.yaml)")
    parser.add_argument("--allow-random-weights", action="store_true", help="tests only")
    args = parser.parse_args()
    try:
        config = FitConfig.from_yaml(args.config)
        sys.exit(run(config, allow_random_weights=args.allow_random_weights))
    except FitError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
