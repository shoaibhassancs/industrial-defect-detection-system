"""The sealed evaluation: the code of the sealed path that opens `test/` and `ground_truth/` files.

    uv run python -m defect_ml.evaluate ml/configs/bottle_wrn50.yaml ml/configs/bottle_resnet18.yaml

(Phase 3 step 6; PLAN.md sections 5.6 and 5.7; ADR-0002.) Several settings files make one planned
batch: every file is checked first, nothing is opened until all pass, and then the models are
evaluated one after the other with identical code and settings. The command prints no metric, only
where the results were written, so no number can steer a choice for the next model.

The sealed path is this module and ``visualize.py`` (figures), which imports this module's reader,
path check and log, so both obey the same rules and write to the same log.

Rules this code enforces, each tested:

* **Calibrated artifacts only.** ``load_artifact`` must report stage ``calibrated`` with a
  calibration block that matches the weights, and the version must be the requested one. A fitted or
  frozen-only artifact is refused before any test file is touched.
* **Its own reader and its own path check** (``assert_evaluation_path``). The training loader and
  its guard (``defect_ml.data.mvtec``) are untouched and still refuse these files. Only the files
  named in the split file are read: the listed test images and, for defective images, the mask
  ``ground_truth/<type>/<stem>_mask.png``.
* **Every file opened is logged** to ``evaluation/test_access_log.jsonl`` (append-only; each line
  carries the SHA-256 of the line before it, so a changed, removed or reordered line is detectable
  with ``verify_access_log``). A ``start`` line is written before the first file is opened and a
  ``finish`` line after the last. The data manifest holds no hashes of test files, so the log
  records each file's SHA-256 as read.
* **An audit hook** compares the files the process actually opened inside the dataset folder with
  the expected set (exactly the listed images and masks, each once, no folder listings, nothing
  from `train/`) and the result is written into the record.
* **One evaluation per version.** An existing result folder is never overwritten. If a result
  looks wrong, that is treated as a suspected bug; the fix makes a new version and every evaluated
  version stays listed.
* **No tuning.** Nothing computed here feeds back into the artifact, the threshold, the heatmap
  limits or the settings.

Pixel metrics are computed at the model's map size (256 x 256): masks are resized to that size with
nearest-neighbour, and every result says so. At full size the bottle test set would be tens of
millions of pixels. Resizing can change localisation numbers (small regions can shrink or vanish);
the record states the mask pixel counts before and after.

Defect types come from the dataset's folder names and are used for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from defect_core.artifacts import ArtifactError, load_artifact
from defect_core.calibration import Calibration
from defect_core.model import infer
from defect_core.preprocessing import PreprocessConfig, load_image, to_tensor
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from defect_ml import metrics, runinfo
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import sha256_bytes, sha256_of
from defect_ml.data.splits import read_split
from defect_ml.fit import FitConfig, FitError, git_state
from defect_ml.freeze import DEFAULT_VERSION, base_name, stage_paths

EVAL_ROOT = Path("evaluation")
LOG_NAME = "test_access_log.jsonl"
HEADLINE_BACKBONE = "wide_resnet50_2.tv_in1k"  # ADR-0002 (F3 skipped for the headline, 2026-11-10)
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 1000
FPR_LIMIT = 0.3
CURVE_POINTS = 200
RESOLUTION_NOTE = (
    "pixel metrics at 256 x 256: ground-truth masks resized to the model's map size with "
    "nearest-neighbour; not computed at the original image size"
)


class EvaluationError(Exception):
    """The evaluation is not allowed to run, or its inputs are not what was expected."""


# --- the evaluation's own path check ---------------------------------------------------------


def assert_evaluation_path(data_dir: Path, path: Path) -> None:
    """Raise ``PermissionError`` unless ``path`` is a file directly in
    ``<category>/test/<type>/`` or ``<category>/ground_truth/<type>/``.

    This is the check of the sealed evaluation. It is separate from, and does not weaken, the
    training guard ``defect_ml.data.mvtec.assert_training_path`` (which refuses exactly these
    files). The path is resolved first, so ``..`` and links that lead elsewhere are refused.
    """
    try:
        parts = path.resolve().relative_to(data_dir.resolve()).parts
    except ValueError:
        msg = f"refusing to open a file outside the dataset folder: {path}"
        raise PermissionError(msg) from None
    if len(parts) != 4 or parts[1] not in ("test", "ground_truth"):
        raise PermissionError(f"refusing to open a file outside test/ and ground_truth/: {path}")


# --- the access log --------------------------------------------------------------------------

_ZERO = "0" * 64


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class AccessLog:
    """Append-only JSON lines with a hash chain. Opening it never rewrites earlier lines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lines = 0
        self._previous = _ZERO
        if path.exists():
            problems = verify_access_log(path)
            if problems:
                raise EvaluationError(f"the access log is damaged: {problems[0]}")
            for raw in path.read_text().splitlines():
                self._previous = sha256_bytes(raw.encode())
                self.lines += 1

    def append(self, event: str, **fields: object) -> None:
        line = json.dumps(
            {"event": event, "time_utc": _now(), "prev_sha256": self._previous, **fields},
            sort_keys=True,
        )
        with self.path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
        self._previous = sha256_bytes(line.encode())
        self.lines += 1


def verify_access_log(path: Path) -> list[str]:
    """Return the problems found in an access log (empty means the chain is intact)."""
    problems: list[str] = []
    previous = _ZERO
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        try:
            entry = json.loads(raw)
        except ValueError:
            problems.append(f"line {number} is not valid JSON")
            break
        if entry.get("prev_sha256") != previous:
            problems.append(f"line {number} does not follow the line before it (log was changed)")
            break
        previous = sha256_bytes(raw.encode())
    return problems


class SealedReader:
    """Reads the listed test images and masks, and logs each file as it is read."""

    def __init__(self, data_dir: Path, category: str, log: AccessLog, context: dict) -> None:
        self.data_dir, self.category, self.log, self.context = data_dir, category, log, context
        self.allowed: set[str] = set()
        self.opened: list[str] = []

    def allow(self, relative_names: list[str]) -> None:
        self.allowed = set(relative_names)

    def read(self, relative: str, role: str) -> bytes:
        if relative not in self.allowed:
            raise PermissionError(f"{relative} is not one of the listed test files")
        path = self.data_dir / self.category / relative
        assert_evaluation_path(self.data_dir, path)
        data = path.read_bytes()
        self.opened.append(f"{self.category}/{relative}")
        self.log.append(
            "open", path=f"{self.category}/{relative}", role=role,
            sha256=sha256_bytes(data), bytes=len(data), **self.context,
        )  # fmt: skip
        return data


# --- what is evaluated -----------------------------------------------------------------------


def listed_test_files(split: dict) -> list[tuple[str, str, str | None]]:
    """(relative image path, defect type, relative mask path or None), normal images first."""
    listing = split["test_files_not_opened"]
    types = ["good"] + sorted(t for t in listing if t != "good") if "good" in listing else []
    if not types:
        raise EvaluationError("the split file lists no normal test images")
    out = []
    for defect in types:
        for rel in listing[defect]:
            mask = None
            if defect != "good":
                mask = f"ground_truth/{defect}/{Path(rel).stem}_mask.png"
            out.append((rel, defect, mask))
    return out


def prepare_mask(data: bytes, image_size: tuple[int, int], size: int) -> tuple[np.ndarray, dict]:
    """Decode a mask, check it matches its image, resize it to ``size`` with nearest-neighbour."""
    with Image.open(io.BytesIO(data)) as handle:
        mask = handle.convert("L")
    if mask.size != image_size:
        raise EvaluationError(f"mask size {mask.size} differs from its image size {image_size}")
    full = np.asarray(mask) > 0
    small = np.asarray(mask.resize((size, size), Image.NEAREST)) > 0
    return small, {"pixels_original": int(full.sum()), "pixels_resized": int(small.sum())}


def _thin(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    if len(x) > CURVE_POINTS:
        pick = np.unique(np.linspace(0, len(x) - 1, CURVE_POINTS).astype(int))
        x, y = x[pick], y[pick]
    return [float(v) for v in x], [float(v) for v in y]


def compute_metrics(
    labels: np.ndarray, defect_types: list[str], scores: np.ndarray,
    masks: np.ndarray, maps: np.ndarray, calibration: Calibration,
) -> tuple[dict, dict, dict]:  # fmt: skip
    """All reported numbers, the curves and the histograms. Pure: no file is read."""
    anomalous_types = [t for t, y in zip(defect_types, labels, strict=True) if y]
    anomalous_scores = scores[labels.astype(bool)]
    at = metrics.threshold_metrics(labels, scores, calibration.threshold)
    n_good = int((labels == 0).sum())
    image = {
        "n_normal": n_good,
        "n_anomalous": int(labels.sum()),
        "auroc": metrics.bootstrap_interval(
            labels, scores, metrics.image_auroc,
            n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        ).to_dict(),
        "aupr": metrics.bootstrap_interval(
            labels, scores, metrics.image_aupr,
            n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        ).to_dict(),
        "at_calibrated_threshold": at.to_dict(),
        "false_alarm_on_normal_test_images": {
            "fpr": at.fpr, "n_normal": n_good, "false_alarms": at.fp,
            "alpha": calibration.alpha,
            "bound_for_new_normal_images": calibration.false_alarm_bound,
        },
        "per_defect_recall": {
            name: r.to_dict()
            for name, r in metrics.per_defect_recall(
                anomalous_types, anomalous_scores, calibration.threshold
            ).items()
        },
    }  # fmt: skip
    pixel = {
        "resolution": RESOLUTION_NOTE,
        "n_images": int(masks.shape[0]),
        "n_pixels": int(masks.size),
        "n_anomalous_pixels": int(masks.sum()),
        "auroc": metrics.pixel_auroc(masks, maps),
        f"aupro_fpr_{FPR_LIMIT}": metrics.aupro(masks, maps, FPR_LIMIT),
        "highlight_level": metrics.highlight_pixel_metrics(
            masks, maps, calibration.pixel.highlight
        ).to_dict(),
    }

    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pro_fpr, pro = metrics.pro_curve(masks, maps, FPR_LIMIT)
    curves = {
        "note": "image ROC and precision-recall at every distinct score; PRO curve thinned to "
        f"at most {CURVE_POINTS} points; " + RESOLUTION_NOTE,
        "image_roc": {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]},
        "image_precision_recall": {
            "recall": [float(v) for v in recall],
            "precision": [float(v) for v in precision],
        },
        "pixel_pro": dict(zip(("fpr", "pro"), _thin(pro_fpr, pro), strict=True)),
    }
    edges = np.histogram_bin_edges(scores, bins=20)
    histograms = {
        "note": "image scores, 20 equal-width bins over the test scores",
        "bin_edges": [float(v) for v in edges],
        "normal": np.histogram(scores[labels == 0], bins=edges)[0].tolist(),
        "anomalous": np.histogram(anomalous_scores, bins=edges)[0].tolist(),
        "calibrated_threshold": calibration.threshold,
    }
    return {"image": image, "pixel": pixel}, curves, histograms


# --- one evaluation --------------------------------------------------------------------------


def _paths(config: FitConfig, version: str, eval_root: Path) -> dict[str, Path]:
    paths = stage_paths(config, version)
    model = base_name(config.run_name).removeprefix(f"{config.category}_")
    paths["result"] = eval_root / model / f"v{version}" / config.category
    paths["log"] = eval_root / LOG_NAME
    return paths


def preflight(config: FitConfig, version: str, eval_root: Path) -> dict:
    """Everything that must hold before any test file may be opened. Opens no dataset file."""
    paths = _paths(config, version, eval_root)
    if paths["result"].exists():
        raise EvaluationError(f"{paths['result']} already exists; one evaluation per version")
    loaded = load_artifact(paths["calibrated"])
    manifest = loaded.manifest
    if manifest.get("stage") != "calibrated" or loaded.calibration is None:
        raise EvaluationError(f"{paths['calibrated']} is not a calibrated artifact")
    freeze = manifest["freeze"]
    if freeze["version"] != version:  # type: ignore[index]
        raise EvaluationError(f"the artifact is version {freeze['version']}, not {version}")  # type: ignore[index]
    if loaded.model_config != config.model:
        raise EvaluationError("the settings file's model differs from the artifact's")
    used = {k: float(config.raw["calibration"][k]) for k in metrics_keys()}
    if any(float(freeze["frozen_settings"][k]) != used[k] for k in used):  # type: ignore[index]
        raise EvaluationError("the settings file's calibration differs from the frozen settings")
    reason = runinfo.conditions_ok(config.max_swap_used_mb, config.min_free_disk_gib)
    if reason:
        raise EvaluationError(f"not starting: {reason}")

    split = read_split(Path(config.split_dir) / f"{config.category}.json")
    files = listed_test_files(split)
    counts = json.loads(Path(config.manifest).read_text())["categories"][config.category]
    n_defective = sum(1 for _, _, m in files if m)
    if (len(files) - n_defective, n_defective) != (counts["test_good"], counts["test_defective"]):
        raise EvaluationError("the split file's test lists do not match the verified counts")
    if n_defective != counts["masks"]:
        raise EvaluationError("the number of listed masks does not match the verified count")
    return {"paths": paths, "loaded": loaded, "split": split, "files": files}


def metrics_keys() -> tuple[str, ...]:
    return ("alpha", "lower_percentile", "highlight_percentile", "upper_multiple")


def evaluate_one(
    config: FitConfig, plan: dict, version: str, eval_root: Path, config_path: str | Path
) -> Path:
    paths, loaded, files = plan["paths"], plan["loaded"], plan["files"]
    calibration: Calibration = loaded.calibration
    pre: PreprocessConfig = loaded.preprocess_config
    weights_sha = loaded.manifest["files"]["model.safetensors"]["sha256"]
    run_id = f"{paths['calibrated'].name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    data_dir = Path(config.data_dir).resolve()
    context = {"run_id": run_id, "weights_sha256": weights_sha}

    torch.set_num_threads(config.torch_threads)
    before = runinfo.system_state()
    watchdog = runinfo.Watchdog(config.max_swap_used_mb, config.min_free_disk_gib)
    watchdog.start()
    log = AccessLog(paths["log"])
    reader = SealedReader(data_dir, config.category, log, context)
    reader.allow([rel for rel, _, _ in files] + [m for _, _, m in files if m])
    try:
        log.append(
            "start", **context, artifact=str(paths["calibrated"]), version=version,
            frozen_at_utc=loaded.manifest["freeze"].get("frozen_at_utc"),  # type: ignore[union-attr]
            calibrated_manifest_sha256=sha256_of(paths["calibrated"] / "manifest.json"),
            n_images=len(files), n_masks=sum(1 for _, _, m in files if m),
        )  # fmt: skip
        labels, types, names, scores = [], [], [], []
        masks, maps, mask_info = [], [], {}
        with OpenAudit() as audit:
            for rel, defect, mask_rel in files:
                data = reader.read(rel, "image")
                image = load_image(data)
                score, anomaly_map = infer(loaded.model, to_tensor(image, pre)[None])
                scores.append(float(score[0]))
                maps.append(anomaly_map[0, 0].numpy())
                labels.append(0 if defect == "good" else 1)
                types.append(defect)
                names.append(f"{config.category}/{rel}")
                if mask_rel is None:
                    masks.append(np.zeros((pre.size, pre.size), dtype=bool))
                else:
                    mask, info = prepare_mask(reader.read(mask_rel, "mask"), image.size, pre.size)
                    masks.append(mask)
                    mask_info[f"{config.category}/{mask_rel}"] = info
            opened, listed = audit.inside(data_dir)
        log.append("finish", **context, files_read=len(reader.opened))
    finally:
        watchdog.stop()

    expected = sorted(reader.opened)
    opened_rel = sorted(str(p.resolve().relative_to(data_dir)) for p in opened)
    audit_ok = opened_rel == expected and not listed and len(set(expected)) == len(expected)
    only_sealed = all("/test/" in p or "/ground_truth/" in p for p in opened_rel)

    y, s = np.array(labels), np.array(scores)
    results, curves, histograms = compute_metrics(
        y, types, s, np.stack(masks), np.stack(maps), calibration
    )
    del maps, masks
    gc.collect()

    out = paths["result"]
    out.mkdir(parents=True, exist_ok=False)
    flagged = [calibration.is_anomalous(v) for v in scores]
    with (out / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "defect_type", "score", "flagged", "normal_rank"])
        for row in zip(names, labels, types, scores, flagged, scores, strict=True):
            writer.writerow([*row[:4], int(row[4]), calibration.normal_rank(row[5])])
    (out / "curves.json").write_text(json.dumps(curves, indent=1) + "\n")
    (out / "histograms.json").write_text(json.dumps(histograms, indent=1) + "\n")

    backbone = loaded.model_config.backbone
    record = {
        "step": "Phase 3 step 6: sealed test evaluation (first and only for this version)",
        "status": "evaluated on the official test set; nothing tuned on it",
        "headline": backbone == HEADLINE_BACKBONE,
        "headline_rule": "docs/adr/ADR-0002-model-and-serving.md (F3 skipped for the headline)",
        "run_id": run_id,
        "category": config.category,
        "version": version,
        "artifact": str(paths["calibrated"]),
        "weights_sha256": weights_sha,
        "calibrated_manifest_sha256": sha256_of(paths["calibrated"] / "manifest.json"),
        "model": loaded.model_config.to_dict(),
        "calibration_used": {
            "alpha": calibration.alpha,
            "n": calibration.n,
            "k": calibration.k,
            "threshold": calibration.threshold,
            "false_alarm_bound": calibration.false_alarm_bound,
            "highlight": calibration.pixel.highlight,
        },  # fmt: skip
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },  # fmt: skip
        "results": results,
        "mask_pixel_counts": mask_info,
        "data_access": {
            "images_read": len(files),
            "masks_read": len(mask_info),
            "files_opened_inside_dataset_folder": len(opened),
            "exactly_the_listed_files_each_once": audit_ok,
            "only_test_and_ground_truth_opened": only_sealed,
            "directory_listings_inside_dataset_folder": len(listed),
            "access_log": str(paths["log"]),
            "method": "sys.addaudithook: open, os.listdir, os.scandir during scoring",
        },
        "settings_file": str(config_path),
        "settings_file_sha256": sha256_of(Path(config_path)),
        "git": git_state(Path.cwd()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },  # fmt: skip
        "peak_rss_mb": runinfo.rss_peak_mb(),
        "system_before": before,
        "system_after": runinfo.system_state(),
        "watchdog": {
            "max_swap_used_mb_seen": round(watchdog.max_swap_seen, 1),
            "min_free_disk_gib_seen": round(watchdog.min_free_seen, 2),
        },
        "limitations": [
            "One category (bottle), one split seed, one calibration set of 42 images.",
            RESOLUTION_NOTE + ".",
            "The false-alarm rate on the 20 normal test images is a small-sample estimate next to "
            "the guarantee for new normal images drawn like the calibration images.",
            "Scores are anomaly scores, not probabilities. Verdicts are demo decisions.",
            "Test-file hashes are as read; the data manifest holds none for test files.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=1) + "\n")
    if not (audit_ok and only_sealed):
        raise EvaluationError(f"the data-access audit did not match the log; see {out}")
    return out


def run_batch(
    configs: list[str | Path], version: str = DEFAULT_VERSION, eval_root: Path = EVAL_ROOT
) -> list[Path]:
    """Check every settings file, then evaluate them one after the other."""
    loaded = [(FitConfig.from_yaml(c), c) for c in configs]
    names = [c.run_name for c, _ in loaded]
    if len(set(names)) != len(names):
        raise EvaluationError("the same run is listed twice")
    plans = [(c, p, preflight(c, version, eval_root)) for c, p in loaded]  # nothing opened yet
    written = []
    for config, path, plan in plans:
        written.append(evaluate_one(config, plan, version, eval_root, path))
        del plan
        gc.collect()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", help="settings files, evaluated in this order")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    try:
        for folder in run_batch(args.configs, args.version):
            print(f"written: {folder}")
    except (EvaluationError, FitError, ArtifactError, PermissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
