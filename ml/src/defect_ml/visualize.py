"""Pictures and a failure analysis for the finished sealed evaluation (Phase 3 step 7).

    uv run python -m defect_ml.visualize ml/configs/bottle_wrn50.yaml \
        ml/configs/bottle_resnet18.yaml

This is descriptive. It changes no model, threshold, metric or setting, and nothing it shows may be
used to tune them: any change would be a new version with a new sealed evaluation, listed next to
the old one.

What it does, per model:

1. Re-reads the finished evaluation (``predictions.csv``, ``metrics.json``) and the calibrated
   artifact, and refuses if they do not belong together or if ``figures/`` already exists.
2. Chooses the example tiles from ``predictions.csv`` **before** it scores anything, by a fixed rule
   (``select_examples``): all false positives, all false negatives, and for each defect type the
   lowest-scoring and the median-scoring correct detection. No choice is made by eye.
3. Re-scores all 83 test images through the **same sealed, logged path** as ``evaluate.py``
   (``SealedReader``, ``AccessLog``, ``assert_evaluation_path``). The reads are appended to the
   existing ``evaluation/test_access_log.jsonl`` with ``purpose: "visualize"``; the hash chain
   continues. The audit hook checks the opened files again.
4. Requires every recomputed score to **equal** the one in ``predictions.csv`` exactly; otherwise it
   stops, logs a ``failed`` line and writes nothing.
5. Writes ``figures/``: the example sheets as PNG (Pillow only), ``mask_size_analysis.json`` and
   ``index.json``.

Tiles are drawn with ``defect_core.render`` (the fixed-scale heatmap, inferno colours and the
outline the app uses) at the original image size and then shrunk for display. Panels: the
original; the heatmap overlay with the highlight outline in white; the original with the
ground-truth region outlined in green (normal images have none).
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from defect_core.artifacts import ArtifactError, load_artifact
from defect_core.model import infer
from defect_core.preprocessing import load_image, to_tensor
from defect_core.render import RenderSettings, outline, render_result
from PIL import Image, ImageDraw, ImageFont

from defect_ml import runinfo
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import sha256_of
from defect_ml.evaluate import (
    EVAL_ROOT,
    AccessLog,
    EvaluationError,
    SealedReader,
    _paths,
    listed_test_files,
    prepare_mask,
)
from defect_ml.fit import FitConfig, FitError
from defect_ml.freeze import DEFAULT_VERSION

PANEL = 192  # side of one panel in the sheets, pixels
CAPTION = 44  # height of the text strip under a tile
GAP = 6
GREEN = (60, 220, 90)
SELECTION_RULE = (
    "all false positives; all false negatives; for each defect type (alphabetical) the "
    "lowest-scoring and the median-scoring correct detection (median = element (n-1)//2 of the "
    "scores sorted ascending, ties broken by path); tiles ordered by group, then path"
)


class VisualizeError(Exception):
    """The pictures cannot be made, or a check against the finished evaluation failed."""


# --- the fixed selection rule ----------------------------------------------------------------


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VisualizeError(f"{path} has no rows")
    return [
        {
            "path": r["path"], "label": int(r["label"]), "type": r["defect_type"],
            "score": float(r["score"]), "flagged": int(r["flagged"]),
        }
        for r in rows
    ]  # fmt: skip


def select_examples(rows: list[dict]) -> list[dict]:
    """The example tiles, by the fixed rule. Depends only on the saved predictions."""
    chosen: list[dict] = []
    false_pos = sorted(
        (r for r in rows if r["label"] == 0 and r["flagged"]), key=lambda r: r["path"]
    )
    false_neg = sorted(
        (r for r in rows if r["label"] == 1 and not r["flagged"]), key=lambda r: r["path"]
    )
    chosen += [{**r, "group": "false_positive", "why": "false positive"} for r in false_pos]
    chosen += [{**r, "group": "false_negative", "why": "false negative"} for r in false_neg]
    for name in sorted({r["type"] for r in rows if r["label"] == 1}):
        hits = sorted(
            (r for r in rows if r["label"] == 1 and r["flagged"] and r["type"] == name),
            key=lambda r: (r["score"], r["path"]),
        )
        if not hits:
            continue
        picks = [("lowest-scoring detection", hits[0])]
        median = hits[(len(hits) - 1) // 2]
        if median is not hits[0]:
            picks.append(("median-scoring detection", median))
        chosen += [{**r, "group": "true_positive", "why": f"{name}: {why}"} for why, r in picks]
    return chosen


# --- drawing ---------------------------------------------------------------------------------


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_tile(
    image: Image.Image, anomaly_map: np.ndarray, calibration, mask: np.ndarray | None,
    lines: list[str],
) -> Image.Image:  # fmt: skip
    """One tile: original | heatmap overlay with outline | original with the true region."""
    rendered = render_result(image, anomaly_map, calibration.pixel, RenderSettings())
    array = np.asarray(image.convert("RGB")).copy()
    if mask is not None and mask.any():
        array[outline(mask, 3)] = GREEN
    panels = [np.asarray(image.convert("RGB")), rendered.overlay, array]
    tile = Image.new("RGB", (3 * PANEL + 2 * GAP, PANEL + CAPTION), (24, 24, 24))
    for index, panel in enumerate(panels):
        shrunk = Image.fromarray(panel).resize((PANEL, PANEL), Image.LANCZOS)
        tile.paste(shrunk, (index * (PANEL + GAP), 0))
    draw, font = ImageDraw.Draw(tile), _font()
    for number, text in enumerate(lines):
        draw.text((4, PANEL + 4 + 13 * number), text, fill=(235, 235, 235), font=font)
    return tile


def stack(tiles: list[Image.Image]) -> Image.Image:
    sheet = Image.new(
        "RGB", (tiles[0].width, sum(t.height + GAP for t in tiles) - GAP), (24, 24, 24)
    )
    top = 0
    for tile in tiles:
        sheet.paste(tile, (0, top))
        top += tile.height + GAP
    return sheet


# --- mask-size analysis ----------------------------------------------------------------------


def _median(values: list[float]) -> float:
    return float(np.median(values))


def mask_size_analysis(records: list[dict]) -> dict:
    """Defect size against score and detection. ``records`` are the defective test images."""
    by_type: dict[str, dict] = {}
    for name in sorted({r["type"] for r in records}):
        group = [r for r in records if r["type"] == name]
        fractions = [r["area_fraction"] for r in group]
        by_type[name] = {
            "n": len(group),
            "area_fraction_min": min(fractions),
            "area_fraction_median": _median(fractions),
            "area_fraction_max": max(fractions),
            "detected": sum(r["flagged"] for r in group),
            "score_min": min(r["score"] for r in group),
            "score_median": _median([r["score"] for r in group]),
            "median_pixel_recall_at_highlight": _median([r["pixel_recall"] for r in group]),
        }
    ordered = sorted(records, key=lambda r: (r["area_fraction"], r["path"]))
    size = len(ordered)
    cuts = [0, size // 3, 2 * size // 3, size]  # fixed rule: three groups of near-equal count
    terciles = []
    names = ("smallest third", "middle third", "largest third")
    for name, lo, hi in zip(names, cuts[:-1], cuts[1:], strict=True):
        part = ordered[lo:hi]
        if part:
            terciles.append({
                "group": name, "n": len(part),
                "area_fraction_range": [part[0]["area_fraction"], part[-1]["area_fraction"]],
                "detected": sum(r["flagged"] for r in part),
                "score_min": min(r["score"] for r in part),
                "score_median": _median([r["score"] for r in part]),
                "median_pixel_recall_at_highlight": _median([r["pixel_recall"] for r in part]),
            })  # fmt: skip
    return {
        "note": "area_fraction = ground-truth pixels / image pixels at the original size; "
        "pixel_recall = share of the ground-truth region (256 x 256 mask) above the highlight "
        "level. Descriptive only.",
        "by_defect_type": by_type,
        "by_size_third": terciles,
        "images": records,
    }


# --- one model -------------------------------------------------------------------------------


def preflight(config: FitConfig, version: str, eval_root: Path) -> dict:
    """Checks that the finished evaluation is there and consistent. Opens no dataset file."""
    paths = _paths(config, version, eval_root)
    result = paths["result"]
    for name in ("metrics.json", "predictions.csv"):
        if not (result / name).is_file():
            raise VisualizeError(f"{result / name} not found; run the evaluation first")
    if (result / "figures").exists():
        raise VisualizeError(f"{result / 'figures'} already exists; not overwritten")
    loaded = load_artifact(paths["calibrated"])
    record = json.loads((result / "metrics.json").read_text())
    weights = loaded.manifest["files"]["model.safetensors"]["sha256"]  # type: ignore[index]
    if loaded.manifest.get("stage") != "calibrated" or loaded.calibration is None:
        raise VisualizeError(f"{paths['calibrated']} is not a calibrated artifact")
    if record["weights_sha256"] != weights:
        raise VisualizeError("the evaluation was made with different weights than this artifact")
    if abs(record["calibration_used"]["threshold"] - loaded.calibration.threshold) > 0:
        raise VisualizeError("the evaluation used another threshold than this artifact holds")
    reason = runinfo.conditions_ok(config.max_swap_used_mb, config.min_free_disk_gib)
    if reason:
        raise VisualizeError(f"not starting: {reason}")
    rows = read_predictions(result / "predictions.csv")
    split = json.loads((Path(config.split_dir) / f"{config.category}.json").read_text())
    files = listed_test_files(split)
    if [f"{config.category}/{rel}" for rel, _, _ in files] != [r["path"] for r in rows]:
        raise VisualizeError("predictions.csv does not list the split file's test images in order")
    return {"paths": paths, "loaded": loaded, "record": record, "rows": rows, "files": files}


def visualize_one(config: FitConfig, plan: dict, version: str) -> Path:
    paths, loaded, rows, files = plan["paths"], plan["loaded"], plan["rows"], plan["files"]
    calibration, pre = loaded.calibration, loaded.preprocess_config
    weights = loaded.manifest["files"]["model.safetensors"]["sha256"]  # type: ignore[index]
    chosen = select_examples(rows)
    wanted = {c["path"] for c in chosen}
    run_id = f"{paths['calibrated'].name}_visualize_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    context = {"run_id": run_id, "weights_sha256": weights, "purpose": "visualize"}
    data_dir = Path(config.data_dir).resolve()

    torch.set_num_threads(config.torch_threads)
    before = runinfo.system_state()
    watchdog = runinfo.Watchdog(config.max_swap_used_mb, config.min_free_disk_gib)
    watchdog.start()
    log = AccessLog(paths["log"])
    reader = SealedReader(data_dir, config.category, log, context)
    reader.allow([rel for rel, _, _ in files] + [m for _, _, m in files if m])
    size_records: list[dict] = []
    kept: dict[str, tuple[Image.Image, np.ndarray, np.ndarray | None]] = {}
    try:
        log.append(
            "start", **context, version=version, artifact=str(paths["calibrated"]),
            predictions_sha256=sha256_of(paths["result"] / "predictions.csv"),
            n_images=len(files), n_masks=sum(1 for _, _, m in files if m),
        )  # fmt: skip
        with OpenAudit() as audit:
            for (rel, defect, mask_rel), row in zip(files, rows, strict=True):
                image = load_image(reader.read(rel, "image"))
                score, anomaly_map = infer(loaded.model, to_tensor(image, pre)[None])
                if float(score[0]) != row["score"]:
                    raise VisualizeError(
                        f"recomputed score for {row['path']} is {float(score[0])!r}, but "
                        f"predictions.csv has {row['score']!r}"
                    )
                map256 = anomaly_map[0, 0].numpy()
                small = full = None
                if mask_rel is not None:
                    data = reader.read(mask_rel, "mask")
                    small, _ = prepare_mask(data, image.size, pre.size)
                    with Image.open(io.BytesIO(data)) as handle:
                        full = np.asarray(handle.convert("L")) > 0
                    size_records.append({
                        "path": row["path"], "type": defect, "score": row["score"],
                        "flagged": row["flagged"],
                        "area_fraction": float(full.sum()) / full.size,
                        "pixel_recall": float((map256[small] > calibration.pixel.highlight).mean()),
                    })  # fmt: skip
                if row["path"] in wanted:
                    kept[row["path"]] = (image, map256, full)
            opened, listed = audit.inside(data_dir)
        log.append("finish", **context, files_read=len(reader.opened), scores_match=True)
    except BaseException as error:
        log.append("failed", **context, files_read=len(reader.opened), reason=repr(error)[:300])
        raise
    finally:
        watchdog.stop()

    expected = sorted(reader.opened)
    opened_rel = sorted(str(p.resolve().relative_to(data_dir)) for p in opened)
    audit_ok = opened_rel == expected and not listed and len(set(expected)) == len(expected)
    if not audit_ok:
        raise VisualizeError("the data-access audit did not match the log")

    figures = paths["result"] / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    sheets: dict[str, list[Image.Image]] = {}
    index_tiles = []
    for c in chosen:
        image, anomaly_map, full = kept[c["path"]]
        rendered = render_result(image, anomaly_map, calibration.pixel, RenderSettings())
        highlighted = f"{100 * rendered.area_fraction:.1f}% above highlight"
        lines = [
            f"{c['why']}: {c['path'].split('bottle/')[1]}",
            f"score {c['score']:.3f}  threshold {calibration.threshold:.3f}  {highlighted}",
        ]
        sheets.setdefault(c["group"], []).append(
            draw_tile(image, anomaly_map, calibration, full, lines)
        )
        entry = {k: c[k] for k in ("group", "why", "path", "score")}
        entry["highlighted_area_fraction"] = rendered.area_fraction
        entry["sheet"] = f"{c['group']}s.png"
        index_tiles.append(entry)
    for group, tiles in sheets.items():
        stack(tiles).save(figures / f"{group}s.png", optimize=True)

    analysis = mask_size_analysis(size_records)
    (figures / "mask_size_analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    sizes = {p.name: p.stat().st_size for p in sorted(figures.iterdir())}
    index = {
        "step": "Phase 3 step 7: figures and failure analysis (descriptive; nothing tuned)",
        "run_id": run_id,
        "version": version,
        "weights_sha256": weights,
        "selection_rule": SELECTION_RULE,
        "threshold": calibration.threshold,
        "tiles": index_tiles,
        "counts": {g: len(t) for g, t in sheets.items()},
        "scores_identical_to_predictions_csv": True,
        "data_access": {
            "images_read": len(files),
            "masks_read": len(size_records),
            "files_opened_inside_dataset_folder": len(opened),
            "exactly_the_listed_files_each_once": audit_ok,
            "directory_listings_inside_dataset_folder": len(listed),
            "access_log": str(paths["log"]),
        },  # fmt: skip
        "file_bytes": sizes,
        "peak_rss_mb": runinfo.rss_peak_mb(),
        "system_before": before,
        "system_after": runinfo.system_state(),
        "watchdog": {
            "max_swap_used_mb_seen": round(watchdog.max_swap_seen, 1),
            "min_free_disk_gib_seen": round(watchdog.min_free_seen, 2),
        },
        "limitations": [
            "Tiles are examples chosen by a fixed rule, not a random or complete sample.",
            "Heatmaps use the calibration's fixed scale; the 1.5 multiple is an unvalidated "
            "display choice.",
            "What the pictures seem to show is a reading by a person or by Claude, not a "
            "measured cause.",
        ],
    }
    (figures / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    gc.collect()
    return figures


def run_batch(
    configs: list[str | Path], version: str = DEFAULT_VERSION, eval_root: Path = EVAL_ROOT
) -> list[Path]:
    loaded = [FitConfig.from_yaml(c) for c in configs]
    plans = [(c, preflight(c, version, eval_root)) for c in loaded]  # nothing opened yet
    written = []
    for config, plan in plans:
        written.append(visualize_one(config, plan, version))
        del plan
        gc.collect()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    try:
        for folder in run_batch(args.configs, args.version):
            print(f"written: {folder}")
    except (VisualizeError, EvaluationError, FitError, ArtifactError, PermissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
