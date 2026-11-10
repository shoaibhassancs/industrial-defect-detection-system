"""Phase 1, step 6: train-only exploration (EDA) of one category's fit and calib images.

What it does
------------
* Reads `splits/<category>.json` (made by make_splits.py) and opens ONLY the training images that
  list names: the `fit` and `calib` groups. Every path goes through the same guard as
  verify_dataset.py, which refuses anything outside `<category>/train/good`.
* Per image it measures brightness (mean and spread), the mean of each colour channel, a simple
  sharpness number (mean absolute difference between neighbouring pixels), and the file size.
* It compares the `fit` group with the `calib` group, metric by metric. The comparison is a ROUGH
  WARNING CHECK, not a proof that the groups are alike: with about 42 calib images a
  Kolmogorov-Smirnov (KS) test has little power, so "no strong difference" is weak evidence, and the
  raw p-values are not corrected for the several metrics tested (a Bonferroni level is shown).
* It flags unusual images (robust z-score above 4) by file name.
* If `data/data_manifest.json` covers the category, it also checks that each file's SHA-256 equals
  the one verify_dataset.py recorded, so the images analysed are the verified ones.

Outputs
-------
* `experiments/eda_<category>.json`: small, deterministic (no timestamp), meant to be committed.
* `data/processed/eda/<category>_*.png`: sample sheets and histograms. Not committed (`/data/` is
  ignored), so dataset images are not republished. They are thumbnails of the original images, NOT
  the model's preprocessing, which does not exist yet.

Nothing here tunes anything. Findings are observations; any change they suggest is decided in a
later phase and logged.

Usage
-----
    uv run python scripts/data/eda_train.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import make_splits  # sibling script: split validation
import numpy as np
import verify_dataset  # sibling script: the train/good guard and sha256_of
from PIL import Image
from scipy import stats

EDA_VERSION = 1
LUMA = np.array([0.299, 0.587, 0.114])  # ITU-R BT.601 luma weights
METRICS = ("luma_mean", "luma_std", "mean_r", "mean_g", "mean_b", "sharpness")
SHEET_COUNT = 16  # thumbnails per sample sheet
THUMB = 128
OUTLIER_Z = 4.0
ROUND = 4


def measure(
    path: Path, want_thumbnail: bool
) -> tuple[dict, Image.Image | None, tuple[int, int, str]]:
    with Image.open(path) as image:
        image.load()
        size_mode = (image.width, image.height, image.mode)
        rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.float64)
    luma = rgb @ LUMA
    values = {
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "mean_r": float(rgb[..., 0].mean()),
        "mean_g": float(rgb[..., 1].mean()),
        "mean_b": float(rgb[..., 2].mean()),
        "sharpness": float(
            (np.abs(np.diff(luma, axis=1)).mean() + np.abs(np.diff(luma, axis=0)).mean()) / 2
        ),
        "file_bytes": float(path.stat().st_size),
    }
    thumb = None
    if want_thumbnail:
        thumb = rgb_image.copy()
        thumb.thumbnail((THUMB, THUMB))
    return values, thumb, size_mode


def describe(values: list[float]) -> dict:
    a = np.asarray(values)
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), ROUND),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, ROUND),
        "min": round(float(a.min()), ROUND),
        "p5": round(float(p5), ROUND),
        "median": round(float(p50), ROUND),
        "p95": round(float(p95), ROUND),
        "max": round(float(a.max()), ROUND),
    }


def outliers(names: list[str], values: list[float]) -> list[dict]:
    """Robust z-score (median and MAD). Returns the files beyond OUTLIER_Z."""
    a = np.asarray(values)
    median = np.median(a)
    mad = np.median(np.abs(a - median))
    if mad == 0:
        return []
    z = 0.6745 * (a - median) / mad
    return [
        {"file": names[i], "robust_z": round(float(z[i]), 2)}
        for i in np.argsort(-np.abs(z))
        if abs(z[i]) > OUTLIER_Z
    ]


def make_sheet(thumbs: list[Image.Image], path: Path) -> None:
    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * THUMB, rows * THUMB), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * THUMB, (i // columns) * THUMB))
    sheet.save(path)


def make_histograms(per_group: dict[str, dict[str, list[float]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no window; write files only
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, metric in zip(axes, ("luma_mean", "luma_std", "sharpness"), strict=True):
        for group, color in (("fit", "tab:blue"), ("calib", "tab:orange")):
            ax.hist(per_group[group][metric], bins=20, alpha=0.55, density=True, color=color,
                    label=f"{group} (n={len(per_group[group][metric])})")  # fmt: skip
        ax.set_title(metric)
        ax.legend()
    fig.suptitle("Training images: fit vs calib (density; thumbnails/statistics of the originals)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    category = args.category
    split_path = Path(args.split_dir) / f"{category}.json"
    split = json.loads(split_path.read_text())
    problems = make_splits.validate_split(split)
    if problems:
        print(f"REFUSING to run: {split_path} is inconsistent: {problems}", file=sys.stderr)
        return 1

    manifest_hashes: dict[str, str] | None = None
    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        deep = json.loads(manifest_path.read_text()).get("deep_check", {}).get(category)
        if deep is not None:
            manifest_hashes = {r["path"]: r["sha256"] for r in deep["images"]}

    per_group: dict[str, dict[str, list[float]]] = {}
    names: dict[str, list[str]] = {}
    thumbs: dict[str, list[Image.Image]] = {}
    sizes: Counter = Counter()
    modes: Counter = Counter()
    opened = 0
    hash_mismatches: list[str] = []
    for group in ("fit", "calib"):
        per_group[group] = {m: [] for m in (*METRICS, "file_bytes")}
        names[group], thumbs[group] = [], []
        for rel in split[group]:
            path = data_dir / category / rel
            verify_dataset.assert_allowed(data_dir, path)  # sealed-test-set guard
            if manifest_hashes is not None:
                key = f"{category}/{rel}"
                if manifest_hashes.get(key) != verify_dataset.sha256_of(path):
                    hash_mismatches.append(rel)
            values, thumb, (width, height, mode) = measure(path, len(thumbs[group]) < SHEET_COUNT)
            opened += 1
            sizes[f"{width}x{height}"] += 1
            modes[mode] += 1
            names[group].append(rel)
            for metric, value in values.items():
                per_group[group][metric].append(value)
            if thumb is not None:
                thumbs[group].append(thumb)
    if hash_mismatches:
        print(f"REFUSING to report: {len(hash_mismatches)} file(s) differ from the verified "
              f"manifest: {hash_mismatches[:5]}", file=sys.stderr)  # fmt: skip
        return 1

    n_tests = len(METRICS)
    bonferroni = 0.05 / n_tests
    metrics_out: dict[str, dict] = {}
    all_names = names["fit"] + names["calib"]
    for metric in (*METRICS, "file_bytes"):
        fit_v, calib_v = per_group["fit"][metric], per_group["calib"][metric]
        ks = stats.ks_2samp(fit_v, calib_v)
        metrics_out[metric] = {
            "fit": describe(fit_v),
            "calib": describe(calib_v),
            "ks_statistic": round(float(ks.statistic), ROUND),
            "ks_p_value": round(float(ks.pvalue), ROUND),
            "outliers_all_209": outliers(all_names, fit_v + calib_v),
        }
    flagged = [m for m in METRICS if metrics_out[m]["ks_p_value"] < bonferroni]
    smallest_p = min(metrics_out[m]["ks_p_value"] for m in METRICS)

    out = {
        "eda_version": EDA_VERSION,
        "category": category,
        "split_file": str(split_path),
        "split_lists_sha256": split["train_files_sha256"],
        "images": {
            "fit": len(split["fit"]),
            "calib": len(split["calib"]),
            "sizes": dict(sizes),
            "modes": dict(modes),
        },
        "files_opened": {"train_good": opened, "any_other_folder": 0},
        "inputs_match_verified_manifest": None if manifest_hashes is None else True,
        "metrics": metrics_out,
        "fit_vs_calib_summary": {
            "tests_run": n_tests,
            "tests_run_note": "the six image metrics; file_bytes is reported for information only",
            "bonferroni_level": round(bonferroni, ROUND),
            "metrics_below_bonferroni_level": flagged,
            "smallest_ks_p_value": smallest_p,
            "reading": (
                "worth a closer look: " + ", ".join(flagged)
                if flagged
                else "no strong evidence that fit and calib differ on these metrics"
            ),
            "caution": (
                "A rough warning check, not a proof of equivalence. With "
                f"{len(split['calib'])} calib images a KS test has little power, p-values are "
                "not corrected beyond the stated Bonferroni level, and the metrics are simple "
                "global numbers, not what the model sees."
            ),
        },
        "outlier_rule": f"robust z-score (median and MAD) above {OUTLIER_Z}",
        "not_the_models_preprocessing": True,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + "\n")

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for group in ("fit", "calib"):
        make_sheet(thumbs[group], fig_dir / f"{category}_{group}_sample_sheet.png")
    make_histograms(per_group, fig_dir / f"{category}_fit_vs_calib_histograms.png")

    print(f"{category}: opened {opened} training images (fit {len(split['fit'])}, "
          f"calib {len(split['calib'])}); no other folder was read")  # fmt: skip
    print(
        f"{'metric':<11} {'fit mean':>10} {'calib mean':>11} {'KS stat':>8} {'KS p':>8}  outliers"
    )
    for metric in (*METRICS, "file_bytes"):
        m = metrics_out[metric]
        n_out = len(m["outliers_all_209"])
        print(f"{metric:<11} {m['fit']['mean']:>10} {m['calib']['mean']:>11} "
              f"{m['ks_statistic']:>8} {m['ks_p_value']:>8}  {n_out}")  # fmt: skip
    print("reading:", out["fit_vs_calib_summary"]["reading"])
    print("caution:", out["fit_vs_calib_summary"]["caution"])
    print(f"wrote {out_path} and sheets/histograms in {fig_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/raw/mvtec-ad")
    parser.add_argument("--category", default="bottle", choices=verify_dataset.CATEGORIES)
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--manifest", default="data/data_manifest.json")
    parser.add_argument("--out", default=None, help="default: experiments/eda_<category>.json")
    parser.add_argument("--figures-dir", default="data/processed/eda")
    args = parser.parse_args()
    args.out = args.out or f"experiments/eda_{args.category}.json"
    sys.exit(run(args))


if __name__ == "__main__":
    main()
