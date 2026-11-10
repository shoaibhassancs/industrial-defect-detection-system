"""Phase 1, step 4: verify the local MVTec AD folder, while keeping the test set sealed.

What it does
------------
* For ALL 15 categories: lists file NAMES only (never opens a file) and counts
  train/good, test/good, test/<defect type> and ground_truth masks. It checks that every
  defective test image has a mask with the matching name (names only) and reports
  anything unexpected.
* For ONE category (default `bottle`): fully checks the TRAINING images in `train/good`.
  Every image is decoded, its size and mode are recorded, and its SHA-256 is computed.
* It never opens or hashes anything under `test/` or `ground_truth/`. A guard raises an error
  if the code ever tries. Mask-size checks and test-file hashes are left to Phase 3, after
  the model is frozen (see PLAN.md, sealed test set rules).

Outputs
-------
* `data/data_manifest.json`: full detail (gitignored, because `data/` is). Deterministic:
  no timestamps or absolute paths, so its hash is stable.
* `experiments/dataset_verification.json`: small summary, including the manifest hash
  (meant to be committed).

Comparison with published counts is optional. By default the script reads
`scripts/data/expected_counts.json` (next to this file); pass `--expected FILE` to use another
file. Format:
    {"source": "<where the numbers come from>", "source_is_official_mvtec": true or false,
     "categories": {"bottle": {"train_good": N, "test_good": N, "test_defective": N}, ...},
     "totals": {"train_good": N, "test_good": N, "test_defective": N}}
Only those three fields can be compared: they count training images, normal test images and
DEFECTIVE TEST IMAGES. They are not mask-file counts, so a "masks" field is rejected.
If no file exists the comparison is reported as NOT RUN. Never fill the file in from memory.
The result says where the numbers came from; PASS only means "matches that source".

Usage
-----
    uv run python scripts/data/verify_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from defect_ml.data.mvtec import CATEGORIES, assert_training_path, sha256_of
from PIL import Image

SCRIPT_VERSION = "1"
EXPECTED_TOP_LEVEL = {"train", "test", "ground_truth", "license.txt", "readme.txt"}
MASK_SUFFIX = "_mask"


# --------------------------------------------------------------------------------------
# Listing (names only)
# --------------------------------------------------------------------------------------
def list_visible(path: Path) -> tuple[list[str], int]:
    """Sorted names in a folder, without hidden files such as .DS_Store. Returns (names, hidden)."""
    if not path.is_dir():
        return [], 0
    names = sorted(os.listdir(path))
    visible = [n for n in names if not n.startswith(".")]
    return visible, len(names) - len(visible)


def scan_category(cat_dir: Path) -> dict:
    """Count files and check mask pairing using file NAMES only."""
    issues: list[str] = []
    hidden = 0

    def names_in(path: Path) -> list[str]:
        nonlocal hidden
        names, h = list_visible(path)
        hidden += h
        return names

    top_level, h = list_visible(cat_dir)
    hidden += h
    for entry in top_level:
        if entry not in EXPECTED_TOP_LEVEL:
            issues.append(f"unexpected top-level entry: {entry}")
    for required in ("train", "test", "ground_truth"):
        if not (cat_dir / required).is_dir():
            issues.append(f"missing folder: {required}")

    train_good = names_in(cat_dir / "train" / "good")
    test_types = names_in(cat_dir / "test")
    mask_types = names_in(cat_dir / "ground_truth")

    non_png = [n for n in train_good if not n.lower().endswith(".png")]
    test_counts: dict[str, int] = {}
    mask_counts: dict[str, int] = {}
    pairing: dict[str, dict[str, list[str]]] = {}

    for defect in test_types:
        images = names_in(cat_dir / "test" / defect)
        test_counts[defect] = len(images)
        non_png += [f"test/{defect}/{n}" for n in images if not n.lower().endswith(".png")]
        if defect == "good":
            continue
        masks = names_in(cat_dir / "ground_truth" / defect)
        mask_counts[defect] = len(masks)
        image_stems = {Path(n).stem for n in images}
        bad_names = [n for n in masks if not Path(n).stem.endswith(MASK_SUFFIX)]
        mask_stems = {Path(n).stem.removesuffix(MASK_SUFFIX) for n in masks}
        missing = sorted(image_stems - mask_stems)
        extra = sorted(mask_stems - image_stems)
        pairing[defect] = {"images_without_mask": missing, "masks_without_image": extra}
        if missing:
            issues.append(f"{defect}: {len(missing)} test image(s) without a mask")
        if extra:
            issues.append(f"{defect}: {len(extra)} mask(s) without a test image")
        if bad_names:
            issues.append(f"{defect}: {len(bad_names)} mask name(s) not ending in '{MASK_SUFFIX}'")
    for defect in mask_types:
        if defect not in test_types:
            issues.append(f"ground_truth/{defect} has no matching test folder")
    if "good" in mask_types:
        issues.append("ground_truth/good exists (normal images should have no masks)")
    if non_png:
        issues.append(f"{len(non_png)} file(s) are not .png")

    defect_types = sorted(d for d in test_types if d != "good")
    return {
        "train_good": len(train_good),
        "test_good": test_counts.get("good", 0),
        "test_defective": sum(test_counts[d] for d in defect_types),
        "test_defect_types": {d: test_counts[d] for d in defect_types},
        "masks": sum(mask_counts.values()),
        "mask_pairing_problems": {
            d: p for d, p in pairing.items() if p["images_without_mask"] or p["masks_without_image"]
        },
        "hidden_files_ignored": hidden,
        "issues": issues,
    }, train_good  # fmt: skip


# --------------------------------------------------------------------------------------
# Deep check of TRAINING images only (the guard and sha256_of are defined once, in
# defect_ml.data.mvtec)
# --------------------------------------------------------------------------------------
def check_training_images(data_dir: Path, category: str, names: list[str]) -> dict:
    records, errors = [], []
    for name in names:
        path = data_dir / category / "train" / "good" / name
        assert_training_path(data_dir, path)
        rel = f"{category}/train/good/{name}"
        try:
            with Image.open(path) as image:
                image.verify()  # structural check
            with Image.open(path) as image:
                image.load()  # full decode
                records.append(
                    {
                        "path": rel,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_of(path),
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "format": image.format,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - any decode problem must be reported
            errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"[:200]})
    return {"split": "train/good", "images": records, "errors": errors}


def summarise_training(deep: dict) -> dict:
    images = deep["images"]
    hashes = Counter(r["sha256"] for r in images)
    size_counts = Counter((r["width"], r["height"]) for r in images)
    return {
        "images_ok": len(images),
        "images_failed": len(deep["errors"]),
        "sizes": {f"{w}x{h}": n for (w, h), n in size_counts.items()},
        "modes": dict(Counter(r["mode"] for r in images)),
        "formats": dict(Counter(r["format"] for r in images)),
        "bytes_total": sum(r["bytes"] for r in images),
        "duplicate_files": sum(n - 1 for n in hashes.values() if n > 1),
    }  # fmt: skip


# --------------------------------------------------------------------------------------
# Optional comparison with official counts
# --------------------------------------------------------------------------------------
COMPARABLE_FIELDS = ("train_good", "test_good", "test_defective")


def compare_with_expected(observed: dict, totals: dict, expected_path: Path | None) -> dict:
    """Compare observed file counts with a file of published counts (see the module docstring)."""
    if expected_path is None or not expected_path.is_file():
        return {"status": "NOT RUN", "reason": "no file with published counts was found"}
    expected = json.loads(expected_path.read_text())
    wanted = {**expected["categories"], "TOTAL": expected.get("totals", {})}
    for name, fields in wanted.items():
        unknown = sorted(set(fields) - set(COMPARABLE_FIELDS))
        if unknown:  # for example "masks": defect regions or mask files must not be compared
            raise ValueError(
                f"{name}: cannot compare field(s) {unknown}; allowed: {COMPARABLE_FIELDS}"
            )
    mismatches = []
    for name, fields in wanted.items():
        got = totals if name == "TOTAL" else observed.get(name)
        if got is None:
            mismatches.append(f"{name}: not found on disk")
            continue
        for key, value in fields.items():
            if got.get(key) != value:
                mismatches.append(f"{name}.{key}: expected {value}, found {got.get(key)}")
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "source": expected.get("source"),
        "source_is_official_mvtec": expected.get("source_is_official_mvtec"),
        "fields_compared": list(COMPARABLE_FIELDS),
        "categories_compared": sorted(expected["categories"]),
        "totals_compared": bool(expected.get("totals")),
        "mismatches": mismatches,
    }


# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"data directory not found: {data_dir}", file=sys.stderr)
        return 2

    observed: dict[str, dict] = {}
    train_names: dict[str, list[str]] = {}
    for category in CATEGORIES:
        if not (data_dir / category).is_dir():
            observed[category] = {"issues": ["category folder not found"]}
            continue
        observed[category], train_names[category] = scan_category(data_dir / category)

    deep_category = args.category
    if deep_category in train_names:
        deep = check_training_images(data_dir, deep_category, train_names[deep_category])
    else:
        missing = {"path": deep_category, "error": "category not found"}
        deep = {"split": "train/good", "images": [], "errors": [missing]}

    manifest = {
        "manifest_version": 1,
        "dataset_folder_name": data_dir.name,
        "categories": observed,
        "deep_check": {deep_category: deep},
    }
    manifest_bytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)

    totals = {
        key: sum(o.get(key, 0) for o in observed.values())
        for key in ("train_good", "test_good", "test_defective", "masks")
    }
    comparison = compare_with_expected(
        observed, totals, Path(args.expected) if args.expected else None
    )
    all_issues = {c: o["issues"] for c, o in observed.items() if o.get("issues")}
    ok = not all_issues and not deep["errors"] and comparison["status"] != "FAIL"

    summary = {
        "script": "scripts/data/verify_dataset.py",
        "script_version": SCRIPT_VERSION,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "pillow": Image.__version__,
            "platform": platform.platform(),
        },
        "download_source": args.source,
        "manifest_file": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_bytes": len(manifest_bytes),
        "files_opened": {
            "train_good_of_" + deep_category: len(deep["images"]) + len(deep["errors"]),
            "test": 0,
            "ground_truth": 0,
        },  # fmt: skip
        "totals_all_categories_by_filename_count": totals,
        "per_category_counts": {
            c: {k: o.get(k) for k in ("train_good", "test_good", "test_defective", "masks")}
            for c, o in observed.items()
        },
        "deep_check": {deep_category: summarise_training(deep)},
        "issues": all_issues,
        "official_comparison": comparison,
        "ok": ok,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    h = ("category", "train/good", "test/good", "defective", "masks", "issues")
    print(f"{h[0]:<12} {h[1]:>10} {h[2]:>10} {h[3]:>10} {h[4]:>6}  {h[5]}")
    for category, obs in observed.items():
        row = [obs.get(k, "-") for k in ("train_good", "test_good", "test_defective", "masks")]
        n_issues = len(obs.get("issues", []))
        print(f"{category:<12} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>6}  {n_issues}")
    t = totals
    print(f"{'TOTAL':<12} {t['train_good']:>10} {t['test_good']:>10} "
          f"{t['test_defective']:>10} {t['masks']:>6}")  # fmt: skip
    print(f"\ndeep check ({deep_category}/train/good): {summary['deep_check'][deep_category]}")
    print(f"official comparison: {comparison['status']}")
    for category, items in all_issues.items():
        print(f"ISSUES in {category}: {items}")
    print(f"manifest: {manifest_path} sha256={summary['manifest_sha256']}")
    print(f"summary:  {summary_path}")
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/raw/mvtec-ad")
    parser.add_argument("--category", default="bottle", choices=CATEGORIES)
    parser.add_argument("--manifest-out", default="data/data_manifest.json")
    parser.add_argument("--summary-out", default="experiments/dataset_verification.json")
    default_expected = Path(__file__).with_name("expected_counts.json")
    parser.add_argument(
        "--expected",
        default=str(default_expected) if default_expected.is_file() else None,
        help="JSON file with published counts to compare with (default: expected_counts.json)",
    )
    parser.add_argument(
        "--source",
        default=(
            "Kaggle (owner's statement, 2026-11-10); the specific Kaggle dataset page was not "
            "stated, and the files were not compared with the official MVTec archive"
        ),
        help="where the local dataset folder was downloaded from (recorded in the summary)",
    )
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
