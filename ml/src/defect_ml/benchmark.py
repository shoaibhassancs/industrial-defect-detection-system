"""Serving-style benchmark of a fitted artifact: NATIVE macOS PROXY, not a container measurement.

    uv run python -m defect_ml.benchmark artifacts/bottle_resnet18_tv_in1k_fit-v0

For one artifact it starts a fresh Python process per thread setting (1 and 2 by default) that:
imports the code, loads the artifact, runs a few warm-up inferences, then times 50 end-to-end
inferences (read the file, decode, preprocess, score) on `fit` images, one image at a time. It
records import and load time, resident memory after each stage, peak and steady-state resident
memory, and median / 95th-percentile latency.

How to read the results (they are stated in every record):

* **Native macOS arm64 on this 8 GB Mac, not a Linux container.** Render runs Linux on other CPUs
  with a memory limit; this is a proxy. The container measurements (Phase 0F step F3 and the
  container part of Phase 2 step 6) are separate and still open.
* **1 and 2 threads are rough settings.** They are not equivalent to a container CPU quota such as
  0.5 or 1 CPU.
* Only files of the `fit` group are read (through the training guard); the records include an audit
  of the files the measuring process opened. The calibration images, the official test images and
  the masks are not touched.
* The scores of the first images are recorded so later runs can be compared; nothing here says
  anything about detection quality.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from defect_ml import runinfo

LABEL = (
    "NATIVE macOS PROXY measurement (arm64, this 8 GB Mac), not a container measurement. "
    "1-2 threads are rough settings, not equivalent to container CPU quotas. "
    "F3 and the container measurements are still open."
)
DEFAULT_THREADS = (1, 2)
DEFAULT_IMAGES = 50
WARMUP = 3
SCORES_RECORDED = 3
DEFAULT_PATHS = {
    "data_dir": "data/raw/mvtec-ad",
    "split_dir": "splits",
    "manifest": "data/data_manifest.json",
    "record_dir": "experiments",
}


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rss_now_mb() -> float:
    kib = runinfo.shell(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return round(int(kib) / 1024, 1) if kib.isdigit() else float("nan")


# --------------------------------------------------------------------------------------
# The measuring process (fresh Python; one per thread setting)
# --------------------------------------------------------------------------------------
def measure(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    stages: dict[str, dict[str, object]] = {}

    def mark(name: str, **extra: object) -> None:
        stages[name] = {
            "elapsed_s": round(time.perf_counter() - started, 3),
            "rss_now_mb": rss_now_mb(),
            "rss_peak_mb": runinfo.rss_peak_mb(),
            **extra,
        }

    watchdog = runinfo.Watchdog(args.max_swap_used_mb, args.min_free_disk_gib)
    watchdog.start()
    mark("python_started")

    import torch  # noqa: PLC0415 - imported here so each import stage can be measured

    mark("after_import_torch", torch_import_seconds=round(time.perf_counter() - started, 3))
    threads = args.threads[0]
    torch.set_num_threads(threads)

    from defect_core.artifacts import load_artifact
    from defect_core.model import infer
    from defect_core.preprocessing import load_image, to_tensor

    from defect_ml.audit import OpenAudit
    from defect_ml.data.mvtec import CATEGORIES, TrainingImages, assert_training_path

    mark("after_import_defect_core")  # includes anomalib, imported by defect_core

    t = time.perf_counter()
    loaded = load_artifact(args.artifact)
    mark("after_load_artifact", seconds=round(time.perf_counter() - t, 3))

    category = loaded.manifest["provenance"]["category"]
    if category not in CATEGORIES:
        raise SystemExit(f"unexpected category {category!r} in the artifact")
    data_dir = Path(args.data_dir).resolve()
    images = TrainingImages.open(  # the `fit` group only; opens no image
        data_dir, category, "fit", split_dir=args.split_dir, manifest_path=args.manifest
    )
    names = list(images.names[: args.images + WARMUP])
    if len(names) < args.images + WARMUP:
        raise SystemExit("not enough fit images for the requested number of timings")

    def one(name: str) -> tuple[float, float, float, float]:
        path = data_dir / category / name
        assert_training_path(data_dir, path)
        t0 = time.perf_counter()
        data = path.read_bytes()
        t1 = time.perf_counter()
        tensor = to_tensor(load_image(data), loaded.preprocess_config).unsqueeze(0)
        t2 = time.perf_counter()
        score, _ = infer(loaded.model, tensor)
        t3 = time.perf_counter()
        return (t3 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, float(score[0])

    with OpenAudit() as audit:
        for name in names[:WARMUP]:
            one(name)
        mark("after_warmup")
        totals, prep, scoring, scores = [], [], [], []
        rss_samples: list[float] = []
        for i, name in enumerate(names[WARMUP:]):
            total, p, s, score = one(name)
            totals.append(total)
            prep.append(p)
            scoring.append(s)
            scores.append(score)
            if (i + 1) % 10 == 0 and i + 1 < len(names) - WARMUP:
                rss_samples.append(rss_now_mb())
        rss_samples.append(rss_now_mb())  # always one sample after the last image
        opened, listed = audit.inside(data_dir)

    expected = sorted(f"{category}/{n}" for n in names)
    opened_rel = sorted(str(p.resolve().relative_to(data_dir)) for p in opened)
    mark("after_timed_run")

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "median_ms": round(percentile(values, 50), 1),
            "p95_ms": round(percentile(values, 95), 1),
            "min_ms": round(min(values), 1),
            "max_ms": round(max(values), 1),
        }

    result = {
        "artifact": str(args.artifact),
        "threads_requested": threads,
        "torch_threads": torch.get_num_threads(),
        "timed_images": len(totals),
        "warmup_images": WARMUP,
        "stages": stages,
        "latency_end_to_end": stats(totals),
        "latency_preprocess_only": stats(prep),
        "latency_scoring_only": stats(scoring),
        "rss_mb": {
            "after_imports": stages["after_import_defect_core"]["rss_now_mb"],
            "after_load": stages["after_load_artifact"]["rss_now_mb"],
            "steady_samples_every_10_images_and_at_end": rss_samples,
            "steady_max": max(rss_samples),
            "peak_whole_process": runinfo.rss_peak_mb(),
        },
        "first_scores": scores[:SCORES_RECORDED],
        "data_access": {
            "group_read": "fit",
            "files_opened_inside_dataset_folder": len(opened),
            "exactly_the_files_used_each_once": opened_rel == expected,
            "directory_listings_inside_dataset_folder": len(listed),
        },
        "watchdog": {
            "max_swap_used_mb_seen": round(watchdog.max_swap_seen, 1),
            "min_free_disk_gib_seen": round(watchdog.min_free_seen, 2),
        },
    }
    watchdog.stop()
    Path(args.result).write_text(json.dumps(result, indent=1))


# --------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact)
    if not (artifact / "manifest.json").is_file():
        print(f"ERROR: {artifact} is not an artifact folder", file=sys.stderr)
        return 2
    record_path = Path(args.record_dir) / f"benchmark_{artifact.name}.json"
    reason = runinfo.conditions_ok(args.max_swap_used_mb, args.min_free_disk_gib)
    if reason:
        print(f"NOT STARTING: {reason}", file=sys.stderr)
        return 1

    manifest = json.loads((artifact / "manifest.json").read_text())
    before = runinfo.system_state()
    runs, code = [], 0
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    for threads in args.threads:
        result_path = scratch / f"{artifact.name}_threads{threads}.json"
        result_path.unlink(missing_ok=True)
        cmd = [
            sys.executable, "-m", "defect_ml.benchmark", str(artifact), "--measure",
            "--threads", str(threads), "--images", str(args.images),
            "--data-dir", args.data_dir, "--split-dir", args.split_dir,
            "--manifest", args.manifest, "--result", str(result_path),
            "--max-swap-used-mb", str(args.max_swap_used_mb),
            "--min-free-disk-gib", str(args.min_free_disk_gib),
        ]  # fmt: skip
        state = runinfo.system_state()
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        entry = {
            "threads_requested": threads,
            "exit_code": proc.returncode,
            "stopped_by_guard": proc.returncode == runinfo.GUARD_EXIT_CODE,
            "process_wall_seconds": round(time.perf_counter() - t0, 1),
            "system_before_this_process": state,
        }
        if result_path.exists():
            entry["measurement"] = json.loads(result_path.read_text())
        else:
            entry["stderr_tail"] = proc.stderr[-1500:]
        runs.append(entry)
        print(f"{artifact.name}: threads={threads} exit {proc.returncode}")
        if proc.returncode != 0:
            code = 1
            break

    record = {
        "step": "Phase 2 step 6 (native proxy): serving-style benchmark of a fitted artifact",
        "label": LABEL,
        "artifact": str(artifact),
        "artifact_files_bytes": {p.name: p.stat().st_size for p in sorted(artifact.iterdir())},
        "backbone": manifest["model"]["backbone"],
        "weights": manifest["provenance"]["weights"],
        "status_of_artifact": manifest["provenance"]["status"],
        "settings": {
            "timed_images_per_run": args.images,
            "warmup_images": WARMUP,
            "thread_settings": list(args.threads),
            "one_image_at_a_time": True,
            "per_image_work": "read file, decode, preprocess, score (no rendering, no network)",
            "image_group": "fit (never calib, test or masks)",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": manifest["libraries"],
        },
        "system_before": before,
        "system_after": runinfo.system_state(),
        "runs": runs,
        "limitations": [
            "Native macOS proxy: a Linux container with a CPU quota and a memory limit (Render) "
            "will differ in speed, memory use and libraries.",
            "1 and 2 threads are rough settings, not equivalents of container CPU quotas.",
            "Other apps were open and the Mac was under memory pressure; see system_before/after.",
            "Resident memory excludes memory the OS compressed or swapped out.",
            "Single measurement series, no repeats, no variance estimate.",
            "Timing includes file reading, decoding and preprocessing but not rendering, "
            "encoding or any web framework.",
        ],
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"record: {record_path}")
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", help="artifact folder, for example artifacts/<run_name>")
    parser.add_argument("--threads", type=int, nargs="+", default=list(DEFAULT_THREADS))
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    parser.add_argument("--data-dir", default=DEFAULT_PATHS["data_dir"])
    parser.add_argument("--split-dir", default=DEFAULT_PATHS["split_dir"])
    parser.add_argument("--manifest", default=DEFAULT_PATHS["manifest"])
    parser.add_argument("--record-dir", default=DEFAULT_PATHS["record_dir"])
    parser.add_argument("--scratch-dir", default="data/processed/benchmark")
    parser.add_argument("--max-swap-used-mb", type=float, default=5000.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=7.0)
    # used internally: the driver starts itself with --measure to get a fresh process per setting
    parser.add_argument("--measure", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.measure:
        measure(args)
    else:
        sys.exit(run(args))


if __name__ == "__main__":
    main()
