"""Phase 0F, step F1: resource measurement on this Mac (SYNTHETIC data only).

Question: what does fitting and running PatchCore cost in time and memory on an 8 GB MacBook Air,
for `wide_resnet50_2` and `resnet18` (layers layer2+layer3, coreset ratio 0.1)?

Random weights (`pre_trained=False`) and random 256x256 images are used. Memory and time are
assumed not to depend on weight values or image content ([Unverified]; Phase 2 re-measures on real
data). Nothing here measures detection quality, and nothing here selects a backbone, a serving
route or a hosting plan.

Each backbone runs in its own Python process (so peak memory is per run). The run is CPU-only. A
watchdog thread inside the measuring process stops the run if swap use or free disk reaches the
limits below, so a heavy run cannot fill the disk.

Usage:
    uv run python scripts/feasibility/f1_resources.py run --work-dir <empty scratch dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

BACKBONES = ["resnet18", "wide_resnet50_2"]  # small first: it checks the script cheaply
IMAGE_SIZE = 256
NUM_FIT = 167  # size of the bottle `fit` split (PLAN.md Phase 0F rules)
NUM_TEST = 30  # single-image inferences timed after warm-up
WARMUP = 3
BATCH_SIZE = 4
THREADS = 4  # the M1 has 4 performance cores; same as F2
SEED = 0
CORESET_RATIO = 0.1
DEFAULT_OUT_DIR = Path("experiments/feasibility")

# Guard limits: the run is stopped if either is reached.
MAX_SWAP_USED_MB = 5000.0
MIN_FREE_DISK_GIB = 7.0
DATA_VOLUME = "/System/Volumes/Data"
GUARD_EXIT_CODE = 3


# --------------------------------------------------------------------------------------
# System state (macOS commands; every value is read-only and recorded as plain text/numbers)
# --------------------------------------------------------------------------------------
def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return "unavailable"


def swap_used_mb() -> float | None:
    # Output looks like: "total = 3072.00M  used = 1919.88M  free = 1152.12M  (encrypted)"
    text = shell(["sysctl", "-n", "vm.swapusage"])
    try:
        return float(text.split("used =")[1].split()[0].rstrip("M"))
    except (IndexError, ValueError):
        return None


def free_disk_gib() -> float:
    return shutil.disk_usage(DATA_VOLUME).free / 2**30


def vm_counters() -> dict[str, int]:
    wanted = ("Pageins", "Pageouts", "Swapins", "Swapouts")
    out: dict[str, int] = {}
    for line in shell(["vm_stat"]).splitlines():
        key, _, value = line.partition(":")
        if key.strip() in wanted:
            out[key.strip()] = int(value.strip().rstrip("."))
    return out


def system_state() -> dict[str, object]:
    free_line = [x for x in shell(["memory_pressure"]).splitlines() if "free percentage" in x]
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "swap_used_mb": swap_used_mb(),
        "swap_usage_raw": shell(["sysctl", "-n", "vm.swapusage"]),
        "free_disk_gib": round(free_disk_gib(), 2),
        "load_average": [round(x, 2) for x in os.getloadavg()],
        "memory_free_percentage": free_line[0].strip() if free_line else "unavailable",
        "power": shell(["pmset", "-g", "batt"]).splitlines()[:2],
        "thermal": shell(["pmset", "-g", "therm"]).splitlines()[:3],
        "vm_counters": vm_counters(),
    }


def rss_now_mb() -> float:
    kib = shell(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return round(int(kib) / 1024, 1) if kib.isdigit() else float("nan")


def rss_peak_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)  # bytes on macOS


# --------------------------------------------------------------------------------------
# The measuring process (one per backbone)
# --------------------------------------------------------------------------------------
class Watchdog(threading.Thread):
    """Stops the process if swap or disk pressure becomes concerning; records the extremes."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.max_swap_mb = swap_used_mb() or 0.0
        self.min_free_gib = free_disk_gib()

    def run(self) -> None:
        while True:
            swap, free = swap_used_mb(), free_disk_gib()
            if swap is not None:
                self.max_swap_mb = max(self.max_swap_mb, swap)
            self.min_free_gib = min(self.min_free_gib, free)
            if (swap or 0) >= MAX_SWAP_USED_MB or free <= MIN_FREE_DISK_GIB:
                msg = f"GUARD: swap used {swap} MB, free disk {free:.1f} GiB; stopping the run\n"
                sys.stderr.write(msg)
                sys.stderr.flush()
                os._exit(GUARD_EXIT_CODE)
            time.sleep(2.0)


def measure(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()
    torch.set_num_threads(args.threads)
    stages: dict[str, dict[str, object]] = {}

    def mark(name: str, **extra: object) -> None:
        stages[name] = {
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "rss_now_mb": rss_now_mb(),
            "rss_peak_mb": rss_peak_mb(),
            **extra,
        }

    watchdog = Watchdog()
    watchdog.start()
    mark("process_start_after_torch_import")

    from defect_core.artifacts import load_artifact, save_artifact
    from defect_core.model import ModelConfig, build_model, infer
    from defect_core.preprocessing import DEFAULT_CONFIG

    mark("after_defect_core_import")

    config = ModelConfig(backbone=args.backbone, coreset_ratio=CORESET_RATIO)
    torch.manual_seed(SEED)
    np.random.seed(SEED)  # the coreset sampler needs numpy seeded too (step 0.2 finding)
    model = build_model(config, pre_trained=False)
    params = sum(p.numel() for p in model.parameters())
    mark("after_build", parameters=params)

    generator = torch.Generator().manual_seed(SEED + 1)

    def images(n: int) -> torch.Tensor:
        return torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)

    # Stage 1: feature collection (train-mode pass), as fit_memory_bank does it
    model.train()
    t = time.perf_counter()
    with torch.no_grad():
        for start in range(0, args.fit_images, BATCH_SIZE):
            model(images(min(BATCH_SIZE, args.fit_images - start)))
    features_s = time.perf_counter() - t
    store = model.embedding_store
    patches = int(sum(e.shape[0] for e in store))
    width = int(store[0].shape[1])
    store_mb = round(sum(e.numel() * e.element_size() for e in store) / 2**20, 1)
    mark(
        "after_features",
        seconds=round(features_s, 2),
        patch_features=patches,
        feature_width=width,
        embedding_store_mb=store_mb,
    )

    # Stage 2: coreset subsampling
    t = time.perf_counter()
    model.subsample_embedding(sampling_ratio=CORESET_RATIO)
    coreset_s = time.perf_counter() - t
    model.eval()
    bank = model.memory_bank
    mark(
        "after_coreset",
        seconds=round(coreset_s, 2),
        memory_bank_shape=list(bank.shape),
        memory_bank_dtype=str(bank.dtype),
        memory_bank_mb=round(bank.numel() * bank.element_size() / 2**20, 1),
    )

    # Stage 3: per-image inference latency (batch of 1, after warm-up)
    for _ in range(WARMUP):
        infer(model, images(1))
    times_ms = []
    for _ in range(args.test_images):
        x = images(1)
        t = time.perf_counter()
        infer(model, x)
        times_ms.append((time.perf_counter() - t) * 1000)
    latency = {
        "images": len(times_ms),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 1),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 1),
        "min_ms": round(min(times_ms), 1),
        "max_ms": round(max(times_ms), 1),
    }
    mark("after_inference", latency=latency)

    # Stage 4: save the artifact, reload it, compare outputs (full-size check of artifacts.py)
    art_dir = Path(args.work_dir) / "artifact"
    t = time.perf_counter()
    save_artifact(model, config, DEFAULT_CONFIG, art_dir, provenance={"synthetic": True})
    save_s = time.perf_counter() - t
    sizes = {p.name: p.stat().st_size for p in art_dir.iterdir()}
    mark("after_save", seconds=round(save_s, 2), file_bytes=sizes)

    t = time.perf_counter()
    loaded = load_artifact(art_dir)
    load_s = time.perf_counter() - t
    probe = images(4)
    ref_score, ref_map = infer(model, probe)
    new_score, new_map = infer(loaded.model, probe)
    identical = bool(torch.equal(ref_score, new_score) and torch.equal(ref_map, new_map))
    mark("after_reload", seconds=round(load_s, 2), reload_outputs_bit_identical=identical)
    shutil.rmtree(art_dir)

    result = {
        "backbone": args.backbone,
        "process_pid": os.getpid(),
        "torch_threads": torch.get_num_threads(),
        "stages": stages,
        "fit_total_s": round(features_s + coreset_s, 2),
        "peak_rss_mb_whole_run": rss_peak_mb(),
        "watchdog": {
            "max_swap_used_mb_seen": round(watchdog.max_swap_mb, 1),
            "min_free_disk_gib_seen": round(watchdog.min_free_gib, 2),
        },
    }
    Path(args.result).write_text(json.dumps(result, indent=2))


# --------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        sys.exit(f"--work-dir must be empty: {work}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import anomalib

    script = str(Path(__file__).resolve())
    for backbone in args.backbones:
        run_dir = work / backbone
        run_dir.mkdir()
        result_path = run_dir / "result.json"
        before = system_state()
        print(
            f"[{backbone}] start. swap {before['swap_used_mb']} MB, "
            f"disk {before['free_disk_gib']} GiB"
        )
        cmd = [
            sys.executable, script, "_measure",
            "--backbone", backbone, "--work-dir", str(run_dir), "--result", str(result_path),
            "--threads", str(args.threads), "--fit-images", str(args.fit_images),
            "--test-images", str(args.test_images),
        ]  # fmt: skip
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - t0
        after = system_state()
        print(
            f"[{backbone}] exit {proc.returncode} in {wall:.1f}s. swap {after['swap_used_mb']} MB"
        )

        record: dict[str, object] = {
            "step": "Phase 0F F1: resource measurement",
            "label": "SYNTHETIC: random weights and random images, CPU. Says nothing about "
            "detection quality or real-data memory, and selects no backbone or hosting plan.",
            "exit_code": proc.returncode,
            "stopped_by_guard": proc.returncode == GUARD_EXIT_CODE,
            "guard_limits": {
                "max_swap_used_mb": MAX_SWAP_USED_MB,
                "min_free_disk_gib": MIN_FREE_DISK_GIB,
            },
            "wall_time_s_whole_process": round(wall, 1),
            "settings": {
                "layers": ["layer2", "layer3"],
                "coreset_ratio": CORESET_RATIO,
                "image_size": IMAGE_SIZE,
                "fit_images": args.fit_images,
                "batch_size": BATCH_SIZE,
                "single_image_timings": args.test_images,
                "warmup_inferences": WARMUP,
                "seed": SEED,
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "anomalib": anomalib.__version__,
                "numpy": np.__version__,
                "requested_torch_threads": args.threads,
            },
            "system_before": before,
            "system_after": after,
            "limitations": [
                "Timings include any slowdown from other apps and from swapping; see "
                "system_before/system_after and the watchdog extremes.",
                "Peak RSS (ru_maxrss) counts resident memory only; memory the OS compressed or "
                "swapped out is not included, so under memory pressure it can understate demand.",
                "The Mac has no fan; long runs may be heat-throttled (thermal notes recorded).",
                "Random weights and noise images; real data may differ ([Unverified]).",
                "Not a Render measurement: different CPU, memory limit and Linux.",
            ],
        }
        if result_path.exists():
            record["measurement"] = json.loads(result_path.read_text())
        else:
            record["stderr_tail"] = proc.stderr[-1500:]
        path = out_dir / f"f1_{backbone}.json"
        path.write_text(json.dumps(record, indent=2))
        print(f"wrote {path}")
        if proc.returncode != 0:
            sys.exit(f"{backbone}: stopped (exit {proc.returncode}); later backbones not run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--work-dir", required=True)
    run.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    run.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    run.add_argument("--threads", type=int, default=THREADS)
    run.add_argument("--fit-images", type=int, default=NUM_FIT)
    run.add_argument("--test-images", type=int, default=NUM_TEST)
    inner = sub.add_parser("_measure")
    inner.add_argument("--backbone", required=True)
    inner.add_argument("--work-dir", required=True)
    inner.add_argument("--result", required=True)
    inner.add_argument("--threads", type=int, default=THREADS)
    inner.add_argument("--fit-images", type=int, default=NUM_FIT)
    inner.add_argument("--test-images", type=int, default=NUM_TEST)
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
