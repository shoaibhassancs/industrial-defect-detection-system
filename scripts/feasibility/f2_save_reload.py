"""Phase 0F, step F2: save and reload verification (SYNTHETIC data only).

Question: which tensors of anomalib's `PatchcoreModel` must be saved so that a model reloaded
in a *fresh process* reproduces the original outputs exactly, and which file format is the
simplest reliable one?

Formats compared:
  safetensors        state dict in a .safetensors file (candidate, no pickle)
  state_dict_pt      state dict saved with torch.save, loaded with weights_only=True
  full_module_pt     the whole module pickled with torch.save (stand-in for the format that
                     anomalib's TorchInferencer reads; it needs TRUST_REMOTE_CODE=1 and
                     weights_only=False)

This measures nothing about detection quality. Random weights and random images are used.
Batch-norm buffers are filled with non-default values so that dropping them is detectable
(pretrained weights would have such values; freshly initialised ones do not).

Usage (one command; it starts a separate Python process for every load):
    uv run python scripts/feasibility/f2_save_reload.py run --work-dir <empty scratch dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

LAYERS = ["layer2", "layer3"]
NUM_NEIGHBORS = 9
CORESET_RATIO = 0.1
IMAGE_SIZE = 256
NUM_FIT = 8
NUM_TEST = 4
THREADS = 4
BACKBONES = ["resnet18", "wide_resnet50_2"]
FORMATS = ["safetensors", "state_dict_pt", "full_module_pt"]
BN_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
OUT_PATH = Path("experiments/feasibility/f2_save_reload.json")


def build_model(backbone: str):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    return PatchcoreModel(
        layers=LAYERS, backbone=backbone, pre_trained=False, num_neighbors=NUM_NEIGHBORS
    )


def bn_snapshot(model) -> dict[str, torch.Tensor]:
    return {n: b.detach().clone() for n, b in model.named_buffers() if n.endswith(BN_SUFFIXES)}


def perturb_bn(model, seed: int) -> None:
    """Give batch-norm buffers non-default values, as pretrained weights would have."""
    g = torch.Generator().manual_seed(seed)
    for name, buf in model.named_buffers():
        if name.endswith("running_mean"):
            buf.copy_(torch.randn(buf.shape, generator=g) * 0.1)
        elif name.endswith("running_var"):
            buf.copy_(torch.rand(buf.shape, generator=g) + 0.5)


def score(model, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    return out.pred_score, out.anomaly_map


def account_bytes(state: dict[str, torch.Tensor]) -> dict[str, int]:
    groups = {"backbone_params": 0, "backbone_bn_buffers": 0, "memory_bank": 0, "blur_kernel": 0}
    for key, t in state.items():
        size = t.numel() * t.element_size()
        if key == "memory_bank":
            groups["memory_bank"] += size
        elif "blur.kernel" in key:
            groups["blur_kernel"] += size
        elif key.endswith(BN_SUFFIXES):
            groups["backbone_bn_buffers"] += size
        else:
            groups["backbone_params"] += size
    return groups


# --------------------------------------------------------------------------------------
# Step 1: build, fit on tiny synthetic data, save in every format (one process)
# --------------------------------------------------------------------------------------
def cmd_save(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    np.random.seed(0)  # anomalib's coreset sampler needs numpy seeded too (step 0.2 finding)

    model = build_model(args.backbone)
    perturb_bn(model, seed=1)
    bn_before = bn_snapshot(model)

    g = torch.Generator().manual_seed(2)
    fit_images = torch.rand(NUM_FIT, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    test_images = torch.rand(NUM_TEST, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)

    model.train()
    with torch.no_grad():
        for start in range(0, NUM_FIT, 4):
            model(fit_images[start : start + 4])
    model.subsample_embedding(sampling_ratio=CORESET_RATIO)

    bn_after = bn_snapshot(model)
    bn_changed_by_fit = any(not torch.equal(bn_before[k], bn_after[k]) for k in bn_before)

    ref_score, ref_map = score(model, test_images)
    np.save(work / "inputs.npy", test_images.numpy(), allow_pickle=False)
    np.savez(work / "ref.npz", score=ref_score.numpy(), anomaly_map=ref_map.numpy())

    state = model.state_dict()
    safe_state = {k: v.detach().clone().contiguous() for k, v in state.items()}
    save_file(safe_state, str(work / "model.safetensors"), metadata={"backbone": args.backbone})
    torch.save(state, work / "state_dict.pt")
    torch.save({"model": model}, work / "full_module.pt")

    info = {
        "backbone": args.backbone,
        "num_state_tensors": len(state),
        "bytes_by_group": account_bytes(state),
        "memory_bank_shape": list(model.memory_bank.shape),
        "memory_bank_dtype": str(model.memory_bank.dtype),
        "batch_norm_buffers_changed_by_fit_pass": bn_changed_by_fit,
        "file_sizes_bytes": {
            "safetensors": (work / "model.safetensors").stat().st_size,
            "state_dict_pt": (work / "state_dict.pt").stat().st_size,
            "full_module_pt": (work / "full_module.pt").stat().st_size,
        },
        "reference_scores": [float(x) for x in ref_score],
    }
    (work / "save_info.json").write_text(json.dumps(info, indent=2))


# --------------------------------------------------------------------------------------
# Step 2: reload in a NEW process and compare with the reference outputs
# --------------------------------------------------------------------------------------
def reload_from_state(state: dict[str, torch.Tensor], backbone: str, drop: tuple[str, ...] = ()):
    """Construct with pre_trained=False, load all tensors except the bank, assign the bank."""
    model = build_model(backbone)
    rest = {k: v for k, v in state.items() if k != "memory_bank" and not k.endswith(drop)}
    result = model.load_state_dict(rest, strict=False)
    model.memory_bank = state["memory_bank"].clone()
    return model, result


def compare(model, inputs: torch.Tensor, ref) -> dict:
    s, m = score(model, inputs)
    ref_s, ref_m = torch.from_numpy(ref["score"]), torch.from_numpy(ref["anomaly_map"])
    return {
        "scores_identical": torch.equal(s, ref_s),
        "maps_identical": torch.equal(m, ref_m),
        "max_abs_score_diff": float((s - ref_s).abs().max()),
        "max_abs_map_diff": float((m - ref_m).abs().max()),
    }


def attempt(fn) -> dict:
    """Run fn(); report either its dict result or the exception type and first message line."""
    try:
        return {"ok": True, **fn()}
    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        first_line = (str(exc).splitlines() or [""])[0][:240]
        return {"ok": False, "error_type": type(exc).__name__, "error": first_line}


def cmd_verify(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    torch.set_num_threads(THREADS)
    inputs = torch.from_numpy(np.load(work / "inputs.npy", allow_pickle=False))
    ref = np.load(work / "ref.npz", allow_pickle=False)
    res: dict = {"format": args.fmt, "backbone": args.backbone, "pid": os.getpid()}

    if args.fmt in ("safetensors", "state_dict_pt"):
        t0 = time.perf_counter()
        if args.fmt == "safetensors":
            state = load_file(str(work / "model.safetensors"))
        else:
            state = torch.load(work / "state_dict.pt", weights_only=True)
        res["load_seconds"] = round(time.perf_counter() - t0, 4)
        if args.fmt == "state_dict_pt":
            res["torch_load_weights_only"] = True  # it loaded without unpickling arbitrary code

        def naive_strict_load():
            # Plain load_state_dict of EVERYTHING, including the bank, into an empty-bank model.
            # (Plain PyTorch would reject the shape mismatch; anomalib resizes dynamic buffers.)
            model = build_model(args.backbone)
            model.load_state_dict(state, strict=True)
            return compare(model, inputs, ref)

        res["naive_strict_load_state_dict"] = attempt(naive_strict_load)

        def proper():
            model, r = reload_from_state(state, args.backbone)
            return {
                "missing_keys": list(r.missing_keys),
                "unexpected_keys": list(r.unexpected_keys),
                **compare(model, inputs, ref),
            }

        res["proper_reload"] = attempt(proper)

        if args.fmt == "safetensors":  # the subset tests need only one format

            def without_blur_kernel():
                model, r = reload_from_state(state, args.backbone, drop=("blur.kernel",))
                return {"missing_keys": list(r.missing_keys), **compare(model, inputs, ref)}

            def without_bn_buffers():
                model, r = reload_from_state(state, args.backbone, drop=BN_SUFFIXES)
                return {"missing_keys_count": len(r.missing_keys), **compare(model, inputs, ref)}

            def bank_only_into_fresh_model():
                torch.manual_seed(123)
                model = build_model(args.backbone)
                model.memory_bank = state["memory_bank"].clone()
                return compare(model, inputs, ref)

            def weights_without_bank():
                model = build_model(args.backbone)
                model.load_state_dict(
                    {k: v for k, v in state.items() if k != "memory_bank"}, strict=False
                )
                score(model, inputs)
                return {}

            res["subset_without_blur_kernel"] = attempt(without_blur_kernel)
            res["subset_without_batch_norm_buffers"] = attempt(without_bn_buffers)
            res["bank_only_into_fresh_random_model"] = attempt(bank_only_into_fresh_model)
            res["weights_without_bank"] = attempt(weights_without_bank)
    else:  # full_module_pt
        path = work / "full_module.pt"

        def safe_load():
            torch.load(path, weights_only=True)
            return {}

        res["load_with_weights_only_true"] = attempt(safe_load)

        def pickle_load():
            t0 = time.perf_counter()
            obj = torch.load(path, weights_only=False)  # noqa: S614 - own file, test only
            res["load_seconds"] = round(time.perf_counter() - t0, 4)
            return compare(obj["model"], inputs, ref)

        res["load_with_weights_only_false"] = attempt(pickle_load)

    (work / f"verify_{args.fmt}.json").write_text(json.dumps(res, indent=2))


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return "unavailable"


def cmd_run(args: argparse.Namespace) -> None:
    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        sys.exit(f"--work-dir must be empty: {root}")
    script = str(Path(__file__).resolve())

    import anomalib
    import safetensors

    results = {
        "step": "Phase 0F F2: save and reload verification",
        "label": "SYNTHETIC: random weights and random images. Says nothing about detection "
        "quality, real-data memory, or Render.",
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "anomalib": anomalib.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
            "torch_threads": THREADS,
            "swap_before": shell(["sysctl", "vm.swapusage"]),
            "power": shell(["pmset", "-g", "batt"]).splitlines()[0:1],
        },
        "settings": {
            "layers": LAYERS,
            "num_neighbors": NUM_NEIGHBORS,
            "coreset_ratio": CORESET_RATIO,
            "image_size": IMAGE_SIZE,
            "fit_images": NUM_FIT,
            "test_images": NUM_TEST,
        },
        "backbones": {},
    }

    for backbone in args.backbones:
        work = root / backbone
        work.mkdir()
        procs = [("save", ["_save"])] + [(f, ["_verify", "--fmt", f]) for f in FORMATS]
        for label, extra in procs:
            cmd = [sys.executable, script, extra[0]]
            cmd += ["--backbone", backbone, "--work-dir", str(work)] + extra[1:]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            print(f"[{backbone}] {label}: exit {proc.returncode} ({time.perf_counter() - t0:.1f}s)")
            if proc.returncode != 0:
                print(proc.stderr[-1500:])
                sys.exit(f"{label} failed for {backbone}")
        entry = json.loads((work / "save_info.json").read_text())
        entry["verify"] = {f: json.loads((work / f"verify_{f}.json").read_text()) for f in FORMATS}
        results["backbones"][backbone] = entry

    results["environment"]["swap_after"] = shell(["sysctl", "vm.swapusage"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--work-dir", required=True)
    run.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    for name in ("_save", "_verify"):
        p = sub.add_parser(name)
        p.add_argument("--backbone", required=True, choices=BACKBONES)
        p.add_argument("--work-dir", required=True)
        if name == "_verify":
            p.add_argument("--fmt", required=True, choices=FORMATS)
    args = parser.parse_args()
    {"run": cmd_run, "_save": cmd_save, "_verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
