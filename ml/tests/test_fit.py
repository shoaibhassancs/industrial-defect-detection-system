"""Tests for defect_ml.fit, defect_ml.audit and the run settings (Phase 2 step 5).

A tiny fake dataset and RANDOM weights are used (the network is blocked): they test the wiring, the
refusals and the records, not the quality of a real fit. The real fits are separate runs.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import torch
import yaml
from defect_core.artifacts import load_artifact
from defect_core.model import ModelConfig, build_model
from defect_ml import fit, runinfo
from defect_ml.audit import OpenAudit
from defect_ml.fit import FitConfig, FitError, compare_with_model, weights_record
from safetensors.torch import save_file

_FAKE = Path(__file__).resolve().parent / "test_mvtec.py"
_spec = importlib.util.spec_from_file_location("ml_fake_dataset_fit", _FAKE)
fake = importlib.util.module_from_spec(_spec)
sys.modules["ml_fake_dataset_fit"] = fake
_spec.loader.exec_module(fake)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


CALIBRATION = {
    "alpha": 0.05,
    "lower_percentile": 1,
    "highlight_percentile": 99.5,
    "upper_multiple": 1.5,
    "upper_multiple_status": "unvalidated display choice",
}


def settings(root: Path, **model_overrides: object) -> dict:
    data, split_dir, manifest = fake.make_dataset(root)
    model = {"backbone": "resnet18", "layers": ["layer2", "layer3"], "num_neighbors": 9,
             "coreset_ratio": 0.1}  # fmt: skip
    model.update(model_overrides)
    return {
        "run_name": "test_run",
        "category": "bottle",
        "model": model,
        "weights": {"pretrained": False},
        "fit": {"seed": 0, "batch_size": 3, "torch_threads": 2},
        "paths": {
            "data_dir": str(data),
            "split_dir": str(split_dir),
            "manifest": str(manifest),
            "artifact_dir": str(root / "artifacts"),
            "record_dir": str(root / "records"),
        },  # fmt: skip
        "guard": {"max_swap_used_mb": 1e9, "min_free_disk_gib": 0},
        "calibration": CALIBRATION,
    }


# --- a complete run ------------------------------------------------------------------------


def test_a_run_fits_only_the_fit_group_and_records_everything(tmp_path: Path) -> None:
    config = FitConfig.from_dict(settings(tmp_path))
    assert fit.run(config, allow_random_weights=True, repo=REPO) == 0

    record = json.loads((tmp_path / "records" / "fit_test_run.json").read_text())
    access = record["data_access"]
    assert access["group_read"] == "fit" and access["images_loaded"] == 8
    assert access["files_opened_inside_dataset_folder"] == 8
    assert access["exactly_the_listed_fit_files_each_once"] is True
    assert access["directory_listings_inside_dataset_folder"] == 0
    assert "NOT frozen and NOT calibrated" in record["status"]
    assert record["device"] == "cpu" and record["torch_threads"] == 2
    assert record["reload_check"]["outputs_bit_identical"] is True
    assert record["memory_bank"]["dtype"] == "torch.float32"
    assert set(record["timings_seconds"]) == {"features", "coreset", "fit_total"}
    assert record["weights"]["pretrained"] is False and "RANDOM" in record["weights"]["note"]
    assert record["peak_rss_mb"] > 0 and record["system_before"]["free_disk_gib"] > 0
    assert set(record["watchdog"]) == {"max_swap_used_mb_seen", "min_free_disk_gib_seen", "limits"}

    # the calibration images were not read: only the listed fit files were opened
    split = json.loads(Path(config.split_dir, "bottle.json").read_text())
    assert access["images_loaded"] == len(split["fit"]) != len(split["fit"] + split["calib"])

    # the saved artifact loads and carries the provenance
    loaded = load_artifact(tmp_path / "artifacts" / "test_run")
    prov = loaded.manifest["provenance"]
    assert prov["group"] == "fit" and prov["seed"] == 0 and prov["fit_images"] == 8
    assert prov["split_lists_sha256"] == split["train_files_sha256"]
    manifest_hash = hashlib.sha256(Path(config.manifest).read_bytes()).hexdigest()
    assert prov["data_manifest_sha256"] == manifest_hash
    assert "NOT frozen" in prov["status"] and set(prov["git"]) == {"commit", "uncommitted_changes"}
    assert loaded.model_config.backbone == "resnet18"


def test_random_weights_need_an_explicit_flag(tmp_path: Path) -> None:
    config = FitConfig.from_dict(settings(tmp_path))
    assert fit.run(config) == 1
    assert not (tmp_path / "artifacts").exists() and not (tmp_path / "records").exists()


def test_an_existing_artifact_is_never_overwritten(tmp_path: Path) -> None:
    config = FitConfig.from_dict(settings(tmp_path))
    assert fit.run(config, allow_random_weights=True, repo=REPO) == 0
    before = (tmp_path / "artifacts" / "test_run" / "model.safetensors").read_bytes()
    assert fit.run(config, allow_random_weights=True, repo=REPO) == 1
    assert (tmp_path / "artifacts" / "test_run" / "model.safetensors").read_bytes() == before


def test_a_run_does_not_start_when_the_machine_is_already_too_tight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FitConfig.from_dict(settings(tmp_path))
    monkeypatch.setattr(runinfo, "swap_used_mb", lambda: 99999.0)
    tight = {"max_swap_used_mb": 5000, "min_free_disk_gib": 0}
    config = FitConfig.from_dict({**config.raw, "guard": tight})
    with OpenAudit() as audit:
        assert fit.run(config, allow_random_weights=True) == 1
    assert audit.inside(config.data_dir) == ([], [])  # nothing under the dataset was touched
    assert not (tmp_path / "artifacts").exists()


# --- the settings --------------------------------------------------------------------------


def test_untagged_pretrained_backbones_are_refused(tmp_path: Path) -> None:
    raw = settings(tmp_path)
    raw["weights"] = {"pretrained": True}
    with pytest.raises(FitError, match="weight tag"):
        FitConfig.from_dict(raw)
    raw["model"]["backbone"] = "resnet18.tv_in1k"
    assert FitConfig.from_dict(raw).model.backbone == "resnet18.tv_in1k"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("guard"),
        lambda d: d.pop("calibration"),
        lambda d: d["calibration"].update(alpha=0),
        lambda d: d["calibration"].update(alpha=1),
        lambda d: d["calibration"].update(alpha="0.05"),
        lambda d: d["calibration"].update(lower_percentile=99.5),
        lambda d: d["calibration"].update(highlight_percentile=101),
        lambda d: d["calibration"].update(upper_multiple=0),
        lambda d: d["calibration"].update(upper_multiple_status=""),
        lambda d: d["calibration"].update(extra=1),
        lambda d: d["model"].update(extra=1),
        lambda d: d["model"].pop("layers"),
        lambda d: d["fit"].update(seed=-1),
        lambda d: d["fit"].update(batch_size=0),
        lambda d: d["fit"].update(seed=True),
        lambda d: d["weights"].update(pretrained="yes"),
        lambda d: d.update(category="bottles"),
        lambda d: d["paths"].update(data_dir=""),
        lambda d: d["model"].update(layers=[]),
        lambda d: d["model"].update(coreset_ratio="x"),
    ],
)
def test_bad_settings_are_refused(tmp_path: Path, change) -> None:
    raw = copy.deepcopy(settings(tmp_path))
    change(raw)
    with pytest.raises(FitError):
        FitConfig.from_dict(raw)


def test_settings_files_are_read_strictly(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(settings(tmp_path)))
    assert FitConfig.from_yaml(path).run_name == "test_run"
    path.write_text("a: [unclosed")
    with pytest.raises(FitError):
        FitConfig.from_yaml(path)
    with pytest.raises(FitError):
        FitConfig.from_yaml(tmp_path / "missing.yaml")


def test_the_committed_settings_pin_the_torchvision_weights_and_fit_only() -> None:
    expected = {
        "bottle_resnet18.yaml": "resnet18.tv_in1k",
        "bottle_wrn50.yaml": "wide_resnet50_2.tv_in1k",
    }
    for name, backbone in expected.items():
        config = FitConfig.from_yaml(REPO / "ml" / "configs" / name)
        assert config.pretrained is True and config.model.backbone == backbone
        assert config.model.layers == ("layer2", "layer3") and config.model.coreset_ratio == 0.1
        assert config.category == "bottle" and config.artifact_dir == "artifacts"
        c = config
        assert (c.alpha, c.lower_percentile, c.highlight_percentile, c.upper_multiple) == (
            0.05, 1.0, 99.5, 1.5)  # fmt: skip
        assert "unvalidated" in c.raw["calibration"]["upper_multiple_status"]
    # no setting can name another data group: the run always reads the `fit` group
    assert "group" not in json.dumps(FitConfig.from_yaml(REPO / "ml/configs/bottle_wrn50.yaml").raw)


# --- pretrained weight records -------------------------------------------------------------


def _backbone_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    prefix = fit.BACKBONE_PREFIX
    state = model.state_dict()
    return {k[len(prefix) :]: v.clone() for k, v in state.items() if k.startswith(prefix)}


def test_comparison_with_a_checkpoint_counts_matches_and_differences() -> None:
    model = build_model(ModelConfig(backbone="resnet18"), pre_trained=False)
    tensors = _backbone_tensors(model)
    matched, differing = compare_with_model(model, tensors)
    assert matched == len(tensors) and differing == 0
    victim = next(k for k in tensors if k.endswith("conv1.weight"))
    tensors[victim] = tensors[victim] + 1.0
    tensors["fc.weight"] = torch.zeros(3)  # a tensor the model does not keep is ignored
    assert compare_with_model(model, tensors) == (matched - 1, 1)


def test_the_weights_record_names_the_file_and_proves_it_is_the_one_in_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    model = build_model(ModelConfig(backbone="resnet18.tv_in1k"), pre_trained=False)
    # the layout of the real cache: the snapshot file is a link into a shared blob store
    repo = tmp_path / "models--timm--resnet18.tv_in1k"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (tmp_path / "blobs" / "1e").mkdir(parents=True)
    blob = tmp_path / "blobs" / "1e" / "blobname"
    save_file(_backbone_tensors(model), str(blob))
    checkpoint = snapshot / "model.safetensors"
    checkpoint.symlink_to(blob)

    def fake_download(repo_id: str, filename: str, **kwargs: object) -> str:
        assert repo_id == "timm/resnet18.tv_in1k" and kwargs == {"local_files_only": True}
        if filename != "model.safetensors":
            raise FileNotFoundError(filename)
        return str(checkpoint)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    record = weights_record("resnet18.tv_in1k", model)
    assert record["hf_hub_id"] == "timm/resnet18.tv_in1k" and record["file"] == "model.safetensors"
    assert record["cache_revision"] == "abc123"  # not the blob folder's name
    assert record["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record["bytes"] == checkpoint.stat().st_size
    assert record["original_source_url"].startswith("https://download.pytorch.org/models/")
    assert record["tensors_differing"] == 0 and record["tensors_compared_with_model"] > 0
    assert "separate from the terms of the ImageNet" in record["terms"]

    # a different model (changed weights) does not match the cached file: refused
    with torch.no_grad():
        next(p for p in model.parameters()).add_(1.0)
    with pytest.raises(FitError, match="matching the model"):
        weights_record("resnet18.tv_in1k", model)


def test_the_weights_record_refuses_when_nothing_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def not_cached(*args: object, **kwargs: object) -> str:
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", not_cached)
    model = build_model(ModelConfig(backbone="resnet18.tv_in1k"), pre_trained=False)
    with pytest.raises(FitError, match="cached checkpoint"):
        weights_record("resnet18.tv_in1k", model)


# --- the audit helper ----------------------------------------------------------------------


def test_the_audit_records_opens_and_listings_only_while_active(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (tmp_path / "a.txt").read_text()  # before: not recorded
    with OpenAudit() as audit:
        (tmp_path / "a.txt").read_text()
        list(tmp_path.iterdir())
        (other / "b.txt").write_text("y")
    (tmp_path / "a.txt").read_text()  # after: not recorded
    opened, listed = audit.inside(tmp_path)
    assert [p.name for p in opened].count("a.txt") == 1 and "b.txt" in [p.name for p in opened]
    assert any(str(tmp_path) in p for p in listed)
    only_other, _ = audit.inside(other)
    assert [p.name for p in only_other] == ["b.txt"]
