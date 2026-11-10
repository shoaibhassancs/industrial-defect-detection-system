"""Tests for defect_ml.benchmark on a tiny fake dataset and a RANDOM-weight artifact.

They test the wiring, the records and the data access, not real speed or memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from defect_core.artifacts import load_artifact
from defect_core.model import infer
from defect_ml import benchmark, fit, runinfo
from defect_ml.data.mvtec import TrainingImages

_FAKE = Path(__file__).resolve().parent / "test_fit.py"
_spec = importlib.util.spec_from_file_location("ml_fake_fit_for_benchmark", _FAKE)
fit_tests = importlib.util.module_from_spec(_spec)
sys.modules["ml_fake_fit_for_benchmark"] = fit_tests
_spec.loader.exec_module(fit_tests)

IMAGES = 4  # plus 3 warm-up images = 7 of the 8 fit images


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory):
    """A fitted random-weight artifact and the settings that go with the fake dataset."""
    root = tmp_path_factory.mktemp("bench")
    config = fit.FitConfig.from_dict(fit_tests.settings(root))
    assert fit.run(config, allow_random_weights=True, repo=Path.cwd()) == 0
    return root, config, root / "artifacts" / config.run_name


def args_for(root: Path, config: fit.FitConfig, artifact: Path, **extra) -> argparse.Namespace:
    values = {
        "artifact": str(artifact), "threads": [1], "images": IMAGES, "data_dir": config.data_dir,
        "split_dir": config.split_dir, "manifest": config.manifest,
        "record_dir": str(root / "bench_records"), "scratch_dir": str(root / "bench_scratch"),
        "max_swap_used_mb": 1e9, "min_free_disk_gib": 0.0, "measure": False, "result": None,
    }  # fmt: skip
    values.update(extra)
    return argparse.Namespace(**values)


def test_a_run_records_a_labelled_proxy_measurement(prepared) -> None:
    root, config, artifact = prepared
    assert benchmark.run(args_for(root, config, artifact)) == 0
    record = json.loads((root / "bench_records" / f"benchmark_{artifact.name}.json").read_text())

    assert record["label"].startswith("NATIVE macOS PROXY")
    assert "not equivalent to container CPU quotas" in record["label"]
    assert "F3" in record["label"]
    assert record["settings"]["image_group"].startswith("fit (never calib")
    assert record["backbone"] == "resnet18" and "NOT frozen" in record["status_of_artifact"]
    assert "container" in " ".join(record["limitations"]).lower()
    assert record["system_before"]["power"]
    # Swap is read with a macOS-only command; on Linux it is recorded as None, which is valid.
    assert "swap_used_mb" in record["system_after"]

    (run,) = record["runs"]
    assert run["exit_code"] == 0 and run["stopped_by_guard"] is False
    m = run["measurement"]
    assert m["threads_requested"] == 1 and m["torch_threads"] == 1  # the thread count is recorded
    assert m["timed_images"] == IMAGES and m["warmup_images"] == 3
    stages = m["stages"]
    assert list(stages) == [
        "python_started", "after_import_torch", "after_import_defect_core",
        "after_load_artifact", "after_warmup", "after_timed_run",
    ]  # fmt: skip
    assert stages["after_import_defect_core"]["rss_now_mb"] > stages["python_started"]["rss_now_mb"]
    for key in ("latency_end_to_end", "latency_preprocess_only", "latency_scoring_only"):
        s = m[key]
        assert 0 < s["min_ms"] <= s["median_ms"] <= s["p95_ms"] <= s["max_ms"]
    assert m["latency_end_to_end"]["median_ms"] >= m["latency_scoring_only"]["median_ms"]
    assert m["rss_mb"]["peak_whole_process"] >= m["rss_mb"]["after_load"] > 0
    assert m["rss_mb"]["steady_max"] > 0 and len(m["first_scores"]) == 3
    assert set(m["watchdog"]) == {"max_swap_used_mb_seen", "min_free_disk_gib_seen"}


def test_only_fit_files_are_read_and_the_scores_match_an_in_process_run(prepared) -> None:
    root, config, artifact = prepared
    benchmark.run(args_for(root, config, artifact))
    record = json.loads((root / "bench_records" / f"benchmark_{artifact.name}.json").read_text())
    m = record["runs"][0]["measurement"]
    access = m["data_access"]
    assert access["group_read"] == "fit"
    assert access["files_opened_inside_dataset_folder"] == IMAGES + 3
    assert access["exactly_the_files_used_each_once"] is True
    assert access["directory_listings_inside_dataset_folder"] == 0

    # the same images scored in this process give the same scores (the files are the first fit ones)
    images = TrainingImages.open(
        config.data_dir, "bottle", "fit", split_dir=config.split_dir, manifest_path=config.manifest
    )
    loaded = load_artifact(artifact)
    first = images.names[3 : 3 + 3]  # after the 3 warm-up images
    expected = [float(infer(loaded.model, images.load(n)[None])[0][0]) for n in first]
    assert m["first_scores"] == pytest.approx(expected, rel=1e-5)


def test_more_than_one_thread_setting_gives_one_run_each(prepared) -> None:
    root, config, artifact = prepared
    args = args_for(root, config, artifact, threads=[1, 2], images=2, record_dir=str(root / "r2"))
    assert benchmark.run(args) == 0
    record = json.loads((root / "r2" / f"benchmark_{artifact.name}.json").read_text())
    assert [r["measurement"]["torch_threads"] for r in record["runs"]] == [1, 2]


def test_not_an_artifact_and_a_tight_machine_are_refused(
    prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, artifact = prepared
    assert benchmark.run(args_for(root, config, root / "nothing")) == 2
    monkeypatch.setattr(runinfo, "conditions_ok", lambda *a: "swap used is too high")
    args = args_for(root, config, artifact, record_dir=str(root / "r3"))
    assert benchmark.run(args) == 1
    assert not (root / "r3").exists()


def test_the_percentile_helper() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert benchmark.percentile(values, 50) == 2.5
    assert benchmark.percentile(values, 0) == 1.0 and benchmark.percentile(values, 100) == 4.0
    assert benchmark.percentile([7.0], 95) == 7.0
    assert benchmark.percentile(values, 95) == pytest.approx(3.85)
    assert torch.isfinite(torch.tensor(benchmark.percentile(values, 33)))
