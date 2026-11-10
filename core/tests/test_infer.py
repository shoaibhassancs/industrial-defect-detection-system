"""Tests for defect_core.infer (Phase 4.5): the shared inspection function and the command line.

Synthetic images only (no dataset file is read) and a random-weight model: the point is that the
CLI verifies the artifact, reports what the calibration says, writes only what was asked for, and
fails clearly. Detection quality is measured elsewhere (docs/evaluation.md).
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from defect_core import infer as infer_module
from defect_core.artifacts import load_artifact
from defect_core.infer import InferenceError, format_report, inspect_image, main, ordinal
from defect_core.model import infer
from defect_core.preprocessing import ImageLoadError, load_image, to_tensor
from defect_core.render import render_result
from PIL import Image

_SRC = Path(__file__).resolve().parent / "test_stages.py"
_spec = importlib.util.spec_from_file_location("core_test_stages_for_infer", _SRC)
stages = importlib.util.module_from_spec(_spec)
sys.modules["core_test_stages_for_infer"] = stages
_spec.loader.exec_module(stages)
fitted_dir, frozen, calibrated, packaged = (
    stages.fitted_dir, stages.frozen, stages.calibrated, stages.packaged,
)  # fmt: skip


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(scope="module")
def image_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(4)
    array = (rng.random((72, 96, 3)) * 120 + 60).astype(np.uint8)
    array[20:40, 30:60] = 250
    path = tmp_path_factory.mktemp("img") / "sample.png"
    Image.fromarray(array).save(path)
    return path


# --- the shared function --------------------------------------------------------------------


def test_the_result_follows_the_calibration_and_the_direct_scoring(
    packaged: Path, image_file: Path
) -> None:
    loaded = load_artifact(packaged)
    result = inspect_image(loaded, image_file)
    expected, expected_map = infer(
        loaded.model, to_tensor(load_image(image_file), loaded.preprocess_config)[None]
    )
    calibration = loaded.calibration
    assert result.score == float(expected[0])  # exactly the direct scoring
    assert np.array_equal(result.anomaly_map, expected_map[0, 0].numpy())
    assert result.threshold == calibration.threshold
    assert result.verdict == ("anomalous" if result.score > calibration.threshold else "normal")
    assert (result.rank_m, result.rank_n) == (calibration.normal_rank(result.score), calibration.n)
    assert result.image_size == (96, 72)
    assert result.decision_rule.startswith("anomalous if anomaly_score > threshold")
    assert f"{ordinal(calibration.k)} smallest of {calibration.n}" in result.decision_rule


def test_a_path_bytes_and_a_picture_give_the_same_result(packaged: Path, image_file: Path) -> None:
    loaded = load_artifact(packaged)
    from_path = inspect_image(loaded, image_file)
    from_bytes = inspect_image(loaded, image_file.read_bytes())
    from_image = inspect_image(loaded, Image.open(image_file))
    assert from_path.score == from_bytes.score == from_image.score
    assert from_path.verdict == from_bytes.verdict == from_image.verdict


def test_the_same_image_gives_the_same_result_every_time(packaged: Path, image_file: Path) -> None:
    loaded = load_artifact(packaged)
    first, second = inspect_image(loaded, image_file), inspect_image(loaded, image_file)
    assert first.score == second.score and np.array_equal(first.anomaly_map, second.anomaly_map)


def test_the_rendering_is_the_shared_rendering_at_the_original_size(
    packaged: Path, image_file: Path
) -> None:
    loaded = load_artifact(packaged)
    result = inspect_image(loaded, image_file)
    direct = render_result(load_image(image_file), result.anomaly_map, loaded.calibration.pixel)
    assert result.rendering.heatmap.shape == (72, 96, 3) == result.rendering.overlay.shape
    assert np.array_equal(result.rendering.overlay, direct.overlay)
    assert result.region_bbox == direct.bbox
    assert result.highlighted_area_fraction == direct.area_fraction


def test_images_of_other_modes_are_prepared_like_files(packaged: Path) -> None:
    loaded = load_artifact(packaged)
    gray = Image.fromarray((np.random.default_rng(1).random((40, 50)) * 255).astype(np.uint8))
    buffer = io.BytesIO()
    gray.save(buffer, format="PNG")
    assert inspect_image(loaded, gray).score == inspect_image(loaded, buffer.getvalue()).score


def test_a_calibrated_artifact_works_and_a_fitted_or_frozen_one_is_refused(
    calibrated: Path, frozen: Path, fitted_dir: Path, image_file: Path
) -> None:
    assert inspect_image(load_artifact(calibrated), image_file).rank_n > 0
    for folder in (frozen, fitted_dir):
        with pytest.raises(InferenceError, match="no calibration"):
            inspect_image(load_artifact(folder), image_file)


def test_a_file_that_is_not_an_image_is_refused(packaged: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        inspect_image(load_artifact(packaged), bad)


# --- the report -----------------------------------------------------------------------------


def test_the_report_states_the_rule_the_rank_and_the_limits_and_never_says_pass_or_fail(
    packaged: Path, image_file: Path
) -> None:
    text = format_report(inspect_image(load_artifact(packaged), image_file), str(image_file))
    for needed in (
        "demo decision",
        "Score:",
        "Threshold:",
        "Rank:",
        "calibration normal images",
        "not a probability or a confidence",
        "Rule:",
        "out of scope",
        "not a quality",
    ):
        assert needed in text
    assert re.search(r"score higher than \d+ of the \d+", text)
    assert not re.search(r"\b(pass|passed|fail|failed)\b", text, re.IGNORECASE)


# --- the command line -----------------------------------------------------------------------


def test_the_cli_prints_the_report_and_writes_nothing_without_out(
    packaged: Path, image_file: Path, capsys: pytest.CaptureFixture, tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert main(["--artifact", str(packaged), str(image_file)]) == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out and "demo decision" in out
    assert list(work.iterdir()) == []  # nothing written


def test_the_cli_saves_a_heatmap_and_an_overlay_at_the_image_size_and_never_overwrites(
    packaged: Path, image_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out_dir = tmp_path / "results" / "nested"
    assert main(["--artifact", str(packaged), str(image_file), "--out", str(out_dir)]) == 0
    heatmap, overlay = out_dir / "sample_heatmap.png", out_dir / "sample_overlay.png"
    assert Image.open(heatmap).size == (96, 72) == Image.open(overlay).size
    assert "Saved:" in capsys.readouterr().out
    before = heatmap.read_bytes()
    assert main(["--artifact", str(packaged), str(image_file), "--out", str(out_dir)]) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert heatmap.read_bytes() == before


def test_the_json_output_has_the_planned_fields_and_matches_the_function(
    packaged: Path, image_file: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["--artifact", str(packaged), str(image_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    result = inspect_image(load_artifact(packaged), image_file)
    assert payload["anomaly_score"] == result.score and payload["prediction"] == result.verdict
    assert payload["threshold"] == result.threshold
    assert payload["calibration_rank"] == {
        "score_higher_than": result.rank_m, "of_calibration_normals": result.rank_n,
    }  # fmt: skip
    expected_keys = {"decision_rule", "limitation", "disclaimer", "model", "image", "timing_ms"}
    assert expected_keys <= set(payload)
    assert payload["image"] == {"width": 96, "height": 72} and payload["saved_files"] == []


@pytest.mark.parametrize("case", ["missing_image", "not_an_image", "missing_artifact", "fitted"])
def test_the_cli_refuses_clearly_with_exit_code_2(
    case: str, packaged: Path, fitted_dir: Path, image_file: Path, tmp_path: Path, capsys
) -> None:
    artifact, image = str(packaged), str(image_file)
    if case == "missing_image":
        image = str(tmp_path / "nope.png")
    elif case == "not_an_image":
        bad = tmp_path / "bad.txt"
        bad.write_text("hello")
        image = str(bad)
    elif case == "missing_artifact":
        artifact = str(tmp_path / "no_such_folder")
    elif case == "fitted":
        artifact = str(fitted_dir)
    assert main(["--artifact", artifact, image]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR:") and "Verdict" not in captured.out


@pytest.mark.parametrize(
    "tamper",
    ["weights", "notice", "calibration", "package"],
)
def test_a_damaged_or_edited_artifact_is_refused_before_anything_is_scored(
    tamper: str, packaged: Path, image_file: Path, capsys: pytest.CaptureFixture
) -> None:
    if tamper == "weights":
        data = bytearray((packaged / "model.safetensors").read_bytes())
        data[-1] ^= 1
        (packaged / "model.safetensors").write_bytes(bytes(data))
        message = "SHA-256"
    elif tamper == "notice":
        (packaged / "NOTICE.md").write_text("changed\n")
        message = "NOTICE.md"
    else:
        manifest = json.loads((packaged / "manifest.json").read_text())
        if tamper == "calibration":
            manifest["calibration"]["block"]["threshold"] = 0.0
            message = "threshold|changed after"
        else:
            manifest["package"]["version"] = "9.9.9"
            message = "package version"
        (packaged / "manifest.json").write_text(json.dumps(manifest))
    assert main(["--artifact", str(packaged), str(image_file)]) == 2
    captured = capsys.readouterr()
    assert re.search(message, captured.err) and "Verdict" not in captured.out


def test_the_module_runs_as_a_command_in_a_fresh_process(packaged: Path, image_file: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "defect_core.infer", "--artifact", str(packaged), str(image_file),
         "--json"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    in_process = inspect_image(load_artifact(packaged), image_file)
    assert payload["anomaly_score"] == in_process.score  # a fresh process gives the same score
    assert infer_module.DISCLAIMER == payload["disclaimer"] and torch.is_tensor(torch.zeros(1))


def test_ordinals_read_correctly() -> None:
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 41, 42, 101, 111)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "41st", "42nd",
        "101st", "111th",
    ]  # fmt: skip
