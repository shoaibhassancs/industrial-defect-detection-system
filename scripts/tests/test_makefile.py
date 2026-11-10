"""Tests for the pipeline targets of the Makefile.

They use ``make -n`` (print the commands, run nothing) and the two refusals that run no Python at
all, so they never open a dataset file, fit a model or touch an artifact.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")

WRN = "ml/configs/bottle_wrn50.yaml"
R18 = "ml/configs/bottle_resnet18.yaml"


def make(*args: str, dry: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["make", *(["-n"] if dry else []), *args]
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)


def test_each_step_runs_the_documented_module_for_each_model_in_order() -> None:
    expected = {
        "fit": "defect_ml.fit",
        "freeze": "defect_ml.freeze",
        "calibrate": "defect_ml.calibrate",
    }
    for target, module in expected.items():
        out = make(target).stdout
        assert f"python -m {module} $c" in out
        assert out.index(WRN) < out.index(R18)  # the headline model first, one at a time
    assert "--version 1.0.0" in make("freeze").stdout


def test_the_version_and_the_models_can_be_overridden() -> None:
    assert "--version 2.0.0" in make("calibrate", "VERSION=2.0.0").stdout
    only = make("fit", f"CONFIGS={R18}").stdout
    assert R18 in only and WRN not in only


def test_package_runs_the_packaging_without_any_sealed_guard() -> None:
    out = make("package").stdout
    assert f"python -m defect_ml.package {WRN} {R18} --version 1.0.0" in out
    assert "CONFIRM_OPEN_TEST_SET" not in out


@pytest.mark.parametrize(("target", "module"), [("evaluate", "evaluate"), ("figures", "visualize")])
def test_the_steps_that_open_the_sealed_test_set_refuse_without_confirmation(
    target: str, module: str
) -> None:
    done = make(target, dry=False)  # runs only the guard: no Python, no dataset access
    assert done.returncode != 0
    assert "REFUSING" in done.stdout and "CONFIRM_OPEN_TEST_SET=yes" in done.stdout
    assert f"defect_ml.{module}" not in done.stdout + done.stderr.replace("make:", "")
    # the confirmed form would run all models in one batch (printed only, not run)
    confirmed = make(target, "CONFIRM_OPEN_TEST_SET=yes").stdout
    assert f"python -m defect_ml.{module} {WRN} {R18} --version 1.0.0" in confirmed


def test_only_the_exact_confirmation_word_is_accepted() -> None:
    for wrong in ("YES", "true", "1", ""):
        done = make("evaluate", f"CONFIRM_OPEN_TEST_SET={wrong}", dry=False)
        assert done.returncode != 0 and "REFUSING" in done.stdout


def test_infer_needs_an_artifact_and_an_image_and_passes_them_on() -> None:
    done = make("infer", dry=False)
    assert done.returncode != 0 and "usage: make infer" in done.stdout
    out = make("infer", "ARTIFACT=pkg", "IMAGE=pic.png", "OUT=results").stdout
    assert 'python -m defect_core.infer --artifact "pkg" "pic.png" --out "results"' in out
    assert "--out" not in make("infer", "ARTIFACT=pkg", "IMAGE=pic.png").stdout
