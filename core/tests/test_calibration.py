"""Tests for defect_core.calibration (PLAN.md section 5.6).

Synthetic numbers only: no model, image or dataset file is used, except one test that scores random
images with a small random model to check that float32 model outputs flow through correctly.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
import torch
from defect_core.calibration import (
    Calibration,
    CalibrationError,
    build_calibration,
    conformal_rank,
    false_alarm_bound,
    is_anomalous,
    normal_rank,
    pixel_levels,
)

PIXEL = {"lower_percentile": 1.0, "highlight_percentile": 99.5, "upper_multiple": 2.0}


# --- the conformal index -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "alpha", "k"),
    [
        (42, 0.05, 41),  # the bottle example in the plan: ceil(43 * 0.95) = ceil(40.85)
        (42, 0.1, 39),  # ceil(43 * 0.9) = ceil(38.7)
        (99, 0.05, 95),  # 100 * 0.95 is exactly 95
        (19, 0.05, 19),  # 20 * 0.95 is exactly 19: a boundary case where k must not round up
        (9, 0.1, 9),  # 10 * 0.9 is exactly 9
        (20, 0.05, 20),  # 21 * 0.95 = 19.95
        (1, 0.5, 1),  # 2 * 0.5 = 1
    ],
)
def test_conformal_rank_known_values(n: int, alpha: float, k: int) -> None:
    assert conformal_rank(n, alpha) == k


def test_the_bottle_example_matches_the_plan() -> None:
    k = conformal_rank(42, 0.05)
    assert k == 41  # the 2nd-highest of 42 calibration scores
    assert false_alarm_bound(42, k) == pytest.approx(2 / 43)
    assert false_alarm_bound(42, k) <= 0.05


def test_too_small_an_alpha_raises_and_never_falls_back_to_the_maximum() -> None:
    for alpha in (0.01, 0.02, 0.0232):  # all below 1/43 = 0.02326 for n = 42
        with pytest.raises(CalibrationError, match="too small"):
            conformal_rank(42, alpha)
    assert conformal_rank(42, 0.0233) == 42  # just above the limit: the largest score is right


def test_the_smallest_k_is_always_chosen_and_the_bound_holds() -> None:
    for n in range(1, 201):
        for alpha in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            if alpha < 1 / (n + 1) - 1e-12:
                with pytest.raises(CalibrationError):
                    conformal_rank(n, alpha)
                continue
            k = conformal_rank(n, alpha)
            assert 1 <= k <= n
            assert false_alarm_bound(n, k) <= alpha + 1e-12  # guaranteed bound
            if k > 1:  # k - 1 would break the bound, so k is the smallest valid index
                assert false_alarm_bound(n, k - 1) > alpha


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, float("nan"), True, "0.05"])
def test_invalid_alpha_is_rejected(alpha: object) -> None:
    with pytest.raises(CalibrationError):
        conformal_rank(42, alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [0, -3, 4.0, True, "42"])
def test_invalid_n_is_rejected(n: object) -> None:
    with pytest.raises(CalibrationError):
        conformal_rank(n, 0.05)  # type: ignore[arg-type]


def test_the_false_alarm_rate_matches_the_bound_in_a_simulation() -> None:
    """Exchangeable continuous scores: flagging rate of a new normal image is about 2/43."""
    rng = np.random.default_rng(0)
    n, trials = 42, 40_000
    k = conformal_rank(n, 0.05)
    draws = rng.random((trials, n + 1))
    thresholds = np.sort(draws[:, :n], axis=1)[:, k - 1]
    flagged = (draws[:, n] > thresholds).mean()
    assert flagged == pytest.approx(false_alarm_bound(n, k), abs=0.006)
    assert flagged <= 0.05 + 0.006


# --- threshold, verdict and rank -----------------------------------------------------------


def test_threshold_is_the_kth_smallest_score_whatever_the_order() -> None:
    scores = np.arange(1.0, 43.0)  # 1 ... 42
    np.random.default_rng(1).shuffle(scores)
    block = build_calibration(scores, [np.ones((2, 2))] * 42, alpha=0.05, **PIXEL)
    assert (block.n, block.k, block.threshold) == (42, 41, 41.0)
    assert block.calibration_scores == tuple(float(x) for x in range(1, 43))


def test_the_verdict_is_strict_so_a_tie_is_not_flagged() -> None:
    assert not is_anomalous(41.0, 41.0)
    assert not is_anomalous(40.999, 41.0)
    assert is_anomalous(41.0001, 41.0)
    assert is_anomalous(42.0, 41.0)


def test_ties_in_the_calibration_scores_are_handled() -> None:
    scores = [1.0] * 10 + [2.0] * 10 + [3.0] * 22  # n = 42, k = 41 -> the 41st smallest is 3.0
    block = build_calibration(scores, [np.ones((2, 2))] * 42, alpha=0.05, **PIXEL)
    assert block.threshold == 3.0
    assert not block.is_anomalous(3.0)  # ties never flag


def test_normal_rank_counts_only_strictly_lower_calibration_scores() -> None:
    table = [float(x) for x in range(1, 43)]
    assert normal_rank(0.5, table) == 0
    assert normal_rank(1.0, table) == 0  # equal is not "higher than"
    assert normal_rank(1.5, table) == 1
    assert normal_rank(41.0, table) == 40
    assert normal_rank(41.5, table) == 41
    assert normal_rank(42.0, table) == 41
    assert normal_rank(99.0, table) == 42  # "higher than 42 of 42"


def test_bad_scores_are_rejected() -> None:
    maps = [np.ones((2, 2))] * 3
    for bad in ([], [1.0, float("nan"), 2.0], [1.0, float("inf"), 2.0], [[1.0, 2.0], [3.0, 4.0]]):
        with pytest.raises(CalibrationError):
            build_calibration(bad, maps, alpha=0.5, **PIXEL)
    with pytest.raises(CalibrationError, match="anomaly maps"):
        build_calibration([1.0, 2.0, 3.0], maps[:2], alpha=0.5, **PIXEL)
    with pytest.raises(CalibrationError, match="too small"):
        build_calibration([1.0, 2.0, 3.0], maps, alpha=0.1, **PIXEL)  # needs n >= 9


def test_float32_model_scores_are_used_exactly() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    block = build_calibration(scores, [np.ones((2, 2))] * 4, alpha=0.4, **PIXEL)
    # n = 4, alpha = 0.4 -> k = ceil(5 * 0.6) = 3 -> the 3rd smallest, exactly as a float32 value
    assert block.threshold == float(torch.tensor(0.3, dtype=torch.float32))
    assert not block.is_anomalous(float(scores[2]))  # the same float32 score is not above it


# --- pixel levels --------------------------------------------------------------------------


def test_pixel_levels_on_known_values() -> None:
    pixels = np.linspace(0, 100, 10001)  # percentile p is exactly p
    maps = [pixels[:5000], pixels[5000:]]  # maps of any shape are flattened
    levels = pixel_levels(maps, **PIXEL)
    assert levels.lower == pytest.approx(1.0)
    assert levels.highlight == pytest.approx(99.5)
    assert levels.upper == pytest.approx(2.0 * 99.5)
    assert (levels.lower_percentile, levels.highlight_percentile) == (1.0, 99.5)


def test_pixel_levels_accept_tensors_and_numpy_alike() -> None:
    values = np.random.default_rng(2).random((3, 8, 8))
    from_numpy = pixel_levels(list(values), **PIXEL)
    from_torch = pixel_levels([torch.from_numpy(v) for v in values], **PIXEL)
    assert from_numpy == from_torch


def test_pixel_settings_have_no_defaults() -> None:
    with pytest.raises(TypeError):
        pixel_levels([np.ones((2, 2))])  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "settings",
    [
        {"lower_percentile": 99.5, "highlight_percentile": 1.0, "upper_multiple": 2.0},
        {"lower_percentile": -1.0, "highlight_percentile": 99.5, "upper_multiple": 2.0},
        {"lower_percentile": 1.0, "highlight_percentile": 101.0, "upper_multiple": 2.0},
        {"lower_percentile": 1.0, "highlight_percentile": 99.5, "upper_multiple": 0.0},
        {"lower_percentile": 1.0, "highlight_percentile": 99.5, "upper_multiple": -2.0},
    ],
)
def test_bad_pixel_settings_are_rejected(settings: dict[str, float]) -> None:
    with pytest.raises(CalibrationError):
        pixel_levels([np.arange(100.0).reshape(10, 10)], **settings)


def test_an_empty_heatmap_scale_is_rejected() -> None:
    flat = [np.full((4, 4), 5.0)]  # every percentile is 5.0, so upper = 2 * 5 > lower = 5 is fine
    assert pixel_levels(flat, **PIXEL).upper == 10.0
    with pytest.raises(CalibrationError, match="scale is empty"):
        pixel_levels(flat, lower_percentile=1.0, highlight_percentile=99.5, upper_multiple=1.0)
    with pytest.raises(CalibrationError, match="NaN"):
        pixel_levels([np.array([[1.0, np.nan]])], **PIXEL)
    with pytest.raises(CalibrationError, match="no anomaly maps"):
        pixel_levels([], **PIXEL)


# --- the block that goes into the manifest -------------------------------------------------


def _block(n: int = 42, alpha: float = 0.05) -> Calibration:
    rng = np.random.default_rng(3)
    return build_calibration(
        rng.random(n) + 1.0, [rng.random((8, 8)) for _ in range(n)], alpha=alpha, **PIXEL
    )


def test_the_block_round_trips_through_json_unchanged() -> None:
    block = _block()
    restored = Calibration.from_dict(json.loads(json.dumps(block.to_dict())))
    assert restored == block
    assert block.to_dict()["false_alarm_bound"] == pytest.approx(2 / 43)
    assert block.to_dict()["method"] == "split-conformal"
    with pytest.raises(dataclasses.FrozenInstanceError):
        block.alpha = 0.1  # type: ignore[misc]


def test_the_block_reports_rank_and_verdict_consistently() -> None:
    block = _block()
    assert block.normal_rank(block.threshold) == block.k - 1  # strictly lower scores only
    assert not block.is_anomalous(block.threshold)
    assert block.is_anomalous(np.nextafter(block.threshold, np.inf))
    assert sum(block.is_anomalous(s) for s in block.calibration_scores) == block.n - block.k


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(k=40), "k does not follow"),
        (lambda d: d.update(alpha=0.1), "k does not follow"),
        (lambda d: d.update(n=41), "n does not equal"),
        (lambda d: d.update(threshold=d["threshold"] + 1e-9), "threshold is not"),
        (lambda d: d["calibration_scores"].reverse(), "not sorted"),
        (lambda d: d["calibration_scores"].__setitem__(0, float("nan")), "NaN"),
        (lambda d: d.update(false_alarm_bound=0.0), "false_alarm_bound"),
        (lambda d: d["pixel"].update(upper=d["pixel"]["upper"] * 2), "pixel levels"),
        (lambda d: d["pixel"].update(lower=d["pixel"]["highlight"] + 1), "pixel levels"),
        (lambda d: d.update(method="other"), "Unsupported"),
        (lambda d: d.update(schema_version=2), "Unsupported"),
    ],
)
def test_an_inconsistent_block_is_rejected(change, message: str) -> None:
    data = copy.deepcopy(_block().to_dict())
    change(data)
    with pytest.raises(CalibrationError, match=message):
        Calibration.from_dict(data)


def test_a_malformed_block_is_rejected() -> None:
    data = _block().to_dict()
    del data["threshold"]
    with pytest.raises(CalibrationError, match="malformed"):
        Calibration.from_dict(data)
    with pytest.raises(CalibrationError, match="malformed"):
        Calibration.from_dict({"schema_version": 1, "method": "split-conformal"})


def test_calibrating_the_outputs_of_a_small_random_model() -> None:
    """Float32 model outputs flow through; exactly n - k calibration images are flagged."""
    from defect_core.model import ModelConfig, build_model, fit_memory_bank, infer

    config = ModelConfig(backbone="resnet18")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = build_model(config, pre_trained=False)
    fit_memory_bank(model, [torch.rand(4, 3, 64, 64, generator=gen)], 0.1, seed=0)
    scores, maps = infer(model, torch.rand(20, 3, 64, 64, generator=gen))
    block = build_calibration(scores, list(maps), alpha=0.1, **PIXEL)
    assert (block.n, block.k) == (20, 19)  # ceil(21 * 0.9) = ceil(18.9)
    flagged = sum(block.is_anomalous(float(s)) for s in scores)
    assert flagged == 1  # n - k, when the scores have no ties
    assert block.pixel.upper > block.pixel.lower
