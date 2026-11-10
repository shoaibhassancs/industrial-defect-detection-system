"""Tests for defect_ml.metrics (Phase 3 step 5). Synthetic inputs only: no dataset file is read.

Expected values are worked out by hand in the comments. AUPRO is also checked against anomalib's
implementation on a small random fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from defect_ml import metrics
from defect_ml.metrics import MetricsError

# --- image AUROC and AUPR -------------------------------------------------------------------


def test_auroc_and_aupr_on_a_hand_worked_example() -> None:
    labels, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    # 3 of the 4 (anomalous, normal) pairs are ordered correctly
    assert metrics.image_auroc(labels, scores) == pytest.approx(0.75)
    # sorted by score: 0.8 (hit) -> precision 1 at recall 0.5; 0.35 (hit) -> precision 2/3 at 1.0
    assert metrics.image_aupr(labels, scores) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_perfect_reversed_and_tied_scores() -> None:
    labels = [0, 0, 0, 1, 1]
    assert metrics.image_auroc(labels, [1, 2, 3, 8, 9]) == 1.0
    assert metrics.image_auroc(labels, [9, 8, 7, 2, 1]) == 0.0
    assert metrics.image_auroc(labels, [5, 5, 5, 5, 5]) == 0.5
    assert metrics.image_aupr(labels, [1, 2, 3, 8, 9]) == 1.0


def test_boolean_labels_are_accepted() -> None:
    assert metrics.image_auroc([False, True], [0.1, 0.9]) == 1.0


@pytest.mark.parametrize("function", [metrics.image_auroc, metrics.image_aupr])
def test_one_class_only_is_an_error_not_a_number(function) -> None:
    with pytest.raises(MetricsError, match="both normal and anomalous"):
        function([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(MetricsError, match="both normal and anomalous"):
        function([1, 1], [0.1, 0.2])


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1], [0.1], "equally long"),
        ([], [], "no images"),
        ([0, 2], [0.1, 0.2], "0 or 1"),
        ([0, 1], [0.1, float("nan")], "NaN or infinite"),
        ([0, 1], [0.1, float("inf")], "NaN or infinite"),
        ([[0, 1]], [[0.1, 0.2]], "1-D"),
    ],
)
def test_bad_inputs_are_refused(labels, scores, message) -> None:
    with pytest.raises(MetricsError, match=message):
        metrics.image_auroc(labels, scores)


# --- bootstrap ------------------------------------------------------------------------------


def _example(n_normal: int = 12, n_anomalous: int = 8):
    rng = np.random.default_rng(5)
    labels = np.array([0] * n_normal + [1] * n_anomalous)
    scores = np.concatenate([rng.normal(1.0, 0.3, n_normal), rng.normal(1.6, 0.4, n_anomalous)])
    return labels, scores


def test_the_same_seed_gives_the_same_interval_and_another_seed_a_different_one() -> None:
    labels, scores = _example()
    a = metrics.bootstrap_interval(labels, scores, seed=3)
    b = metrics.bootstrap_interval(labels, scores, seed=3)
    c = metrics.bootstrap_interval(labels, scores, seed=4)
    assert a == b
    assert (a.low, a.high) != (c.low, c.high)
    assert a.estimate == c.estimate == metrics.image_auroc(labels, scores)
    assert a.low <= a.estimate <= a.high and 0 <= a.low and a.high <= 1
    assert a.seed == 3 and a.n_resamples == 1000 and a.confidence == 0.95
    assert set(a.to_dict()) == {
        "estimate", "low", "high", "confidence", "n_resamples", "seed", "method"
    }  # fmt: skip


def test_every_resample_keeps_the_class_counts() -> None:
    labels, scores = _example(12, 8)
    seen: list[tuple[int, int]] = []

    def statistic(y: np.ndarray, s: np.ndarray) -> float:
        seen.append((int((~y).sum()), int(y.sum())))
        return 0.5

    metrics.bootstrap_interval(labels, scores, statistic, n_resamples=100, seed=0)
    assert set(seen) == {(12, 8)}  # also the call on the original data


def test_a_perfectly_separated_set_has_a_degenerate_interval_at_one() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    result = metrics.bootstrap_interval(labels, [1, 2, 3, 7, 8, 9], seed=1)
    assert (result.estimate, result.low, result.high) == (1.0, 1.0, 1.0)


def test_the_bootstrap_works_with_aupr_too() -> None:
    labels, scores = _example()
    result = metrics.bootstrap_interval(labels, scores, metrics.image_aupr, seed=0)
    assert result.estimate == metrics.image_aupr(labels, scores)
    assert result.low <= result.high


@pytest.mark.parametrize(
    ("labels", "kwargs", "message"),
    [
        ([0, 0, 1], {}, "at least 2"),
        ([0, 1, 1], {}, "at least 2"),
        ([0, 0, 1, 1], {"n_resamples": 99}, "n_resamples"),
        ([0, 0, 1, 1], {"n_resamples": 100.0}, "n_resamples"),
        ([0, 0, 1, 1], {"confidence": 1.0}, "confidence"),
        ([0, 0, 1, 1], {"confidence": 0.0}, "confidence"),
    ],
)
def test_bootstrap_refusals(labels, kwargs, message) -> None:
    with pytest.raises(MetricsError, match=message):
        metrics.bootstrap_interval(labels, np.arange(len(labels), dtype=float), **kwargs)


# --- verdicts at the threshold --------------------------------------------------------------


def test_threshold_metrics_by_hand_and_the_verdict_is_strict() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    scores = [1.0, 1.0, 2.0, 2.0, 3.0, 0.5]
    result = metrics.threshold_metrics(labels, scores, threshold=1.0)
    # flagged (score > 1.0): the normal 2.0, the anomalous 2.0 and 3.0. A score equal to 1.0 is
    # normal, and so is the anomalous 0.5.
    assert (result.tp, result.fp, result.tn, result.fn) == (2, 1, 2, 1)
    assert result.tpr == pytest.approx(2 / 3) and result.fpr == pytest.approx(1 / 3)
    assert result.precision == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert result.to_dict()["threshold"] == 1.0


def test_the_verdict_comes_from_the_shared_rule() -> None:
    from defect_core.calibration import is_anomalous

    scores = [0.9, 1.0, 1.1]
    result = metrics.threshold_metrics([1, 1, 1], scores, threshold=1.0)
    assert result.tp == sum(is_anomalous(s, 1.0) for s in scores) == 1


def test_undefined_rates_are_none_not_zero() -> None:
    nothing_flagged = metrics.threshold_metrics([0, 1], [0.1, 0.2], threshold=5.0)
    assert nothing_flagged.precision is None and nothing_flagged.tpr == 0.0
    assert nothing_flagged.f1 == 0.0  # there is a missed anomaly, so F1 is defined and zero
    no_anomalous = metrics.threshold_metrics([0, 0], [0.1, 0.2], threshold=5.0)
    assert no_anomalous.tpr is None and no_anomalous.f1 is None and no_anomalous.fpr == 0.0
    no_normal = metrics.threshold_metrics([1, 1], [0.1, 9.0], threshold=5.0)
    assert no_normal.fpr is None and no_normal.precision == 1.0


def test_threshold_metrics_refuse_a_bad_threshold() -> None:
    with pytest.raises(MetricsError, match="finite"):
        metrics.threshold_metrics([0, 1], [0.1, 0.2], float("nan"))


def test_per_defect_recall() -> None:
    types = ["crack", "crack", "hole", "crack", "hole"]
    scores = [2.0, 1.0, 3.0, 0.5, 1.0]
    result = metrics.per_defect_recall(types, scores, threshold=1.0)
    assert list(result) == ["crack", "hole"]  # sorted
    assert (result["crack"].n, result["crack"].detected) == (3, 1)  # only 2.0 is strictly above
    assert result["crack"].recall == pytest.approx(1 / 3)
    assert (result["hole"].n, result["hole"].detected, result["hole"].recall) == (2, 1, 0.5)
    assert result["hole"].to_dict() == {"n": 2, "detected": 1, "recall": 0.5}


@pytest.mark.parametrize(
    ("types", "scores"),
    [(["a"], [1.0, 2.0]), ([], []), (["a", ""], [1.0, 2.0]), (["a", "b"], [1.0, float("nan")])],
)
def test_per_defect_recall_refusals(types, scores) -> None:
    with pytest.raises(MetricsError):
        metrics.per_defect_recall(types, scores, 1.0)


# --- pixel AUROC and highlight pixels -------------------------------------------------------


def test_pixel_auroc_and_its_refusals() -> None:
    masks = np.zeros((2, 4, 4), dtype=int)
    masks[0, :2, :2] = 1
    maps = masks * 5.0 + 1.0
    assert metrics.pixel_auroc(masks, maps) == 1.0
    assert metrics.pixel_auroc(masks, -maps) == 0.0
    with pytest.raises(MetricsError, match="both normal and anomalous"):
        metrics.pixel_auroc(np.zeros((1, 3, 3)), np.ones((1, 3, 3)))
    with pytest.raises(MetricsError, match="N, H, W"):
        metrics.pixel_auroc(masks, maps[:1])
    with pytest.raises(MetricsError, match="only 0 and 1"):
        metrics.pixel_auroc(masks * 2, maps)
    bad = maps.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(MetricsError, match="NaN"):
        metrics.pixel_auroc(masks, bad)


def test_highlight_pixels_are_strictly_above_the_level() -> None:
    mask = np.array([[[1, 1, 0, 0]]])
    maps = np.array([[[2.0, 1.0, 1.5, 0.0]]])
    result = metrics.highlight_pixel_metrics(mask, maps, highlight=1.0)
    # strictly above 1.0: pixels 0 (hit) and 2 (false alarm); pixel 1 (equal) is not flagged
    assert (result.tp, result.fp, result.fn) == (1, 1, 1)
    assert result.precision == 0.5 and result.recall == 0.5
    none = metrics.highlight_pixel_metrics(mask, maps, highlight=9.0)
    assert none.precision is None and none.recall == 0.0
    empty_masks = metrics.highlight_pixel_metrics(np.zeros((1, 1, 4)), maps, highlight=1.0)
    assert empty_masks.recall is None and empty_masks.precision == 0.0
    assert set(result.to_dict()) == {"highlight", "tp", "fp", "fn", "precision", "recall"}


# --- connected regions ----------------------------------------------------------------------


def test_regions_are_eight_connected() -> None:
    diagonal = np.array([[1, 0], [0, 1]], dtype=bool)
    labels, count = metrics.label_regions(diagonal)
    assert count == 1 and labels[0, 0] == labels[1, 1] == 1
    apart = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)
    assert metrics.label_regions(apart)[1] == 2
    empty_labels, empty_count = metrics.label_regions(np.zeros((3, 3), dtype=bool))
    assert empty_count == 0 and not empty_labels.any()
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    assert metrics.label_regions(ring)[1] == 1
    with pytest.raises(MetricsError, match="2-D"):
        metrics.label_regions(np.zeros((2, 2, 2), dtype=bool))


def test_region_labelling_agrees_with_anomalibs_on_random_masks() -> None:
    import torch
    from anomalib.utils.cv import connected_components_cpu

    rng = np.random.default_rng(2)
    for _ in range(5):
        mask = rng.random((20, 20)) < 0.35
        ours, count = metrics.label_regions(mask)
        theirs = connected_components_cpu(torch.from_numpy(mask.astype(np.float32))[None, None])
        theirs = theirs[0, 0].numpy()
        assert count == len(np.unique(theirs)) - 1
        # same partition of the pixels, whatever the label numbers are
        pairs = {(int(a), int(b)) for a, b in zip(ours.ravel(), theirs.ravel(), strict=True)}
        assert len(pairs) == count + 1


# --- AUPRO ----------------------------------------------------------------------------------


def test_aupro_extremes() -> None:
    masks = np.zeros((2, 6, 6), dtype=int)
    masks[0, 1:3, 1:3] = 1
    masks[1, 4, 3:6] = 1
    perfect = masks * 10.0 + np.random.default_rng(0).random(masks.shape)
    assert metrics.aupro(masks, perfect) == pytest.approx(1.0)
    assert metrics.aupro(masks, -perfect) == pytest.approx(0.0)
    # every pixel tied: one threshold, a straight line from (0,0) to (1,1): the area up to 0.3
    # is 0.3 * 0.3 / 2, divided by 0.3
    flat = np.full(masks.shape, 0.7)
    assert metrics.aupro(masks, flat) == pytest.approx(0.15)
    assert metrics.aupro(masks, flat, fpr_limit=1.0) == pytest.approx(0.5)


def test_aupro_hand_worked_example() -> None:
    # one image, 10 pixels; the region is pixels 0 and 1 (scores 0.9 and 0.5); the 8 normal
    # pixels score 0.8 0.7 0.6 0.4 0.3 0.2 0.1 0.0. Walking the threshold down: after 0.9 the
    # region overlap is 1/2 at FPR 0; it stays 1/2 until 0.5, which is at FPR 4/8, beyond 0.3.
    masks = np.array([[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]])
    maps = np.array([[[0.9, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0]]])
    fpr, pro = metrics.pro_curve(masks, maps)
    assert fpr[0] == 0 and pro[0] == 0 and fpr[-1] == pytest.approx(0.3)
    assert pro[-1] == pytest.approx(0.5) and np.all(np.diff(fpr) >= 0) and np.all(np.diff(pro) >= 0)
    assert metrics.aupro(masks, maps) == pytest.approx(0.5)


def test_aupro_averages_regions_equally_not_by_size() -> None:
    # region A is 1 pixel and found first; region B is 3 pixels and found last. The macro average
    # over regions is 1/2 at low FPR; averaging over pixels would give 1/4.
    masks = np.array([[[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]]])
    maps = np.array([[[0.9, 0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]]])
    assert metrics.aupro(masks, maps) == pytest.approx(0.5)


def test_aupro_keeps_regions_of_different_images_apart() -> None:
    # the same pixel position is a region in both images: two regions, one found, one missed
    masks = np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]])
    maps = np.array([[[0.9, 0.5, 0.5, 0.5]], [[0.1, 0.5, 0.5, 0.5]]])
    assert metrics.aupro(masks, maps) == pytest.approx(0.5)


def test_aupro_refusals() -> None:
    masks = np.zeros((1, 3, 3))
    masks[0, 0, 0] = 1
    maps = np.random.default_rng(0).random((1, 3, 3))
    with pytest.raises(MetricsError, match="at least one anomalous region"):
        metrics.aupro(np.zeros((1, 3, 3)), maps)
    with pytest.raises(MetricsError, match="normal"):
        metrics.aupro(np.ones((1, 3, 3)), maps)
    for limit in (0, -0.1, 1.5):
        with pytest.raises(MetricsError, match="fpr_limit"):
            metrics.aupro(masks, maps, limit)


@pytest.mark.parametrize("fpr_limit", [0.05, 0.3, 1.0])
def test_aupro_equals_anomalibs_on_a_small_fixture(fpr_limit: float) -> None:
    import torch
    from anomalib.metrics.aupro import _AUPRO  # the metric's own algorithm, without batch wiring

    rng = np.random.default_rng(11)
    masks = np.zeros((6, 24, 24), dtype=np.float32)
    masks[0, 3:9, 4:10] = 1
    masks[0, 9:11, 10:12] = 1  # touches the first region only at a corner: one region (8-connected)
    masks[0, 18:22, 2:5] = 1
    masks[1, 10:16, 10:20] = 1
    masks[2, 0:3, 20:24] = 1
    masks[3, 12, 12] = 1  # a single-pixel region
    # images 4 and 5 are normal
    signal = masks * 0.8 + rng.random(masks.shape).astype(np.float32)
    maps = (signal + 0.3 * rng.random(masks.shape)).astype(np.float32)

    reference = float(_AUPRO(fpr_limit=fpr_limit)(torch.from_numpy(maps), torch.from_numpy(masks)))
    assert metrics.aupro(masks, maps, fpr_limit) == pytest.approx(reference, abs=1e-5)
