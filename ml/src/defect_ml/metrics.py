"""Evaluation metrics (PLAN.md section 5.7, Phase 3 step 5).

Plain arrays in, plain numbers out. Nothing here reads a file, an image or the dataset, so it can be
written and tested before the sealed test set is opened, and the evaluation (``evaluate.py``, not
written yet) only has to call it.

Conventions (each is tested):

* **Labels** are 0/1 or bool, with ``1`` = anomalous (the positive class). **Scores** are the raw
  anomaly scores (higher = more anomalous) and must be finite.
* **The verdict** is ``defect_core.calibration.is_anomalous``: ``score > threshold``, strictly. A
  score equal to the threshold is normal. There is no second copy of that rule.
* **Undefined is not zero.** A rate whose denominator is zero (for example precision when nothing
  was flagged) is ``None``, never ``0.0`` or ``nan``. A curve metric that needs both classes
  (AUROC, AUPR, pixel AUROC, AUPRO, the bootstrap) raises ``MetricsError`` when a class is missing.
* **AUROC / AUPR** use scikit-learn. AUPR is average precision (``average_precision_score``, the
  step-wise sum, not the trapezoid rule), with anomalous as positive. Tied scores are handled by
  scikit-learn (all-equal scores give AUROC 0.5).
* **Bootstrap**: percentile interval, resampling **images** (never pixels) with replacement,
  separately within the normal and the anomalous images, so every draw keeps the original class
  counts. The generator is ``numpy.random.default_rng(seed)``; the same seed gives the same
  interval. It needs at least 2 images of each class.
* **Pixel metrics** take masks and maps of the same shape ``(N, H, W)``; the caller chooses the
  resolution and resizes with ``defect_core.render.resize_map``. A pixel is anomalous if the mask
  is 1.
* **AUPRO** follows MVTec's definition (as does anomalib's implementation, which it is checked
  against): regions are the 8-connected components of each ground-truth mask, found per image; for
  every threshold the per-region overlap (fraction of the region's pixels at or above the
  threshold) is averaged over all regions (a small region counts as much as a large one); this is
  plotted against the false-positive rate over all normal pixels; the area under that curve up to
  ``fpr_limit`` is divided by ``fpr_limit``. Every distinct score is a threshold; the curve is cut
  at ``fpr_limit`` by linear interpolation.
* **Highlight precision/recall** use the same strict rule as the outline in ``render``: pixels
  strictly above the highlight level.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from defect_core.calibration import is_anomalous
from defect_core.render import highlight_mask
from sklearn.metrics import average_precision_score, roc_auc_score


class MetricsError(ValueError):
    """The inputs to a metric are invalid, or the metric is not defined for them."""


# --- checking inputs ------------------------------------------------------------------------


def _labels_and_scores(labels: object, scores: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape:
        msg = f"labels and scores must be 1-D and equally long, got {y.shape}, {s.shape}"
        raise MetricsError(msg)
    if y.size == 0:
        raise MetricsError("no images given")
    if not np.isin(y, (0, 1)).all():
        raise MetricsError("labels must be 0 or 1 (1 = anomalous)")
    if not np.isfinite(s).all():
        raise MetricsError("scores contain NaN or infinite values")
    return y.astype(bool), s


def _both_classes(y: np.ndarray, what: str) -> None:
    if y.all() or not y.any():
        raise MetricsError(f"{what} needs both normal and anomalous items; got only one class")


# --- image level: AUROC, AUPR, bootstrap ----------------------------------------------------


def image_auroc(labels: object, scores: object) -> float:
    y, s = _labels_and_scores(labels, scores)
    _both_classes(y, "AUROC")
    return float(roc_auc_score(y, s))


def image_aupr(labels: object, scores: object) -> float:
    """Average precision with anomalous as the positive class."""
    y, s = _labels_and_scores(labels, scores)
    _both_classes(y, "AUPR")
    return float(average_precision_score(y, s))


@dataclass(frozen=True)
class Interval:
    estimate: float  # the metric on the original data
    low: float
    high: float
    confidence: float
    n_resamples: int
    seed: int
    method: str = "percentile bootstrap, images resampled within each class"

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate, "low": self.low, "high": self.high,
            "confidence": self.confidence, "n_resamples": self.n_resamples,
            "seed": self.seed, "method": self.method,
        }  # fmt: skip


def bootstrap_interval(
    labels: object,
    scores: object,
    statistic: Callable[[np.ndarray, np.ndarray], float] = image_auroc,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap interval of an image-level statistic, stratified by class."""
    y, s = _labels_and_scores(labels, scores)
    positives, negatives = np.flatnonzero(y), np.flatnonzero(~y)
    if len(positives) < 2 or len(negatives) < 2:
        raise MetricsError(
            "the bootstrap needs at least 2 normal and 2 anomalous images, got "
            f"{len(negatives)} and {len(positives)}"
        )
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 100:
        raise MetricsError(f"n_resamples must be a whole number >= 100, got {n_resamples!r}")
    if not 0 < confidence < 1:
        raise MetricsError(f"confidence must be between 0 and 1, got {confidence}")
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = np.concatenate([
            rng.choice(positives, size=len(positives), replace=True),
            rng.choice(negatives, size=len(negatives), replace=True),
        ])  # fmt: skip
        values[i] = statistic(y[pick], s[pick])
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return Interval(float(statistic(y, s)), float(low), float(high), confidence, n_resamples, seed)


# --- image level: at the deployed threshold -------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class ThresholdMetrics:
    """Counts and rates at one threshold. A rate with a zero denominator is ``None``."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def tpr(self) -> float | None:  # recall of anomalous images
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def fpr(self) -> float | None:  # false alarms among normal images
        return _ratio(self.fp, self.fp + self.tn)

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def f1(self) -> float | None:
        return None if (d := 2 * self.tp + self.fp + self.fn) == 0 else 2 * self.tp / d

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold, "tp": self.tp, "fp": self.fp, "tn": self.tn,
            "fn": self.fn, "tpr": self.tpr, "fpr": self.fpr, "precision": self.precision,
            "f1": self.f1,
        }  # fmt: skip


def threshold_metrics(labels: object, scores: object, threshold: float) -> ThresholdMetrics:
    """Confusion counts and rates for the verdict ``score > threshold`` (strict)."""
    y, s = _labels_and_scores(labels, scores)
    if not math.isfinite(threshold):
        raise MetricsError("the threshold must be finite")
    flagged = np.array([is_anomalous(float(v), threshold) for v in s], dtype=bool)
    return ThresholdMetrics(
        threshold=float(threshold),
        tp=int((flagged & y).sum()), fp=int((flagged & ~y).sum()),
        tn=int((~flagged & ~y).sum()), fn=int((~flagged & y).sum()),
    )  # fmt: skip


@dataclass(frozen=True)
class DefectRecall:
    n: int
    detected: int

    @property
    def recall(self) -> float:
        return self.detected / self.n

    def to_dict(self) -> dict[str, object]:
        return {"n": self.n, "detected": self.detected, "recall": self.recall}


def per_defect_recall(
    defect_types: Sequence[str], scores: object, threshold: float
) -> dict[str, DefectRecall]:
    """Recall at the threshold for each defect type, over anomalous images only.

    ``defect_types[i]`` is the dataset folder name of anomalous image ``i`` (used for evaluation
    only); ``scores[i]`` is its score. The result is sorted by type name.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or len(defect_types) != s.size or s.size == 0:
        raise MetricsError("defect_types and scores must be non-empty and equally long")
    if not np.isfinite(s).all() or not math.isfinite(threshold):
        raise MetricsError("scores and threshold must be finite")
    if any(not isinstance(t, str) or not t for t in defect_types):
        raise MetricsError("every defect type must be a non-empty name")
    result: dict[str, DefectRecall] = {}
    for name in sorted(set(defect_types)):
        pairs = zip(defect_types, s, strict=True)
        hits = [is_anomalous(float(v), threshold) for t, v in pairs if t == name]
        result[name] = DefectRecall(len(hits), sum(hits))
    return result


# --- pixel level ----------------------------------------------------------------------------


def _masks_and_maps(masks: object, maps: object) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(masks)
    a = np.asarray(maps, dtype=np.float64)
    if m.ndim != 3 or m.shape != a.shape or m.size == 0:
        raise MetricsError(f"masks and maps must both be (N, H, W); got {m.shape} and {a.shape}")
    if not np.isin(m, (0, 1)).all():
        raise MetricsError("masks must contain only 0 and 1")
    if not np.isfinite(a).all():
        raise MetricsError("anomaly maps contain NaN or infinite values")
    return m.astype(bool), a


def pixel_auroc(masks: object, maps: object) -> float:
    """AUROC over every pixel of every image (easy background pixels inflate it; see AUPRO)."""
    m, a = _masks_and_maps(masks, maps)
    _both_classes(m.reshape(-1), "pixel AUROC")
    return float(roc_auc_score(m.reshape(-1), a.reshape(-1)))


@dataclass(frozen=True)
class HighlightPixels:
    """Precision and recall of the pixels strictly above the highlight level."""

    highlight: float
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    def to_dict(self) -> dict[str, object]:
        return {
            "highlight": self.highlight, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall,
        }  # fmt: skip


def highlight_pixel_metrics(masks: object, maps: object, highlight: float) -> HighlightPixels:
    m, a = _masks_and_maps(masks, maps)
    if not math.isfinite(highlight):
        raise MetricsError("the highlight level must be finite")
    flagged = highlight_mask(a, highlight)
    tp, fp, fn = (int((flagged & m).sum()), int((flagged & ~m).sum()), int((~flagged & m).sum()))
    return HighlightPixels(float(highlight), tp, fp, fn)


def label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the 8-connected regions of a 2-D bool mask; returns (labels 0..n, n).

    Pixels that touch diagonally belong to one region. Works on horizontal runs of pixels, so it is
    fast on large masks. (Same connectivity as OpenCV's default, which anomalib uses.)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise MetricsError("mask must be 2-D")
    runs: list[tuple[int, int, int]] = []  # (row, first column, last column)
    spans: list[tuple[int, int]] = []
    for y in range(mask.shape[0]):
        change = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1
        first = len(runs)
        runs.extend((y, int(a), int(b)) for a, b in zip(starts, ends, strict=True))
        spans.append((first, len(runs)))
    parent = list(range(len(runs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(1, mask.shape[0]):
        (a0, a1), (b0, b1) = spans[y], spans[y - 1]
        for i in range(a0, a1):
            for j in range(b0, b1):
                if runs[j][1] <= runs[i][2] + 1 and runs[i][1] <= runs[j][2] + 1:
                    parent[find(i)] = find(j)
    labels = np.zeros(mask.shape, dtype=np.int32)
    ids: dict[int, int] = {}
    for index, (y, x0, x1) in enumerate(runs):
        labels[y, x0 : x1 + 1] = ids.setdefault(find(index), len(ids) + 1)
    return labels, len(ids)


def pro_curve(masks: object, maps: object, fpr_limit: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """The PRO curve ``(fpr, pro)`` from (0, 0) up to ``fpr_limit``, both increasing."""
    m, a = _masks_and_maps(masks, maps)
    if not 0 < fpr_limit <= 1:
        raise MetricsError(f"fpr_limit must be in (0, 1], got {fpr_limit}")
    region_of = np.zeros(m.shape, dtype=np.int64)
    n_regions = 0
    for i in range(m.shape[0]):  # regions are found per image, so equal positions never merge
        labels, count = label_regions(m[i])
        region_of[i] = np.where(labels > 0, labels + n_regions, 0)
        n_regions += count
    if n_regions == 0:
        raise MetricsError("AUPRO needs at least one anomalous region in the masks")
    background = region_of == 0
    n_background = int(background.sum())
    if n_background == 0:
        raise MetricsError("AUPRO needs normal (background) pixels")

    flat_region, flat_score = region_of.reshape(-1), a.reshape(-1)
    sizes = np.bincount(flat_region, minlength=n_regions + 1).astype(np.float64)
    gain = np.zeros(flat_score.size)  # what flagging this pixel adds to the averaged overlap
    foreground = flat_region > 0
    gain[foreground] = 1.0 / sizes[flat_region[foreground]]
    order = np.argsort(-flat_score, kind="stable")  # highest score first
    sorted_score = flat_score[order]
    fpr = np.cumsum(background.reshape(-1)[order]) / n_background
    pro = np.cumsum(gain[order]) / n_regions
    last_of_tie = np.append(sorted_score[:-1] != sorted_score[1:], True)  # a threshold flags ties
    fpr = np.concatenate(([0.0], np.minimum(fpr[last_of_tie], 1.0)))
    pro = np.concatenate(([0.0], np.minimum(pro[last_of_tie], 1.0)))

    inside = np.flatnonzero(fpr <= fpr_limit)
    last = int(inside[-1])  # fpr[0] == 0, so this always exists
    if fpr[last] < fpr_limit and last + 1 < fpr.size:
        share = (fpr_limit - fpr[last]) / (fpr[last + 1] - fpr[last])
        cut = pro[last] + (pro[last + 1] - pro[last]) * share
        return np.append(fpr[: last + 1], fpr_limit), np.append(pro[: last + 1], cut)
    return fpr[: last + 1], pro[: last + 1]


def aupro(masks: object, maps: object, fpr_limit: float = 0.3) -> float:
    """Area under the PRO curve up to ``fpr_limit``, divided by ``fpr_limit`` (so 1 is perfect)."""
    fpr, pro = pro_curve(masks, maps, fpr_limit)
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (pro[1:] + pro[:-1]) / 2))
    return area / float(fpr[-1])
