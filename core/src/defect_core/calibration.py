"""Calibration: the split-conformal image threshold and the pixel levels (PLAN.md section 5.6).

Pure functions on plain numbers. No model, no image and no dataset file is touched here: the caller
scores the held-out ``calib`` images (training normals, never test images) and passes the scores.

Image threshold (exact rule):

    k = ceil((n + 1) * (1 - alpha)),   threshold = the k-th smallest of the n calibration scores.

The verdict is strict: anomalous only if ``score > threshold``, so a tie is not flagged. Under
exchangeability of the calibration normals and a new normal image, and for scores without ties, the
chance that a new normal image is flagged is (n + 1 - k) / (n + 1), which is at most alpha. That is
a marginal statement about normal images in MVTec-like conditions. It says nothing about the miss
rate, about other cameras or lighting, or about any single calibration set.

A finite threshold needs k <= n, i.e. alpha >= 1 / (n + 1). A smaller alpha raises an error: the
code never falls back to the largest score, because that would deliver only 1 / (n + 1).

Pixel levels (for display only): the lower end of the heatmap scale and the highlight level are
percentiles of all calibration pixel scores, and the upper end is a multiple of the highlight
level. The percentiles and the multiple are configuration values chosen before any test access, so
they have no defaults here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

METHOD = "split-conformal"
SCHEMA_VERSION = 1


class CalibrationError(ValueError):
    """The calibration inputs are invalid, or a calibration block is inconsistent."""


# --- image threshold -----------------------------------------------------------------------


def _exact(value: float) -> Fraction:
    """The decimal the caller wrote (0.05 means exactly 1/20), not the nearest binary float."""
    return Fraction(str(float(value)))


def _check_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not 0 < alpha < 1:
        raise CalibrationError(f"alpha must be a number strictly between 0 and 1, got {alpha!r}")


def _check_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise CalibrationError(f"n must be a positive integer, got {n!r}")


def conformal_rank(n: int, alpha: float) -> int:
    """Return k = ceil((n + 1) * (1 - alpha)), computed in exact arithmetic.

    Raises ``CalibrationError`` if alpha < 1 / (n + 1), where k would exceed n.
    """
    _check_n(n)
    _check_alpha(alpha)
    k = math.ceil((n + 1) * (1 - _exact(alpha)))
    if k > n:
        raise CalibrationError(
            f"alpha={alpha} is too small for n={n} calibration images: the smallest supported "
            f"alpha is 1/(n+1) = {1 / (n + 1):.4f}. Use a larger calibration set or a larger alpha."
        )
    return k


def false_alarm_bound(n: int, k: int) -> float:
    """The marginal false-alarm probability (n + 1 - k) / (n + 1) for scores without ties."""
    _check_n(n)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n:
        raise CalibrationError(f"k must be an integer from 1 to n={n}, got {k!r}")
    return (n + 1 - k) / (n + 1)


def _clean_scores(scores: Iterable[float] | torch.Tensor | np.ndarray) -> np.ndarray:
    array = np.asarray(
        scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores,
        dtype=np.float64,  # exact for float32 inputs
    )
    if array.ndim != 1 or array.size == 0:
        raise CalibrationError("scores must be a non-empty 1-D sequence")
    if not np.isfinite(array).all():
        raise CalibrationError("scores contain NaN or infinite values")
    return array


def is_anomalous(score: float, threshold: float) -> bool:
    """The strict verdict rule: anomalous only if the score is above the threshold."""
    return float(score) > float(threshold)


def normal_rank(score: float, sorted_calibration_scores: Sequence[float]) -> int:
    """How many calibration normals this score is strictly higher than (the "m" of "m of n").

    It is a rank among the calibration normals and nothing more: not a probability and not a
    confidence. Calibration scores equal to ``score`` are not counted.
    """
    scores = np.asarray(sorted_calibration_scores, dtype=np.float64)
    return int(np.searchsorted(scores, float(score), side="left"))


# --- pixel levels --------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelLevels:
    """Display levels computed from calibration pixel scores."""

    lower_percentile: float
    highlight_percentile: float
    upper_multiple: float
    lower: float  # bottom of the heatmap scale
    highlight: float  # level used for the outline and area fraction
    upper: float  # top of the heatmap scale = upper_multiple * highlight


def pixel_levels(
    anomaly_maps: Iterable[torch.Tensor | np.ndarray],
    *,
    lower_percentile: float,
    highlight_percentile: float,
    upper_multiple: float,
) -> PixelLevels:
    """Compute the heatmap scale and highlight level from the calibration anomaly maps.

    Percentiles are taken over every pixel of every map, with linear interpolation
    (``numpy.percentile``'s default). All three settings are required: they are configuration
    chosen before test access, not defaults of this code.
    """
    if not 0 <= lower_percentile < highlight_percentile <= 100:
        raise CalibrationError(
            "need 0 <= lower_percentile < highlight_percentile <= 100, got "
            f"{lower_percentile} and {highlight_percentile}"
        )
    if not upper_multiple > 0:
        raise CalibrationError(f"upper_multiple must be positive, got {upper_multiple}")
    parts = [
        np.asarray(m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m, np.float64)
        .reshape(-1)
        .copy()
        for m in anomaly_maps
    ]
    if not parts:
        raise CalibrationError("no anomaly maps given")
    pixels = np.concatenate(parts)
    if not np.isfinite(pixels).all():
        raise CalibrationError("anomaly maps contain NaN or infinite values")
    lower, highlight = (
        float(x) for x in np.percentile(pixels, [lower_percentile, highlight_percentile])
    )
    upper = float(upper_multiple) * highlight
    if not upper > lower:
        raise CalibrationError(
            f"heatmap scale is empty: upper={upper} is not above lower={lower} "
            "(check the multiple and the percentiles)"
        )
    return PixelLevels(
        float(lower_percentile),
        float(highlight_percentile),
        float(upper_multiple),
        lower,
        highlight,
        upper,
    )


# --- the calibration block (what goes into the manifest) -----------------------------------


@dataclass(frozen=True)
class Calibration:
    """Everything calibration decides, in a form that can be stored and checked."""

    alpha: float
    n: int
    k: int
    threshold: float
    calibration_scores: tuple[float, ...]  # sorted ascending: the rank table
    pixel: PixelLevels

    @property
    def false_alarm_bound(self) -> float:
        return false_alarm_bound(self.n, self.k)

    def is_anomalous(self, score: float) -> bool:
        return is_anomalous(score, self.threshold)

    def normal_rank(self, score: float) -> int:
        return normal_rank(score, self.calibration_scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "alpha": self.alpha,
            "n": self.n,
            "k": self.k,
            "threshold": self.threshold,
            "false_alarm_bound": self.false_alarm_bound,
            "calibration_scores": list(self.calibration_scores),
            "pixel": {
                "lower_percentile": self.pixel.lower_percentile,
                "highlight_percentile": self.pixel.highlight_percentile,
                "upper_multiple": self.pixel.upper_multiple,
                "lower": self.pixel.lower,
                "highlight": self.pixel.highlight,
                "upper": self.pixel.upper,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Calibration:
        """Read a stored block and check that it is internally consistent.

        Raises ``CalibrationError`` if, for example, ``k`` or ``threshold`` do not follow from
        ``alpha`` and the stored scores, or the scores are not sorted. This is the check a loader
        uses to catch a calibration block that does not match itself.
        """
        try:
            if data["schema_version"] != SCHEMA_VERSION or data["method"] != METHOD:
                raise CalibrationError("Unsupported calibration schema_version or method")
            scores = tuple(float(x) for x in data["calibration_scores"])  # type: ignore[union-attr]
            pixel_data = data["pixel"]
            pixel = PixelLevels(**{k: float(v) for k, v in pixel_data.items()})  # type: ignore[union-attr]
            block = cls(
                alpha=float(data["alpha"]),  # type: ignore[arg-type]
                n=int(data["n"]),  # type: ignore[call-overload]
                k=int(data["k"]),  # type: ignore[call-overload]
                threshold=float(data["threshold"]),  # type: ignore[arg-type]
                calibration_scores=scores,
                pixel=pixel,
            )
            stored_bound = data["false_alarm_bound"]
        except CalibrationError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CalibrationError(f"Calibration block is malformed: {error!r}") from error

        problems = []
        if block.n != len(block.calibration_scores):
            problems.append("n does not equal the number of stored scores")
        if list(block.calibration_scores) != sorted(block.calibration_scores):
            problems.append("stored scores are not sorted")
        if not all(math.isfinite(s) for s in block.calibration_scores):
            problems.append("stored scores contain NaN or infinite values")
        if not problems:
            if block.k != conformal_rank(block.n, block.alpha):
                problems.append("k does not follow from n and alpha")
            elif block.threshold != block.calibration_scores[block.k - 1]:
                problems.append("threshold is not the k-th smallest stored score")
            elif stored_bound != block.false_alarm_bound:
                problems.append("false_alarm_bound does not follow from n and k")
        if not (p := block.pixel).lower < p.highlight or p.upper != p.upper_multiple * p.highlight:
            problems.append("pixel levels are inconsistent")
        if problems:
            raise CalibrationError("Inconsistent calibration block: " + "; ".join(problems))
        return block


def build_calibration(
    image_scores: Iterable[float] | torch.Tensor | np.ndarray,
    anomaly_maps: Iterable[torch.Tensor | np.ndarray],
    *,
    alpha: float,
    lower_percentile: float,
    highlight_percentile: float,
    upper_multiple: float,
) -> Calibration:
    """Calibrate from the frozen model's outputs on the held-out calibration images.

    ``image_scores`` are the n image scores and ``anomaly_maps`` the n anomaly maps (one per
    calibration image). Nothing here reads any image.
    """
    scores = np.sort(_clean_scores(image_scores))
    n = int(scores.size)
    k = conformal_rank(n, alpha)
    maps = list(anomaly_maps)
    if len(maps) != n:
        raise CalibrationError(f"got {n} image scores but {len(maps)} anomaly maps")
    pixel = pixel_levels(
        maps,
        lower_percentile=lower_percentile,
        highlight_percentile=highlight_percentile,
        upper_multiple=upper_multiple,
    )
    return Calibration(
        alpha=float(alpha),
        n=n,
        k=k,
        threshold=float(scores[k - 1]),
        calibration_scores=tuple(float(s) for s in scores),
        pixel=pixel,
    )
