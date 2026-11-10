"""Turn an anomaly map into the pictures and numbers the app shows (PLAN.md Phase 3 step 4).

Plain arrays in, plain arrays out: no file, no network, no matplotlib. The caller loads the image
and encodes the results (for example as PNG).

What it produces, from one image and its raw anomaly map:

* a **fixed-scale heatmap**: ``(map - lower) / (upper - lower)`` clipped to 0..1 with the
  calibration levels, never rescaled per image, coloured with the inferno table;
* an **overlay**: the heatmap blended over the original image, with an outline around the pixels
  above the highlight level;
* the **bounding box** of the largest highlighted region, ``(x, y, width, height)`` in image
  pixels, and the **highlighted area fraction**.

All of it is computed at the original image size: the 256 x 256 anomaly map is first resized to the
image with bilinear interpolation, so the outline, the box and the fraction describe the same
pixels. The highlighted area is the pixels *above* the highlight level (strict), and regions are
8-connected (touching diagonally counts as one region).

The highlight level and the heatmap limits come from calibration (``calibration.PixelLevels``). They
are display aids; they are not part of the verdict, which uses the image score alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from defect_core.calibration import PixelLevels

LUT_FILE = "inferno_lut.json"


class RenderError(ValueError):
    """The inputs to rendering are invalid."""


@dataclass(frozen=True)
class RenderSettings:
    """Cosmetic settings. They change how results look, never the verdict."""

    overlay_alpha: float = 0.45  # weight of the heatmap in the overlay
    outline_color: tuple[int, int, int] = (255, 255, 255)
    outline_width: int = 1  # pixels

    def __post_init__(self) -> None:
        if not 0 < self.overlay_alpha < 1:
            raise RenderError(f"overlay_alpha must be between 0 and 1, got {self.overlay_alpha}")
        if self.outline_width < 1 or not float(self.outline_width).is_integer():
            msg = f"outline_width must be a whole number >= 1, got {self.outline_width}"
            raise RenderError(msg)
        if len(self.outline_color) != 3 or not all(0 <= c <= 255 for c in self.outline_color):
            msg = f"outline_color must be three values 0..255, got {self.outline_color}"
            raise RenderError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "overlay_alpha": self.overlay_alpha,
            "outline_color": list(self.outline_color),
            "outline_width": self.outline_width,
        }


@dataclass(frozen=True)
class RenderResult:
    heatmap: np.ndarray  # uint8 (H, W, 3): the fixed-scale heatmap alone
    overlay: np.ndarray  # uint8 (H, W, 3): image + heatmap + outline
    bbox: tuple[int, int, int, int] | None  # (x, y, width, height) of the largest region
    area_fraction: float  # highlighted pixels / all pixels


# --- colour table and fixed-scale normalisation --------------------------------------------


@cache
def inferno_table() -> np.ndarray:
    """The 256-entry inferno lookup table as a read-only uint8 array of shape (256, 3)."""
    text = resources.files("defect_core").joinpath(LUT_FILE).read_text()
    table = np.array(json.loads(text)["rgb"], dtype=np.int64)
    if table.shape != (256, 3) or table.min() < 0 or table.max() > 255:
        raise RenderError(f"{LUT_FILE} must hold 256 RGB entries of 0..255")
    table = table.astype(np.uint8)
    table.setflags(write=False)
    return table


def normalise_fixed_scale(anomaly_map: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map scores to 0..1 with the fixed calibration limits; values outside are clipped."""
    if not upper > lower:
        raise RenderError(f"upper ({upper}) must be above lower ({lower})")
    values = np.asarray(anomaly_map, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RenderError("anomaly map contains NaN or infinite values")
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def colorize(normalised: np.ndarray) -> np.ndarray:
    """Colour values in 0..1 with the inferno table; returns uint8 (H, W, 3)."""
    values = np.asarray(normalised, dtype=np.float64)
    if values.min() < 0 or values.max() > 1 or not np.isfinite(values).all():
        raise RenderError("colorize expects finite values between 0 and 1")
    return inferno_table()[np.rint(values * 255).astype(np.intp)]


# --- masks, outline, regions ---------------------------------------------------------------


def resize_map(anomaly_map: torch.Tensor | np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a 2-D anomaly map to (height, width) with bilinear interpolation (float32)."""
    tensor = torch.as_tensor(np.asarray(anomaly_map), dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == 1:  # (1, H, W), as the model returns per image
        tensor = tensor[0]
    if tensor.ndim != 2:
        msg = f"anomaly map must be 2-D (or 1 x H x W), got shape {tuple(tensor.shape)}"
        raise RenderError(msg)
    if tuple(tensor.shape) != (height, width):
        tensor = F.interpolate(
            tensor[None, None], size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
    return tensor.numpy()


def highlight_mask(anomaly_map: np.ndarray, highlight: float) -> np.ndarray:
    """Pixels strictly above the highlight level."""
    return np.asarray(anomaly_map) > highlight


def outline(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """The inner border of the mask, ``width`` pixels thick.

    A region that reaches the image edge is not outlined along that edge (the region is treated as
    continuing beyond it), so no line is drawn on the picture's border.
    """
    inner = mask
    for _ in range(width):
        padded = np.pad(inner, 1, mode="edge")
        inner = inner & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~inner


def largest_region_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x, y, width, height)`` of the largest 8-connected region, or None.

    "Largest" means most pixels. If two regions tie, the one whose first pixel comes earlier in
    reading order (top to bottom, then left to right) wins, so the result is deterministic.
    The search works on horizontal runs of pixels, which keeps it fast on large images.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise RenderError("mask must be 2-D")
    runs: list[tuple[int, int, int]] = []  # (row, first column, last column), in reading order
    row_slices: list[tuple[int, int]] = []  # runs[start:end] belong to one row
    for y in range(mask.shape[0]):
        padded = np.concatenate(([0], mask[y].astype(np.int8), [0]))
        change = np.diff(padded)
        starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1
        first = len(runs)
        runs.extend((y, int(a), int(b)) for a, b in zip(starts, ends, strict=True))
        row_slices.append((first, len(runs)))
    if not runs:
        return None

    parent = list(range(len(runs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(1, mask.shape[0]):
        (a0, a1), (b0, b1) = row_slices[y], row_slices[y - 1]
        for i in range(a0, a1):
            for j in range(b0, b1):
                # runs touch if their column ranges overlap or meet diagonally
                if runs[j][1] <= runs[i][2] + 1 and runs[i][1] <= runs[j][2] + 1:
                    parent[find(i)] = find(j)

    # per region: area, box, and the index of its first run (for the tie-break)
    stats: dict[int, list[int]] = {}
    for index, (y, x0, x1) in enumerate(runs):
        root = find(index)
        entry = stats.get(root)
        if entry is None:
            stats[root] = [x1 - x0 + 1, x0, y, x1, y, index]
        else:
            entry[0] += x1 - x0 + 1
            entry[1], entry[2] = min(entry[1], x0), min(entry[2], y)
            entry[3], entry[4] = max(entry[3], x1), max(entry[4], y)
            entry[5] = min(entry[5], index)
    area, x0, y0, x1, y1, _ = min(stats.values(), key=lambda e: (-e[0], e[5]))
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def area_fraction(mask: np.ndarray) -> float:
    """Highlighted pixels divided by all pixels."""
    mask = np.asarray(mask, dtype=bool)
    return float(mask.sum()) / mask.size


# --- putting it together -------------------------------------------------------------------


def make_overlay(
    image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray, settings: RenderSettings
) -> np.ndarray:
    """Blend the heatmap over the image and draw the outline of ``mask`` on top."""
    alpha = settings.overlay_alpha
    blended = np.rint(alpha * heatmap.astype(np.float64) + (1 - alpha) * image.astype(np.float64))
    result = blended.astype(np.uint8)
    result[outline(mask, settings.outline_width)] = settings.outline_color
    return result


def render_result(
    image: Image.Image | np.ndarray,
    anomaly_map: torch.Tensor | np.ndarray,
    levels: PixelLevels,
    settings: RenderSettings | None = None,
) -> RenderResult:
    """Render one inspection result at the original image size.

    ``image`` is the RGB original (a PIL image or a uint8 array of shape (H, W, 3)); ``anomaly_map``
    is the model's raw map for it; ``levels`` are the calibration pixel levels.
    """
    settings = settings or RenderSettings()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise RenderError("image must be an RGB uint8 array of shape (H, W, 3)")
    height, width = array.shape[:2]

    resized = resize_map(anomaly_map, height, width)
    heatmap = colorize(normalise_fixed_scale(resized, levels.lower, levels.upper))
    mask = highlight_mask(resized, levels.highlight)
    return RenderResult(
        heatmap=heatmap,
        overlay=make_overlay(array, heatmap, mask, settings),
        bbox=largest_region_bbox(mask),
        area_fraction=area_fraction(mask),
    )
