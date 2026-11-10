"""Tests for defect_core.render (PLAN.md Phase 3 step 4). Synthetic arrays only."""

from __future__ import annotations

import dataclasses

import matplotlib
import numpy as np
import pytest
import torch
from defect_core.calibration import PixelLevels
from defect_core.render import (
    RenderError,
    RenderSettings,
    area_fraction,
    colorize,
    highlight_mask,
    inferno_table,
    largest_region_bbox,
    make_overlay,
    normalise_fixed_scale,
    outline,
    render_result,
    resize_map,
)
from PIL import Image
from scipy import ndimage

LEVELS = PixelLevels(
    lower_percentile=1.0,
    highlight_percentile=99.5,
    upper_multiple=2.0,
    lower=1.0,
    highlight=5.0,
    upper=10.0,
)


def _blob_map(height: int = 64, width: int = 64) -> np.ndarray:
    """A map that is 0 everywhere except a 10 x 6 block (rows 20..29, columns 30..35) of 8.0."""
    array = np.zeros((height, width), dtype=np.float32)
    array[20:30, 30:36] = 8.0
    return array


# --- colour table and fixed scale ----------------------------------------------------------


def test_the_colour_table_matches_matplotlib_inferno() -> None:
    table = inferno_table()
    assert table.shape == (256, 3) and table.dtype == np.uint8
    rgba = matplotlib.colormaps["inferno"](np.arange(256))
    expected = np.rint(rgba[:, :3] * 255).astype(np.uint8)
    assert np.array_equal(table, expected)
    assert tuple(table[0]) == (0, 0, 4) and tuple(table[255]) == (252, 255, 164)
    assert not table.flags.writeable


def test_fixed_scale_normalisation_and_clipping() -> None:
    values = np.array([[0.0, 1.0, 5.5, 10.0, 99.0]])
    out = normalise_fixed_scale(values, lower=1.0, upper=10.0)
    assert out.dtype == np.float32
    assert out[0].tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])  # clipped at both ends


def test_the_scale_is_fixed_not_stretched_per_image() -> None:
    weak, strong = np.full((4, 4), 3.0), np.full((4, 4), 9.0)
    weak_out = normalise_fixed_scale(weak, 1.0, 10.0)
    strong_out = normalise_fixed_scale(strong, 1.0, 10.0)
    assert weak_out.max() < strong_out.min()
    assert weak_out.max() < 0.3  # a flat, normal-looking map is not stretched into a "hot" one


def test_bad_scale_or_values_are_refused() -> None:
    with pytest.raises(RenderError, match="above lower"):
        normalise_fixed_scale(np.ones((2, 2)), 5.0, 5.0)
    with pytest.raises(RenderError, match="NaN"):
        normalise_fixed_scale(np.array([[np.nan]]), 1.0, 2.0)
    with pytest.raises(RenderError, match="between 0 and 1"):
        colorize(np.array([[1.5]]))


def test_colorize_uses_the_table_ends() -> None:
    out = colorize(np.array([[0.0, 1.0]]))
    assert out.shape == (1, 2, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == tuple(inferno_table()[0])
    assert tuple(out[0, 1]) == tuple(inferno_table()[255])


# --- regions, area, outline ----------------------------------------------------------------


def test_a_known_rectangle_gives_its_exact_box_and_area() -> None:
    mask = highlight_mask(_blob_map(), highlight=5.0)
    assert largest_region_bbox(mask) == (30, 20, 6, 10)  # (x, y, width, height)
    assert area_fraction(mask) == 60 / (64 * 64)


def test_the_highlight_rule_is_strictly_above() -> None:
    values = np.array([[4.9, 5.0, 5.1]])
    assert highlight_mask(values, 5.0).tolist() == [[False, False, True]]


def test_of_two_regions_the_larger_wins() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:5, 2:5] = True  # 9 pixels
    mask[20:30, 15:25] = True  # 100 pixels
    assert largest_region_bbox(mask) == (15, 20, 10, 10)


def test_equal_regions_tie_to_the_one_first_in_reading_order() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[30:33, 2:5] = True  # lower on the page
    mask[5:8, 30:33] = True  # higher up, further right: comes first in reading order
    assert largest_region_bbox(mask) == (30, 5, 3, 3)


def test_diagonal_neighbours_form_one_region_but_gaps_do_not() -> None:
    diagonal = np.zeros((6, 6), dtype=bool)
    diagonal[1, 1] = diagonal[2, 2] = diagonal[3, 3] = True
    assert largest_region_bbox(diagonal) == (1, 1, 3, 3)  # one region of 3 pixels
    gap = np.zeros((6, 6), dtype=bool)
    gap[1, 1] = gap[1, 3] = True  # one empty column between them
    assert largest_region_bbox(gap)[2:] == (1, 1)  # two separate single pixels


def test_no_highlighted_pixels_gives_no_box_and_zero_area() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    assert largest_region_bbox(mask) is None
    assert area_fraction(mask) == 0.0
    assert not outline(mask).any()


def test_region_finding_matches_scipy_on_random_masks() -> None:
    rng = np.random.default_rng(0)
    eight = np.ones((3, 3), dtype=int)
    for density in (0.1, 0.3, 0.5, 0.65):
        for _ in range(15):
            mask = rng.random((30, 45)) < density
            labels, count = ndimage.label(mask, structure=eight)
            if count == 0:
                assert largest_region_bbox(mask) is None
                continue
            sizes = np.bincount(labels.ravel())[1:]
            # reading-order first pixel of each label decides ties, as in the implementation
            first = [np.flatnonzero(labels.ravel() == i + 1)[0] for i in range(count)]
            best = min(range(count), key=lambda i: (-sizes[i], first[i]))
            ys, xs = np.nonzero(labels == best + 1)
            expected = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            assert largest_region_bbox(mask) == tuple(int(v) for v in expected)


def test_outline_is_the_one_pixel_inner_border() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True  # 5 x 5 square
    border = outline(mask, 1)
    assert border.sum() == 16  # the 5x5 ring
    assert not border[4, 4]  # the centre is not outlined
    assert outline(mask, 2).sum() == 24  # a ring two pixels thick
    assert (border & ~mask).sum() == 0  # always inside the region


def test_a_region_touching_the_image_edge_is_not_outlined_along_the_edge() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, :4] = True  # touches the left, top and bottom edges
    border = outline(mask)
    assert not border[:, 0].any()  # no line on the left edge
    assert border[:, 3].all()  # the inner side facing the rest of the image is outlined


# --- resize, overlay, whole pipeline -------------------------------------------------------


def test_resize_keeps_values_and_shape() -> None:
    small = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.array_equal(resize_map(small, 4, 4), small)
    big = resize_map(small, 8, 12)
    assert big.shape == (8, 12) and big.dtype == np.float32
    assert small.min() <= big.min() and big.max() <= small.max()
    assert resize_map(torch.from_numpy(small)[None], 8, 12).shape == (8, 12)  # (1, H, W) accepted
    with pytest.raises(RenderError, match="2-D"):
        resize_map(np.zeros((2, 3, 4)), 8, 8)


def test_overlay_blends_with_the_given_weight_and_draws_the_outline() -> None:
    image = np.full((9, 9, 3), 100, dtype=np.uint8)
    heat = np.full((9, 9, 3), 200, dtype=np.uint8)
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    settings = RenderSettings(overlay_alpha=0.5, outline_color=(1, 2, 3))
    overlay = make_overlay(image, heat, mask, settings)
    assert overlay.dtype == np.uint8 and overlay.shape == image.shape
    assert tuple(overlay[0, 0]) == (150, 150, 150)  # 0.5 * 200 + 0.5 * 100
    assert tuple(overlay[2, 2]) == (1, 2, 3)  # on the outline
    assert tuple(overlay[4, 4]) == (150, 150, 150)  # inside, not outlined
    assert image.max() == 100 and heat.min() == 200  # inputs are not changed


def test_the_whole_pipeline_on_a_known_blob() -> None:
    anomaly = _blob_map(64, 64)
    image = Image.fromarray(np.full((128, 128, 3), 90, dtype=np.uint8))  # twice the map size
    snapshot = anomaly.copy()
    result = render_result(image, anomaly, LEVELS)
    assert result.heatmap.shape == (128, 128, 3) and result.heatmap.dtype == np.uint8
    assert result.overlay.shape == (128, 128, 3) and result.overlay.dtype == np.uint8
    x, y, w, h = result.bbox
    # block at columns 30..35, rows 20..29 of the 64 x 64 map, scaled by 2 (edges are interpolated)
    assert (x, y, w, h) == pytest.approx((60, 40, 12, 20), abs=3)
    assert result.area_fraction == pytest.approx(60 * 4 / (128 * 128), rel=0.2)
    assert np.array_equal(anomaly, snapshot)  # the input map is not changed
    # a pixel far from the blob is cold (low end of the table), inside the blob it is hotter
    assert result.heatmap[5, 5].sum() < result.heatmap[50, 66].sum()


def test_a_normal_looking_map_has_no_highlight() -> None:
    quiet = np.full((64, 64), 2.0, dtype=np.float32)  # between lower (1.0) and highlight (5.0)
    image = np.full((50, 70, 3), 128, dtype=np.uint8)
    result = render_result(image, quiet, LEVELS)
    assert result.bbox is None and result.area_fraction == 0.0
    assert result.overlay.shape == (50, 70, 3)


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(RenderError, match="RGB uint8"):
        render_result(np.zeros((8, 8), dtype=np.uint8), np.zeros((4, 4)), LEVELS)
    with pytest.raises(RenderError, match="RGB uint8"):
        render_result(np.zeros((8, 8, 3), dtype=np.float32), np.zeros((4, 4)), LEVELS)


def test_settings_are_validated_frozen_and_serialisable() -> None:
    settings = RenderSettings()
    assert (settings.overlay_alpha, settings.outline_color, settings.outline_width) == (
        0.45,
        (255, 255, 255),
        1,
    )
    assert settings.to_dict() == {
        "overlay_alpha": 0.45,
        "outline_color": [255, 255, 255],
        "outline_width": 1,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.overlay_alpha = 0.5  # type: ignore[misc]
    for bad in (
        {"overlay_alpha": 0.0},
        {"overlay_alpha": 1.0},
        {"outline_width": 0},
        {"outline_color": (256, 0, 0)},
        {"outline_color": (1, 2)},
    ):
        with pytest.raises(RenderError):
            RenderSettings(**bad)  # type: ignore[arg-type]
