"""Tests for defect_core.preprocessing (Phase 2, step 1).

Only small synthetic images made in memory are used. No dataset file is read.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from defect_core.preprocessing import (
    DEFAULT_CONFIG,
    PREPROCESSING_VERSION,
    ImageLoadError,
    PreprocessConfig,
    load_image,
    to_tensor,
)
from PIL import Image
from torchvision.transforms import v2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _png_bytes(image: Image.Image, **save_kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **save_kwargs)
    return buffer.getvalue()


def _gradient_rgb(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


# --- the config ----------------------------------------------------------------------------


def test_default_config_matches_the_plan() -> None:
    config = PreprocessConfig()
    assert config == DEFAULT_CONFIG
    assert config.version == PREPROCESSING_VERSION
    assert config.size == 256
    assert config.mean == (0.485, 0.456, 0.406)
    assert config.std == (0.229, 0.224, 0.225)
    assert config.interpolation == "bilinear"
    assert config.antialias is True


def test_config_is_frozen_and_serialises_to_plain_data() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.size = 128  # type: ignore[misc]
    data = DEFAULT_CONFIG.to_dict()
    assert data == {
        "version": PREPROCESSING_VERSION,
        "size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
        "antialias": True,
    }


# --- loading: orientation, colour modes, errors --------------------------------------------


def test_load_image_returns_rgb_from_path_and_from_bytes(tmp_path: Path) -> None:
    image = _gradient_rgb(30, 20)
    path = tmp_path / "x.png"
    image.save(path)
    from_path = load_image(path)
    from_str = load_image(str(path))
    from_bytes = load_image(path.read_bytes())
    for loaded in (from_path, from_str, from_bytes):
        assert loaded.mode == "RGB"
        assert loaded.size == (30, 20)
        assert np.array_equal(np.asarray(loaded), np.asarray(image))


def test_exif_orientation_is_applied() -> None:
    image = _gradient_rgb(40, 20)  # wider than tall
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    loaded = load_image(_png_bytes(image, exif=exif))
    assert loaded.size == (20, 40)  # now taller than wide
    expected = np.asarray(image.rotate(-90, expand=True))
    assert np.array_equal(np.asarray(loaded), expected)


def test_grayscale_is_copied_to_three_channels() -> None:
    gray = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8), mode="L")
    loaded = np.asarray(load_image(_png_bytes(gray)))
    assert loaded.shape == (8, 8, 3)
    for channel in range(3):
        assert np.array_equal(loaded[:, :, channel], np.asarray(gray))


def test_transparent_pixels_become_white() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = (200, 0, 0)  # red ...
    rgba[:2, :, 3] = 0  # ... top half fully transparent
    rgba[2:, :, 3] = 255  # ... bottom half opaque
    loaded = np.asarray(load_image(_png_bytes(Image.fromarray(rgba, mode="RGBA"))))
    assert (loaded[:2] == 255).all()
    assert (loaded[2:] == (200, 0, 0)).all()


def test_half_transparent_pixels_are_blended_with_white() -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[:, :, 3] = 128
    loaded = np.asarray(load_image(_png_bytes(Image.fromarray(rgba, mode="RGBA"))))
    # black at alpha 128/255 over white: about 255 * (1 - 128/255) = 127
    assert abs(int(loaded[0, 0, 0]) - 127) <= 1


def test_not_an_image_raises_a_clear_error() -> None:
    with pytest.raises(ImageLoadError):
        load_image(b"this is not an image")


def test_truncated_image_raises_a_clear_error() -> None:
    data = _png_bytes(_gradient_rgb(64, 64))
    with pytest.raises(ImageLoadError):
        load_image(data[: len(data) // 2])


def test_unsupported_mode_is_refused_not_clipped() -> None:
    sixteen_bit = Image.fromarray(np.full((4, 4), 40000, dtype=np.uint16))
    assert sixteen_bit.mode == "I;16"
    with pytest.raises(ImageLoadError, match="Unsupported image mode"):
        load_image(_png_bytes(sixteen_bit))


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


# --- tensor: shape, dtype, values, determinism ---------------------------------------------


def test_output_shape_and_dtype_for_non_square_input() -> None:
    tensor = to_tensor(_gradient_rgb(300, 200))
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_no_crop_the_whole_image_is_kept() -> None:
    # Left half black, right half white. With no crop both halves must remain after resizing.
    array = np.zeros((100, 200, 3), dtype=np.uint8)
    array[:, 100:] = 255
    tensor = to_tensor(Image.fromarray(array, mode="RGB"))
    black = (0 - MEAN) / STD
    white = (1 - MEAN) / STD
    assert np.allclose(tensor[:, 128, 5].numpy(), black, atol=1e-4)
    assert np.allclose(tensor[:, 128, 250].numpy(), white, atol=1e-4)


def test_normalisation_matches_the_manifest_values() -> None:
    colour = (10, 128, 250)
    image = Image.new("RGB", (64, 64), colour)
    tensor = to_tensor(image)
    expected = (np.array(colour, dtype=np.float32) / 255 - MEAN) / STD
    assert np.allclose(tensor[:, 0, 0].numpy(), expected, atol=1e-5)
    assert np.allclose(tensor[:, 255, 255].numpy(), expected, atol=1e-5)


def test_same_input_gives_identical_tensors() -> None:
    image = _gradient_rgb(123, 87)
    assert torch.equal(to_tensor(image), to_tensor(image))


def test_resizing_is_antialiased() -> None:
    # A 1-pixel checkerboard averages to mid-grey when downsized with antialiasing.
    # Without antialiasing it would keep strong black/white pixels.
    checker = (np.indices((1024, 1024)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = Image.fromarray(np.stack([checker] * 3, axis=-1), mode="RGB")
    plain = v2.functional.pil_to_tensor(image).float() / 255
    tensor = to_tensor(image)
    restored = tensor.numpy() * STD[:, None, None] + MEAN[:, None, None]
    assert abs(float(restored.mean()) - 0.5) < 0.02
    assert float(restored.std()) < 0.05
    assert float(plain.std()) > 0.4  # the input really was high contrast


def test_matches_anomalibs_pipeline_resize_then_normalise() -> None:
    """The plan says this mirrors anomalib's PatchCore pre-processor (Resize with antialias, then
    ImageNet Normalize on a float tensor in [0, 1])."""
    image = _gradient_rgb(173, 311)
    reference = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
    assert torch.allclose(to_tensor(image), reference, atol=1e-6)


def test_config_size_is_honoured_and_non_rgb_input_is_refused() -> None:
    small = PreprocessConfig(size=64)
    assert to_tensor(_gradient_rgb(100, 100), small).shape == (3, 64, 64)
    with pytest.raises(ValueError, match="RGB"):
        to_tensor(Image.new("L", (8, 8)))
    with pytest.raises(ValueError, match="interpolation"):
        to_tensor(_gradient_rgb(8, 8), PreprocessConfig(interpolation="nearest"))


def test_equals_the_installed_anomalib_patchcore_default_and_it_has_no_crop() -> None:
    """Guard for PLAN.md 5.5 ("no crop"): anomalib's PatchCore pre-processor crops only when it is
    given ``center_crop_size``. Its default must stay resize + normalise, and must match ours."""
    from anomalib.models import Patchcore
    from torchvision.transforms.v2 import CenterCrop, Normalize, Resize

    transform = Patchcore.configure_pre_processor().transform
    kinds = [type(step) for step in transform.transforms]
    assert kinds == [Resize, Normalize]  # no CenterCrop, no augmentation
    assert CenterCrop not in kinds

    image = _gradient_rgb(173, 311)
    as_float = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    assert torch.allclose(to_tensor(image), transform(as_float), atol=1e-6)
