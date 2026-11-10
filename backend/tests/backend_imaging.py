"""Synthetic images for the backend tests. No dataset file is ever read."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def make_image(seed: int, size: tuple[int, int] = (128, 96)) -> Image.Image:
    """A smooth random RGB image (width, height) with a brighter patch."""
    rng = np.random.default_rng(seed)
    width, height = size
    base = rng.random((height // 8 + 1, width // 8 + 1, 3))
    array = np.kron(base, np.ones((8, 8, 1)))[:height, :width] * 120 + 60
    array += rng.normal(0, 3, array.shape)
    array[height // 4 : height // 2, width // 4 : width // 2] += 60
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(image: Image.Image, **kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", **kwargs)
    return buffer.getvalue()
