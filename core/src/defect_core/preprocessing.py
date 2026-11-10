"""Image preprocessing: the one shared definition (PLAN.md section 5.5).

Training, calibration, evaluation, the command-line tool and the API all prepare images with this
module, so the model always receives inputs prepared the same way.

Steps, in order:

1. Decode the image, apply the EXIF orientation, and convert it to RGB. Transparent pixels are
   composited on white; grayscale is copied to three channels.
2. Resize to ``size`` x ``size``, bilinear, with antialiasing. No crop, so the heatmap covers the
   whole image.
3. Convert to a float32 tensor in [0, 1] and normalise with the ImageNet mean and standard
   deviation (the same values anomalib's PatchCore pre-processor uses).

There is no augmentation: augmented "normal" images could hide real anomalies.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tvf

PREPROCESSING_VERSION = "1"

# Image modes whose conversion to RGB keeps the pixel values meaningful. Other modes (for example
# 16-bit grayscale "I;16") would be silently clipped by Pillow, so they are refused instead.
_SUPPORTED_MODES = frozenset({"1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK"})


class ImageLoadError(ValueError):
    """The input could not be decoded as a supported image."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Frozen description of the preprocessing. It is stored in the model manifest."""

    version: str = PREPROCESSING_VERSION
    size: int = 256
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"
    antialias: bool = True

    def to_dict(self) -> dict[str, object]:
        """Plain-data form for the manifest (tuples become lists, as in JSON)."""
        data = asdict(self)
        data["mean"] = list(self.mean)
        data["std"] = list(self.std)
        return data


DEFAULT_CONFIG = PreprocessConfig()


def load_image(source: str | Path | bytes) -> Image.Image:
    """Decode an image from a file path or from bytes and return it as an RGB image.

    The EXIF orientation is applied. Transparent pixels are composited on white and grayscale is
    copied to three channels. Raises ``ImageLoadError`` for data that is not a supported image;
    a missing file raises ``FileNotFoundError``.
    """
    try:
        if isinstance(source, bytes | bytearray):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)
        image.load()  # force a full decode, so truncated or corrupt data fails here
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        if isinstance(error, FileNotFoundError):
            raise
        raise ImageLoadError(f"Could not decode image: {error}") from error

    image = ImageOps.exif_transpose(image)
    if image.mode not in _SUPPORTED_MODES:
        raise ImageLoadError(f"Unsupported image mode {image.mode!r}")

    has_alpha = image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def to_tensor(image: Image.Image, config: PreprocessConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """Turn an RGB image into the model input: float32, shape (3, size, size), normalised."""
    if image.mode != "RGB":
        raise ValueError(f"to_tensor expects an RGB image, got mode {image.mode!r}")
    if config.interpolation != "bilinear":
        raise ValueError(f"Unsupported interpolation {config.interpolation!r}")

    tensor = tvf.pil_to_tensor(image)  # uint8, (3, H, W)
    tensor = tvf.to_dtype(tensor, torch.float32, scale=True)  # [0, 1]
    tensor = v2.Resize(
        (config.size, config.size),
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=config.antialias,
    )(tensor)
    return v2.Normalize(mean=list(config.mean), std=list(config.std))(tensor)
