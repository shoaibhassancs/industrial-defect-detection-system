"""Upload validation (PLAN.md section 4.5 "File handling", Phase 5 step 4).

Order of checks, cheapest first, none of which trusts the client's file name or content type:

1. the number of bytes (also enforced earlier, while the body streams in);
2. the **magic bytes**: only JPEG (``FF D8 FF``) and PNG (``89 50 4E 47 0D 0A 1A 0A``);
3. the **header** is read without decoding the pixels: both sides within the allowed range and the
   pixel count under the cap, so an image that would decode to gigabytes is refused before decoding;
4. a **full decode** (``defect_core.preprocessing.load_image``: EXIF orientation applied,
   transparency composited on white, RGB). Truncated or corrupt data fails here.

``reencode_png`` re-encodes the decoded image as PNG, which drops metadata (EXIF, GPS, comments);
it is what the API returns as the "original" view. The file name is never used.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from defect_core.preprocessing import ImageLoadError, load_image
from PIL import Image, UnidentifiedImageError

from app.core.errors import FileTooLarge, ImageDimensions, InvalidImage, UnsupportedMediaType
from app.core.settings import Settings

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image  # RGB, orientation applied
    width: int
    height: int
    byte_size: int
    detected_type: str  # "png" or "jpeg", from the bytes
    sha256: str


def detect_type(data: bytes) -> str | None:
    if data.startswith(_PNG):
        return "png"
    if data.startswith(_JPEG):
        return "jpeg"
    return None


def validate_upload(data: bytes, settings: Settings) -> ValidatedImage:
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(
            f"Image must be {settings.max_upload_mb} MB or smaller.",
            {"max_mb": settings.max_upload_mb},
        )
    kind = detect_type(data)
    if kind is None:
        raise UnsupportedMediaType(
            "Only JPEG and PNG images are accepted.", {"accepted": ["image/jpeg", "image/png"]}
        )
    try:
        with Image.open(io.BytesIO(data)) as header:  # reads the header only
            width, height = header.size
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImage("The image could not be read.") from None
    low, high = settings.min_image_side, settings.max_image_side
    if min(width, height) < low or max(width, height) > high:
        raise ImageDimensions(
            f"Each side of the image must be between {low} and {high} pixels.",
            {"min_side": low, "max_side": high},
        )
    if width * height > settings.max_image_pixels:
        raise ImageDimensions(
            "The image has too many pixels.", {"max_pixels": settings.max_image_pixels}
        )
    try:
        image = load_image(data)
    except ImageLoadError:
        raise InvalidImage(
            "The image could not be decoded (it may be damaged or cut off)."
        ) from None
    return ValidatedImage(
        image, image.width, image.height, len(data), kind, hashlib.sha256(data).hexdigest()
    )


def reencode_png(image: Image.Image) -> bytes:
    """PNG bytes of the decoded pixels only (no metadata)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
