"""Tests for app.services.validation: what an upload must satisfy before it is decoded or scored."""

from __future__ import annotations

import io

import pytest
from app.core.errors import FileTooLarge, ImageDimensions, InvalidImage, UnsupportedMediaType
from app.core.settings import Settings
from app.services.validation import detect_type, reencode_png, validate_upload
from backend_imaging import jpeg_bytes, make_image, png_bytes
from PIL import Image


def settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_png_and_jpeg_are_accepted_and_typed_by_their_bytes() -> None:
    image = make_image(1)
    png = validate_upload(png_bytes(image), settings())
    jpg = validate_upload(jpeg_bytes(image), settings())
    assert (png.detected_type, jpg.detected_type) == ("png", "jpeg")
    assert (png.width, png.height) == (128, 96) and png.image.mode == "RGB"
    assert len(png.sha256) == 64 and png.byte_size == len(png_bytes(image))


@pytest.mark.parametrize(
    "data",
    [b"GIF89a" + b"\x00" * 50, b"hello world", b"%PDF-1.4 ...", b"", b"RIFF\x00\x00\x00\x00WEBP"],
)
def test_other_formats_are_refused_by_magic_bytes(data: bytes) -> None:
    with pytest.raises(UnsupportedMediaType):
        validate_upload(data, settings())


def test_a_wrong_extension_or_content_type_does_not_matter_only_the_bytes() -> None:
    assert detect_type(png_bytes(make_image(1))) == "png"
    assert detect_type(b"not an image") is None


def test_too_many_bytes_are_refused() -> None:
    with pytest.raises(FileTooLarge) as caught:
        validate_upload(b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 1), settings(max_upload_mb=1))
    assert caught.value.details == {"max_mb": 1}


@pytest.mark.parametrize("size", [(32, 96), (128, 63), (63, 63)])
def test_a_side_below_the_minimum_is_refused(size: tuple[int, int]) -> None:
    with pytest.raises(ImageDimensions):
        validate_upload(png_bytes(make_image(1, size)), settings())


def test_a_side_above_the_maximum_is_refused_before_decoding() -> None:
    wide = Image.new("RGB", (8001, 64))
    with pytest.raises(ImageDimensions):
        validate_upload(png_bytes(wide), settings())
    at_limit = Image.new("RGB", (8000, 64))
    assert validate_upload(png_bytes(at_limit), settings()).width == 8000


def test_the_pixel_cap_is_enforced() -> None:
    with pytest.raises(ImageDimensions, match="too many pixels"):
        validate_upload(png_bytes(make_image(1, (128, 96))), settings(max_image_pixels=10_000))


def test_a_damaged_or_cut_off_image_is_invalid() -> None:
    good = png_bytes(make_image(1))
    with pytest.raises(InvalidImage):
        validate_upload(good[: len(good) // 2], settings())  # header fine, data cut off
    with pytest.raises(InvalidImage):
        validate_upload(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, settings())
    with pytest.raises(InvalidImage):
        validate_upload(b"\xff\xd8\xff" + b"\x00" * 40, settings())


def test_exif_orientation_is_applied() -> None:
    image = make_image(1, (128, 96))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees: the stored 128 x 96 shows as 96 x 128
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    validated = validate_upload(buffer.getvalue(), settings())
    assert (validated.width, validated.height) == (96, 128)


def test_transparency_is_composited_on_white() -> None:
    clear = Image.new("RGBA", (128, 96), (255, 0, 0, 0))
    validated = validate_upload(png_bytes(clear), settings())
    assert validated.image.getpixel((5, 5)) == (255, 255, 255)


def test_reencoding_drops_metadata() -> None:
    image = make_image(1)
    exif = Image.Exif()
    exif[0x010E] = "a secret description"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    validated = validate_upload(buffer.getvalue(), settings())
    assert b"secret description" in buffer.getvalue()
    clean = reencode_png(validated.image)
    assert b"secret description" not in clean
    reopened = Image.open(io.BytesIO(clean))
    assert reopened.format == "PNG" and not reopened.getexif() and reopened.size == (128, 96)
