import io
import random

from PIL import Image

from app.core.image_optimizer import MAX_IMAGE_SIDE, optimize_image_bytes


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_optimize_large_opaque_image_to_jpeg() -> None:
    image = Image.effect_noise((2000, 1500), 120).convert("RGB")
    raw = _png_bytes(image)

    optimized, media_type = optimize_image_bytes(raw, media_type="image/png")

    assert media_type == "image/jpeg"
    assert optimized is not None
    assert len(optimized) < len(raw)
    decoded = Image.open(io.BytesIO(optimized))
    assert max(decoded.size) <= MAX_IMAGE_SIDE


def test_optimize_keeps_alpha_images_as_png() -> None:
    pixels = random.Random(20261010).randbytes(1600 * 1200 * 3)
    image = Image.frombytes("RGB", (1600, 1200), pixels).convert("RGBA")
    image.putalpha(128)
    raw = _png_bytes(image)

    optimized, media_type = optimize_image_bytes(raw, media_type="image/png")

    assert media_type == "image/png"
    assert optimized is not None
    decoded = Image.open(io.BytesIO(optimized))
    assert max(decoded.size) <= MAX_IMAGE_SIDE
    assert decoded.mode == "RGBA"
    assert decoded.getchannel("A").getextrema() == (128, 128)


def test_optimize_returns_none_for_undecodable_bytes() -> None:
    assert optimize_image_bytes(b"png-bytes", media_type="image/png") is None


def test_optimize_returns_none_for_small_images() -> None:
    image = Image.new("RGB", (400, 300), "steelblue")
    raw = _png_bytes(image)
    assert optimize_image_bytes(raw, media_type="image/png") is None
