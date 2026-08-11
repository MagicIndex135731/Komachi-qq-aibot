from __future__ import annotations

import io
from typing import Any


MAX_IMAGE_SIDE = 1280
JPEG_QUALITY = 90

# Images below these thresholds are already cheap enough to send as-is.
_SKIP_MAX_SIDE = 800
_SKIP_MAX_BYTES = 200_000


def optimize_image_bytes(
    data: bytes,
    *,
    media_type: str | None = None,
    max_side: int = MAX_IMAGE_SIDE,
    jpeg_quality: int = JPEG_QUALITY,
) -> tuple[bytes, str] | None:
    """Return ``(optimized_bytes, media_type)`` or ``None`` when no change helps.

    Downscales oversized images and re-encodes opaque images as JPEG (alpha
    images stay PNG). Never raises: undecodable input (including synthetic
    test bytes) returns ``None`` so callers fall back to the original bytes.
    """
    if not data:
        return None
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        return None

    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= _SKIP_MAX_SIDE and len(data) <= _SKIP_MAX_BYTES:
        return None

    if largest_side > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)

    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    output = io.BytesIO()
    if has_alpha:
        image.convert("RGBA").save(output, format="PNG", optimize=True)
        out_media_type = "image/png"
    else:
        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=int(jpeg_quality),
            optimize=True,
        )
        out_media_type = "image/jpeg"

    optimized = output.getvalue()
    if not optimized or len(optimized) >= len(data):
        return None
    return optimized, out_media_type
