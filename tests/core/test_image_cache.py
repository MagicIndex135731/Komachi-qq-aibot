import io
from pathlib import Path

import httpx
from PIL import Image

from app.core.image_cache import cache_images_in_raw_payload


def test_cache_images_in_raw_payload_downloads_image_and_persists_local_path(tmp_path) -> None:
    payload = {
        "message_id": "image-msg-1",
        "group_id": 10001,
        "message": [
            {"type": "text", "data": {"text": "look"}},
            {
                "type": "image",
                "data": {
                    "file": "cat.png",
                    "url": "https://img.example.test/cat.png",
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://img.example.test/cat.png"
        return httpx.Response(
            200,
            content=b"png-bytes",
            headers={"content-type": "image/png"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    cache_images_in_raw_payload(payload, cache_dir=tmp_path, http_client=client)

    image_data = payload["message"][1]["data"]
    cached_path = Path(image_data["local_path"])
    assert cached_path.exists()
    assert cached_path.read_bytes() == b"png-bytes"
    assert cached_path.parent == tmp_path / "10001"
    assert cached_path.suffix == ".png"


def test_cache_images_in_raw_payload_downloads_multiple_images_in_parallel_and_compresses(tmp_path) -> None:
    image = Image.effect_noise((1600, 1200), 100).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    payload = {
        "message_id": "multi-image-msg",
        "group_id": 10001,
        "message": [
            {
                "type": "image",
                "data": {
                    "file": "one.png",
                    "url": "https://img.example.test/one.png",
                },
            },
            {
                "type": "image",
                "data": {
                    "file": "two.png",
                    "url": "https://img.example.test/two.png",
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            content=png_bytes,
            headers={"content-type": "image/png"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    cache_images_in_raw_payload(payload, cache_dir=tmp_path, http_client=client)

    for image_data in (payload["message"][0]["data"], payload["message"][1]["data"]):
        cached_path = Path(image_data["local_path"])
        assert cached_path.exists()
        assert cached_path.suffix == ".jpg"
        optimized = cached_path.read_bytes()
        assert optimized != png_bytes
        decoded = Image.open(io.BytesIO(optimized))
        assert max(decoded.size) <= 1280

