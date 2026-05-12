from pathlib import Path

import httpx

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

