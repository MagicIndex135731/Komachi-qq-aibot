from __future__ import annotations

from datetime import UTC, datetime
import io
from pathlib import Path

import httpx
from PIL import Image
import pytest

from app.agent.social_contracts import SocialPost
from app.agent.social_images import allowed_media_url, post_image_inputs


def image_bytes(size=(8, 8)):
    output = io.BytesIO()
    Image.new("RGB", size, "red").save(output, format="PNG")
    return output.getvalue()


def observed(images):
    return SocialPost(900000201, "synthetic-pixels-post", "", datetime(2026, 10, 9, tzinfo=UTC), tuple(images))


@pytest.mark.parametrize("url", [
    "file:///C:/secret.png", "data:image/png;base64,AAAA", "http://m.qpic.cn/image.png",
    "https://127.0.0.1/image.png", "https://qq.com.evil.test/image.png",
    "https://evilqq.com/image.png", "https://user:secret@qpic.cn/image.png", "https://@qpic.cn/image.png",
    "https://qpic.cn:8443/image.png", "https://qpic.cn\\@evil.test/image.png",
    "https://qpic.cn/image.png#fragment", "https://qpic.cn/image\n.png",
])
def test_social_media_rejects_untrusted_urls(url):
    assert not allowed_media_url(url)


def test_social_media_verified_pixels_are_bounded_and_removed(tmp_path):
    requested = []
    def respond(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=image_bytes(), headers={"content-type": "application/octet-stream"})
    cache = tmp_path / "social-media"
    post = observed(f"https://m.qpic.cn/{index}.png" for index in range(12))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(post, cache_dir=cache, http_client=client) as images:
            assert len(images) == 9
            paths = [Path(image.local_path) for image in images]
            assert all(path.read_bytes() == image_bytes() for path in paths)
            assert all(image.url == "" and not image.fallback_url for image in images)
            assert all(image.source_user_id == post.owner_id and image.source_message_id == post.post_id for image in images)
            assert all(str(post.owner_id) in image.reference_subject and post.post_id in image.reference_subject for image in images)
            assert all(path.stem.isalnum() and len(path.stem) == 64 for path in paths)
    assert len(requested) == 9
    assert all(not path.exists() for path in paths)
    assert list(cache.iterdir()) == []


def test_three_post_images_are_all_read_in_order(tmp_path):
    colors = ["red", "green", "blue"]
    requested = []
    def respond(request):
        index = int(request.url.path.removesuffix(".png").lstrip("/"))
        requested.append(index)
        stream = io.BytesIO()
        Image.new("RGB", (8, 8), colors[index]).save(stream, format="PNG")
        return httpx.Response(200, content=stream.getvalue())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(observed(f"https://m.qpic.cn/{i}.png" for i in range(3)),
                               cache_dir=tmp_path, http_client=client) as images:
            assert len(images) == 3
            assert [Image.open(image.local_path).getpixel((0, 0)) for image in images] == [
                (255, 0, 0), (0, 128, 0), (0, 0, 255)]
    assert requested == [0, 1, 2]


def test_failed_middle_post_image_does_not_skip_third(tmp_path):
    requested = []
    def respond(request):
        requested.append(request.url.path)
        return httpx.Response(200, content=b"bad pixels" if request.url.path == "/1.png" else image_bytes())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(observed(f"https://m.qpic.cn/{i}.png" for i in range(3)),
                               cache_dir=tmp_path, http_client=client) as images:
            assert len(images) == 2
    assert requested == ["/0.png", "/1.png", "/2.png"]


def test_unknown_redirect_target_is_never_requested(tmp_path):
    requested = []
    def respond(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://untrusted.example.test/image.png"})
    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/first.png",)), cache_dir=tmp_path, http_client=client) as images:
            assert images == []
    assert requested == ["https://m.qpic.cn/first.png"]


def test_valid_official_redirect_is_followed_with_same_pixel_contract(tmp_path):
    def respond(request):
        if request.url.path == "/first.png":
            return httpx.Response(302, headers={"location": "https://photo.qq.com/final.png"})
        return httpx.Response(200, content=image_bytes())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/first.png",)), cache_dir=tmp_path, http_client=client) as images:
            assert len(images) == 1


@pytest.mark.parametrize("failure", ["bad_pixels", "too_large", "timeout", "too_many_pixels"])
def test_bad_or_expensive_media_does_not_become_vision_input(tmp_path, monkeypatch, failure):
    monkeypatch.setattr("app.agent.social_images.MAX_TOTAL_BYTES", 1024)
    monkeypatch.setattr("app.agent.social_images.MAX_PIXELS", 100)
    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout")
        if failure == "too_large":
            return httpx.Response(200, content=b"x", headers={"content-length": "2048"})
        return httpx.Response(200, content=image_bytes((11, 11)) if failure == "too_many_pixels" else b"not actual pixels")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/image.png",)), cache_dir=tmp_path, http_client=client) as images:
            assert images == []
    assert list(tmp_path.iterdir()) == []


def test_total_download_budget_is_shared_between_images(tmp_path, monkeypatch):
    data = image_bytes()
    monkeypatch.setattr("app.agent.social_images.MAX_TOTAL_BYTES", len(data) + 1)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data))) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/1.png", "https://m.qpic.cn/2.png")), cache_dir=tmp_path, http_client=client) as images:
            assert len(images) == 1


def test_verified_large_image_uses_existing_optimizer(tmp_path):
    data = image_bytes((1600, 1600))
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data))) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/image.png",)), cache_dir=tmp_path, http_client=client) as images:
            with Image.open(images[0].local_path) as result:
                assert max(result.size) <= 1280


def test_provider_failure_still_cleans_local_media(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=image_bytes()))) as client:
        with pytest.raises(RuntimeError):
            with post_image_inputs(observed(("https://m.qpic.cn/image.png",)), cache_dir=tmp_path, http_client=client) as images:
                path = Path(images[0].local_path)
                raise RuntimeError("synthetic provider failure")
    assert not path.exists()


def test_media_headers_alone_do_not_count_as_decoded_pixels(tmp_path):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="JPEG")
    # JPEG headers remain readable, but its truncated image body cannot load.
    truncated = output.getvalue()[:-10]
    with Image.open(io.BytesIO(truncated)) as picture:
        picture.verify()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=truncated))) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/image.jpg",)), cache_dir=tmp_path, http_client=client) as images:
            assert images == []


def test_media_read_deadline_prevents_the_next_fetch(tmp_path, monkeypatch):
    requested, instants = [], iter((0.0, 1.0, 12.0, 12.0))
    monkeypatch.setattr("app.agent.social_images.time.monotonic", lambda: next(instants))
    def respond(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=image_bytes())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(observed(("https://m.qpic.cn/1.png", "https://m.qpic.cn/2.png")), cache_dir=tmp_path, http_client=client) as images:
            assert images == []
    assert requested == ["https://m.qpic.cn/1.png"]


def test_failed_media_does_not_log_sensitive_url_from_exception(tmp_path, caplog):
    secret = "synthetic-private-query-token"
    url = "https://m.qpic.cn/image.png?token=" + secret
    def respond(request):
        raise httpx.ReadTimeout("failed at " + str(request.url))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with post_image_inputs(observed((url,)), cache_dir=tmp_path, http_client=client) as images:
            assert images == []
    assert all(secret not in record.getMessage() for record in caplog.records if record.name.startswith("app."))
