"""Bounded QQ media pixels for one observed space post, removed after use."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import tempfile
import time
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

from app.agent.social_contracts import SocialPost
from app.core.image_optimizer import optimize_image_bytes
from app.core.message_content import ImageAttachment


MAX_IMAGES = 2
MAX_POST_IMAGES = 9
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
FETCH_SECONDS = 10.0
FORMATS = {"PNG": ("image/png", ".png"), "JPEG": ("image/jpeg", ".jpg"), "WEBP": ("image/webp", ".webp"), "GIF": ("image/gif", ".gif")}


def allowed_media_url(value: str) -> bool:
    if not isinstance(value, str) or len(value) > 4096 or any(char.isspace() or ord(char) < 32 for char in value) or "\\" in value:
        return False
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        return (
            parsed.scheme == "https" and parsed.username is None and parsed.password is None
            and parsed.port in {None, 443} and not parsed.fragment
            and (host == "multimedia.nt.qq.com.cn"
                 or any(host == domain or host.endswith("." + domain) for domain in ("qpic.cn", "qq.com")))
        )
    except ValueError:
        return False


def _fetch_pixels(client: httpx.Client, url: str, *, budget: list[int], deadline: float) -> tuple[bytes, str, str] | None:
    for _ in range(3):
        remaining = deadline - time.monotonic()
        if not allowed_media_url(url) or remaining <= 0 or budget[0] <= 0:
            return None
        with client.stream("GET", url, follow_redirects=False, timeout=min(remaining, 5.0)) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and (not content_length.isdigit() or int(content_length) > budget[0]):
                return None
            chunks = []
            for chunk in response.iter_bytes(chunk_size=65536):
                budget[0] -= len(chunk)
                if budget[0] < 0 or time.monotonic() >= deadline:
                    return None
                chunks.append(chunk)
            data = b"".join(chunks)
        with Image.open(io.BytesIO(data)) as picture:
            if picture.format not in FORMATS or picture.width * picture.height > MAX_PIXELS:
                return None
            media_type, suffix = FORMATS[picture.format]
            picture.verify()
        # Some codecs (notably JPEG) verify only the container metadata.
        # The provider must receive a body that actually decodes to pixels.
        with Image.open(io.BytesIO(data)) as picture:
            picture.load()
        optimized = optimize_image_bytes(data, media_type=media_type)
        if optimized is not None:
            data, media_type = optimized
            suffix = ".png" if media_type == "image/png" else ".jpg"
        return data, media_type, suffix
    return None


@contextmanager
def post_image_inputs(post: SocialPost, *, cache_dir: Path, http_client: httpx.Client | None = None):
    """Yield local validated attachments; URLs never enter provider fallbacks."""
    urls = tuple(dict.fromkeys(url for url in post.images if allowed_media_url(url)))[:MAX_POST_IMAGES]
    if not urls:
        yield []
        return
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=5.0, follow_redirects=False, trust_env=False)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="turn-", dir=cache_dir) as directory:
            attachments, budget = [], [MAX_TOTAL_BYTES]
            deadline = time.monotonic() + FETCH_SECONDS
            for url in urls:
                try:
                    pixels = _fetch_pixels(client, url, budget=budget, deadline=deadline)
                    if pixels is None:
                        continue
                    data, _, suffix = pixels
                    path = Path(directory) / (hashlib.sha256(data).hexdigest() + suffix)
                    path.write_bytes(data)
                    attachments.append(ImageAttachment(
                        url="", local_path=str(path.resolve()), source_user_id=post.owner_id,
                        source_message_id=post.post_id,
                        reference_subject=f"Qzone observed post owner_id={post.owner_id} post_id={post.post_id}",
                    ))
                except Exception:
                    continue
            yield attachments
    finally:
        if owns_client:
            client.close()
