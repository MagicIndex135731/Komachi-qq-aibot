"""Public X single-post downloads, shared by the standalone CLI and bot worker."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from uuid import uuid4

import httpx

from app.core.url_policy import public_web_url


MIB = 1024 * 1024
_POST_HOSTS = {"x.com", "www.x.com", "mobile.x.com", "twitter.com",
               "www.twitter.com", "mobile.twitter.com", "m.twitter.com"}
_MEDIA_HOSTS = {"pbs.twimg.com", "video.twimg.com"}
_POST_PATH = re.compile(
    r"/(?:[A-Za-z0-9_]{1,15}/status|i/(?:web/)?status)/([1-9][0-9]{0,24})"
    r"(?:/(?:photo|video)/[1-9][0-9]*)?/?\Z"
)
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_PHOTO_FALLBACK_CODES = {"http_400", "http_404", "http_410", "http_500", "http_502", "http_503", "http_504",
                         "network_error", "incomplete_download", "empty_download", "size_limit"}


class TwitterMediaError(ValueError):
    """Safe error code without upstream bodies, account details, or credentials."""

    def __init__(self, code: str, *, details: dict | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(code)


class _Retryable(TwitterMediaError):
    pass


@dataclass(frozen=True)
class TwitterPost:
    post_id: str
    source_url: str


@dataclass(frozen=True)
class TwitterMedia:
    kind: str
    url: str
    variants: tuple[dict, ...] = ()


@dataclass(frozen=True)
class TwitterDownload:
    directory: Path
    manifest: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"


def parse_post_url(value: str) -> TwitterPost:
    """Accept one post, including share parameters and photo/video subpages."""
    try:
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError()
        parts = urlsplit(public_web_url(value))
        match = _POST_PATH.fullmatch(parts.path)
        if parts.hostname not in _POST_HOSTS or not match:
            raise ValueError()
    except (ValueError, TypeError, AttributeError) as exc:
        raise TwitterMediaError("invalid_post_url") from exc
    post_id = match[1]
    return TwitterPost(post_id, f"https://x.com/i/status/{post_id}")


def _allowed_url(value: str, hosts: set[str], code: str) -> str:
    try:
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError()
        url = public_web_url(value)
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in hosts or parts.port not in {None, 443}:
            raise ValueError()
        return url
    except (ValueError, TypeError, AttributeError) as exc:
        raise TwitterMediaError(code) from exc


def _original_photo(value: str) -> str:
    url = _allowed_url(value, {"pbs.twimg.com"}, "invalid_media_url")
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["name"] = "orig"
    return urlunsplit(parts._replace(query=urlencode(query)))


def _media_item(item: dict[str, Any], *, syndication: bool) -> TwitterMedia:
    kind = item.get("type")
    if kind == "photo":
        return TwitterMedia("photo", _original_photo(item.get("media_url_https") if syndication else item.get("url")))
    if kind not in {"video", "animated_gif", "gif"}:
        raise TwitterMediaError("unsupported_media_type")
    info = item.get("video_info") if syndication else item
    if not isinstance(info, dict):
        raise TwitterMediaError("invalid_media_metadata")
    variants = info.get("variants") or []
    if not isinstance(variants, list) or len(variants) > 64:
        raise TwitterMediaError("invalid_media_metadata")
    candidates = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise TwitterMediaError("invalid_media_metadata")
        if variant.get("content_type", variant.get("contentType")) != "video/mp4":
            continue
        url = _allowed_url(variant.get("url"), {"video.twimg.com"}, "invalid_media_url")
        bitrate = variant.get("bitrate", 0)
        if type(bitrate) is not int or bitrate < 0:
            raise TwitterMediaError("invalid_media_metadata")
        if urlsplit(url).path.lower().endswith(".mp4"):
            resolution=re.search(r'/(\d{2,4})x(\d{2,4})/',urlsplit(url).path)
            candidates.append({'bitrate':bitrate, 'url':url,
                'duration_seconds':float(info.get('duration_millis') or 0)/1000,
                'width':int(resolution[1]) if resolution else 0,
                'height':int(resolution[2]) if resolution else 0,
                'codec':'h264'})
    if candidates:
        url = max(candidates, key=lambda entry: entry['bitrate'])['url']
    elif not syndication:
        url = _allowed_url(item.get("url"), {"video.twimg.com"}, "invalid_media_url")
        if not urlsplit(url).path.lower().endswith(".mp4"):
            raise TwitterMediaError("no_mp4_stream")
    else:
        raise TwitterMediaError("no_mp4_stream")
    return TwitterMedia("gif" if kind in {"gif", "animated_gif"} else "video", url, tuple(candidates))


def _parse_media(payload: dict[str, Any], post: TwitterPost, resolver: str) -> list[TwitterMedia]:
    if resolver == "syndication":
        tweet = payload
        if not tweet.get("id_str"):
            raise TwitterMediaError("post_unavailable")
        post_id = str(tweet["id_str"])
        items = tweet.get("mediaDetails") or []
    else:
        if payload.get("code") != 200 or not isinstance(payload.get("tweet"), dict):
            raise TwitterMediaError("post_unavailable")
        tweet = payload["tweet"]
        post_id = str(tweet.get("id") or "")
        if not post_id and tweet.get("url"):
            post_id = parse_post_url(tweet["url"]).post_id
        media = tweet.get("media") or {}
        if not isinstance(media, dict):
            raise TwitterMediaError("invalid_media_metadata")
        if "all" in media:
            items = media["all"]
        else:
            photos, videos = media.get("photos") or [], media.get("videos") or []
            if not isinstance(photos, list) or not isinstance(videos, list):
                raise TwitterMediaError("invalid_media_metadata")
            if photos and videos:
                # Separate arrays do not prove the original mixed order.
                raise TwitterMediaError("media_order_unavailable")
            items = photos + videos
    if post_id != post.post_id:
        raise TwitterMediaError("post_id_mismatch")
    if not isinstance(items, list) or len(items) > 16 or any(not isinstance(i, dict) for i in items):
        raise TwitterMediaError("invalid_media_metadata")
    if not items:
        raise TwitterMediaError("no_media")
    # Never recurse into quoted posts, replies, cards, avatars, or preview mosaics.
    return [_media_item(item, syndication=resolver == "syndication") for item in items]


def _signature_suffix(prefix: bytes, kind: str) -> str:
    if kind != "photo":
        if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
            return ".mp4"
    else:
        if prefix.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if prefix.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
            return ".webp"
    raise TwitterMediaError("invalid_media_signature")


class TwitterMediaDownloader:
    """Caller owns the HTTP client; no bot settings, cookies or database are loaded.

    Downloads are sequential and each successful invocation creates a new directory.
    Ordinary item failures retain completed siblings. Cancellation still propagates.
    """

    def __init__(self, client: httpx.AsyncClient, *, resolver: str = "auto",
                 max_file_bytes: int = 100 * MIB, max_total_bytes: int = 300 * MIB,
                 timeout_seconds: float = 180, retry_delay: float = 0.5, check=None, metadata_client=None,
                 on_file=None, transfer_budget=None, original_only=False):
        if resolver not in {"auto", "syndication", "fxtwitter"}:
            raise TwitterMediaError("invalid_resolver")
        if (type(max_file_bytes) is not int or type(max_total_bytes) is not int
                or min(max_file_bytes, max_total_bytes) <= 0
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 1800
                or not math.isfinite(retry_delay) or not 0 <= retry_delay <= 5):
            raise TwitterMediaError("invalid_download_limits")
        self.client, self.resolver = client, resolver
        self.max_file_bytes, self.max_total_bytes = max_file_bytes, max_total_bytes
        self.timeout_seconds, self.retry_delay = timeout_seconds, retry_delay
        self.check = check or (lambda: None)
        self.metadata_client = metadata_client
        self.on_file = on_file
        from app.core.media_variants import TransferBudget
        from app.core.request_cost import current_request_budget
        self.transfer_budget=TransferBudget(max_total_bytes,check=self.check,deadline=time.monotonic()+timeout_seconds,
            parent=transfer_budget or current_request_budget.get())
        self.original_only=original_only

    async def _transfer(self, url: str, *, limit: int, target: Path | None = None,
                        kind: str | None = None) -> dict[str, Any]:
        hosts = _MEDIA_HOSTS if target else {urlsplit(url).hostname}
        code = "invalid_media_url" if target else "invalid_resolver_url"
        for attempt in range(3):
            self.check()
            try:
                current = url
                for redirect in range(4):
                    self.transfer_budget.request()
                    current = _allowed_url(current, hosts, code)
                    async with self.client.stream("GET", current, follow_redirects=False, timeout=30,
                            headers={"User-Agent": "KomachiMediaDownloader/1.0", "Accept-Encoding": "identity"}) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            if redirect == 3 or not response.headers.get("location"):
                                raise TwitterMediaError("redirect_limit")
                            current = _allowed_url(urljoin(current, response.headers["location"]), hosts, code)
                            continue
                        if response.status_code in _RETRY_STATUSES:
                            raise _Retryable(f"http_{response.status_code}")
                        if response.status_code != 200:
                            raise TwitterMediaError(f"http_{response.status_code}")
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"} if kind == "photo" else {"video/mp4"}
                        if content_type not in (allowed | {"application/octet-stream"} if target else {"application/json"}):
                            raise TwitterMediaError("unexpected_content_type")
                        length = response.headers.get("content-length")
                        if length is not None and (not length.isdigit() or len(length) > 20):
                            raise TwitterMediaError("invalid_content_length")
                        if length is not None and int(length) > limit:
                            raise TwitterMediaError("size_limit")
                        total, prefix, chunks, digest = 0, b"", [], hashlib.sha256()
                        stream = target.open("wb") if target else None
                        try:
                            async for chunk in response.aiter_bytes(64 * 1024):
                                self.check()
                                total += len(chunk)
                                if target:self.transfer_budget.charge(len(chunk))
                                elif self.transfer_budget.parent:self.transfer_budget.parent.charge(len(chunk))
                                if total > limit:
                                    raise TwitterMediaError("size_limit")
                                prefix = (prefix + chunk[:32])[:32]
                                digest.update(chunk)
                                if stream:
                                    stream.write(chunk)
                                else:
                                    chunks.append(chunk)
                        finally:
                            if stream:
                                stream.close()
                        if not total:
                            raise TwitterMediaError("empty_download")
                        if (length is not None and not response.headers.get("content-encoding")
                                and total != int(length)):
                            raise TwitterMediaError("incomplete_download")
                        result = {"bytes": total, "sha256": digest.hexdigest(), "url": current}
                        if target:
                            result["suffix"] = _signature_suffix(prefix, kind)
                        else:
                            result["body"] = b"".join(chunks)
                        return result
            except (httpx.RequestError, _Retryable) as exc:
                if attempt == 2:
                    raise TwitterMediaError(exc.code if isinstance(exc, _Retryable) else "network_error") from exc
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        raise TwitterMediaError("redirect_limit")  # Defensive: all paths above return or raise.

    async def _resolve(self, post: TwitterPost) -> tuple[list[TwitterMedia], str, list[dict[str, str]], dict]:
        if self.metadata_client is not None:
            observed = await asyncio.to_thread(self.metadata_client.inspect, post.source_url, check=self.check)
            _, items, metadata = self._resolved(post.source_url, observed,
                evidence_ref=observed.get('evidence_ref', ''))
            return items, 'x_crawler', [], metadata
        resolvers = ["syndication", "fxtwitter"] if self.resolver == "auto" else [self.resolver]
        failures = []
        for resolver in resolvers:
            url = (f"https://cdn.syndication.twimg.com/tweet-result?id={post.post_id}&token=0"
                   if resolver == "syndication" else f"https://api.fxtwitter.com/status/{post.post_id}")
            try:
                response = await self._transfer(url, limit=2 * MIB)
                try:
                    payload = json.loads(response["body"])
                    if not isinstance(payload, dict):
                        raise ValueError()
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise TwitterMediaError("invalid_metadata_json") from exc
                items = _parse_media(payload, post, resolver)
                tweet = payload if resolver == 'syndication' else payload['tweet']
                author = tweet.get('user' if resolver == 'syndication' else 'author') or {}
                metadata = {'post_id': post.post_id, 'author': str(author.get('screen_name') or author.get('username') or '')[:160],
                    'published_at': str(tweet.get('created_at') or '')[:80],
                    'text': str(tweet.get('text') or '')[:2000], 'identity_status': 'matched'}
                return items, resolver, failures, metadata
            except TwitterMediaError as exc:
                if exc.code in {"post_id_mismatch", "invalid_post_url", "invalid_media_url"}:
                    raise
                failures.append({"resolver": resolver, "code": exc.code})
        error = TwitterMediaError("resolve_failed:" + ",".join(f"{i['resolver']}={i['code']}" for i in failures))
        raise error

    async def inspect(self, url: str) -> dict:
        """Read root-post identity and media descriptors without downloading pixels."""
        post = parse_post_url(url)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                items, resolver, failures, metadata = await self._resolve(post)
                return {**metadata, 'source_url': post.source_url, 'title': metadata['text'][:240],
                    'media': [{'kind': 'image' if item.kind == 'photo' else 'video',
                        'role': 'post_image' if item.kind == 'photo' else 'post_video', 'url': item.url,
                        'declared_bytes': 0} for item in items], 'extractor': resolver}
        except TimeoutError as exc:
            raise TwitterMediaError('download_timeout') from exc

    async def download(self, url: str, output_dir: Path) -> TwitterDownload:
        post = parse_post_url(url)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            async with asyncio.timeout(self.timeout_seconds):
                items, resolver, failures, metadata = await self._resolve(post)
            return await self._save(post, output_dir, items, resolver, failures, metadata, deadline=deadline)
        except TimeoutError as exc:
            raise TwitterMediaError("download_timeout") from exc
        except OSError as exc:
            raise TwitterMediaError("output_io_error") from exc

    async def download_resolved(self, url: str, output_dir: Path, observation: dict,
                                *, evidence_ref: str) -> TwitterDownload:
        """Download a crawler adapter's sanitized root-post descriptors.

        No credentials or authenticated HTTP client are accepted. The caller owns the
        observation's trust boundary; every media URL is revalidated here.
        """
        post, items, metadata = self._resolved(url, observation, evidence_ref=evidence_ref)
        try:
            return await self._save(post, output_dir, items, 'x_crawler', [], metadata,
                deadline=time.monotonic() + self.timeout_seconds)
        except TimeoutError as exc:
            raise TwitterMediaError('download_timeout') from exc
        except OSError as exc:
            raise TwitterMediaError('output_io_error') from exc

    @staticmethod
    def _resolved(url: str, observation: dict, *, evidence_ref: str):
        post = parse_post_url(url)
        if (observation.get('post_id') != post.post_id
                or observation.get('identity_status') != 'matched' or observation.get('protected') is not False
                or not isinstance(evidence_ref, str) or not re.fullmatch(r'x-posts-[a-f0-9]{24}:post:[0-9]+', evidence_ref)):
            raise TwitterMediaError('post_id_mismatch')
        media = observation.get('media')
        if (not isinstance(media, list) or not 1 <= len(media) <= 16
                or observation.get('media_count') != len(media) or observation.get('media_error')):
            raise TwitterMediaError('incomplete_resolved_media')
        items = []
        for item in media:
            if not isinstance(item, dict): raise TwitterMediaError('invalid_media_metadata')
            kind = item.get('kind')
            if kind == 'photo':
                media_url = _original_photo(item.get('url'))
            elif kind in {'video', 'gif'}:
                media_url = _allowed_url(item.get('url'), {'video.twimg.com'}, 'invalid_media_url')
                if not urlsplit(media_url).path.lower().endswith('.mp4'):
                    raise TwitterMediaError('no_mp4_stream')
            else:
                raise TwitterMediaError('unsupported_media_type')
            variants=[]
            for variant in item.get('variants',[])[:16]:
                candidate={**variant,'url':_allowed_url(variant.get('url'),{'video.twimg.com'},'invalid_media_url')}
                if not urlsplit(candidate['url']).path.lower().endswith('.mp4'):raise TwitterMediaError('no_mp4_stream')
                variants.append(candidate)
            items.append(TwitterMedia(kind, media_url, tuple(variants)))
        metadata = {k: observation.get(k, '') for k in
            ('post_id', 'author', 'published_at', 'text', 'identity_status')}
        metadata['evidence_ref'] = evidence_ref
        return post, items, metadata

    async def _save(self, post: TwitterPost, output_dir: Path, items: list[TwitterMedia],
                    resolver: str, failures: list, metadata: dict, *, deadline: float) -> TwitterDownload:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"{post.post_id}-{uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(prefix=".twitter-", dir=root) as staging:
            directory = Path(staging)
            files, total, item_failures = [], 0, []
            for index, item in enumerate(items, 1):
                self.check()
                target = directory / f"{index:02d}.part"
                remaining = min(self.max_file_bytes, self.max_total_bytes - total,
                    8 * MIB if item.kind == 'photo' else self.max_file_bytes)
                quality = {"image_size": "orig"} if item.kind == "photo" else {}
                try:
                    if remaining <= 0:
                        raise TwitterMediaError('size_limit')
                    async with asyncio.timeout(max(0, deadline - time.monotonic())):
                        try:
                            if item.kind=='photo':
                                saved = await self._transfer(item.url, limit=min(remaining,self.transfer_budget.remaining), target=target, kind=item.kind)
                            else:
                                from app.core.media_variants import native_variants
                                diagnostics={'stage':'video_variant_selection'}
                                choices=native_variants(list(item.variants) or [{'url':item.url}],
                                    max_bytes=min(remaining,self.transfer_budget.remaining),original_only=self.original_only,
                                    diagnostics=diagnostics)
                                if not choices:raise TwitterMediaError('no_mp4_stream_within_budget',details=diagnostics)
                                last=None
                                for choice in choices:
                                    try:
                                        saved=await self._transfer(choice['url'],limit=min(remaining,self.transfer_budget.remaining),target=target,kind=item.kind)
                                        quality={key:choice[key] for key in ('quality','fallback_reason','estimated_bytes')}
                                        break
                                    except TwitterMediaError as exc:
                                        if exc.code not in {'size_limit','incomplete_download','http_404','http_410','network_error'}:raise
                                        last=exc;target.unlink(missing_ok=True)
                                else:raise last
                        except TwitterMediaError as exc:
                            if item.kind != "photo" or self.original_only or exc.code not in _PHOTO_FALLBACK_CODES:
                                raise
                            self.check()
                            target.unlink(missing_ok=True)
                            parts = urlsplit(item.url)
                            fallback = urlunsplit(parts._replace(query=urlencode({**dict(parse_qsl(parts.query)), "name": "large"})))
                            saved = await self._transfer(fallback, limit=min(remaining,self.transfer_budget.remaining), target=target, kind=item.kind)
                            quality = {"image_size": "large", "fallback_reason": exc.code}
                    filename = f"{index:02d}_{item.kind}{saved['suffix']}"
                    path = directory / filename
                    target.rename(path)
                    entry = {"index": index, "kind": item.kind, "filename": filename,
                        "media_url": item.url, "download_url": saved["url"],
                        "bytes": saved["bytes"], "sha256": saved["sha256"], **quality}
                    # The bot commits a validated artifact before starting its sibling.
                    if self.on_file is not None:
                        await self.on_file(entry, path)
                    total += saved["bytes"]
                    files.append(entry)
                except (TwitterMediaError, TimeoutError, OSError, ValueError) as exc:
                    self.check()  # Revocation never becomes an ordinary item failure.
                    item_failures.append({'index': index, 'kind': item.kind,
                        'error': exc.code if isinstance(exc, TwitterMediaError) else
                            'download_timeout' if isinstance(exc, TimeoutError) else 'media_validation_or_storage_failed',
                        **({'details':exc.details} if isinstance(exc,TwitterMediaError) and exc.details else {})})
                    target.unlink(missing_ok=True)
            if not files and item_failures:
                raise TwitterMediaError(item_failures[0]['error'],details=item_failures[0].get('details'))
            manifest = {"schema_version": 1, "post_id": post.post_id, "source_url": post.source_url,
                        "resolver": resolver, "resolver_failures": failures,
                        "created_at": datetime.now(UTC).isoformat(), "post": metadata, "total_bytes": total,
                        "files": files, 'received_bytes':self.transfer_budget.received,
                        'requested_count': len(items), 'failures': item_failures,
                        'all_requested_downloaded': not item_failures}
            (directory / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            directory.rename(destination)
        return TwitterDownload(destination, manifest)
