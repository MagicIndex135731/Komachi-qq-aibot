"""Bounded SnowLuma Qzone actions with verified post identities.

Feed handles are not post IDs. Reads resolve them against the author's real
post list; writes recheck the target and never retry an unknown result.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import re
from collections.abc import Callable
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlsplit

from app.adapters.sender import QQMessageDeliveryUncertainError
from app.agent.social_contracts import SocialPost


QzonePost = SocialPost
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MAX_TEXT = 2000
_MAX_READ_TEXT = 65536
_MAX_IMAGES = 9
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_FEED_AUTHORS = 5
_MAX_FRIENDS = 10000
_HTML_VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                            "link", "meta", "param", "source", "track", "wbr"})


class QzoneProtocolError(RuntimeError):
    """A read response cannot be trusted as a typed observation."""


class QzoneRejectedError(RuntimeError):
    """A request was rejected explicitly or its target failed validation."""


class QzoneUnavailableError(RuntimeError):
    """The gateway was unavailable before writing, or a read failed."""


class QzoneAdapter:
    def __init__(self, gateway, *, timeout_seconds: float = 20) -> None:
        if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
            raise ValueError("Qzone deadline must be finite and bounded")
        self.gateway = gateway
        self.timeout_seconds = timeout_seconds

    async def get_friend_ids(self) -> tuple[int, ...]:
        data = await self._read("get_friend_list", {}, allow_list=True)
        try:
            if not isinstance(data, list) or len(data) > _MAX_FRIENDS:
                raise ValueError("invalid friend list")
            ids = []
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError("invalid friend")
                _integer(row["user_id"], maximum=2**63 - 1)
                ids.append(row["user_id"])
            return tuple(dict.fromkeys(ids))
        except (KeyError, ValueError, TypeError):
            raise QzoneProtocolError("get_friend_list returned malformed friend data") from None

    async def _require_friend(self, owner_id: int) -> None:
        if owner_id not in await self.get_friend_ids():
            raise QzoneRejectedError("Qzone author is no longer a current QQ friend")

    async def get_friend_contacts(self) -> tuple[dict, ...]:
        data = await self._read('get_friend_list', {}, allow_list=True)
        try:
            if not isinstance(data, list) or len(data) > _MAX_FRIENDS:
                raise ValueError('invalid friend list')
            result = []
            for row in data:
                _integer(row['user_id'], maximum=2**63-1)
                result.append({'owner_id': row['user_id'],
                    'nickname': _string(row.get('nickname', ''), maximum=1000),
                    'remark': _string(row.get('remark', ''), maximum=1000)})
            return tuple(result)
        except (KeyError, ValueError, TypeError):
            raise QzoneProtocolError('get_friend_list returned malformed friend contacts') from None

    async def get_posts(self, user_id: int, count: int = 20) -> tuple[SocialPost, ...]:
        _integer(user_id, maximum=2**63 - 1)
        _integer(count, maximum=100)
        data = await self._read("get_qzone_msg_list", {"target_uin": user_id, "pos": 0, "num": count})
        try:
            _integer(data["total"], minimum=0, maximum=2**63 - 1)
            rows = data["msglist"]
            if not isinstance(rows, list) or len(rows) > count:
                raise ValueError("invalid post list")
            result = []
            seen = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("invalid post")
                tid = _identifier(row["tid"])
                text = _plain_text(_string(row["content"], maximum=_MAX_READ_TEXT))
                stamp = _timestamp(row["time"])
                _integer(row["comment_num"], minimum=0, maximum=2**63 - 1)
                if type(row["is_private"]) is not bool:
                    raise ValueError("invalid post privacy")
                images = _read_images(row["images"])
                # Future bridges may echo scope: reject a conflicting author.
                if "uin" in row:
                    _integer(row["uin"], maximum=2**63 - 1)
                    if row["uin"] != user_id:
                        raise ValueError("post author differs from requested space")
                if tid not in seen:
                    result.append(SocialPost(user_id, tid, text, stamp, images))
                    seen.add(tid)
            return tuple(result)
        except (KeyError, ValueError, TypeError, OverflowError):
            raise QzoneProtocolError("get_qzone_msg_list returned malformed post data") from None

    async def get_feeds(self, count: int = 20, *,
                        allowed_owner_ids: tuple[int, ...] | None = None,
                        require_friends: bool = False) -> tuple[SocialPost, ...]:
        _integer(count, maximum=50)
        if allowed_owner_ids is not None:
            if not isinstance(allowed_owner_ids, (tuple, list)) or len(allowed_owner_ids) > (_MAX_FRIENDS if require_friends else 100):
                raise ValueError("Qzone feed author allowlist must be bounded")
            for owner in allowed_owner_ids:
                _integer(owner, maximum=2**63 - 1)
            allowed = frozenset(allowed_owner_ids)
            if not allowed:
                return ()
        else:
            allowed = None
        data = await self._read("get_qzone_feeds", {"page_num": 1, "count": count})
        try:
            feeds = data["feeds"]
            if not isinstance(feeds, list) or len(feeds) > count or type(data["has_more"]) is not bool:
                raise ValueError("invalid feed list")
            candidates = []
            for feed in feeds:
                if not isinstance(feed, dict):
                    raise ValueError("invalid feed")
                _integer(feed["appid"], minimum=0, maximum=2**31 - 1)
                # Non-mood entries can have no author; SnowLuma then emits
                # uin=0. They cannot become comment targets, so filter them
                # before validating the positive author required for moods.
                if feed["appid"] != 311:
                    continue
                owner = feed["uin"]
                _integer(owner, maximum=2**63 - 1)
                _string(feed["nickname"], maximum=1000)
                stamp = _timestamp(feed["time"])
                key = _string(feed["key"], maximum=256)
                html = _string(feed["html"], maximum=_MAX_READ_TEXT)
                if allowed is not None and owner not in allowed:
                    continue
                parser = _FeedHTML(owner)
                parser.feed(html)
                parser.close()
                if parser.conflicting_owner or len(parser.post_ids) > 1:
                    continue
                # A key is only a candidate until exact equality with a real
                # msglist tid is proved. No text/time-based ID guessing.
                candidate = next(iter(parser.post_ids), key)
                if _ID.fullmatch(candidate):
                    candidates.append((owner, candidate, stamp))
        except (KeyError, ValueError, TypeError, OverflowError):
            raise QzoneProtocolError("get_qzone_feeds returned malformed feed data") from None

        verified = {}
        result = []
        seen = set()
        for owner, candidate, stamp in candidates:
            if owner not in verified:
                if len(verified) >= _MAX_FEED_AUTHORS:
                    continue
                if require_friends and owner not in await self.get_friend_ids():
                    continue
                verified[owner] = {post.post_id: post for post in await self.get_posts(owner, 100)}
            actual = verified[owner].get(candidate)
            if actual is not None and actual.published_at == stamp and (owner, candidate) not in seen:
                result.append(actual)
                seen.add((owner, candidate))
        if require_friends:
            current = frozenset(await self.get_friend_ids())
            result = [post for post in result if post.owner_id in current]
        return tuple(result)

    async def publish(self, content: str, visibility: int = 4, images: tuple[str, ...] = (),
                      *, before_dispatch: Callable[[], None] | None = None) -> str | None:
        content = _content(content)
        if type(visibility) is not int or visibility not in {4, 64}:
            raise ValueError("Qzone visibility must be friends or self")
        params = {"content": content, "ugc_right": visibility}
        media = _write_images(images)
        if media:
            params["images"] = media
        data = await self._write("send_qzone_msg", params, before_dispatch=before_dispatch)
        # The fixed SnowLuma publisher promises a server-assigned tid. A
        # success-shaped body missing it may follow a real accepted write.
        try:
            return _identifier(data["tid"])
        except (KeyError, TypeError, ValueError):
            raise QQMessageDeliveryUncertainError("Qzone publish acknowledged without a valid post receipt") from None

    async def comment(self, post: SocialPost, content: str, images: tuple[str, ...] = (),
                      *, before_dispatch: Callable[[], None] | None = None,
                      require_friend: bool = False) -> str | None:
        content = _content(content)
        media = _write_images(images)
        if require_friend:
            await self._require_friend(post.owner_id)
        actual = await self._verify_post(post)
        params = {"target_uin": actual.owner_id, "tid": actual.post_id, "content": content}
        if media:
            params["images"] = media
        if require_friend:
            await self._require_friend(actual.owner_id)
        data = await self._write("comment_qzone", params, before_dispatch=before_dispatch)
        try:
            # SnowLuma explicitly permits '' when QQ did not supply an ID.
            receipt = _string(data["comment_id"], maximum=128)
            return _identifier(receipt) if receipt else None
        except (KeyError, TypeError, ValueError):
            raise QQMessageDeliveryUncertainError("Qzone comment returned a malformed acknowledgement") from None

    async def like(self, post: SocialPost, *, before_dispatch: Callable[[], None] | None = None) -> str | None:
        actual = await self._verify_post(post)
        data = await self._write("like_qzone", {"target_uin": actual.owner_id, "tid": actual.post_id,
                                               "abstime": int(actual.published_at.timestamp())}, allow_null=True,
                                 before_dispatch=before_dispatch)
        if data is not None:
            raise QQMessageDeliveryUncertainError("Qzone like returned a malformed acknowledgement")
        return None

    async def _verify_post(self, post: SocialPost) -> SocialPost:
        if not isinstance(post, SocialPost):
            raise ValueError("Qzone target must be a typed post")
        _integer(post.owner_id, maximum=2**63 - 1)
        _identifier(post.post_id)
        if not isinstance(post.published_at, datetime) or post.published_at.tzinfo is None:
            raise ValueError("Qzone target timestamp must have a timezone")
        for actual in await self.get_posts(post.owner_id, 100):
            if actual == post:
                return actual
        raise QzoneRejectedError("Qzone target is absent from the verified author post list")

    def _require_connection(self) -> None:
        if hasattr(self.gateway, "websocket") and self.gateway.websocket is None:
            raise QzoneUnavailableError("Qzone gateway is disconnected before submission")

    async def _read(self, action: str, params: dict, *, allow_list: bool = False) -> dict | list:
        self._require_connection()
        try:
            response = await asyncio.wait_for(self.gateway.call_api(action, params), self.timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise QzoneUnavailableError(f"{action} read failed") from None
        data = _envelope(response, action=action)
        if not isinstance(data, list if allow_list else dict):
            raise QzoneProtocolError(f"{action} returned malformed read data")
        return data

    async def _write(self, action: str, params: dict, *, allow_null: bool = False,
                     before_dispatch: Callable[[], None] | None = None) -> dict | None:
        self._require_connection()
        submitted = False

        async def submit():
            nonlocal submitted
            self._require_connection()
            if before_dispatch is not None:
                before_dispatch()
            # No scheduling boundary between the final fence and API entry.
            submitted = True
            return await self.gateway.call_api(action, params)

        try:
            response = await asyncio.wait_for(submit(), self.timeout_seconds)
        except (asyncio.CancelledError, Exception):
            if not submitted:
                raise
            raise QQMessageDeliveryUncertainError(f"{action} result unknown after submission") from None
        # SnowLuma ACTION_FAILED=100 also wraps exceptions AFTER its HTTP
        # write. The generic code alone does not prove a business rejection.
        if (isinstance(response, dict) and response.get("status") == "failed"
                and type(response.get("retcode")) is int and response["retcode"] == 100
                and not _explicit_rejection(response)):
            raise QQMessageDeliveryUncertainError(f"{action} bridge result unknown after submission")
        try:
            data = _envelope(response, action=action)
            if not isinstance(data, dict) and not (allow_null and data is None):
                raise QzoneProtocolError("invalid write data")
            return data
        except QzoneProtocolError:
            raise QQMessageDeliveryUncertainError(f"{action} returned an invalid acknowledgement") from None


def _envelope(response, *, action: str):
    if (not isinstance(response, dict) or type(response.get("status")) is not str
            or response["status"] not in {"ok", "failed"}):
        raise QzoneProtocolError(f"{action} returned an invalid response envelope")
    retcode = response.get("retcode")
    if type(retcode) is not int or not -(2**31) <= retcode <= 2**31 - 1:
        raise QzoneProtocolError(f"{action} returned an invalid response code")
    if retcode != 0 and response["status"] == "failed":
        raise QzoneRejectedError(f"{action} rejected with retcode {retcode}")
    if response["status"] != "ok" or retcode != 0 or "data" not in response:
        raise QzoneProtocolError(f"{action} returned an inconsistent response envelope")
    return response["data"]


def _explicit_rejection(response: dict) -> bool:
    for key in ("wording", "message"):
        text = response.get(key)
        if not isinstance(text, str) or len(text) > 8192:
            continue
        codes = re.findall(r"qzone (?:publish|comment|like) failed: (?:code|subcode)=(-?\d{1,10})(?:\s|$)", text)
        if any(int(code) != 0 for code in codes):
            return True
        # The fixed action uploads all images before calling publish/comment.
        if re.match(r"第 \d{1,2} 张图片上传失败:", text):
            return True
    return False


def _integer(value, *, minimum: int = 1, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Qzone integer parameter is outside its bounds")


def _string(value, *, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("Qzone string parameter is invalid")
    return value


def _identifier(value) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("Qzone identifier is invalid")
    return value


def _timestamp(value) -> datetime:
    _integer(value, maximum=4102444800)
    return datetime.fromtimestamp(value, timezone.utc)


def _content(value) -> str:
    _string(value, maximum=_MAX_TEXT)
    if not value.strip() or "\x00" in value:
        raise ValueError("Qzone content must be nonempty bounded text")
    return value


def _http_image(value: str) -> str:
    _string(value, maximum=4096)
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Qzone image reference must be an approved HTTP URL")
    if any(ord(char) < 32 for char in value):
        raise ValueError("Qzone image reference contains control characters")
    return value


def _read_images(values) -> tuple[str, ...]:
    if not isinstance(values, list) or len(values) > _MAX_IMAGES:
        raise ValueError("Qzone images must be a bounded list")
    return tuple(_http_image(value) for value in values)


def _write_images(values) -> list[str]:
    if not isinstance(values, (tuple, list)) or len(values) > _MAX_IMAGES:
        raise ValueError("Qzone images must be a bounded sequence")
    result = []
    for value in values:
        _string(value, maximum=_MAX_IMAGE_BYTES * 4 // 3 + 100)
        if value.startswith("base64://"):
            try:
                decoded = base64.b64decode(value[9:], validate=True)
            except (ValueError, binascii.Error):
                raise ValueError("Qzone base64 image is invalid") from None
            if not decoded or len(decoded) > _MAX_IMAGE_BYTES:
                raise ValueError("Qzone base64 image is outside size bounds")
            result.append(value)
        elif value.startswith(("http://", "https://")):
            result.append(_http_image(value))
        elif value.startswith("file://"):
            parsed = urlsplit(value)
            if parsed.netloc not in {"", "localhost"} or not parsed.path or parsed.query or parsed.fragment:
                raise ValueError("Qzone file image must be local")
            if any(ord(char) < 32 for char in unquote(parsed.path)) or unquote(parsed.path).startswith("//"):
                raise ValueError("Qzone file image must be local")
            result.append(value)
        elif Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            if PureWindowsPath(value).drive.startswith("\\\\"):
                raise ValueError("Qzone file image must be local")
            result.append(PureWindowsPath(value).as_uri() if PureWindowsPath(value).is_absolute()
                          else Path(value).resolve().as_uri())
        else:
            raise ValueError("Qzone image must be a local path or approved image reference")
    return result


class _FeedHTML(HTMLParser):
    """Treat HTML as inert markup; never run scripts or follow URLs."""

    def __init__(self, owner: int | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.owner = owner
        self.post_ids: set[str] = set()
        self.conflicting_owner = False
        self.text: list[str] = []
        self.blocked: list[str] = []
        self.comment_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "iframe", "object", "noscript"}:
            self.blocked.append(tag)
        if self.blocked:
            return
        values = dict(attrs)
        # SnowLuma comment rows carry the COMMENTER's data-uin/data-tid.
        # Their nested links likewise cannot establish the enclosing post's
        # identity. Original-post candidates still require author-list proof.
        comment_region = bool(self.comment_tags) or bool(
            {"comments-list", "comments-item"}.intersection((values.get("class") or "").split()))
        if comment_region and tag not in _HTML_VOID_TAGS:
            self.comment_tags.append(tag)
        if self.owner is not None and not comment_region:
            tid, owner = values.get("data-tid"), values.get("data-uin") or values.get("data-tuin")
            if tid is not None and owner is not None and _ID.fullmatch(tid):
                if owner == str(self.owner):
                    self.post_ids.add(tid)
                else:
                    self.conflicting_owner = True
            if tag == "a" and values.get("href"):
                parsed = urlsplit(values["href"])
                match = re.fullmatch(r"/(\d+)/mood/([A-Za-z0-9_-]{1,128})/?", unquote(parsed.path))
                if parsed.scheme in {"http", "https"} and parsed.hostname == "user.qzone.qq.com" and match:
                    if int(match[1]) == self.owner:
                        self.post_ids.add(match[2])
                    else:
                        self.conflicting_owner = True
        if tag in {"br", "p", "div", "li"}:
            self.text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self.blocked and tag == self.blocked[-1]:
            self.blocked.pop()
        if not self.blocked and tag in self.comment_tags:
            last = len(self.comment_tags) - 1 - self.comment_tags[::-1].index(tag)
            del self.comment_tags[last:]
        if not self.blocked and tag in {"p", "div", "li"}:
            self.text.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.blocked:
            self.text.append(data)


def _plain_text(html: str) -> str:
    parser = _FeedHTML()
    parser.feed(html)
    parser.close()
    return " ".join("".join(parser.text).split())
