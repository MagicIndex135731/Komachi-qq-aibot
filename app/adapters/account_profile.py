"""Own-account profile actions with bounded sources and honest acknowledgements.

SnowLuma's profile setters return ``okResponse()`` (``data: null``), not an
operation ID. A receipt therefore confirms the bridge acknowledgement only.
Persona switching remains independent of these explicitly enabled actions.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from PIL import Image

from app.adapters.sender import QQMessageDeliveryUncertainError


_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_PIXELS = 16_000_000
_KNOWN_PRE_WRITE_REJECTIONS = frozenset({1400, 1404})


class ProfileRejectedError(RuntimeError):
    """Policy or a known pre-write bridge check rejected the action."""


class ProfileUnavailableError(RuntimeError):
    """The gateway is disconnected before any write was attempted."""


class ProfileResultUncertainError(QQMessageDeliveryUncertainError):
    """A profile write may have succeeded; retain its ledger and do not replay."""


@dataclass(frozen=True, slots=True)
class ProfileReceipt:
    action: str
    account_id: int
    group_id: int | None = None
    acknowledged: bool = True
    artifact_sha256: str | None = None


class AccountProfileAdapter:
    """Expose only own nickname, own avatar, and own card in allowed groups.

    ``media_roots`` must come from configured artifact output, never from model
    arguments. Avatar bytes are decoded and captured before submitting, so a
    gateway cannot fetch an arbitrary URL or a different deployment path.
    Runtime permission, frequency, and durable action ownership remain in the
    caller; ``before_dispatch`` applies its last fence at actual API entry.
    """

    def __init__(
        self,
        gateway,
        *,
        account_id: int,
        allowed_group_ids: Iterable[int] = (),
        media_roots: Iterable[Path | str] = (),
        timeout_seconds: float = 20,
    ) -> None:
        _identifier(account_id)
        if (isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Profile timeout must be finite and positive")
        groups = tuple(allowed_group_ids)
        for group_id in groups:
            _identifier(group_id)
        roots = []
        for raw in media_roots:
            path = Path(raw)
            if not path.is_absolute() or PureWindowsPath(str(raw)).drive.startswith("\\\\"):
                raise ValueError("Profile media root must be an absolute local path")
            roots.append(path.resolve())
        self.gateway = gateway
        self.account_id = account_id
        self.allowed_group_ids = frozenset(groups)
        self.media_roots = tuple(roots)
        self.timeout_seconds = float(timeout_seconds)

    async def nickname(
        self, name: str, *, before_dispatch: Callable[[], None] | None = None
    ) -> ProfileReceipt:
        _name(name, maximum=32)
        await self._write("set_qq_profile", {"nickname": name}, before_dispatch=before_dispatch)
        return ProfileReceipt(action="set_qq_profile", account_id=self.account_id)

    async def card(
        self,
        group_id: int,
        bot_id: int,
        name: str,
        *,
        before_dispatch: Callable[[], None] | None = None,
    ) -> ProfileReceipt:
        _identifier(group_id)
        _identifier(bot_id)
        _name(name, maximum=60)
        if group_id not in self.allowed_group_ids or bot_id != self.account_id:
            raise ProfileRejectedError("Profile card target is outside the own-account scope")
        await self._write(
            "set_group_card",
            {"group_id": group_id, "user_id": self.account_id, "card": name},
            before_dispatch=before_dispatch,
        )
        return ProfileReceipt(
            action="set_group_card", account_id=self.account_id, group_id=group_id
        )

    async def avatar(
        self, file: str, *, before_dispatch: Callable[[], None] | None = None
    ) -> ProfileReceipt:
        raw = self._avatar_bytes(file)
        await self._write(
            "set_qq_avatar",
            {"file": "base64://" + base64.b64encode(raw).decode("ascii")},
            before_dispatch=before_dispatch,
        )
        return ProfileReceipt(
            action="set_qq_avatar", account_id=self.account_id,
            artifact_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def _avatar_bytes(self, file: str) -> bytes:
        if (not isinstance(file, str) or not file or len(file) > 4096
                or any(ord(char) < 32 for char in file)):
            raise ValueError("Profile avatar must be a bounded local artifact path")
        path = Path(file)
        if not path.is_absolute() or PureWindowsPath(file).drive.startswith("\\\\"):
            raise ValueError("Profile avatar must be an absolute local artifact path")
        actual = path.resolve()
        if not any(actual.is_relative_to(root) for root in self.media_roots):
            raise ProfileRejectedError("Profile avatar is outside authorized artifact output")
        try:
            if not actual.is_file():
                raise ValueError("Profile avatar artifact is absent")
            with actual.open("rb") as stream:
                raw = stream.read(_MAX_IMAGE_BYTES + 1)
            if not raw or len(raw) > _MAX_IMAGE_BYTES:
                raise ValueError("Profile avatar exceeds its byte budget")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if (image.format not in {"PNG", "JPEG", "WEBP", "GIF"}
                            or image.width * image.height > _MAX_IMAGE_PIXELS):
                        raise ValueError("Profile avatar has unsupported image dimensions or format")
                    image.verify()
                # verify() checks container structure; load() additionally
                # rejects truncated pixel data, including truncated JPEGs.
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
        except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise ValueError("Profile avatar is not a decodable bounded image") from None
        return raw

    def _require_connection(self) -> None:
        if hasattr(self.gateway, "websocket") and self.gateway.websocket is None:
            raise ProfileUnavailableError("Profile gateway is disconnected before submission")

    async def _write(
        self, action: str, params: dict, *, before_dispatch: Callable[[], None] | None
    ) -> None:
        if before_dispatch is not None and not callable(before_dispatch):
            raise ValueError("Profile dispatch fence must be callable")
        self._require_connection()
        submitted = False

        async def submit():
            nonlocal submitted
            self._require_connection()
            if before_dispatch is not None:
                before_dispatch()
            # Keep the ledger/fence and gateway API entry in one coroutine
            # segment. No await or scheduled callback intervenes here.
            submitted = True
            return await self.gateway.call_api(action, params)

        try:
            response = await asyncio.wait_for(submit(), self.timeout_seconds)
        except (asyncio.CancelledError, Exception):
            if not submitted:
                raise
            raise ProfileResultUncertainError(
                f"{action} result unknown after submission"
            ) from None
        if (isinstance(response, dict) and response.get("status") == "failed"
                and type(response.get("retcode")) is int
                and response["retcode"] in _KNOWN_PRE_WRITE_REJECTIONS
                and "data" in response and response["data"] is None):
            raise ProfileRejectedError(
                f"{action} rejected before write with retcode {response['retcode']}"
            )
        if not (isinstance(response, dict) and response.get("status") == "ok"
                and type(response.get("retcode")) is int and response["retcode"] == 0
                and "data" in response and response["data"] is None):
            # Generic SnowLuma errors may wrap a failure after native write;
            # missing acknowledgement data likewise does not prove no write.
            raise ProfileResultUncertainError(
                f"{action} returned an unconfirmed acknowledgement"
            )


def _identifier(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise ValueError("Profile scope identifier is invalid")


def _name(value: str, *, maximum: int) -> None:
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("Profile display name must be bounded nonempty text")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("Profile display name must contain valid Unicode text") from None
