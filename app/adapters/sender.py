from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

from app.adapters.outbound_guard import GroupDispatchRevokedError, current_group_dispatch


class QQMessageBlockedError(RuntimeError):
    """QQ accepted the request but never echoed the bot message back."""


class QQMessageDeliveryUncertainError(RuntimeError):
    """The send request may have been accepted, so repeating it is unsafe."""


class QQMessageRejectedError(RuntimeError):
    """The gateway explicitly rejected a send; expose a bounded safe category."""
    def __init__(self, code='qq_send_rejected'):
        self.code = code
        super().__init__(code)


@dataclass(slots=True)
class OutboundMessage:
    group_id: int
    text: str
    allow_chunking: bool = False


@dataclass(slots=True)
class OutboundPrivateMessage:
    user_id: int
    text: str
    allow_chunking: bool = False


class Sender:
    MAX_TEXT_CHUNK_LENGTH = 180
    MAX_IMAGE_BYTES = 8 * 1024 * 1024

    def __init__(self, gateway) -> None:
        self.gateway = gateway

    async def send_group_text(self, outbound: OutboundMessage) -> tuple[str, ...]:
        return await self._send_text_message(
            action="send_group_msg",
            target_params={"group_id": outbound.group_id},
            text=outbound.text,
            allow_chunking=outbound.allow_chunking,
        )

    async def send_private_text(self, outbound: OutboundPrivateMessage) -> tuple[str, ...]:
        return await self._send_text_message(
            action="send_private_msg",
            target_params={"user_id": outbound.user_id},
            text=outbound.text,
            allow_chunking=outbound.allow_chunking,
        )

    async def send_group_image(self, *, group_id: int, image_file: str) -> None:
        scope = current_group_dispatch(group_id)
        if scope is not None and not scope.can_send():
            raise GroupDispatchRevokedError("Group Agent stopped before QQ dispatch")
        image_payload = self._image_file_payload(image_file)
        await self._send_once(
            action="send_group_msg",
            params={
                "group_id": group_id,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": image_payload},
                    }
                ],
            },
        )

    async def send_private_image(self, *, user_id: int, image_file: str) -> str | None:
        image_payload = self._image_file_payload(image_file)
        return await self._send_once(
            action="send_private_msg",
            params={
                "user_id": user_id,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": image_payload},
                    }
                ],
            },
        )

    async def send_private_media(self, *, user_id: int, text: str, image_files: tuple[str, ...]) -> tuple[str, ...]:
        """A single acknowledged caption/image message, without partial sends."""
        if not 1 <= len(image_files) <= 2 or len(text) > 12000 or '[CQ:' in text:
            raise ValueError('invalid private media message')
        segments = [{'type': 'text', 'data': {'text': text}}] if text else []
        segments.extend({'type': 'image', 'data': {'file': self._image_file_payload(name)}} for name in image_files)
        receipt = await self._send_once(action='send_private_msg', params={'user_id': user_id, 'message': segments})
        return (receipt,) if receipt is not None else ()

    async def send_video(self, *, video_file: str, text: str = '', group_id=None, user_id=None,
                         self_id=None, name='小町') -> tuple[str, ...]:
        if (group_id is None) == (user_id is None) or len(text) > 12000 or '[CQ:' in text:
            raise ValueError('invalid video message')
        if text:
            # A QQ video cannot share a message with a caption. Keep both in
            # separate forward nodes, within one acknowledged external action.
            return await self.send_media_bundle(self_id=self_id, name=name, text=text,
                media_files=(('video', video_file),), group_id=group_id, user_id=user_id)
        segments = [{'type': 'video', 'data': {'file': self._video_file_payload(video_file)}}]
        target = {'group_id': group_id} if group_id is not None else {'user_id': user_id}
        receipt = await self._send_once(action='send_group_msg' if group_id is not None else 'send_private_msg',
            params={**target, 'message': segments})
        if receipt is None:
            raise QQMessageDeliveryUncertainError('QQ video acknowledged without message receipt')
        return (receipt,)

    @staticmethod
    def _video_file_payload(video_file):
        from app.core.video_file import MAX_VIDEO_BYTES
        path = Path(video_file)
        if path.is_symlink() or not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
            raise ValueError('video exceeds send budget')
        with path.open('rb') as stream:
            content = stream.read(MAX_VIDEO_BYTES + 1)
        if not content or len(content) > MAX_VIDEO_BYTES:
            raise ValueError('video exceeds send budget')
        return 'base64://' + base64.b64encode(content).decode('ascii')

    async def send_media_bundle(self, *, self_id: int, name: str, text: str, media_files,
                                group_id=None, user_id=None) -> tuple[str, ...]:
        """One acknowledged forward, preserving mixed image/video node order."""
        if ((group_id is None) == (user_id is None) or not 1 <= len(media_files) <= 20
                or len(text) > 12000 or '[CQ:' in text or type(self_id) is not int or self_id <= 0
                or any(kind not in {'image', 'video'} or Path(path).is_symlink() for kind, path in media_files)):
            raise ValueError('invalid media bundle')
        if sum(Path(path).stat().st_size for _, path in media_files) > 128 * 1024 * 1024:
            raise ValueError('media bundle exceeds send budget')
        nodes = []
        if text:
            nodes.append({'type': 'node', 'data': {'uin': str(self_id), 'name': str(name)[:60],
                'content': [{'type': 'text', 'data': {'text': text}}]}})
        for kind, path in media_files:
            payload = self._image_file_payload(path) if kind == 'image' else self._video_file_payload(path)
            nodes.append({'type': 'node', 'data': {'uin': str(self_id), 'name': str(name)[:60],
                'content': [{'type': kind, 'data': {'file': payload}}]}})
        target = {'group_id': group_id} if group_id is not None else {'user_id': user_id}
        receipt = await self._send_once(action='send_group_forward_msg' if group_id is not None else 'send_private_forward_msg',
            params={**target, 'messages': nodes})
        if receipt is None:
            raise QQMessageDeliveryUncertainError('QQ forward acknowledged without message receipt')
        return (receipt,)

    async def send_image_bundle(self, *, self_id: int, name: str, text: str, image_files: tuple[str, ...],
                                group_id: int | None = None, user_id: int | None = None) -> tuple[str, ...]:
        """Send one merged forward with real bot-authored nodes, one image each."""
        if ((group_id is None) == (user_id is None) or not 1 <= len(image_files) <= 20
                or len(text) > 12000 or '[CQ:' in text or type(self_id) is not int or self_id <= 0):
            raise ValueError('invalid image bundle')
        if sum(Path(path).stat().st_size for path in image_files) > 32 * 1024 * 1024:
            raise ValueError('image bundle exceeds send budget')
        nodes = []
        for index, path in enumerate(image_files):
            caption = (text + '\n' if index == 0 and text else '') + f'{index + 1}/{len(image_files)}'
            content = [{'type': 'text', 'data': {'text': caption}},
                {'type': 'image', 'data': {'file': self._image_file_payload(path)}}]
            nodes.append({'type': 'node', 'data': {'uin': str(self_id), 'name': str(name)[:60], 'content': content}})
        action = 'send_group_forward_msg' if group_id is not None else 'send_private_forward_msg'
        target = {'group_id': group_id} if group_id is not None else {'user_id': user_id}
        receipt = await self._send_once(action=action, params={**target, 'messages': nodes})
        if receipt is None:
            raise QQMessageDeliveryUncertainError('QQ forward acknowledged without message receipt')
        return (receipt,)

    async def send_group_media(self, *, group_id: int, text: str, image_files: tuple[str, ...], max_text_chars=300) -> tuple[str, ...]:
        """One caption/image message, one acknowledgement and one dispatch marker."""
        if max_text_chars not in {300, 12000} or not 1 <= len(image_files) <= 2 or len(text) > max_text_chars or "[CQ:" in text:
            raise ValueError("invalid group media message")
        segments = [{"type": "text", "data": {"text": text}}] if text else []
        for name in image_files:
            segments.append({"type": "image", "data": {"file": self._image_file_payload(name)}})
        receipt = await self._send_once(action="send_group_msg", params={"group_id": group_id, "message": segments})
        return (receipt,) if receipt is not None else ()

    @classmethod
    def _image_file_payload(cls, name: str) -> str:
        # The QQ gateway may run in a different container or on another host.
        # Its filesystem cannot be assumed to contain an application file URI.
        path = Path(name)
        if not 0 < path.stat().st_size <= cls.MAX_IMAGE_BYTES:
            raise ValueError("image exceeds send budget")
        with path.open("rb") as stream:
            content = stream.read(cls.MAX_IMAGE_BYTES + 1)
        if not content or len(content) > cls.MAX_IMAGE_BYTES:
            raise ValueError("image exceeds send budget")
        return "base64://" + base64.b64encode(content).decode("ascii")

    async def get_qq_avatar(self, *, user_id: int) -> str:
        response = await self._call_profile_action(
            "get_qq_avatar",
            {"user_id": int(user_id)},
        )
        return _extract_avatar_file(response)

    async def set_qq_avatar(self, *, file: str) -> None:
        await self._call_profile_action(
            "set_qq_avatar",
            {"file": str(file)},
        )

    async def set_group_card(self, *, group_id: int, user_id: int, card: str) -> None:
        await self._call_profile_action(
            "set_group_card",
            {
                "group_id": int(group_id),
                "user_id": int(user_id),
                "card": str(card),
            },
        )

    async def get_group_member_info(self, *, group_id: int, user_id: int) -> dict:
        return await self._call_profile_action(
            "get_group_member_info",
            {
                "group_id": int(group_id),
                "user_id": int(user_id),
                # LLBot caches member cards by default; a cached card would
                # poison the pre-switch snapshot used for restoration.
                "no_cache": True,
            },
        )

    async def _call_profile_action(self, action: str, params: dict) -> dict:
        try:
            response = await self.gateway.call_api(action, params)
        except Exception as exc:
            raise RuntimeError(f"{action} failed: {type(exc).__name__}") from exc
        self._require_ok(response, action=action)
        return response or {}

    async def _send_once(self, *, action: str, params: dict) -> str | None:
        scope = current_group_dispatch(params["group_id"]) if action in {"send_group_msg", 'send_group_forward_msg'} else None
        if scope is not None:
            if not scope.can_send():
                raise GroupDispatchRevokedError("Group Agent stopped before QQ dispatch")
            scope.on_dispatch()
            scope.dispatched = True
        try:
            response = await self.gateway.call_api(action, params)
            self._require_ok(response, action=action)
        except asyncio.CancelledError as exc:
            if scope is not None:
                scope.uncertain = True
            raise QQMessageDeliveryUncertainError("QQ send cancelled after dispatch") from exc
        except Exception as exc:
            if self._is_delivery_uncertain_failure(exc, action=action):
                if scope is not None:
                    scope.uncertain = True
                raise QQMessageDeliveryUncertainError(str(exc)) from exc
            if scope is not None:
                scope.failed = True
            raise
        data = (response or {}).get("data") or {}
        message_id = data.get("message_id") if isinstance(data, dict) else None
        receipt = str(message_id) if message_id is not None else None
        if scope is not None and receipt is None:
            scope.uncertain = True
            raise QQMessageDeliveryUncertainError("QQ acknowledged send without a message_id receipt")
        if receipt is not None and scope is not None:
            scope.receipts.append(receipt)
        return receipt

    def _require_ok(self, response: dict | None, *, action: str) -> None:
        payload = response if isinstance(response, dict) else {}
        status = str(payload.get("status", "")).strip().lower()
        retcode = payload.get("retcode", 0)
        try:
            numeric_retcode = int(retcode) if retcode is not None else None
        except (TypeError, ValueError, OverflowError):
            numeric_retcode = None
        if status == "ok" and numeric_retcode == 0:
            return
        message = str(payload.get("message") or payload.get("wording") or "").strip()
        if action in {'send_group_msg','send_private_msg','send_group_forward_msg','send_private_forward_msg'}:
            if self._is_delivery_uncertain_failure(RuntimeError(message), action=action):
                raise QQMessageDeliveryUncertainError('qq_send_result_unknown')
            if status in {'failed', 'error'} or numeric_retcode is not None and numeric_retcode != 0:
                raise QQMessageRejectedError('qq_video_requires_standalone_message'
                    if 'video' in message and 'only segment' in message else 'qq_send_rejected')
            raise QQMessageDeliveryUncertainError('invalid_qq_send_response')
        raise RuntimeError(f"{action} failed: status={status or 'unknown'} retcode={retcode} message={message}")

    def _is_delivery_uncertain_failure(self, error: Exception, *, action: str) -> bool:
        if action not in {"send_group_msg", "send_private_msg", 'send_group_forward_msg', 'send_private_forward_msg'}:
            return False
        if isinstance(error, QQMessageDeliveryUncertainError):
            return True
        if isinstance(error, asyncio.TimeoutError):
            return True
        if isinstance(error, ConnectionError):
            return True
        text = str(error).lower()
        return (
            "waitforselfecho timeout" in text
            or ("nodeikernelmsgservice/sendmsg" in text and "timeout" in text)
        )

    async def _send_text_message(self, *, action: str, target_params: dict, text: str, allow_chunking: bool) -> tuple[str, ...]:
        normalized = str(text).strip()
        if allow_chunking:
            receipts = []
            for chunk in self._split_text_chunks(normalized):
                receipt = await self._send_once(
                    action=action,
                    params={**target_params, "message": chunk},
                )
                if receipt is not None:
                    receipts.append(receipt)
            return tuple(receipts)

        receipt = await self._send_once(
            action=action,
            params={**target_params, "message": normalized},
        )
        return (receipt,) if receipt is not None else ()

    def _split_text_chunks(self, text: str) -> list[str]:
        normalized = str(text).strip()
        if not normalized:
            return [""]
        if len(normalized) <= self.MAX_TEXT_CHUNK_LENGTH:
            return [normalized]

        chunks: list[str] = []
        current = ""
        for raw_line in normalized.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            candidate = line if not current else f"{current}\n{line}"
            if len(candidate) <= self.MAX_TEXT_CHUNK_LENGTH:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            while len(line) > self.MAX_TEXT_CHUNK_LENGTH:
                chunks.append(line[: self.MAX_TEXT_CHUNK_LENGTH])
                line = line[self.MAX_TEXT_CHUNK_LENGTH :].lstrip()
            current = line
        if current:
            chunks.append(current)
        return chunks or [normalized]


def _extract_avatar_file(payload: dict | None) -> str:
    """Extract a usable avatar source from a OneBot response payload.

    LLBot/NapCat implementations disagree on the response shape (``data.url``,
    ``data.file``, a plain base64 string, ...). Accept every documented shape
    and keep the extracted value verbatim so it can be passed back to
    ``set_qq_avatar`` without re-encoding.
    """

    payload = payload or {}
    data = payload.get("data", payload)
    if isinstance(data, str):
        value = data.strip()
        if value:
            return value
        raise RuntimeError("get_qq_avatar returned empty avatar data")
    if not isinstance(data, dict):
        raise RuntimeError("get_qq_avatar returned no usable avatar data")
    for key in ("url", "file", "path", "base64"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inner = data.get("data")
    if isinstance(inner, str) and inner.strip():
        return inner.strip()
    raise RuntimeError("get_qq_avatar returned no usable avatar data")


def extract_group_card(payload: dict | None) -> str:
    payload = payload or {}
    data = payload.get("data", payload)
    if isinstance(data, dict):
        card = data.get("card")
        if isinstance(card, str):
            return card
    return ""
