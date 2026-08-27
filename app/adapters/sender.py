from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


class QQMessageBlockedError(RuntimeError):
    """QQ accepted the request but never echoed the bot message back."""


class QQMessageDeliveryUncertainError(RuntimeError):
    """The send request may have been accepted, so repeating it is unsafe."""


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

    def __init__(self, gateway) -> None:
        self.gateway = gateway

    async def send_group_text(self, outbound: OutboundMessage) -> None:
        await self._send_text_message(
            action="send_group_msg",
            target_params={"group_id": outbound.group_id},
            text=outbound.text,
            allow_chunking=outbound.allow_chunking,
        )

    async def send_private_text(self, outbound: OutboundPrivateMessage) -> None:
        await self._send_text_message(
            action="send_private_msg",
            target_params={"user_id": outbound.user_id},
            text=outbound.text,
            allow_chunking=outbound.allow_chunking,
        )

    async def send_group_image(self, *, group_id: int, image_file: str) -> None:
        image_uri = Path(image_file).resolve().as_uri()
        await self._send_once(
            action="send_group_msg",
            params={
                "group_id": group_id,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": image_uri},
                    }
                ],
            },
        )

    async def send_private_image(self, *, user_id: int, image_file: str) -> None:
        image_uri = Path(image_file).resolve().as_uri()
        await self._send_once(
            action="send_private_msg",
            params={
                "user_id": user_id,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": image_uri},
                    }
                ],
            },
        )

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
            },
        )

    async def _call_profile_action(self, action: str, params: dict) -> dict:
        try:
            response = await self.gateway.call_api(action, params)
        except Exception as exc:
            raise RuntimeError(f"{action} failed: {type(exc).__name__}") from exc
        self._require_ok(response, action=action)
        return response or {}

    async def _send_once(self, *, action: str, params: dict) -> None:
        try:
            response = await self.gateway.call_api(action, params)
            self._require_ok(response, action=action)
        except Exception as exc:
            if self._is_delivery_uncertain_failure(exc, action=action):
                raise QQMessageDeliveryUncertainError(str(exc)) from exc
            raise

    def _require_ok(self, response: dict | None, *, action: str) -> None:
        payload = response or {}
        status = str(payload.get("status", "")).strip().lower()
        retcode = payload.get("retcode", 0)
        if status == "ok" and int(retcode or 0) == 0:
            return
        message = str(payload.get("message") or payload.get("wording") or "").strip()
        raise RuntimeError(f"{action} failed: status={status or 'unknown'} retcode={retcode} message={message}")

    def _is_delivery_uncertain_failure(self, error: Exception, *, action: str) -> bool:
        if action not in {"send_group_msg", "send_private_msg"}:
            return False
        if isinstance(error, asyncio.TimeoutError):
            return True
        if isinstance(error, ConnectionError):
            return True
        text = str(error).lower()
        return (
            "waitforselfecho timeout" in text
            or ("nodeikernelmsgservice/sendmsg" in text and "timeout" in text)
        )

    async def _send_text_message(self, *, action: str, target_params: dict, text: str, allow_chunking: bool) -> None:
        normalized = str(text).strip()
        if allow_chunking:
            for chunk in self._split_text_chunks(normalized):
                await self._send_once(
                    action=action,
                    params={**target_params, "message": chunk},
                )
            return

        await self._send_once(
            action=action,
            params={**target_params, "message": normalized},
        )

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
