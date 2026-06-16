from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


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
    RETRYABLE_RETCODES = {1200}
    MAX_SEND_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 0.25
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
        await self._send_with_retry(
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
        await self._send_with_retry(
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

    async def _send_with_retry(self, *, action: str, params: dict) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.MAX_SEND_ATTEMPTS + 1):
            try:
                response = await self.gateway.call_api(action, params)
                self._require_ok(response, action=action)
                return
            except Exception as exc:
                last_error = exc
                if attempt >= self.MAX_SEND_ATTEMPTS or not self._is_retryable_send_failure(exc):
                    raise
                logger.warning(
                    "sender_retry action=%s attempt=%s reason=%s",
                    action,
                    attempt,
                    type(exc).__name__,
                )
                await asyncio.sleep(self.RETRY_DELAY_SECONDS)
        if last_error is not None:
            raise last_error

    def _require_ok(self, response: dict | None, *, action: str) -> None:
        payload = response or {}
        status = str(payload.get("status", "")).strip().lower()
        retcode = payload.get("retcode", 0)
        if status == "ok" and int(retcode or 0) == 0:
            return
        message = str(payload.get("message") or payload.get("wording") or "").strip()
        raise RuntimeError(f"{action} failed: status={status or 'unknown'} retcode={retcode} message={message}")

    def _is_retryable_send_failure(self, error: Exception) -> bool:
        if isinstance(error, asyncio.TimeoutError):
            return True
        if isinstance(error, ConnectionError):
            return True
        text = str(error)
        if "retcode=1200" in text:
            return True
        return "timeout" in text.lower()

    async def _send_text_message(self, *, action: str, target_params: dict, text: str, allow_chunking: bool) -> None:
        normalized = str(text).strip()
        if allow_chunking:
            for chunk in self._split_text_chunks(normalized):
                await self._send_with_retry(
                    action=action,
                    params={**target_params, "message": chunk},
                )
            return

        try:
            await self._send_with_retry(
                action=action,
                params={**target_params, "message": normalized},
            )
        except Exception as exc:
            if not normalized or len(normalized) <= self.MAX_TEXT_CHUNK_LENGTH or not self._is_retryable_send_failure(exc):
                raise
            for chunk in self._split_text_chunks(normalized):
                await self._send_with_retry(
                    action=action,
                    params={**target_params, "message": chunk},
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
