from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OutboundMessage:
    group_id: int
    text: str


@dataclass(slots=True)
class OutboundPrivateMessage:
    user_id: int
    text: str


class Sender:
    def __init__(self, gateway) -> None:
        self.gateway = gateway

    async def send_group_text(self, outbound: OutboundMessage) -> None:
        response = await self.gateway.call_api(
            "send_group_msg",
            {"group_id": outbound.group_id, "message": outbound.text},
        )
        self._require_ok(response, action="send_group_msg")

    async def send_private_text(self, outbound: OutboundPrivateMessage) -> None:
        response = await self.gateway.call_api(
            "send_private_msg",
            {"user_id": outbound.user_id, "message": outbound.text},
        )
        self._require_ok(response, action="send_private_msg")

    async def send_group_image(self, *, group_id: int, image_file: str) -> None:
        image_uri = Path(image_file).resolve().as_uri()
        response = await self.gateway.call_api(
            "send_group_msg",
            {
                "group_id": group_id,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": image_uri},
                    }
                ],
            },
        )
        self._require_ok(response, action="send_group_msg")

    async def send_private_image(self, *, user_id: int, image_file: str) -> None:
        image_uri = Path(image_file).resolve().as_uri()
        response = await self.gateway.call_api(
            "send_private_msg",
            {
                "user_id": user_id,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": image_uri},
                    }
                ],
            },
        )
        self._require_ok(response, action="send_private_msg")

    def _require_ok(self, response: dict | None, *, action: str) -> None:
        payload = response or {}
        status = str(payload.get("status", "")).strip().lower()
        retcode = payload.get("retcode", 0)
        if status == "ok" and int(retcode or 0) == 0:
            return
        message = str(payload.get("message") or payload.get("wording") or "").strip()
        raise RuntimeError(f"{action} failed: status={status or 'unknown'} retcode={retcode} message={message}")
