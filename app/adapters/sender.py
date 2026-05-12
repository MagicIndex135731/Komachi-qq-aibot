from __future__ import annotations

from dataclasses import dataclass


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
        await self.gateway.call_api(
            "send_group_msg",
            {"group_id": outbound.group_id, "message": outbound.text},
        )

    async def send_private_text(self, outbound: OutboundPrivateMessage) -> None:
        await self.gateway.call_api(
            "send_private_msg",
            {"user_id": outbound.user_id, "message": outbound.text},
        )
