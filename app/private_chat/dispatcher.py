"""Private ingress persistence without group policy, persona or memory wiring."""

from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from app.adapters.onebot_models import PrivateMessageEvent
from app.core.image_cache import cache_images_in_raw_payload
from app.core.message_content import extract_images_from_raw_payload
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository, UserRepository


class PrivateMessageHandler(Protocol):
    async def handle_private_message(self, event: PrivateMessageEvent) -> bool: ...


@dataclass(slots=True)
class PrivateMessageDispatcher:
    engine: object
    data_dir: Path
    private_chat_service: PrivateMessageHandler

    def _persist_inbound(self, event: PrivateMessageEvent) -> bool:
        platform_msg_id = str(getattr(event, "platform_msg_id", "")).strip()
        if not platform_msg_id:
            return True
        # Preserve the existing ledger namespace for upgrades and replays.
        ledger_id = f"private-inbound-{event.user_id}-{event.platform_msg_id}"
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            if messages.get_by_platform_msg_id(ledger_id) is not None:
                return False
            UserRepository(session).upsert_user(
                user_id=event.user_id,
                nickname=event.nickname,
                group_card="",
            )
            if event.images:
                agent = getattr(self.private_chat_service, "agent_handler", None)
                if agent is not None:
                    self._cache_native_images(event, agent.image_input_root(event.user_id))
                else:
                    cache_images_in_raw_payload(event.raw_payload, cache_dir=self.data_dir / "image_cache")
                event.images = extract_images_from_raw_payload(event.raw_payload)
            messages.add_private_message(
                platform_msg_id=ledger_id,
                user_id=event.user_id,
                timestamp=event.timestamp,
                plain_text=event.plain_text,
                raw_json=event.raw_payload,
                msg_type=getattr(event, "msg_type", "text"),
                reply_to_msg_id=getattr(event, "reply_to_msg_id", None),
            )
        return True

    def _cache_native_images(self, event: PrivateMessageEvent, owner_root: Path) -> None:
        """Native private input is canonical QQ data under one user's cache.

        The legacy cache behavior stays unchanged. A gateway local_path from
        any other scope never becomes a host-file capability for the model.
        """
        raw = deepcopy(event.raw_payload)
        raw.pop("group_id", None)
        raw["message_id"] = sha256(f"{event.user_id}:{event.platform_msg_id}".encode()).hexdigest()
        parts = raw.get("message", [])
        if not isinstance(parts, list):
            return
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "image" or not isinstance(part.get("data"), dict):
                continue
            data = part["data"]
            local = data.get("local_path")
            try:
                owned = isinstance(local, str) and Path(local).resolve().is_relative_to(owner_root.resolve())
            except (OSError, ValueError):
                owned = False
            if not owned:
                data.pop("local_path", None)
            try:
                parsed = urlparse(str(data.get("url", "")))
                host = (parsed.hostname or "").lower().rstrip(".")
                trusted = (parsed.scheme in {"http", "https"} and parsed.port in {None, 80, 443}
                    and parsed.username is None and parsed.password is None
                    and any(host == domain or host.endswith("." + domain)
                        for domain in ("qq.com", "qq.com.cn", "qpic.cn", "qlogo.cn")))
            except ValueError:
                trusted = False
            if not trusted:
                data["url"] = ""
        cache_images_in_raw_payload(raw, cache_dir=owner_root)
        event.raw_payload["message"] = parts

    async def handle_private_message(self, event: PrivateMessageEvent) -> None:
        if self._persist_inbound(event):
            await self.private_chat_service.handle_private_message(event)
