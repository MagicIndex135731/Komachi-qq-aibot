"""Group QQ delivery lifecycle, independent of memory retrieval and Agent decisions."""

from __future__ import annotations
import logging
import asyncio
import random
from app.adapters.outbound_guard import GroupDispatchRevokedError
from app.adapters.sender import (
    OutboundMessage,
    QQMessageBlockedError,
    QQMessageDeliveryUncertainError,
)
from app.core.chat_style import (
    burst_delays,
    build_reply_split_config,
    scrub_banned_address_terms,
    split_burst_reply,
)
from app.core.message_archive import append_group_message_archive
from app.core.style_distill import BANNED_ADDRESS_TERMS
from app.core.url_policy import explicitly_requests_urls, filter_reply_urls
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository, UserRepository
from app.core.delivery_contract import (
    QQ_BLOCKED_REPLY_NOTICE,
    QQ_BLOCKED_CONTEXT_NOTE,
    DELIVERY_UNCERTAIN_CONTEXT_NOTE,
)

logger = logging.getLogger(__name__)


class GroupReplyDeliveryMixin:
    """Deliver group replies while the host retains policy and memory callbacks.

    The host supplies engine/runtime/sender plus _group_policy_bool,
    _impersonating, _active_persona and _enqueue_episode_message. The mixin
    owns reservation, delivery states, burst delivery and outbound archives.
    """

    def _outbound_platform_msg_id(self, inbound_platform_msg_id: str) -> str:
        return f"bot-reply-{inbound_platform_msg_id}"

    def _blocked_notice_platform_msg_id(self, inbound_platform_msg_id: str) -> str:
        return f"bot-reply-notice-{inbound_platform_msg_id}"

    async def _send_prebuilt_reply(
        self,
        event,
        reply_text: str,
        *,
        allow_chunking: bool = False,
        platform_msg_id: str | None = None,
    ) -> None:
        reply_text = filter_reply_urls(
            reply_text,
            allow_urls=explicitly_requests_urls(event.plain_text),
        )
        reserved = self._reserve_outbound_reply(event, reply_text, platform_msg_id=platform_msg_id)
        if not reserved:
            return

        try:
            await self.sender.send_group_text(
                OutboundMessage(
                    group_id=event.group_id, text=reply_text, allow_chunking=allow_chunking
                )
            )
        except (asyncio.CancelledError, GroupDispatchRevokedError):
            # Sender maps cancellation after dispatch to UNKNOWN. This path
            # therefore only covers a revoked turn before the API write.
            self._clear_outbound_reply_reservation(event, platform_msg_id=platform_msg_id)
            raise
        except QQMessageDeliveryUncertainError as exc:
            logger.warning(
                "reply_delivery_uncertain group_id=%s msg_id=%s error_type=%s",
                event.group_id,
                event.platform_msg_id,
                type(exc).__name__,
            )
            uncertain_reply_text = self._mark_outbound_reply_uncertain(
                event, reply_text, platform_msg_id=platform_msg_id
            )
            self._archive_outbound_reply(
                event, uncertain_reply_text, platform_msg_id=platform_msg_id
            )
            return
        except QQMessageBlockedError as exc:
            logger.warning(
                "reply_qq_blocked group_id=%s msg_id=%s reason=%s",
                event.group_id,
                event.platform_msg_id,
                str(exc),
            )
            blocked_reply_text = self._mark_outbound_reply_blocked(
                event, reply_text, platform_msg_id=platform_msg_id
            )
            self._archive_outbound_reply(event, blocked_reply_text, platform_msg_id=platform_msg_id)
            await self._send_qq_block_notice(event)
            return
        except Exception:
            logger.exception(
                "reply_send_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
            self._clear_outbound_reply_reservation(event, platform_msg_id=platform_msg_id)
            raise
        logger.info(
            "reply_send_success group_id=%s msg_id=%s",
            event.group_id,
            event.platform_msg_id,
        )

        try:
            self._mark_outbound_reply_sent(event, reply_text, platform_msg_id=platform_msg_id)
        except Exception:
            self._fallback_mark_outbound_reply_sent(
                event, reply_text, platform_msg_id=platform_msg_id
            )
        self._archive_outbound_reply(event, reply_text, platform_msg_id=platform_msg_id)

    def _reply_split_config(self) -> dict:
        """Delivery shape for a chat reply, taken from settings not the persona.

        The persona only describes style and personality; how a long answer is
        broken into QQ messages is a system-level policy.
        """

        return build_reply_split_config(self.runtime.settings)

    async def _send_chat_reply(self, event, reply_text: str) -> None:
        """Send an LLM chat reply, optionally as a short multi-message burst."""

        active_persona = self._active_persona(event.group_id)
        # Delivery shape comes from settings, not from the persona: the persona
        # file only describes style and personality.
        burst = self._reply_split_config()
        allow_urls = explicitly_requests_urls(event.plain_text)
        # A long Markdown citation can contain dots that the burst splitter
        # treats as sentence boundaries. Remove unwanted links while the reply
        # is still whole, before per-segment delivery applies its final guard.
        reply_text = filter_reply_urls(
            reply_text,
            allow_urls=allow_urls,
        )
        if allow_urls and any(marker in reply_text for marker in ("http://", "https://", "www.")):
            # Explicitly requested URLs must remain intact rather than being
            # split on the dots in their hostnames.
            burst = {**burst, "enabled": False}
        segments = split_burst_reply(reply_text, burst)
        impersonating = self._impersonating(event.group_id)
        base_id = self._outbound_platform_msg_id(event.platform_msg_id)
        delay_min, delay_max = burst_delays(burst, segment_count=len(segments))
        for index, segment in enumerate(segments):
            if impersonating:
                segment = scrub_banned_address_terms(segment, BANNED_ADDRESS_TERMS)
            platform_msg_id = base_id if index == 0 else f"{base_id}-b{index}"
            await self._send_prebuilt_reply(
                event,
                segment,
                platform_msg_id=platform_msg_id,
            )
            if index < len(segments) - 1 and delay_max > 0:
                await asyncio.sleep(random.uniform(delay_min, delay_max))

    def _archive_outbound_reply(
        self, event, reply_text: str, *, platform_msg_id: str | None = None
    ) -> None:
        if not self._group_policy_bool(group_id=event.group_id, key="archive", default=False):
            return
        try:
            append_group_message_archive(
                history_dir=self.runtime.settings.data_dir / "history",
                group_id=event.group_id,
                timestamp=event.timestamp,
                platform_msg_id=platform_msg_id
                or self._outbound_platform_msg_id(event.platform_msg_id),
                user_id=self.runtime.settings.bot_qq,
                nickname=str(self.runtime.persona.get("name", "Bot")),
                group_card="",
                plain_text=reply_text,
                msg_type="text",
                mentioned_bot=False,
                reply_to_msg_id=event.platform_msg_id,
                direction="outbound",
                image_local_paths=[],
            )
        except Exception:
            logger.exception(
                "history_archive_outbound_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )

    def _reserve_outbound_reply(
        self,
        event,
        reply_text: str,
        *,
        platform_msg_id: str | None = None,
    ) -> bool:
        resolved_id = platform_msg_id or self._outbound_platform_msg_id(event.platform_msg_id)
        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)

            if messages.get_by_platform_msg_id(resolved_id) is not None:
                return False

            users.upsert_user(
                user_id=self.runtime.settings.bot_qq,
                nickname=str(self.runtime.persona.get("name", "Bot")),
                group_card="",
            )
            messages.add_group_message(
                platform_msg_id=resolved_id,
                group_id=event.group_id,
                user_id=self.runtime.settings.bot_qq,
                timestamp=event.timestamp,
                plain_text=reply_text,
                raw_json={
                    "direction": "outbound",
                    "reply_to_msg_id": event.platform_msg_id,
                    "delivery_state": "reserved",
                },
                msg_type="text",
                reply_to_msg_id=event.platform_msg_id,
                mentioned_bot=False,
            )
            return True

    def _clear_outbound_reply_reservation(
        self, event, *, platform_msg_id: str | None = None
    ) -> None:
        resolved_id = platform_msg_id or self._outbound_platform_msg_id(event.platform_msg_id)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(resolved_id)
            if outbound_message is None:
                return
            session.delete(outbound_message)

    def _mark_outbound_reply_blocked(
        self, event, reply_text: str, *, platform_msg_id: str | None = None
    ) -> str:
        resolved_id = platform_msg_id or self._outbound_platform_msg_id(event.platform_msg_id)
        blocked_reply_text = f"{str(reply_text).strip()}\n\n{QQ_BLOCKED_CONTEXT_NOTE}"
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(resolved_id)
            if outbound_message is None:
                raise RuntimeError("blocked outbound reply reservation is missing")
            outbound_message.plain_text = blocked_reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "reply_to_msg_id": event.platform_msg_id,
                "delivery_state": "blocked",
                "failure_kind": "qq_sensitive_content",
                "delivery_reason": "wait_for_self_echo_timeout",
                "delivery_attempts": 3,
            }
            session.add(outbound_message)
        return blocked_reply_text

    def _mark_outbound_reply_uncertain(
        self, event, reply_text: str, *, platform_msg_id: str | None = None
    ) -> str:
        resolved_id = platform_msg_id or self._outbound_platform_msg_id(event.platform_msg_id)
        uncertain_reply_text = f"{str(reply_text).strip()}\n\n{DELIVERY_UNCERTAIN_CONTEXT_NOTE}"
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(resolved_id)
            if outbound_message is None:
                raise RuntimeError("uncertain outbound reply reservation is missing")
            outbound_message.plain_text = uncertain_reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "reply_to_msg_id": event.platform_msg_id,
                "delivery_state": "uncertain",
                "failure_kind": "delivery_result_unknown",
                "delivery_reason": "gateway_ack_timeout",
                "delivery_attempts": 1,
            }
            session.add(outbound_message)
        return uncertain_reply_text

    def _reserve_block_notice(self, event) -> bool:
        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)
            platform_msg_id = self._blocked_notice_platform_msg_id(event.platform_msg_id)
            if messages.get_by_platform_msg_id(platform_msg_id) is not None:
                return False
            users.upsert_user(
                user_id=self.runtime.settings.bot_qq,
                nickname=str(self.runtime.persona.get("name", "Bot")),
                group_card="",
            )
            messages.add_group_message(
                platform_msg_id=platform_msg_id,
                group_id=event.group_id,
                user_id=self.runtime.settings.bot_qq,
                timestamp=event.timestamp,
                plain_text=QQ_BLOCKED_REPLY_NOTICE,
                raw_json={
                    "direction": "outbound",
                    "reply_to_msg_id": event.platform_msg_id,
                    "delivery_state": "reserved",
                    "notice_kind": "qq_sensitive_content",
                },
                msg_type="text",
                reply_to_msg_id=event.platform_msg_id,
                mentioned_bot=False,
            )
            return True

    async def _send_qq_block_notice(self, event) -> None:
        if not self._reserve_block_notice(event):
            return
        platform_msg_id = self._blocked_notice_platform_msg_id(event.platform_msg_id)
        try:
            await self.sender.send_group_text(
                OutboundMessage(group_id=event.group_id, text=QQ_BLOCKED_REPLY_NOTICE)
            )
        except Exception:
            logger.exception(
                "reply_qq_block_notice_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
            with session_scope(self.engine) as session:
                messages = MessageRepository(session)
                notice = messages.get_by_platform_msg_id(platform_msg_id)
                if notice is not None:
                    session.delete(notice)
            return

        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            notice = messages.get_by_platform_msg_id(platform_msg_id)
            if notice is not None:
                notice.raw_json = {
                    "direction": "outbound",
                    "reply_to_msg_id": event.platform_msg_id,
                    "delivery_state": "sent",
                    "notice_kind": "qq_sensitive_content",
                }
                session.add(notice)
        self._archive_outbound_reply(
            event,
            QQ_BLOCKED_REPLY_NOTICE,
            platform_msg_id=platform_msg_id,
        )
        self._enqueue_episode_message(
            group_id=event.group_id,
            platform_msg_id=platform_msg_id,
            timestamp=event.timestamp,
        )
        logger.info(
            "reply_qq_block_notice_success group_id=%s msg_id=%s",
            event.group_id,
            event.platform_msg_id,
        )

    def _mark_outbound_reply_sent(
        self, event, reply_text: str, *, platform_msg_id: str | None = None
    ) -> None:
        resolved_id = platform_msg_id or self._outbound_platform_msg_id(event.platform_msg_id)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(resolved_id)
            if outbound_message is None:
                return
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "reply_to_msg_id": event.platform_msg_id,
                "delivery_state": "sent",
            }
            session.add(outbound_message)
        self._enqueue_episode_message(
            group_id=event.group_id,
            platform_msg_id=resolved_id,
            timestamp=event.timestamp,
        )

    def _fallback_mark_outbound_reply_sent(
        self, event, reply_text: str, *, platform_msg_id: str | None = None
    ) -> None:
        resolved_id = platform_msg_id or self._outbound_platform_msg_id(event.platform_msg_id)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(resolved_id)
            if outbound_message is None:
                return
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "reply_to_msg_id": event.platform_msg_id,
                "delivery_state": "sent",
            }
            session.add(outbound_message)
        self._enqueue_episode_message(
            group_id=event.group_id,
            platform_msg_id=resolved_id,
            timestamp=event.timestamp,
        )
