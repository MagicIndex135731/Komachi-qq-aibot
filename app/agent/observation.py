"""Read-only observation and source-backed QQ dispatch authorization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from app.adapters.outbound_guard import current_group_dispatch
from app.core.memory_source_guard import is_usable_memory_source
from app.core.reply_policy import PolicyInput
from app.core.search_policy import detect_address_intent
from app.core.time_utils import shanghai_aware
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


class GroupObservationService:
    def __init__(self, router, *, clock: Callable[[], datetime] | None = None) -> None:
        self.router = router
        self.clock = clock or (lambda: datetime.now(UTC))

    def _live_event(self, event, messages):
        source = messages.get_by_platform_msg_id(event.platform_msg_id)
        if not is_usable_memory_source(source, group_id=event.group_id, author_id=event.user_id):
            return None
        # Addressing authority comes from the ledger, never model-mutated fields.
        return replace(
            event,
            plain_text=source.plain_text,
            mentioned_bot=source.mentioned_bot,
            reply_to_msg_id=source.reply_to_msg_id,
        )

    def _recent(self, messages, group_id: int):
        return [
            item
            for item in messages.list_recent_group_messages(group_id=group_id, limit=24)
            if is_usable_memory_source(item, group_id=group_id)
        ]

    def _direct(self, event, messages, recent) -> bool:
        router = self.router
        persona = router._active_persona(event.group_id)
        address = detect_address_intent(
            text=event.plain_text.lower(),
            bot_names=router._build_bot_names(str(persona.get("name", ""))),
            reply_to_bot=router._is_reply_to_bot(
                event=event, messages=messages, quoted_raw_payload=None
            ),
            quoted_bot=False,
            bot_recently_participated=any(
                item.user_id == router.runtime.settings.bot_qq for item in recent[-10:]
            ),
            recent_bot_message_count=sum(
                item.user_id == router.runtime.settings.bot_qq for item in recent[-3:]
            ),
        )
        return bool(event.mentioned_bot or address.is_addressed)

    def _suppression(
        self,
        *,
        group_id,
        event_id,
        messages,
        policy,
        direct=False,
        last_sent_at=None,
        continued_dispatch=False,
    ) -> str:
        enabled, speak, proactive, interval, quiet, _ = policy
        if not enabled or not speak:
            return "group_speak_disabled"
        if direct:
            return ""
        if not proactive:
            return "proactive_disabled"
        timestamps = [
            self.router._normalize_timestamp(value)
            for value in (
                last_sent_at,
                messages.last_bot_reply_at(
                    group_id=group_id, bot_user_id=self.router.runtime.settings.bot_qq,
                    exclude_owner_user_id=self.router.runtime.settings.owner_qq,
                ),
            )
            if value is not None
        ]
        policy_input = PolicyInput(
            speak,
            False,
            False,
            False,
            max(timestamps) if timestamps else None,
            self.clock(),
            quiet,
            proactive,
            0,
            proactive_interval_seconds=interval,
            event_id=event_id,
        )
        if self.router.reply_policy._in_quiet_hours(policy_input):
            return "quiet_hours"
        # The first chunk is an authorized turn; later chunks still check policy
        # and source but must not be stopped by that turn's own first sent row.
        if not continued_dispatch and self.router.reply_policy._in_cooldown(policy_input):
            return "cooldown"
        return ""

    def proactive_send_allowed(self, group_id, event_id) -> bool:
        """Final chat cadence for both event turns and scheduled wakeups."""
        with session_scope(self.router.engine) as session:
            messages = MessageRepository(session)
            policy = self.router._resolve_group_policy(groups=GroupRepository(session), group_id=group_id)
            return not self._suppression(group_id=group_id, event_id=event_id,
                messages=messages, policy=policy)

    def build_agent_observation(self, batch) -> dict:
        router = self.router
        with session_scope(router.engine) as session:
            messages = MessageRepository(session)
            recent = self._recent(messages, batch.group_id)
            users_by_id = UserRepository(session).get_users_by_ids(
                [item.user_id for item in recent]
            )
            policy = router._resolve_group_policy(
                groups=GroupRepository(session), group_id=batch.group_id
            )
            events = [self._live_event(event, messages) for event in batch.events]
            sources_valid = bool(events) and all(event is not None for event in events)
            direct_target = next(
                (
                    event.platform_msg_id
                    for event in events
                    if event is not None and self._direct(event, messages, recent)
                ),
                None,
            )
            suppression = self._suppression(
                group_id=batch.group_id,
                event_id=batch.events[-1].platform_msg_id if batch.events else "",
                messages=messages,
                policy=policy,
                direct=direct_target is not None,
                last_sent_at=batch.last_sent_at,
            )
            if not sources_valid:
                suppression = "source_unavailable"
            if suppression:
                direct_target = None
            persona = router._active_persona(batch.group_id)
            return {
                "direct_target": direct_target,
                "suppression": suppression,
                "memory_enabled": router._group_memory_enabled(group_id=batch.group_id),
                "persona": router._persona_text_for(persona, batch.group_id),
                "source_message_ids": tuple(item.id for item in recent),
                "recent_lines": [
                    f"[{shanghai_aware(item.timestamp).isoformat(timespec='seconds')}] " + router._format_message_line(
                        user_id=item.user_id,
                        plain_text=item.plain_text[:700],
                        users_by_id=users_by_id,
                        group_id=batch.group_id,
                    )
                    for item in recent
                ],
            }

    def dispatch_allowed(self, event, *, decision, continued_dispatch=False) -> bool:
        del decision  # Agent rationale never grants QQ write permission.
        router = self.router
        with session_scope(router.engine) as session:
            messages = MessageRepository(session)
            current = self._live_event(event, messages)
            if current is None:
                return False
            policy = router._resolve_group_policy(
                groups=GroupRepository(session), group_id=event.group_id
            )
            scope = current_group_dispatch(event.group_id)
            return not self._suppression(
                group_id=event.group_id,
                event_id=event.platform_msg_id,
                messages=messages,
                policy=policy,
                direct=self._direct(current, messages, self._recent(messages, event.group_id)),
                continued_dispatch=bool(continued_dispatch or scope is not None and scope.dispatched),
            )

    def image_job_allowed(self, request) -> bool:
        """Authorize an accepted durable image job by its current source and group.

        A long-running image job has its own cancellation lifecycle. It neither
        inherits a short Agent turn guard nor starts a new participation decision.
        """
        with session_scope(self.router.engine) as session:
            source = MessageRepository(session).get_by_platform_msg_id(request.trigger_message_id)
            if not is_usable_memory_source(
                source, group_id=request.group_id, author_id=request.requester_user_id
            ):
                return False
            enabled, speak, *_ = self.router._resolve_group_policy(
                groups=GroupRepository(session), group_id=request.group_id
            )
            return bool(enabled and speak)
