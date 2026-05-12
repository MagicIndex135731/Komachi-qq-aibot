from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
import logging
from pathlib import Path
import re

from app.adapters.sender import OutboundMessage, OutboundPrivateMessage
from app.admin.commands import AdminCommandParser, CommandContext
from app.config import AppSettings, RuntimeConfig
from app.core.chat_style import build_human_chat_style_lines, normalize_chat_reply, normalize_proactive_chat_reply
from app.core.context_builder import ContextBuilder
from app.core.image_cache import cache_images_in_raw_payload
from app.core.message_archive import append_group_message_archive
from app.core.image_turn_resolver import resolve_images_for_turn
from app.core.message_content import ImageAttachment, extract_images_from_raw_payload
from app.core.memory_engine import extract_memory_candidates, retrieve_relevant_memories
from app.core.persona_engine import render_persona, render_safety_lines
from app.core.reply_policy import PolicyInput, ReplyPolicy
from app.core.search_policy import (
    build_forced_search_query,
    build_current_datetime_facts,
    build_search_decision_prompt,
    detect_address_intent,
    is_explicit_search_request,
    is_general_search_decision_candidate,
    needs_external_lookup_search,
    needs_reference_search,
    is_time_sensitive_request,
    needs_current_datetime_context,
    normalize_relative_time_query,
    parse_search_decision,
    SearchDecision,
)
from app.core.summarizer import summarize_window
from app.core.usage_meter import UsageTotals, build_local_day_utc_window, format_daily_admin_usage_report
from app.core.web_grounding import build_grounding_notes
from app.jobs.summary_jobs import format_summary_source_lines, should_schedule_window_summary
from app.providers.web_search import WebSearchClient
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    SummaryRepository,
    UsageRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)
LOOKUP_NORMALIZER = re.compile(r"[\s\u3000`~!@#$%^&*()_+\-=\[\]{}\\|;:'\",<.>/?，。！？：；、“”‘’（）《》【】]")


@dataclass(slots=True)
class PreparedGroupReply:
    should_reply: bool
    prompt_lines: list[str] | None = None
    target_images: list[ImageAttachment] | None = None
    requires_user_visible_failure_reply: bool = False
    proactive_turn: bool = False


@dataclass(slots=True)
class InboundRouter:
    engine: object
    runtime: RuntimeConfig
    sender: object
    llm_client: object
    reply_policy: ReplyPolicy
    context_builder: ContextBuilder
    admin_parser: AdminCommandParser
    web_search_client: WebSearchClient | None = None
    dev_control_service: object | None = None

    @classmethod
    def build_for_test(cls, *, sqlite_engine, sender, llm_client, web_search_client=None, dev_control_service=None):
        settings = AppSettings.model_construct(
            napcat_ws_url="ws://127.0.0.1:3001",
            llm_base_url="https://api.example.test/v1",
            llm_api_key="test-key",
            llm_model="gpt-5.4",
            bot_qq=123456789,
            owner_qq=987654321,
            admin_qqs="",
            search_provider="tavily",
            search_base_url="https://api.tavily.com/search",
            search_api_key="",
            search_timeout_seconds=8.0,
            context_recent_limit=60,
            context_summary_limit=3,
            context_history_limit=8,
            config_dir=Path("configs"),
            data_dir=Path("data"),
        )
        runtime = RuntimeConfig(
            settings=settings,
            persona={
                "name": "Mira",
                "identity": "AI assistant",
                "core_traits": ["calm", "helpful"],
                "speaking_style": {"tone": "natural"},
            },
            group_policy={
                "default_group_behavior": {
                    "enabled": False,
                    "archive": False,
                    "speak": False,
                    "proactive_reply": True,
                    "proactive_interval_seconds": "180-480",
                },
                "groups": {
                    "10001": {
                        "enabled": True,
                        "archive": True,
                        "speak": True,
                        "proactive_reply": True,
                        "proactive_interval_seconds": "180-480",
                    }
                },
            },
            safety={
                "must_disclose_ai_identity": True,
                "deny_prompt_leak": True,
            },
        )
        return cls(
            engine=sqlite_engine,
            runtime=runtime,
            sender=sender,
            llm_client=llm_client,
            web_search_client=web_search_client,
            reply_policy=ReplyPolicy(),
            context_builder=ContextBuilder(),
            admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
            dev_control_service=dev_control_service,
        )

    def _group_runtime_policy(
        self,
        *,
        group_id: int,
    ) -> tuple[bool, bool, bool, tuple[int, int], tuple[time, time] | None, list[str]]:
        defaults = self.runtime.group_policy.get("default_group_behavior", {})
        configured = self.runtime.group_policy.get("groups", {}).get(str(group_id), {})

        enabled = bool(configured.get("enabled", defaults.get("enabled", False)))
        speak_enabled = bool(configured.get("speak", defaults.get("speak", False)))
        proactive_enabled = bool(configured.get("proactive_reply", defaults.get("proactive_reply", True)))
        proactive_interval = self._parse_interval_range(
            configured.get("proactive_interval_seconds", defaults.get("proactive_interval_seconds", "180-480"))
        )
        quiet_hours = self._parse_quiet_hours(configured.get("quiet_hours", defaults.get("quiet_hours")))
        if not enabled:
            speak_enabled = False

        group_policy_lines = [
            "Speak only in allowlisted groups.",
            "Keep replies short in group chat.",
            "Only use web search when the service has marked the turn as eligible.",
        ]
        return enabled, speak_enabled, proactive_enabled, proactive_interval, quiet_hours, group_policy_lines

    def _outbound_platform_msg_id(self, inbound_platform_msg_id: str) -> str:
        return f"bot-reply-{inbound_platform_msg_id}"

    def _private_inbound_platform_msg_id(self, event) -> str:
        return f"private-inbound-{event.user_id}-{event.platform_msg_id}"

    def _parse_interval_range(self, value: object) -> tuple[int, int]:
        if not isinstance(value, str) or "-" not in value:
            return (180, 480)
        minimum_text, maximum_text = value.split("-", maxsplit=1)
        try:
            minimum = int(minimum_text)
            maximum = int(maximum_text)
        except ValueError:
            return (180, 480)
        if minimum <= 0 or maximum <= 0:
            return (180, 480)
        return minimum, max(minimum, maximum)

    def _parse_quiet_hours(self, value: object) -> tuple[time, time] | None:
        if not isinstance(value, str) or "-" not in value:
            return None
        start_text, end_text = value.split("-", maxsplit=1)
        try:
            return time.fromisoformat(start_text), time.fromisoformat(end_text)
        except ValueError:
            return None

    def _normalize_timestamp(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    def _is_admin_usage_query(self, event) -> bool:
        if event.user_id not in self.runtime.settings.admin_whitelist:
            return False
        if not event.mentioned_bot:
            return False
        normalized = event.plain_text.lower()
        return any(
            keyword in normalized
            for keyword in (
                "token",
                "tokens",
                "多少钱",
                "多少刀",
                "花了多少",
                "花费",
                "成本",
                "费用",
                "消耗",
            )
        )

    def _build_admin_usage_report(self, event) -> str:
        current_time = self._normalize_timestamp(event.timestamp)
        start_of_day, end_of_window = build_local_day_utc_window(current_time)
        with session_scope(self.engine) as session:
            usage_summary = UsageRepository(session).summarize_usage(
                start_at=start_of_day,
                end_at=end_of_window,
                model=self.runtime.settings.llm_model,
            )
        totals = UsageTotals(
            call_count=usage_summary["call_count"],
            input_tokens=usage_summary["input_tokens"],
            cached_input_tokens=usage_summary["cached_input_tokens"],
            output_tokens=usage_summary["output_tokens"],
        )
        return format_daily_admin_usage_report(totals)

    def _build_local_generation_failure_reply(self, *, target_images: list[ImageAttachment] | None) -> str:
        if target_images:
            return "我这边刚刚图没读出来，你再发一下或者再叫我一次。"
        return "我这边刚刚卡了一下，结果没拿到。你再叫我一次，我马上接上。"

    async def _send_prebuilt_reply(self, event, reply_text: str) -> None:
        reserved = self._reserve_outbound_reply(event, reply_text)
        if not reserved:
            return

        try:
            await self.sender.send_group_text(OutboundMessage(group_id=event.group_id, text=reply_text))
        except Exception:
            logger.exception(
                "reply_send_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
            self._clear_outbound_reply_reservation(event)
            raise
        logger.info(
            "reply_send_success group_id=%s msg_id=%s",
            event.group_id,
            event.platform_msg_id,
        )

        try:
            self._mark_outbound_reply_sent(event, reply_text)
        except Exception:
            self._fallback_mark_outbound_reply_sent(event, reply_text)
        self._archive_outbound_reply(event, reply_text)

    def _build_bot_names(self, persona_name: str) -> set[str]:
        normalized = persona_name.strip().lower()
        if not normalized:
            return set()

        condensed = normalized.replace(" ", "")
        names = {normalized, condensed}
        if condensed and any("\u4e00" <= char <= "\u9fff" for char in condensed) and len(condensed) >= 2:
            names.add(condensed[-2:])
        return {name for name in names if name}

    def _normalize_lookup_text(self, value: str) -> str:
        return LOOKUP_NORMALIZER.sub("", value).lower()

    def _format_member_label(
        self,
        *,
        nickname: str,
        group_card: str,
        fallback: str,
    ) -> str:
        clean_nickname = nickname.strip()
        clean_group_card = group_card.strip()
        if clean_group_card and clean_nickname:
            return f"{clean_group_card}（QQ昵称：{clean_nickname}）"
        if clean_group_card:
            return clean_group_card
        if clean_nickname:
            return clean_nickname
        return fallback

    def _member_label_for_user(self, *, user_id: int, users_by_id: dict[int, object]) -> str:
        if user_id == self.runtime.settings.bot_qq:
            persona_name = str(self.runtime.persona.get("name", "Bot")).strip()
            return persona_name or "Bot"
        user = users_by_id.get(user_id)
        if user is None:
            return str(user_id)
        return self._format_member_label(
            nickname=str(getattr(user, "nickname", "")),
            group_card=str(getattr(user, "group_card", "")),
            fallback=str(user_id),
        )

    def _format_message_line(self, *, user_id: int, plain_text: str, users_by_id: dict[int, object]) -> str:
        return f"{self._member_label_for_user(user_id=user_id, users_by_id=users_by_id)}: {plain_text}"

    def _target_message_text_for_prompt(self, *, event, resolved_image_count: int = 0) -> str:
        if event.plain_text.strip():
            return event.plain_text
        image_count = len(event.images)
        if image_count <= 0:
            image_count = resolved_image_count
        if image_count <= 0:
            return event.plain_text
        if image_count == 1:
            return "[sent 1 image]" if event.images else "[asked about 1 image]"
        return f"[sent {image_count} images]" if event.images else f"[asked about {image_count} images]"

    def _resolve_referenced_member_id(
        self,
        *,
        query_text: str,
        users_by_id: dict[int, object],
        exclude_user_ids: set[int] | None = None,
    ) -> int | None:
        normalized_query = self._normalize_lookup_text(query_text)
        if not normalized_query:
            return None

        excluded = exclude_user_ids or set()
        best_match: tuple[int, int] | None = None
        for user_id, user in users_by_id.items():
            if user_id in excluded or user_id == self.runtime.settings.bot_qq:
                continue
            raw_aliases = [
                str(getattr(user, "group_card", "")).strip(),
                str(getattr(user, "nickname", "")).strip(),
            ]
            for alias in raw_aliases:
                normalized_alias = self._normalize_lookup_text(alias)
                if len(normalized_alias) < 2:
                    continue
                if normalized_alias not in normalized_query:
                    continue
                score = len(normalized_alias)
                if alias == str(getattr(user, "group_card", "")).strip():
                    score += 100
                if best_match is None or score > best_match[1]:
                    best_match = (user_id, score)
        return None if best_match is None else best_match[0]

    def _humanize_memory_content(self, *, content: str, user_id: int, member_label: str) -> str:
        numeric_prefix = f"{user_id} "
        if content.startswith(numeric_prefix):
            return member_label + content[len(str(user_id)) :]
        return content.replace(str(user_id), member_label)

    def _build_member_focus_lines(
        self,
        *,
        event,
        messages: MessageRepository,
        memories: MemoryRepository,
        users: UserRepository,
    ) -> list[str]:
        candidate_user_ids = messages.list_recent_group_user_ids(group_id=event.group_id, limit=200)
        candidate_user_ids.extend([event.user_id, self.runtime.settings.bot_qq])
        users_by_id = users.get_users_by_ids(candidate_user_ids)
        referenced_user_id = self._resolve_referenced_member_id(
            query_text=event.plain_text,
            users_by_id=users_by_id,
            exclude_user_ids={event.user_id},
        )
        if referenced_user_id is None:
            return []

        member_label = self._member_label_for_user(user_id=referenced_user_id, users_by_id=users_by_id)
        member_lines = [f"Referenced member: {member_label}"]

        member_messages = messages.list_recent_group_messages_for_user(
            group_id=event.group_id,
            user_id=referenced_user_id,
            limit=max(4, self.runtime.settings.context_history_limit),
        )
        if member_messages:
            member_lines.append(
                "Recent messages from this member:\n"
                + "\n".join(
                    self._format_message_line(
                        user_id=message.user_id,
                        plain_text=message.plain_text,
                        users_by_id=users_by_id,
                    )
                    for message in member_messages
                )
            )

        member_memories = memories.list_group_memories_for_subject(
            scope_id=str(event.group_id),
            subject_id=str(referenced_user_id),
            limit=4,
        )
        if member_memories:
            member_lines.append(
                "Known memories about this member:\n"
                + "\n".join(
                    self._humanize_memory_content(
                        content=memory.content,
                        user_id=referenced_user_id,
                        member_label=member_label,
                    )
                    for memory in member_memories
                )
            )

        return member_lines

    def _resolve_group_policy(
        self,
        *,
        groups: GroupRepository,
        group_id: int,
    ) -> tuple[bool, bool, bool, tuple[int, int], tuple[time, time] | None, list[str]]:
        (
            runtime_enabled,
            runtime_speak_enabled,
            proactive_enabled,
            proactive_interval,
            quiet_hours,
            group_policy_lines,
        ) = self._group_runtime_policy(group_id=group_id)
        stored_group = groups.get_group(group_id)
        if stored_group is None:
            return runtime_enabled, runtime_speak_enabled, proactive_enabled, proactive_interval, quiet_hours, group_policy_lines

        enabled = bool(stored_group.enabled)
        speak_enabled = bool(stored_group.speak_enabled) if enabled else False
        return enabled, speak_enabled, proactive_enabled, proactive_interval, quiet_hours, group_policy_lines

    def _archive_inbound_message(self, event) -> None:
        try:
            append_group_message_archive(
                history_dir=self.runtime.settings.data_dir / "history",
                group_id=event.group_id,
                timestamp=event.timestamp,
                platform_msg_id=event.platform_msg_id,
                user_id=event.user_id,
                nickname=event.nickname,
                group_card=event.group_card,
                plain_text=event.plain_text,
                msg_type=event.msg_type,
                mentioned_bot=event.mentioned_bot,
                reply_to_msg_id=event.reply_to_msg_id,
                direction="inbound",
                image_local_paths=[image.local_path for image in event.images if image.local_path],
            )
        except Exception:
            logger.exception(
                "history_archive_inbound_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )

    def _archive_outbound_reply(self, event, reply_text: str) -> None:
        try:
            append_group_message_archive(
                history_dir=self.runtime.settings.data_dir / "history",
                group_id=event.group_id,
                timestamp=event.timestamp,
                platform_msg_id=self._outbound_platform_msg_id(event.platform_msg_id),
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

    def _persist_inbound_message(self, event) -> bool:
        with session_scope(self.engine) as session:
            groups = GroupRepository(session)
            users = UserRepository(session)
            messages = MessageRepository(session)
            summaries = SummaryRepository(session)
            memories = MemoryRepository(session)

            inbound_message = messages.get_by_platform_msg_id(event.platform_msg_id)
            if inbound_message is not None:
                outbound_message = messages.get_by_platform_msg_id(
                    self._outbound_platform_msg_id(event.platform_msg_id)
                )
                return outbound_message is None

            enabled, speak_enabled, _proactive_enabled, _proactive_interval, _quiet_hours, _group_policy_lines = (
                self._resolve_group_policy(
                    groups=groups,
                    group_id=event.group_id,
                )
            )
            group = groups.get_group(event.group_id)
            if group is None:
                groups.upsert_group(
                    group_id=event.group_id,
                    group_name=str(event.group_id),
                    enabled=enabled,
                    speak_enabled=speak_enabled,
                )
            else:
                group.group_name = str(event.group_id)
                groups.session.add(group)
            current_user = users.upsert_user(user_id=event.user_id, nickname=event.nickname, group_card=event.group_card)
            current_users_by_id = {event.user_id: current_user}
            if event.images:
                cache_images_in_raw_payload(
                    event.raw_payload,
                    cache_dir=self.runtime.settings.data_dir / "image_cache",
                )
                event.images = extract_images_from_raw_payload(event.raw_payload)
            messages.add_group_message(
                platform_msg_id=event.platform_msg_id,
                group_id=event.group_id,
                user_id=event.user_id,
                timestamp=event.timestamp,
                plain_text=event.plain_text,
                raw_json=event.raw_payload,
                msg_type=event.msg_type,
                reply_to_msg_id=event.reply_to_msg_id,
                mentioned_bot=event.mentioned_bot,
            )

            current_lines = format_summary_source_lines(
                [
                    self._format_message_line(
                        user_id=event.user_id,
                        plain_text=event.plain_text,
                        users_by_id=current_users_by_id,
                    )
                ]
            )
            for candidate in extract_memory_candidates(
                scope_id=str(event.group_id),
                source_msg_id=event.platform_msg_id,
                lines=current_lines,
            ):
                candidate["subject_id"] = str(event.user_id)
                memories.add_memory(**candidate)

            message_count = messages.count_group_messages(group_id=event.group_id)
            if should_schedule_window_summary(message_count=message_count):
                window_messages = messages.list_recent_group_messages(group_id=event.group_id, limit=25)
                window_users_by_id = users.get_users_by_ids([message.user_id for message in window_messages])
                source_lines = format_summary_source_lines(
                    [
                        self._format_message_line(
                            user_id=item.user_id,
                            plain_text=item.plain_text,
                            users_by_id=window_users_by_id,
                        )
                        for item in window_messages
                    ]
                )
                if source_lines:
                    summaries.add_summary(
                        scope_type="group",
                        scope_id=str(event.group_id),
                        summary_level="window",
                        start_at=window_messages[0].timestamp,
                        end_at=window_messages[-1].timestamp,
                        content=summarize_window(source_lines),
                        source_count=len(source_lines),
                    )
            return True

    def _persist_private_inbound_message(self, event) -> bool:
        platform_msg_id = str(getattr(event, "platform_msg_id", "")).strip()
        if not platform_msg_id:
            return True

        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)
            private_platform_msg_id = self._private_inbound_platform_msg_id(event)

            if messages.get_by_platform_msg_id(private_platform_msg_id) is not None:
                return False

            users.upsert_user(user_id=event.user_id, nickname=event.nickname, group_card="")
            if event.images:
                cache_images_in_raw_payload(
                    event.raw_payload,
                    cache_dir=self.runtime.settings.data_dir / "image_cache",
                )
                event.images = extract_images_from_raw_payload(event.raw_payload)
            messages.add_private_message(
                platform_msg_id=private_platform_msg_id,
                user_id=event.user_id,
                timestamp=event.timestamp,
                plain_text=event.plain_text,
                raw_json=event.raw_payload,
                msg_type=getattr(event, "msg_type", "text"),
                reply_to_msg_id=getattr(event, "reply_to_msg_id", None),
            )
            return True

    def _prepare_group_reply(self, event, *, quoted_raw_payload: dict | None = None) -> PreparedGroupReply:
        with session_scope(self.engine) as session:
            groups = GroupRepository(session)
            users = UserRepository(session)
            messages = MessageRepository(session)
            summaries = SummaryRepository(session)
            memories = MemoryRepository(session)

            (
                _enabled,
                speak_enabled,
                proactive_enabled,
                proactive_interval,
                quiet_hours,
                group_policy_lines,
            ) = self._resolve_group_policy(
                groups=groups,
                group_id=event.group_id,
            )
            recent_messages = messages.list_recent_group_messages(
                group_id=event.group_id,
                limit=self.runtime.settings.context_recent_limit,
            )
            users_by_id = users.get_users_by_ids(
                [message.user_id for message in recent_messages] + [event.user_id, self.runtime.settings.bot_qq]
            )
            recent_lines = [
                self._format_message_line(
                    user_id=message.user_id,
                    plain_text=message.plain_text,
                    users_by_id=users_by_id,
                )
                for message in recent_messages
            ]
            recent_minute_threshold = self._normalize_timestamp(event.timestamp) - timedelta(minutes=1)
            recent_minute_traffic = max(
                1,
                sum(
                    1
                    for message in recent_messages
                    if self._normalize_timestamp(message.timestamp) >= recent_minute_threshold
                ),
            )
            recent_bot_message_count = sum(
                1 for message in recent_messages[-3:] if message.user_id == self.runtime.settings.bot_qq
            )
            bot_recently_participated = any(
                message.user_id == self.runtime.settings.bot_qq for message in recent_messages[-10:]
            )
            lowered_message = event.plain_text.lower()
            persona_name = str(self.runtime.persona.get("name", "")).strip()
            bot_names = self._build_bot_names(persona_name)
            address_decision = detect_address_intent(
                text=lowered_message,
                bot_names=bot_names,
                reply_to_bot=False,
                quoted_bot=False,
                bot_recently_participated=bot_recently_participated,
                recent_bot_message_count=recent_bot_message_count,
            )
            time_sensitive = is_time_sensitive_request(event.plain_text)
            named_bot = address_decision.reason == "named_bot"
            addressed_turn = event.mentioned_bot or address_decision.is_addressed
            addressed_without_at = address_decision.is_addressed and not event.mentioned_bot and not named_bot
            resolved_image_turn = resolve_images_for_turn(
                event=event,
                addressed_turn=addressed_turn,
                bot_names=bot_names,
                messages=messages,
                quoted_raw_payload=quoted_raw_payload,
            )
            image_followup_trigger = (
                resolved_image_turn is not None and resolved_image_turn.followup_from_prior_prompt
            )
            decision = self.reply_policy.decide(
                PolicyInput(
                    group_speak_enabled=speak_enabled,
                    mentioned_bot=event.mentioned_bot,
                    named_bot=named_bot,
                    direct_question=("?" in event.plain_text) or ("？" in event.plain_text),
                    same_thread_followup=(event.reply_to_msg_id is not None) or image_followup_trigger,
                    recent_bot_reply_at=messages.last_bot_reply_at(
                        group_id=event.group_id,
                        bot_user_id=self.runtime.settings.bot_qq,
                    ),
                    now=event.timestamp,
                    quiet_hours=quiet_hours,
                    proactive_enabled=proactive_enabled,
                    group_traffic_last_minute=recent_minute_traffic,
                    addressed_without_at=addressed_without_at,
                    has_interjection_opportunity=address_decision.is_addressed or time_sensitive,
                    proactive_interval_seconds=proactive_interval,
                    event_id=event.platform_msg_id,
                )
            )
            logger.info(
                "reply_decision group_id=%s msg_id=%s should_reply=%s reason=%s score=%s mentioned_bot=%s "
                "addressed=%s time_sensitive=%s recent_messages=%s",
                event.group_id,
                event.platform_msg_id,
                decision.should_reply,
                decision.reason,
                decision.score,
                event.mentioned_bot,
                address_decision.is_addressed,
                time_sensitive,
                recent_minute_traffic,
            )
            if not decision.should_reply:
                return PreparedGroupReply(False)

            relevant_summaries = summaries.list_recent_group_summaries(
                scope_id=str(event.group_id),
                limit=self.runtime.settings.context_summary_limit,
            )
            memory_rows = memories.list_group_memories(scope_id=str(event.group_id), limit=50)
            relevant_memories = retrieve_relevant_memories(
                event.plain_text,
                [{"content": row.content, "importance": row.importance} for row in memory_rows],
                limit=self.runtime.settings.context_history_limit,
            )
            member_focus_lines = self._build_member_focus_lines(
                event=event,
                messages=messages,
                memories=memories,
                users=users,
            )
            runtime_facts: list[str] = []
            grounding_notes: list[str] = []
            current_datetime_context_required = needs_current_datetime_context(event.plain_text)
            if current_datetime_context_required:
                runtime_facts = build_current_datetime_facts(datetime.now().astimezone())
            web_results: list[str] = []
            web_pages: list[str] = []
            search_hits = []
            page_reads = []
            target_images = resolved_image_turn.images if resolved_image_turn is not None else []
            search_reference_time = self._normalize_timestamp(event.timestamp).astimezone()
            explicit_search_request = is_explicit_search_request(event.plain_text)
            reference_search_request = needs_reference_search(event.plain_text)
            external_lookup_search_request = needs_external_lookup_search(event.plain_text)
            general_search_candidate = is_general_search_decision_candidate(event.plain_text)
            proactive_time_sensitive_turn = decision.reason == "proactive_score" and time_sensitive
            forced_search_request = addressed_turn and (
                explicit_search_request or reference_search_request or external_lookup_search_request
            )
            addressed_optional_search_eligible = (
                addressed_turn and (time_sensitive or general_search_candidate) and not forced_search_request
            )
            if (
                self.web_search_client is not None
                and not current_datetime_context_required
                and (forced_search_request or addressed_optional_search_eligible or proactive_time_sensitive_turn)
            ):
                if forced_search_request:
                    parsed_search = SearchDecision(
                        True,
                        normalize_relative_time_query(
                            build_forced_search_query(event.plain_text, bot_names=bot_names),
                            now=search_reference_time,
                        ),
                        (
                            "reference-topic-required"
                            if reference_search_request
                            else "local-lookup-required"
                            if external_lookup_search_request
                            else "explicit-search-request"
                        ),
                    )
                else:
                    search_prompt = build_search_decision_prompt(
                        bot_name=persona_name or "Bot",
                        target_message=self._format_message_line(
                            user_id=event.user_id,
                            plain_text=event.plain_text,
                            users_by_id=users_by_id,
                        ),
                        recent_messages=recent_lines,
                        proactive_turn=not addressed_turn,
                        now=search_reference_time,
                    )
                    try:
                        parsed_search = parse_search_decision(self.llm_client.generate_text(search_prompt))
                        if parsed_search.should_search:
                            parsed_search = SearchDecision(
                                True,
                                normalize_relative_time_query(parsed_search.query, now=search_reference_time),
                                parsed_search.reason,
                            )
                    except Exception:
                        logger.exception(
                            "web_search_decision_failed group_id=%s msg_id=%s",
                            event.group_id,
                            event.platform_msg_id,
                        )
                        parsed_search = SearchDecision(False, "", "search-decision-error")
                logger.info(
                    "web_search_decision group_id=%s msg_id=%s should_search=%s query=%s reason=%s",
                    event.group_id,
                    event.platform_msg_id,
                    parsed_search.should_search,
                    parsed_search.query,
                    parsed_search.reason,
                )
                if parsed_search.should_search:
                    search_result_limit = 5 if reference_search_request or external_lookup_search_request else 3
                    try:
                        search_hits = self.web_search_client.search(parsed_search.query, max_results=search_result_limit)
                    except Exception:
                        logger.exception(
                            "web_search_execute_failed group_id=%s msg_id=%s query=%s",
                            event.group_id,
                            event.platform_msg_id,
                            parsed_search.query,
                        )
                        search_hits = []
                    logger.info(
                        "web_search_execute group_id=%s msg_id=%s query=%s result_count=%s",
                        event.group_id,
                        event.platform_msg_id,
                        parsed_search.query,
                        len(search_hits),
                    )
                    web_results = [
                        f"{hit.title} | {hit.snippet} | {hit.source} | {hit.date}"
                        for hit in search_hits
                    ]
                    try:
                        page_reads = self.web_search_client.read_pages(
                            search_hits,
                            query=parsed_search.query,
                            max_pages=3,
                        )
                    except Exception:
                        logger.exception(
                            "web_page_fetch_failed group_id=%s msg_id=%s query=%s",
                            event.group_id,
                            event.platform_msg_id,
                            parsed_search.query,
                        )
                        page_reads = []
                    logger.info(
                        "web_page_fetch group_id=%s msg_id=%s fetched_count=%s",
                        event.group_id,
                        event.platform_msg_id,
                        len(page_reads),
                    )
                    web_pages = [
                        f"{page.title} | {page.url} | {page.content}"
                        for page in page_reads
                    ]
                    recent_bot_replies = [
                        message.plain_text
                        for message in recent_messages
                        if message.user_id == self.runtime.settings.bot_qq and message.plain_text.strip()
                    ]
                    grounding_notes = build_grounding_notes(
                        target_text=event.plain_text,
                        external_lookup=external_lookup_search_request,
                        web_results=search_hits,
                        web_pages=page_reads,
                        recent_bot_replies=recent_bot_replies,
                    )

            proactive_turn = not addressed_turn
            prompt_lines = self.context_builder.build(
                persona_text=render_persona(self.runtime.persona),
                safety_rules=render_safety_lines(self.runtime.safety),
                group_policy_lines=group_policy_lines,
                reply_style_lines=build_human_chat_style_lines(proactive_turn=proactive_turn),
                recent_messages=recent_lines,
                member_focus_lines=member_focus_lines,
                summaries=relevant_summaries,
                memories=[memory["content"] for memory in relevant_memories],
                runtime_facts=runtime_facts,
                grounding_notes=grounding_notes,
                web_results=web_results,
                web_pages=web_pages,
                target_message=self._format_message_line(
                    user_id=event.user_id,
                    plain_text=self._target_message_text_for_prompt(
                        event=event,
                        resolved_image_count=len(target_images),
                    ),
                    users_by_id=users_by_id,
                ),
            )
            return PreparedGroupReply(
                should_reply=True,
                prompt_lines=prompt_lines,
                target_images=target_images or None,
                requires_user_visible_failure_reply=(
                    addressed_turn or image_followup_trigger or event.reply_to_msg_id is not None
                ),
                proactive_turn=proactive_turn,
            )

    def _reserve_outbound_reply(self, event, reply_text: str) -> bool:
        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)

            if messages.get_by_platform_msg_id(self._outbound_platform_msg_id(event.platform_msg_id)) is not None:
                return False

            users.upsert_user(
                user_id=self.runtime.settings.bot_qq,
                nickname=str(self.runtime.persona.get("name", "Bot")),
                group_card="",
            )
            messages.add_group_message(
                platform_msg_id=self._outbound_platform_msg_id(event.platform_msg_id),
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

    def _clear_outbound_reply_reservation(self, event) -> None:
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(self._outbound_platform_msg_id(event.platform_msg_id))
            if outbound_message is None:
                return
            session.delete(outbound_message)

    def _mark_outbound_reply_sent(self, event, reply_text: str) -> None:
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(self._outbound_platform_msg_id(event.platform_msg_id))
            if outbound_message is None:
                return
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "reply_to_msg_id": event.platform_msg_id,
                "delivery_state": "sent",
            }
            session.add(outbound_message)

    def _fallback_mark_outbound_reply_sent(self, event, reply_text: str) -> None:
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(self._outbound_platform_msg_id(event.platform_msg_id))
            if outbound_message is None:
                return
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "reply_to_msg_id": event.platform_msg_id,
                "delivery_state": "sent",
            }
            session.add(outbound_message)

    async def handle_group_message(self, event) -> None:
        persisted = self._persist_inbound_message(event)
        if not persisted:
            return
        self._archive_inbound_message(event)
        if self._is_admin_usage_query(event):
            await self._send_prebuilt_reply(event, self._build_admin_usage_report(event))
            return
        quoted_raw_payload = await self._fetch_quoted_message_payload(reply_to_msg_id=event.reply_to_msg_id)
        prepared_reply = self._prepare_group_reply(event, quoted_raw_payload=quoted_raw_payload)
        if not prepared_reply.should_reply or prepared_reply.prompt_lines is None:
            return

        try:
            conversation_key = f"group:{event.group_id}"
            if prepared_reply.target_images:
                raw_reply = self.llm_client.generate_text(
                    prepared_reply.prompt_lines,
                    images=prepared_reply.target_images,
                    conversation_key=conversation_key,
                )
            else:
                raw_reply = self.llm_client.generate_text(
                    prepared_reply.prompt_lines,
                    conversation_key=conversation_key,
                )
            reply_text = (
                normalize_proactive_chat_reply(raw_reply)
                if prepared_reply.proactive_turn
                else normalize_chat_reply(raw_reply)
            )
        except Exception:
            logger.exception(
                "reply_generation_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
            if prepared_reply.requires_user_visible_failure_reply:
                try:
                    await self._send_prebuilt_reply(
                        event,
                        self._build_local_generation_failure_reply(target_images=prepared_reply.target_images),
                    )
                except Exception:
                    logger.exception(
                        "reply_fallback_send_failed group_id=%s msg_id=%s",
                        event.group_id,
                        event.platform_msg_id,
                    )
            return

        if not reply_text.strip():
            logger.warning(
                "reply_generation_empty group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
            if prepared_reply.requires_user_visible_failure_reply:
                try:
                    await self._send_prebuilt_reply(
                        event,
                        self._build_local_generation_failure_reply(target_images=prepared_reply.target_images),
                    )
                except Exception:
                    logger.exception(
                        "reply_fallback_send_failed group_id=%s msg_id=%s",
                        event.group_id,
                        event.platform_msg_id,
                    )
            return

        await self._send_prebuilt_reply(event, reply_text)

    async def _fetch_quoted_message_payload(self, *, reply_to_msg_id: str | None) -> dict | None:
        if not reply_to_msg_id:
            return None
        gateway = getattr(self.sender, "gateway", None)
        if gateway is None or not hasattr(gateway, "call_api"):
            return None
        message_id: int | str = int(reply_to_msg_id) if reply_to_msg_id.isdigit() else reply_to_msg_id
        try:
            response = await gateway.call_api("get_msg", {"message_id": message_id})
        except Exception:
            logger.exception("quoted_message_fetch_failed reply_to_msg_id=%s", reply_to_msg_id)
            return None
        if not isinstance(response, dict):
            return None
        payload = response.get("data")
        if not isinstance(payload, dict):
            return None
        return payload

    async def _send_private_text(self, *, user_id: int, text: str) -> None:
        await self.sender.send_private_text(OutboundPrivateMessage(user_id=user_id, text=text))

    def _configured_group_ids(self) -> list[int]:
        return [int(group_id) for group_id in self.runtime.group_policy.get("groups", {})]

    def _runtime_group_speak_value(self, group_id: int) -> bool:
        defaults = self.runtime.group_policy.get("default_group_behavior", {})
        configured = self.runtime.group_policy.get("groups", {}).get(str(group_id), {})
        return bool(configured.get("speak", defaults.get("speak", False)))

    def _execute_private_admin_command(self, *, sender_qq: int, raw_text: str) -> str | None:
        command = self.admin_parser.parse(
            raw_text,
            CommandContext(sender_qq=sender_qq, is_private_chat=True, group_id=None),
        )
        if command is None:
            return None

        with session_scope(self.engine) as session:
            groups = GroupRepository(session)
            if command.name == "group_allow":
                group_id = int(command.arguments["group_id"])
                groups.set_enabled(group_id, True)
                groups.set_speak_enabled(group_id, True)
                return f"已允许群 {group_id} 发言。"
            if command.name == "group_deny":
                group_id = int(command.arguments["group_id"])
                groups.set_enabled(group_id, True)
                groups.set_speak_enabled(group_id, False)
                return f"已禁止群 {group_id} 发言。"
            if command.name == "status":
                configured_count = len(self._configured_group_ids())
                return (
                    f"当前模型是 {self.runtime.settings.llm_model}，"
                    f"Bot QQ 是 {self.runtime.settings.bot_qq}，"
                    f"配置里的群数量是 {configured_count}。"
                )
            if command.name == "off":
                for group_id in self._configured_group_ids():
                    groups.set_enabled(group_id, False)
                    groups.set_speak_enabled(group_id, False)
                return "我先把配置里的群都静音了。"
            if command.name == "on":
                for group_id in self._configured_group_ids():
                    groups.set_enabled(group_id, True)
                    groups.set_speak_enabled(group_id, self._runtime_group_speak_value(group_id))
                return "我把配置里的群发言状态恢复了。"
        return None

    async def handle_private_message(self, event) -> None:
        persisted = self._persist_private_inbound_message(event)
        if not persisted:
            return

        reply_text = self._execute_private_admin_command(sender_qq=event.user_id, raw_text=event.plain_text)
        if reply_text is not None:
            await self._send_private_text(user_id=event.user_id, text=reply_text)
            return

        if self.dev_control_service is not None:
            handled = await self.dev_control_service.handle_private_message(event)
            if handled:
                return

    async def handle_private_command(self, *, sender_qq: int, raw_text: str) -> None:
        event = type(
            "PrivateCommandEvent",
            (),
            {"user_id": sender_qq, "plain_text": raw_text},
        )()
        await self.handle_private_message(event)
