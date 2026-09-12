"""Private chat service: daily persona chat and private drawings only.

The owner used to be able to drive Codex project work over QQ through
``app.dev_control``.  That channel was removed on 2026-09-11 (see
``.trellis/tasks/08-11-memory-test-platform/research/39-private-chat-only-scope.md``),
so this service keeps only the chat surface:

* daily persona chat for the owner and the ``PRIVATE_CHAT_QQS`` allow list,
* private image generation plus its follow-up/quote windows,
* outbound reply de-duplication and the rolling session summary.

The private turn store still uses the ``dev_sessions`` / ``dev_tasks`` tables
through the existing repositories; no schema migration was made.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import random
import re

from app.adapters.onebot_models import PrivateMessageEvent
from app.adapters.sender import OutboundPrivateMessage, QQMessageDeliveryUncertainError
from app.core.chat_style import (
    burst_delays,
    build_human_chat_style_lines,
    build_reply_split_config,
    normalize_chat_reply,
    split_burst_reply,
)
from app.core.group_image_generation import (
    ImageJobResult,
    PrivateImageGenerationRequest,
    PrivateImageGenerationService,
)
from app.core.image_turn_resolver import resolve_private_images_for_turn
from app.core.persona_engine import render_persona, render_safety_lines
from app.core.quoted_message import (
    quoted_message_line_for_prompt,
    quoted_pronoun_referent_note,
)
from app.core.router import (
    AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN,
    AUTO_WEB_REFERENCE_QUERY_PATTERN,
    GROUP_IMAGE_NEGATIVE_PATTERNS,
    GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS,
    GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS,
    GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS,
    GROUP_IMAGE_REQUEST_PATTERNS,
    LOOKUP_NORMALIZER,
)
from app.core.search_policy import (
    SearchDecision,
    build_current_datetime_facts,
    build_forced_search_query,
    build_search_decision_prompt,
    build_search_priority_instructions,
    is_explicit_search_request,
    is_general_search_decision_candidate,
    is_search_verification_query,
    is_time_sensitive_request,
    needs_current_datetime_context,
    needs_external_lookup_search,
    needs_reference_search,
    normalize_relative_time_query,
    parse_search_decision,
)
from app.core.time_utils import ASIA_SHANGHAI
from app.core.url_policy import (
    explicitly_requests_urls,
    filter_reply_urls,
    url_reply_policy_instruction,
)
from app.core.web_grounding import build_grounding_notes
from app.storage.db import session_scope
from app.storage.models import DevSession
from app.storage.repositories import (
    DevSessionRepository,
    DevTaskArtifactRepository,
    DevTaskRepository,
    MessageRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)

SESSION_NEW_COMMANDS = {
    "/bot new-session",
    "/bot reset-session",
    "清空上下文",
    "开新对话",
    "开启新对话",
    "新建对话",
    "重开对话",
    "重置会话",
}
SESSION_STATUS_COMMANDS = {
    "/bot session-status",
    "会话状态",
    "查看会话状态",
}
PRIVATE_DRAW_RESET_COMMANDS = {
    "/bot reset-draw",
    "重置绘画",
    "清空绘画",
    "清空画图",
    "重置画图",
}
PRIVATE_IMAGE_RETOUCH_INTENT_KEYWORDS = (
    "轻度人像优化",
    "人像优化",
    "优化一下人脸",
    "优化人脸",
    "稍微调整一下五官",
    "修一下鼻毛",
    "修下鼻毛",
    "修一下胡须",
    "修下胡须",
    "修图",
    "精修",
    "小修",
    "润色",
)

SESSION_MODE_DAILY = "daily"
PRIVATE_SCOPE_OWNER_DAILY = "owner_daily"
PRIVATE_SCOPE_ALLOWLIST_DAILY = "allowlist_daily"

# The private daily context is the last N *messages* of the session and counts
# both sides: every owner message and every Komachi reply occupies one slot, so
# 20 messages cover up to ten exchanges.  The rolling session summary and the
# recent-turns block in the prompt share the same window.
PRIVATE_CONTEXT_MESSAGE_LIMIT = 20
SUMMARY_LINE_LIMIT = PRIVATE_CONTEXT_MESSAGE_LIMIT
# Half the window: each history turn contributes up to two lines (owner +
# assistant).  The call site fetches two turns beyond this so the recent-turns
# block still fills the window once the in-flight turn is excluded.
RECENT_TURN_LIMIT = PRIVATE_CONTEXT_MESSAGE_LIMIT // 2


@dataclass(slots=True)
class PrivateWebContext:
    runtime_facts: list[str]
    web_results: list[str]
    web_pages: list[str]
    grounding_notes: list[str]
    search_priority: bool = False
    # Built-in provider search (``LLM_BUILTIN_WEB_SEARCH`` on the responses
    # endpoint): the turn rides the provider's own ``web_search`` tool instead
    # of an external search client, so the flags travel to ``generate_text``.
    force_web_search: bool = False
    allow_web_search: bool = False


class PrivateChatService:
    """Serve private chat for the owner and the ``PRIVATE_CHAT_QQS`` allow list."""

    def __init__(
        self,
        *,
        engine,
        sender,
        llm_client,
        image_llm_client=None,
        owner_qq: int,
        bot_qq: int | None = None,
        private_chat_qqs: set[int] | None = None,
        data_dir: Path,
        private_image_followup_window_seconds: float = 1.2,
        web_search_client=None,
        image_reference_search_client=None,
        image_reference_planner_client=None,
        reply_split_config: dict | None = None,
        image_model: str = "gpt-image-2",
        image_size: str | None = "auto",
        image_quality: str | None = "high",
        image_background: str | None = None,
        image_output_format: str | None = "png",
        image_output_compression: int | None = 100,
        image_moderation: str | None = "low",
        image_queue_capacity: int = 3,
        image_max_attempts: int = 1,
        image_timeout_seconds: float = 900.0,
        assistant_name: str = "小町",
        persona: dict | None = None,
        safety: dict | None = None,
    ) -> None:
        self.engine = engine
        self.sender = sender
        self.llm_client = llm_client
        self.image_llm_client = image_llm_client or llm_client
        self.owner_qq = owner_qq
        self.bot_qq = bot_qq
        self.private_chat_qqs = {qq for qq in (private_chat_qqs or set()) if qq != owner_qq}
        self.data_dir = data_dir.resolve()
        self.private_image_followup_window_seconds = max(
            0.0,
            float(private_image_followup_window_seconds),
        )
        self.web_search_client = web_search_client
        # A private drawing must search reference images with the same client and
        # planner the group image service uses; the chat search client stays for
        # the text-side grounding path.
        self.image_reference_search_client = (
            image_reference_search_client
            if image_reference_search_client is not None
            else web_search_client
        )
        self.image_reference_planner_client = image_reference_planner_client
        self.reply_split_config = dict(
            reply_split_config
            if reply_split_config is not None
            else build_reply_split_config()
        )
        self.private_image_service = PrivateImageGenerationService(
            engine=engine,
            llm_client=self.image_llm_client,
            sender=sender,
            web_search_client=self.image_reference_search_client,
            image_reference_planner_client=self.image_reference_planner_client,
            output_dir=self.data_dir / "generated_private_images",
            model=image_model,
            size=image_size,
            quality=image_quality,
            background=image_background,
            output_format=image_output_format,
            output_compression=image_output_compression,
            moderation=image_moderation,
            max_slots=image_queue_capacity,
            image_max_attempts=image_max_attempts,
            image_timeout_seconds=image_timeout_seconds,
            task_result_callback=self._finalize_private_image_task,
        )
        self.assistant_name = assistant_name.strip() or "小町"
        self.persona = dict(persona or {})
        self.safety = dict(safety or {})
        self._pending_private_image_turns: dict[
            tuple[int, str],
            tuple[asyncio.Task, PrivateMessageEvent],
        ] = {}
        self._private_image_turn_overrides: dict[int, list] = {}
        self._private_draw_context_reset_users: set[int] = set()

    async def start(self) -> None:
        await self.private_image_service.start()

    async def stop(self) -> None:
        self._cancel_pending_private_image_turns_for_user(None, preserve_for_next_turn=False)
        await self.private_image_service.stop()

    async def handle_private_message(self, event: PrivateMessageEvent) -> bool:
        """Answer one direct message; return ``False`` when the sender is unknown."""

        if getattr(event, "reply_to_msg_id", None) is not None or bool(
            list(getattr(event, "images", []) or [])
        ):
            self._private_draw_context_reset_users.discard(event.user_id)
        if self._should_hold_private_image_for_followup(event):
            self._cancel_pending_private_image_turns_for_user(
                event.user_id,
                preserve_for_next_turn=False,
            )
            self._private_image_turn_overrides[event.user_id] = list(event.images)
            return True
        if self._should_defer_private_image_turn(event):
            self._private_image_turn_overrides.pop(event.user_id, None)
        self._cancel_pending_private_image_turns_for_user(
            event.user_id,
            preserve_for_next_turn=not self._should_defer_private_image_turn(event),
        )

        is_owner = event.user_id == self.owner_qq
        if not is_owner and event.user_id not in self.private_chat_qqs:
            return False
        private_scope = (
            PRIVATE_SCOPE_OWNER_DAILY if is_owner else PRIVATE_SCOPE_ALLOWLIST_DAILY
        )

        if is_owner:
            draw_reset_reply = self._handle_private_draw_reset_command(
                raw_text=event.plain_text,
                user_id=event.user_id,
            )
            if draw_reset_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=draw_reset_reply,
                    context=f"private_draw_reset:{event.platform_msg_id}",
                )
                return True
            command_reply = self._handle_private_session_command(
                raw_text=event.plain_text,
                user_id=event.user_id,
            )
            if command_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=command_reply,
                    context=f"private_session_command:{event.platform_msg_id}",
                )
                return True

        if await self._schedule_private_image_followup_if_needed(
            event,
            private_scope=private_scope,
        ):
            return True
        await self._handle_private_chat_turn(event, private_scope=private_scope)
        return True

    async def _send_private_text(
        self,
        *,
        user_id: int,
        text: str,
        context: str,
    ) -> bool:
        delivered, _failure_reason = await self._deliver_private_text(
            user_id=user_id,
            text=text,
            context=context,
        )
        return delivered

    async def _deliver_private_text(
        self,
        *,
        user_id: int,
        text: str,
        context: str,
    ) -> tuple[bool, str | None]:
        if not self._reserve_private_outbound_reply(user_id=user_id, reply_text=text, context=context):
            logger.info("private_reply_dedup_skip context=%s user_id=%s", context, user_id)
            return True, None
        try:
            await self.sender.send_private_text(OutboundPrivateMessage(user_id=user_id, text=text))
            self._mark_private_outbound_reply_sent(user_id=user_id, reply_text=text, context=context)
            return True, None
        except QQMessageDeliveryUncertainError as exc:
            self._mark_private_outbound_reply_uncertain(
                user_id=user_id,
                reply_text=text,
                context=context,
            )
            logger.warning(
                "private_reply_delivery_uncertain context=%s user_id=%s error_type=%s",
                context,
                user_id,
                type(exc).__name__,
            )
            return False, "delivery_result_unknown"
        except Exception as exc:
            self._clear_private_outbound_reply_reservation(context=context)
            logger.exception("private_reply_send_failed context=%s user_id=%s", context, user_id)
            failure_reason = str(exc).strip() or exc.__class__.__name__
            return False, failure_reason

    async def _send_private_chat_reply(
        self,
        *,
        user_id: int,
        text: str,
        context: str,
    ) -> None:
        """Send one chat reply, split into a short burst when configured.

        The private surface uses the same delivery shape as the group chat: one
        answer may arrive as up to ``max_messages`` short QQ messages with the
        configured delay between them. Each segment reserves its own outbound
        key, so reply de-duplication still covers the whole burst.
        """

        burst = self.reply_split_config
        segments = split_burst_reply(text, burst)
        # Same window as the group delivery path, from the shared helper.
        delay_min, delay_max = burst_delays(burst, segment_count=len(segments))
        for index, segment in enumerate(segments):
            await self._deliver_private_text(
                user_id=user_id,
                text=segment,
                context=context if index == 0 else f"{context}-b{index}",
            )
            if index < len(segments) - 1 and delay_max > 0:
                await asyncio.sleep(random.uniform(delay_min, delay_max))

    async def _fetch_private_quoted_message_payload(self, *, reply_to_msg_id: str | None) -> dict | None:
        if not reply_to_msg_id:
            return None
        gateway = getattr(self.sender, "gateway", None)
        if gateway is None or not hasattr(gateway, "call_api"):
            return None
        message_id: int | str = (
            int(reply_to_msg_id) if str(reply_to_msg_id).isdigit() else str(reply_to_msg_id)
        )
        try:
            response = await gateway.call_api("get_msg", {"message_id": message_id})
        except Exception:
            logger.exception("private_quoted_message_fetch_failed reply_to_msg_id=%s", reply_to_msg_id)
            return None
        if not isinstance(response, dict):
            return None
        payload = response.get("data")
        if not isinstance(payload, dict):
            return None
        return payload

    def _private_image_turn_key(self, *, user_id: int, private_scope: str) -> tuple[int, str]:
        return (user_id, private_scope)

    def _should_defer_private_image_turn(self, event: PrivateMessageEvent) -> bool:
        if self.private_image_followup_window_seconds <= 0:
            return False
        if str(event.plain_text or "").strip():
            return False
        if getattr(event, "reply_to_msg_id", None) is not None:
            return False
        return bool(list(getattr(event, "images", []) or []))

    def _should_hold_private_image_for_followup(self, event: PrivateMessageEvent) -> bool:
        return (
            getattr(event, "reply_to_msg_id", None) is None
            and not str(event.plain_text or "").strip()
            and len(list(getattr(event, "images", []) or [])) == 1
        )

    def _cancel_pending_private_image_turns_for_user(
        self,
        user_id: int | None,
        *,
        preserve_for_next_turn: bool,
    ) -> None:
        for key, pending in list(self._pending_private_image_turns.items()):
            if user_id is not None and key[0] != user_id:
                continue
            task, pending_event = pending
            if preserve_for_next_turn:
                pending_images = list(getattr(pending_event, "images", []) or [])
                if pending_images:
                    self._private_image_turn_overrides[key[0]] = pending_images
            task.cancel()
            self._pending_private_image_turns.pop(key, None)

    def _consume_private_image_turn_override(self, *, user_id: int) -> list | None:
        override_images = self._private_image_turn_overrides.pop(user_id, None)
        if not override_images:
            return None
        return list(override_images)

    def _reset_private_draw_state(self, *, user_id: int) -> None:
        self._cancel_pending_private_image_turns_for_user(user_id, preserve_for_next_turn=False)
        self._private_image_turn_overrides.pop(user_id, None)
        self._private_draw_context_reset_users.add(user_id)

    async def _schedule_private_image_followup_if_needed(
        self,
        event: PrivateMessageEvent,
        *,
        private_scope: str,
    ) -> bool:
        if not self._should_defer_private_image_turn(event):
            return False

        key = self._private_image_turn_key(user_id=event.user_id, private_scope=private_scope)

        async def _run_deferred_turn() -> None:
            try:
                await asyncio.sleep(self.private_image_followup_window_seconds)
                await self._handle_private_chat_turn(event, private_scope=private_scope)
            except asyncio.CancelledError:
                return
            finally:
                self._pending_private_image_turns.pop(key, None)

        self._pending_private_image_turns[key] = (asyncio.create_task(_run_deferred_turn()), event)
        return True

    def _private_turn_text_for_prompt(self, event: PrivateMessageEvent, *, resolved_image_count: int = 0) -> str:
        plain_text = str(event.plain_text or "").strip()
        if plain_text:
            return plain_text

        current_images = list(getattr(event, "images", []) or [])
        image_count = len(current_images) or resolved_image_count
        if image_count <= 0:
            return plain_text
        if image_count == 1:
            return "[sent 1 image]" if current_images else "[asked about 1 image]"
        return f"[sent {image_count} images]" if current_images else f"[asked about {image_count} images]"

    def _normalize_private_lookup_text(self, value: str) -> str:
        return LOOKUP_NORMALIZER.sub("", value).lower()

    def _extract_auto_web_reference_query(self, *, stripped_text: str) -> str | None:
        match = AUTO_WEB_REFERENCE_QUERY_PATTERN.search(stripped_text)
        if match is not None:
            query = str(match.group("query") or "").strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
            if query:
                return query
        marker_positions = [
            stripped_text.find(marker)
            for marker in ("\u7684\u4eba\u8bbe\u56fe", "\u4eba\u8bbe\u56fe", "\u8bbe\u5b9a\u56fe", "\u53c2\u8003\u56fe")
            if stripped_text.find(marker) >= 0
        ]
        if not marker_positions:
            return None
        end = min(marker_positions)
        action_positions = [
            idx
            for idx in (stripped_text.find("\u627e"), stripped_text.find("\u641c"))
            if 0 <= idx < end
        ]
        if not action_positions:
            return None
        start = min(action_positions) + 1
        query = stripped_text[start:end].strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
        while query.startswith(("\u7f51\u4e0a", "\u4e0a\u7f51", "\u8054\u7f51")):
            query = query[2:].strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
        return query or None

    def _build_auto_web_reference_prompt(self, *, stripped_text: str, query: str) -> str:
        prompt = AUTO_WEB_REFERENCE_QUERY_PATTERN.sub("", stripped_text, count=1)
        prompt = AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN.sub("", prompt).strip(
            " \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A"
        )
        if not prompt:
            return f"\u53c2\u8003\u641c\u7d22\u5230\u7684{query}\u4eba\u8bbe\u56fe\u751f\u6210\u4e00\u5f20\u56fe"
        return f"\u53c2\u8003\u641c\u7d22\u5230\u7684{query}\u4eba\u8bbe\u56fe\uff0c{prompt}"

    def _looks_like_private_reference_image_generation_request(
        self,
        *,
        stripped_text: str,
        target_images: list | None,
    ) -> bool:
        if not target_images:
            return False
        normalized_text = self._normalize_private_lookup_text(stripped_text)
        if not normalized_text:
            return False
        has_transform_intent = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS
        )
        has_reference_context = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS
        )
        has_generation_intent = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS
        )
        has_retouch_intent = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in PRIVATE_IMAGE_RETOUCH_INTENT_KEYWORDS
        )
        if has_transform_intent or (has_reference_context and has_generation_intent):
            return True
        if has_retouch_intent:
            return True
        direct_text = str(stripped_text or "")
        return (
            (
                "\u6784\u56fe" in direct_text
                and ("\u66ff\u6362\u4eba\u7269" in direct_text or "\u6362\u6210\u4eba\u7269" in direct_text)
                and "\u51fa\u56fe" in direct_text
            )
            or ("\u53c2\u8003" in direct_text and "\u51fa\u56fe" in direct_text)
        )

    def _build_private_image_request(
        self,
        *,
        event: PrivateMessageEvent,
        target_images: list | None,
    ) -> PrivateImageGenerationRequest | None:
        stripped = str(event.plain_text or "").strip()
        if not stripped:
            return None
        if any(pattern.search(stripped) for pattern in GROUP_IMAGE_NEGATIVE_PATTERNS):
            return None
        reference_images = list(target_images or [])
        auto_web_reference_query = self._extract_auto_web_reference_query(stripped_text=stripped)
        if auto_web_reference_query is not None:
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=self._build_auto_web_reference_prompt(
                    stripped_text=stripped,
                    query=auto_web_reference_query,
                ),
                reference_images=reference_images,
                web_search_query=auto_web_reference_query,
            )
        if self._looks_like_private_reference_image_generation_request(
            stripped_text=stripped,
            target_images=reference_images,
        ):
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=stripped,
                reference_images=reference_images,
            )
        for pattern in GROUP_IMAGE_REQUEST_PATTERNS:
            match = pattern.match(stripped)
            if match is None:
                continue
            prompt = match.group("prompt").strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
            if not prompt:
                return None
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=prompt,
                reference_images=reference_images,
            )
        if reference_images and any(
            keyword in stripped
            for keyword in ("\u51fa\u56fe", "\u753b\u56fe", "\u7ed8\u56fe", "\u751f\u6210")
        ):
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=stripped,
                reference_images=reference_images,
            )
        return None

    def _finalize_private_image_task(self, task_id: int, result: ImageJobResult) -> None:
        with session_scope(self.engine) as session:
            tasks = DevTaskRepository(session)
            task = tasks.get_task(task_id)
            if task is None or task.status in {"completed", "failed", "rolled_back"}:
                return
            if result.success:
                tasks.mark_completed(
                    task_id=task_id,
                    summary=self._build_turn_summary(task.raw_request_text, result.notice_text),
                    result_text=result.notice_text,
                    files_read=[],
                    files_changed=[],
                    commands_run=["private_image_service.completed"],
                    restart_required=False,
                    restart_result="not-needed",
                    checkpoint_dir="",
                )
                if result.image_path is not None:
                    DevTaskArtifactRepository(session).add_artifact(
                        task_id=task_id,
                        artifact_type="private_image",
                        artifact_path=str(result.image_path),
                        metadata_json={"notice_text": result.notice_text},
                    )
                self._append_session_summary(
                    session_id=task.session_id,
                    owner_text=task.raw_request_text,
                    assistant_text=result.notice_text,
                    sessions=DevSessionRepository(session),
                )
                return
            tasks.mark_failed(task_id=task_id, failure_reason=result.failure_reason or result.notice_text)
            self._append_session_summary(
                session_id=task.session_id,
                owner_text=task.raw_request_text,
                assistant_text=f"失败：{result.notice_text}",
                sessions=DevSessionRepository(session),
            )

    async def _handle_private_chat_turn(self, event: PrivateMessageEvent, *, private_scope: str) -> None:
        initial_request_text = self._private_turn_text_for_prompt(event)
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_or_create_owner_session(
                owner_qq=event.user_id,
                session_mode=SESSION_MODE_DAILY,
            )
            task = tasks.add_task(
                session_id=dev_session.id,
                requested_by_qq=event.user_id,
                raw_request_text=initial_request_text,
                intent_type="private_chat",
            )
            sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
            session_id = dev_session.id
            task_id = task.id

        try:
            quoted_raw_payload: dict | None = None
            override_images = self._consume_private_image_turn_override(user_id=event.user_id)
            if override_images:
                target_images = override_images
            else:
                quoted_raw_payload = await self._fetch_private_quoted_message_payload(
                    reply_to_msg_id=getattr(event, "reply_to_msg_id", None)
                )
                with session_scope(self.engine) as session:
                    target_images_turn = resolve_private_images_for_turn(
                        event=event,
                        messages=MessageRepository(session),
                        quoted_raw_payload=quoted_raw_payload,
                    )
                if (
                    target_images_turn is not None
                    and target_images_turn.source_kind == "recent"
                    and event.user_id in self._private_draw_context_reset_users
                ):
                    target_images = None
                else:
                    target_images = (
                        target_images_turn.images
                        if target_images_turn and target_images_turn.images
                        else None
                    )
            request_text = self._private_turn_text_for_prompt(
                event,
                resolved_image_count=len(target_images or []),
            )
            conversation_key = f"dev-session:{session_id}"
            reply_text = None
            private_image_request = self._build_private_image_request(
                event=event,
                target_images=target_images,
            )
            if private_image_request is not None:
                private_image_request.dev_task_id = task_id
                enqueue_result = await self.private_image_service.enqueue(private_image_request)
                if enqueue_result.accepted:
                    reply_text = "图我接住了，开始画"
                    with session_scope(self.engine) as session:
                        DevTaskRepository(session).mark_status(task_id=task_id, status="running")
                else:
                    reply_text = "现在排队的图太多了，你等一下再发"
            if reply_text is None:
                prompt_lines, web_context = self._build_private_chat_prompt(
                    session_id=session_id,
                    task_id=task_id,
                    request_text=request_text,
                    request_time=event.timestamp,
                    private_scope=private_scope,
                    image_count=len(target_images or []),
                    quoted_raw_payload=quoted_raw_payload,
                )
                reply_text = self._normalize_private_reply(
                    self.llm_client.generate_text(
                        prompt_lines,
                        images=target_images,
                        conversation_key=conversation_key,
                        **self._private_generation_search_kwargs(web_context),
                    )
                )
                if not reply_text:
                    raise ValueError("empty private chat reply")
                reply_text = filter_reply_urls(
                    reply_text,
                    allow_urls=explicitly_requests_urls(request_text),
                )

            if private_image_request is None:
                with session_scope(self.engine) as session:
                    tasks = DevTaskRepository(session)
                    tasks.mark_completed(
                        task_id=task_id,
                        summary=self._build_turn_summary(request_text, reply_text),
                        result_text=reply_text,
                        files_read=[],
                        files_changed=[],
                        commands_run=["llm_client.generate_text"],
                        restart_required=False,
                        restart_result="not-needed",
                        checkpoint_dir="",
                    )
                    self._append_session_summary(
                        session_id=session_id,
                        owner_text=request_text,
                        assistant_text=reply_text,
                        sessions=DevSessionRepository(session),
                    )

            if private_image_request is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=reply_text,
                    context=f"private_chat:{task_id}:accepted",
                )
            else:
                await self._send_private_chat_reply(
                    user_id=event.user_id,
                    text=reply_text,
                    context=f"private_chat:{task_id}:completed",
                )
        except Exception as exc:
            logger.exception("private_chat_failed user_id=%s scope=%s", event.user_id, private_scope)
            failure_reason = str(exc)
            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_failed(task_id=task_id, failure_reason=failure_reason)
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=initial_request_text,
                    assistant_text=f"失败：{failure_reason}",
                    sessions=DevSessionRepository(session),
                )
            await self._send_private_text(
                user_id=event.user_id,
                text="我这边刚刚卡了一下，你再问我一句。",
                context=f"private_chat:{task_id}:failed",
            )

    def _private_daily_persona_text(self) -> str:
        if self.persona:
            return render_persona(self.persona)
        return (
            f"You are {self.assistant_name}. "
            "Speaking tone: natural. Keep replies concise unless asked to expand."
        )

    def _private_daily_safety_line(self, *, extra_rule: str | None = None) -> str:
        rules = render_safety_lines(self.safety)
        rules.append(
            "Stay grounded in the provided session history, runtime facts, and web evidence. "
            "Do not invent runtime state."
        )
        if extra_rule:
            rules.append(extra_rule)
        return "Safety rules: " + " ".join(rule for rule in rules if rule)

    def _private_reply_style_lines(self, *, private_scope: str) -> list[str]:
        del private_scope
        return [
            f"Reply style: Stay in {str(self.persona.get('name', self.assistant_name) or self.assistant_name)}'s daily-chat persona.",
            # Same human/Komachi work-style the group chat runs on; only the
            # opening line names a direct chat instead of a group.
            *build_human_chat_style_lines(
                proactive_turn=False,
                komachi_style=True,
                chat_context="private",
                voice=str(self.persona.get("chat_voice") or "komachi"),
            ),
            "Reply style: If evidence is missing or conflicting, say so plainly instead of smoothing it over.",
        ]

    def _private_image_reasoning_lines(
        self,
        *,
        request_text: str,
        recent_turns: list[str],
        image_count: int,
    ) -> list[str]:
        del request_text
        del recent_turns
        if image_count <= 0:
            return []
        return [
            f"Vision task: {image_count} attached image(s) belong to the current turn. Inspect them directly before replying.",
            "Vision task: Base claims about identity, source, or scene details on visible evidence in the image, not on chat memory alone.",
        ]

    def _private_web_context_lines(self, web_context: PrivateWebContext) -> list[str]:
        lines: list[str] = []
        if web_context.search_priority:
            lines.extend(build_search_priority_instructions())
        if web_context.runtime_facts:
            lines.append(
                "Treat runtime facts as authoritative for the current year, date, weekday, and clock time."
            )
            lines.append("Runtime facts:")
            lines.extend(web_context.runtime_facts)
        if web_context.grounding_notes:
            lines.append("Grounding notes:")
            lines.extend(web_context.grounding_notes)
        if web_context.web_results:
            lines.append("Web search results:")
            lines.extend(web_context.web_results)
        if web_context.web_pages:
            lines.append("Web page extracts:")
            lines.extend(web_context.web_pages)
        return lines

    def _private_search_names(self) -> set[str]:
        candidates = {
            self.assistant_name,
            self.assistant_name.lower(),
            "小町",
            "komachi",
            "xiaomachi",
        }
        return {candidate for candidate in candidates if candidate}

    def _normalize_private_timestamp(self, value):
        if getattr(value, "tzinfo", None) is None:
            return value.replace(tzinfo=ASIA_SHANGHAI)
        return value

    def _recent_private_owner_turns(self, recent_turns: list[str]) -> list[str]:
        return [
            line.split("Owner: ", maxsplit=1)[1].strip()
            for line in recent_turns
            if line.startswith("Owner: ")
        ]

    def _find_recent_private_weather_turn(self, recent_owner_turns: list[str]) -> str | None:
        for turn in reversed(recent_owner_turns):
            normalized = turn.strip()
            if "天气" in normalized:
                return normalized
        return None

    def _normalize_private_weather_followup_location(self, text: str) -> str:
        normalized = re.sub(r"[\n\r\t,，。！？!?:;；、“”‘’（）()《》【】\[\]<>]+", " ", text.strip())
        normalized = re.sub(
            r"^(那就|那|就|改成|换成|那换|那改成|那就按|那就查|那就搜|改查|改搜|查一下|查下|搜一下|搜下|查|搜)\s*",
            "",
            normalized,
        )
        normalized = re.sub(r"(吧|呢|呀|啊|可以吗|行吗)$", "", normalized).strip()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _looks_like_private_location_fragment(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized or len(normalized) < 2 or len(normalized) > 24:
            return False
        if any(
            token in normalized
            for token in ("天气", "联网", "上网", "搜索", "搜", "查", "评价", "口碑", "新闻")
        ):
            return False
        return any(
            token in normalized
            for token in (
                "省",
                "市",
                "区",
                "县",
                "镇",
                "乡",
                "村",
                "校区",
                "大学",
                "学校",
                "路",
                "街",
                "巷",
                "站",
                "机场",
                "广场",
                "商场",
                "医院",
                "酒店",
                "景区",
                "公园",
            )
        )

    def _extract_private_weather_time_hint(self, *texts: str) -> str:
        for text in texts:
            normalized = text.strip()
            for hint in (
                "今天",
                "明天",
                "后天",
                "今晚",
                "今早",
                "今晨",
                "今夜",
                "现在",
                "当前",
                "本周",
                "这周",
                "周末",
            ):
                if hint in normalized:
                    return hint
        return "今天"

    def _build_contextual_private_search(
        self,
        *,
        request_text: str,
        recent_turns: list[str],
        search_reference_time,
    ) -> SearchDecision | None:
        normalized_request = request_text.strip()
        if not normalized_request or "天气" in normalized_request:
            return None
        if (
            is_explicit_search_request(normalized_request)
            or needs_reference_search(normalized_request)
            or needs_external_lookup_search(normalized_request)
            or is_general_search_decision_candidate(normalized_request)
        ):
            return None

        recent_owner_turns = self._recent_private_owner_turns(recent_turns)
        weather_anchor = self._find_recent_private_weather_turn(recent_owner_turns)
        if weather_anchor is None:
            return None

        location_text = self._normalize_private_weather_followup_location(normalized_request)
        if not self._looks_like_private_location_fragment(location_text):
            return None

        time_hint = self._extract_private_weather_time_hint(location_text, weather_anchor)
        query = normalize_relative_time_query(f"{location_text} {time_hint}天气", now=search_reference_time)
        return SearchDecision(True, query, "weather-followup-context")

    def _build_private_web_context(
        self,
        *,
        request_text: str,
        request_time,
        recent_turns: list[str],
    ) -> PrivateWebContext:
        runtime_facts: list[str] = []
        web_results: list[str] = []
        web_pages: list[str] = []
        grounding_notes: list[str] = []

        search_reference_time = self._normalize_private_timestamp(request_time).astimezone()
        # Every private turn carries the current clock, exactly like the group
        # router: without it the model invents stale years for its own search
        # queries and cannot ground "today"/"this year" style answers.
        runtime_facts = build_current_datetime_facts(search_reference_time)
        current_datetime_context_required = needs_current_datetime_context(request_text)

        # The search predicates drive both transports: the external client
        # below, and the provider's own ``web_search`` tool when this
        # deployment has no external client.  The built-in eligibility rules
        # mirror the group router.
        explicit_search_request = is_explicit_search_request(request_text)
        reference_search_request = needs_reference_search(request_text)
        external_lookup_search_request = needs_external_lookup_search(request_text)
        general_search_candidate = is_general_search_decision_candidate(request_text)
        time_sensitive = is_time_sensitive_request(request_text)
        contextual_followup_search = self._build_contextual_private_search(
            request_text=request_text,
            recent_turns=recent_turns,
            search_reference_time=search_reference_time,
        )

        forced_search_request = (
            explicit_search_request
            or reference_search_request
            or external_lookup_search_request
            or contextual_followup_search is not None
        )
        # Ask the provider client instead of duplicating settings: only the
        # client that serves this turn can say whether the built-in tool is
        # configured, and a dedicated search model scopes it to turns that
        # actually need fresh information (same rule as the group router).
        builtin_search_configured = bool(
            getattr(self.llm_client, "builtin_web_search", False)
        )
        scoped_builtin_search = bool(
            str(getattr(self.llm_client, "web_search_model", "") or "").strip()
        )
        builtin_web_search_eligible = (
            self.web_search_client is None
            and builtin_search_configured
            and not is_search_verification_query(request_text)
            and (not scoped_builtin_search or time_sensitive or forced_search_request)
        )
        force_builtin_web_search = (
            forced_search_request
            and self.web_search_client is None
            and builtin_search_configured
        )
        builtin_web_search_active = (
            builtin_web_search_eligible or force_builtin_web_search
        )

        if (
            self.web_search_client is None
            or current_datetime_context_required
            or is_search_verification_query(request_text)
        ):
            if builtin_web_search_active:
                logger.info(
                    "private_web_search_builtin owner_qq=%s force=%s eligible=%s scoped=%s",
                    self.owner_qq,
                    force_builtin_web_search,
                    builtin_web_search_eligible,
                    scoped_builtin_search,
                )
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
                search_priority=builtin_web_search_active,
                force_web_search=force_builtin_web_search,
                allow_web_search=builtin_web_search_eligible,
            )

        optional_search_eligible = (time_sensitive or general_search_candidate) and not forced_search_request
        if not forced_search_request and not optional_search_eligible:
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
            )

        if contextual_followup_search is not None:
            parsed_search = contextual_followup_search
        elif forced_search_request:
            parsed_search = SearchDecision(
                True,
                normalize_relative_time_query(
                    build_forced_search_query(request_text, bot_names=self._private_search_names()),
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
                bot_name=self.assistant_name,
                target_message=f"Owner: {request_text}",
                recent_messages=recent_turns,
                proactive_turn=False,
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
                logger.exception("private_web_search_decision_failed owner_qq=%s", self.owner_qq)
                parsed_search = SearchDecision(False, "", "search-decision-error")

        if not parsed_search.should_search:
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
            )

        search_result_limit = 5 if reference_search_request or external_lookup_search_request else 3
        try:
            search_hits = self.web_search_client.search(parsed_search.query, max_results=search_result_limit)
        except Exception:
            logger.exception(
                "private_web_search_execute_failed owner_qq=%s query=%s",
                self.owner_qq,
                parsed_search.query,
            )
            search_hits = []
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
                "private_web_page_fetch_failed owner_qq=%s query=%s",
                self.owner_qq,
                parsed_search.query,
            )
            page_reads = []
        web_pages = [
            f"{page.title} | {page.url} | {page.content}"
            for page in page_reads
        ]
        recent_assistant_replies = [
            line.split("Assistant: ", maxsplit=1)[1]
            for line in recent_turns
            if line.startswith("Assistant: ")
        ]
        grounding_notes = build_grounding_notes(
            target_text=request_text,
            external_lookup=external_lookup_search_request,
            web_results=search_hits,
            web_pages=page_reads,
            recent_bot_replies=recent_assistant_replies,
        )
        return PrivateWebContext(
            runtime_facts=runtime_facts,
            web_results=web_results,
            web_pages=web_pages,
            grounding_notes=grounding_notes,
            # A turn that goes out with fresh search evidence carries the same
            # search-priority instructions the group chat adds.
            search_priority=True,
        )

    def _build_private_chat_prompt(
        self,
        *,
        session_id: int,
        task_id: int,
        request_text: str,
        request_time,
        private_scope: str,
        image_count: int = 0,
        quoted_raw_payload: dict | None = None,
    ) -> tuple[list[str], PrivateWebContext]:
        session_summary = self._session_summary(session_id=session_id)
        recent_turns = self._recent_turn_lines(session_id=session_id, exclude_task_id=task_id)
        history_block = "\n".join(recent_turns) if recent_turns else "(none)"
        web_context = self._build_private_web_context(
            request_text=request_text,
            request_time=request_time,
            recent_turns=recent_turns,
        )

        daily_safety_extra = (
            "This user is not the owner: do not offer or imply code changes, admin actions, runtime restarts, or project control."
            if private_scope == PRIVATE_SCOPE_ALLOWLIST_DAILY
            else "This private chat is chat-only: you cannot change the repository, run local commands, or restart the runtime. If the owner asks for that, say so plainly instead of implying you did it."
        )
        current_message_label = (
            "Current user message:" if private_scope == PRIVATE_SCOPE_ALLOWLIST_DAILY else "Current owner message:"
        )
        target_text = request_text
        quoted_message_line = quoted_message_line_for_prompt(
            quoted_raw_payload=quoted_raw_payload
        )
        if quoted_message_line is not None:
            target_text = f"{target_text}\nQuoted message: {quoted_message_line}"
            pronoun_referent_note = quoted_pronoun_referent_note(
                query_text=request_text,
                quoted_raw_payload=quoted_raw_payload,
            )
            if pronoun_referent_note is not None:
                target_text = f"{target_text}\n{pronoun_referent_note}"
        return [
            f"System persona: {self._private_daily_persona_text()}",
            self._private_daily_safety_line(extra_rule=daily_safety_extra),
            *self._private_reply_style_lines(private_scope=private_scope),
            url_reply_policy_instruction(request_text),
            "Current private daily session summary:",
            session_summary or "(none)",
            "Recent private daily turns:",
            history_block,
            *self._private_web_context_lines(web_context),
            *self._private_image_reasoning_lines(
                request_text=request_text,
                recent_turns=recent_turns,
                image_count=image_count,
            ),
            f"{current_message_label} {target_text}",
        ], web_context

    def _private_generation_search_kwargs(self, web_context: PrivateWebContext) -> dict:
        """Provider kwargs that let this turn ride the built-in web search.

        Mirrors the group router's capability guards: the selective flag only
        travels to clients that declare it, and only a forced search asks the
        provider to attach the tool unconditionally.  Callers without built-in
        search receive a byte-identical request to before.
        """
        kwargs: dict = {}
        if bool(getattr(self.llm_client, "supports_selective_web_search", False)):
            kwargs["allow_web_search"] = web_context.allow_web_search
        if web_context.force_web_search and bool(
            getattr(self.llm_client, "supports_forced_web_search", False)
        ):
            kwargs["force_web_search"] = True
        return kwargs

    def _normalize_private_reply(self, text: str) -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""
        return normalize_chat_reply(raw).strip()

    def _normalize_private_command_text(self, text: str) -> str:
        return "".join(text.strip().lower().split())

    def _handle_private_session_command(self, *, raw_text: str, user_id: int | None = None) -> str | None:
        target_user_id = self.owner_qq if user_id is None else user_id
        normalized = self._normalize_private_command_text(raw_text)
        normalized_new_commands = {
            self._normalize_private_command_text(item) for item in SESSION_NEW_COMMANDS
        }
        normalized_status_commands = {
            self._normalize_private_command_text(item) for item in SESSION_STATUS_COMMANDS
        }
        if normalized not in normalized_new_commands and normalized not in normalized_status_commands:
            return None

        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            active_session = sessions.get_or_create_owner_session(
                owner_qq=target_user_id,
                session_mode=SESSION_MODE_DAILY,
            )
            open_tasks = tasks.list_tasks_for_session_by_status(
                session_id=active_session.id,
                statuses=["queued", "running"],
            )

            if normalized in normalized_new_commands:
                if open_tasks:
                    return "现在还有正在处理的日常对话任务，等它跑完我再给你开新的日常对话。"
                sessions.create_owner_session(owner_qq=target_user_id, session_mode=SESSION_MODE_DAILY)
                return "好，这里给你重新开了一条新的日常会话。"

            queued_count = len([task for task in open_tasks if task.status == "queued"])
            running_count = len([task for task in open_tasks if task.status == "running"])
            return (
                f"当前日常对话 #{active_session.id}，排队 {queued_count}，进行中 {running_count}。"
                "要重开就发“清空上下文”或者“开新对话”。"
            )

    def _handle_private_draw_reset_command(
        self,
        *,
        raw_text: str,
        user_id: int | None = None,
    ) -> str | None:
        target_user_id = self.owner_qq if user_id is None else user_id
        normalized = self._normalize_private_command_text(raw_text)
        normalized_commands = {
            self._normalize_private_command_text(item) for item in PRIVATE_DRAW_RESET_COMMANDS
        }
        if normalized not in normalized_commands:
            return None
        self._reset_private_draw_state(user_id=target_user_id)
        return "好，绘画上下文已经清空。要重新参考哪张图，直接再发图或重新说。"

    def _session_summary(self, *, session_id: int) -> str:
        with session_scope(self.engine) as session:
            dev_session = session.get(DevSession, session_id)
            if dev_session is None:
                return ""
            return str(getattr(dev_session, "summary", "") or "")

    def _private_outbound_platform_msg_id(self, *, context: str) -> str:
        return f"private-outbound-{context}"

    def _private_sender_user_id(self) -> int:
        return self.bot_qq if isinstance(self.bot_qq, int) and self.bot_qq > 0 else self.owner_qq

    def _reserve_private_outbound_reply(self, *, user_id: int, reply_text: str, context: str) -> bool:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        sender_user_id = self._private_sender_user_id()
        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)
            existing = messages.get_by_platform_msg_id(platform_msg_id)
            if existing is not None:
                return False

            users.upsert_user(
                user_id=sender_user_id,
                nickname=self.assistant_name,
                group_card="",
            )
            messages.add_private_message(
                platform_msg_id=platform_msg_id,
                user_id=sender_user_id,
                timestamp=datetime.now(ASIA_SHANGHAI),
                plain_text=reply_text,
                raw_json={
                    "direction": "outbound",
                    "recipient_user_id": user_id,
                    "delivery_state": "reserved",
                    "context": context,
                },
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
            return True

    def _mark_private_outbound_reply_sent(self, *, user_id: int, reply_text: str, context: str) -> None:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(platform_msg_id)
            if outbound_message is None:
                return
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "recipient_user_id": user_id,
                "delivery_state": "sent",
                "context": context,
            }
            session.add(outbound_message)

    def _mark_private_outbound_reply_uncertain(self, *, user_id: int, reply_text: str, context: str) -> None:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(platform_msg_id)
            if outbound_message is None:
                raise RuntimeError("uncertain private outbound reply reservation is missing")
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "recipient_user_id": user_id,
                "delivery_state": "uncertain",
                "failure_kind": "delivery_result_unknown",
                "delivery_reason": "gateway_ack_timeout",
                "delivery_attempts": 1,
                "context": context,
            }
            session.add(outbound_message)

    def _clear_private_outbound_reply_reservation(self, *, context: str) -> None:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(platform_msg_id)
            if outbound_message is None:
                return
            session.delete(outbound_message)

    def _recent_turn_lines(self, *, session_id: int, exclude_task_id: int | None = None) -> list[str]:
        with session_scope(self.engine) as session:
            recent_tasks = DevTaskRepository(session).list_recent_tasks_for_session(
                session_id=session_id,
                limit=RECENT_TURN_LIMIT + 2,
            )
        lines: list[str] = []
        for task in recent_tasks:
            if exclude_task_id is not None and task.id == exclude_task_id:
                continue
            lines.append(f"Owner: {task.raw_request_text}")
            if task.result_text:
                lines.append(f"Assistant: {task.result_text}")
            elif task.failure_reason:
                lines.append(f"Assistant: {task.failure_reason}")
        return lines[-SUMMARY_LINE_LIMIT:]

    def _append_session_summary(
        self,
        *,
        session_id: int,
        owner_text: str,
        assistant_text: str,
        sessions: DevSessionRepository,
    ) -> None:
        current_session = sessions.session.get(DevSession, session_id)
        current_summary = ""
        if current_session is not None:
            current_summary = str(getattr(current_session, "summary", "") or "")
        lines = current_summary.splitlines() if current_summary else []
        lines.extend(
            [
                f"Owner: {self._truncate_text(owner_text, limit=120)}",
                f"Assistant: {self._truncate_text(assistant_text, limit=180)}",
            ]
        )
        sessions.update_session(session_id=session_id, summary="\n".join(lines[-SUMMARY_LINE_LIMIT:]))

    def _build_turn_summary(self, owner_text: str, assistant_text: str) -> str:
        return (
            f"Owner asked: {self._truncate_text(owner_text, limit=80)} | "
            f"Assistant replied: {self._truncate_text(assistant_text, limit=120)}"
        )

    def _truncate_text(self, text: str, *, limit: int) -> str:
        value = " ".join(text.strip().split())
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."
