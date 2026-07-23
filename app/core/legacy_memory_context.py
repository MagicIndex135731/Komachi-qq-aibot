from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re

from app.config import AppSettings
from app.core.context_builder import ContextBuilder
from app.core.memory_engine import (
    history_recall_limits,
    history_search_terms,
    is_history_detail_query,
    retrieve_relevant_history,
    retrieve_relevant_memories,
)
from app.core.memory_orchestrator import MemoryContextResult
from app.core.memory_v2_context import MemoryV2Request
from app.storage.db import session_scope
from app.storage.repositories import (
    MemoryRepository,
    MessageRepository,
    SummaryRepository,
    UserRepository,
)


_LOOKUP_NORMALIZER = re.compile(
    r"[\s\u3000`~!@#$%^&*()_+\-=\[\]{}\\|;:'\",<.>/?，。！？：；、“”‘’（）《》【】]"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupMemoryContextRequest(MemoryV2Request):
    current_user_id: int
    use_full_history: bool = False

    @property
    def query_text(self) -> str:
        return self.query

    @property
    def current_msg_id(self) -> str:
        return self.target_message_id or ""

    @property
    def current_timestamp(self) -> datetime:
        if self.now is None:
            raise ValueError("memory request timestamp is required")
        return self.now


# The V1 provider and V2 provider intentionally receive the same superset
# request. Keep the old name while external wiring migrates to the neutral one.
LegacyMemoryRequest = GroupMemoryContextRequest


@dataclass(frozen=True, slots=True)
class LegacyMemoryPromptContext:
    recent_messages: list[str]
    full_history_messages: list[str]
    full_history_preamble: list[str]
    full_history_enabled: bool
    member_focus_lines: list[str]
    summaries: list[str]
    relevant_history_messages: list[str]
    memories: list[str]
    history_detail: bool
    blocked_output_present: bool = False
    fts_memory_candidate_count: int = 0
    vector_memory_candidate_count: int = 0


def normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def normalize_lookup_text(value: str) -> str:
    return _LOOKUP_NORMALIZER.sub("", value).casefold()


def format_member_label(*, nickname: str, group_card: str, fallback: str) -> str:
    clean_nickname = nickname.strip()
    clean_group_card = group_card.strip()
    if clean_group_card and clean_nickname:
        return f"{clean_group_card}（QQ昵称：{clean_nickname}）"
    if clean_group_card:
        return clean_group_card
    if clean_nickname:
        return clean_nickname
    return fallback


def member_label_for_user(
    *,
    user_id: int,
    users_by_id: dict[int, object],
    bot_user_id: int,
    bot_display_name: str,
) -> str:
    if user_id == bot_user_id:
        return bot_display_name or "Bot"
    user = users_by_id.get(user_id)
    if user is None:
        return str(user_id)
    return format_member_label(
        nickname=str(getattr(user, "nickname", "")),
        group_card=str(getattr(user, "group_card", "")),
        fallback=str(user_id),
    )


def format_message_line(
    *,
    user_id: int,
    plain_text: str,
    users_by_id: dict[int, object],
    bot_user_id: int,
    bot_display_name: str,
) -> str:
    label = member_label_for_user(
        user_id=user_id,
        users_by_id=users_by_id,
        bot_user_id=bot_user_id,
        bot_display_name=bot_display_name,
    )
    return f"{label}: {plain_text}"


class LegacyMemoryContext:
    """Build the pre-V2 group-memory prompt package without router coupling."""

    def __init__(
        self,
        *,
        engine: object,
        settings: AppSettings,
        bot_user_id: int,
        bot_display_name: str,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.bot_user_id = int(bot_user_id)
        self.bot_display_name = bot_display_name.strip() or "Bot"

    def build_context(self, request: GroupMemoryContextRequest) -> MemoryContextResult:
        return self._build(request, recent_only=False)

    def build_recent_context(self, request: GroupMemoryContextRequest) -> MemoryContextResult:
        return self._build(request, recent_only=True)

    def _build(self, request: GroupMemoryContextRequest, *, recent_only: bool) -> MemoryContextResult:
        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)
            summaries = SummaryRepository(session)
            memories = MemoryRepository(session)

            recent_messages = messages.list_recent_group_messages(
                group_id=request.group_id,
                limit=self.settings.context_recent_limit,
            )
            full_history_messages = (
                messages.list_group_messages_chronological(
                    group_id=request.group_id,
                    exclude_platform_msg_id=request.current_msg_id,
                )
                if request.use_full_history and not recent_only
                else []
            )
            users_by_id = users.get_users_by_ids(
                [message.user_id for message in recent_messages]
                + [message.user_id for message in full_history_messages]
                + [request.current_user_id, self.bot_user_id]
            )
            recent_lines = [
                format_message_line(
                    user_id=message.user_id,
                    plain_text=message.plain_text,
                    users_by_id=users_by_id,
                    bot_user_id=self.bot_user_id,
                    bot_display_name=self.bot_display_name,
                )
                for message in recent_messages
            ]
            full_history_preamble, full_history_lines = self._format_full_history_lines(
                full_history_messages
            )
            blocked_output_present = any(
                messages.is_qq_blocked_outbound(message)
                for message in [*recent_messages, *full_history_messages]
            )
            selected_source_msg_ids = [
                message.platform_msg_id for message in [*recent_messages, *full_history_messages]
            ]

            if recent_only:
                context = LegacyMemoryPromptContext(
                    recent_messages=recent_lines,
                    full_history_messages=[],
                    full_history_preamble=[],
                    full_history_enabled=False,
                    member_focus_lines=[],
                    summaries=[],
                    relevant_history_messages=[],
                    memories=[],
                    history_detail=False,
                    blocked_output_present=blocked_output_present,
                )
                return self._result(
                    request=request,
                    context=context,
                    selected_source_msg_ids=selected_source_msg_ids,
                    mode="recent",
                )

            history_detail = is_history_detail_query(request.query_text)
            relevant_summaries, summary_source_ids = self._select_summaries(
                request=request,
                summaries=summaries,
                history_detail=history_detail,
            )
            relevant_history_lines, history_messages = self._select_relevant_history(
                request=request,
                messages=messages,
                users=users,
                users_by_id=users_by_id,
                recent_messages=recent_messages,
                history_detail=history_detail,
            )
            if any(messages.is_qq_blocked_outbound(message) for message in history_messages):
                blocked_output_present = True
            relevant_memories, memory_source_ids, fts_count, vector_count = self._select_memories(
                request=request,
                memories=memories,
                history_detail=history_detail,
            )
            member_focus_lines, member_source_ids = self._build_member_focus_lines(
                request=request,
                messages=messages,
                memories=memories,
                users=users,
            )
            selected_source_msg_ids.extend(summary_source_ids)
            selected_source_msg_ids.extend(message.platform_msg_id for message in history_messages)
            selected_source_msg_ids.extend(memory_source_ids)
            selected_source_msg_ids.extend(member_source_ids)

            context = LegacyMemoryPromptContext(
                recent_messages=recent_lines,
                full_history_messages=full_history_lines,
                full_history_preamble=full_history_preamble,
                full_history_enabled=request.use_full_history,
                member_focus_lines=member_focus_lines,
                summaries=relevant_summaries,
                relevant_history_messages=relevant_history_lines,
                memories=relevant_memories,
                history_detail=history_detail,
                blocked_output_present=blocked_output_present,
                fts_memory_candidate_count=fts_count,
                vector_memory_candidate_count=vector_count,
            )
            return self._result(
                request=request,
                context=context,
                selected_source_msg_ids=selected_source_msg_ids,
                mode="v1",
            )

    def _select_summaries(
        self,
        *,
        request: GroupMemoryContextRequest,
        summaries: SummaryRepository,
        history_detail: bool,
    ) -> tuple[list[str], list[str]]:
        summary_rows = summaries.list_group_summaries(
            scope_id=str(request.group_id),
            limit=max(24, self.settings.context_summary_limit * 12),
        )
        semantic_rows = [
            summary
            for summary in summary_rows
            if summary.summary_level in {"semantic_window", "semantic_daily"}
        ]
        if semantic_rows:
            summary_rows = semantic_rows
        ranked = retrieve_relevant_history(
            request.query_text,
            [{"id": summary.id, "plain_text": summary.content} for summary in summary_rows],
            limit=(
                self.settings.context_summary_limit * 2
                if history_detail
                else self.settings.context_summary_limit
            ),
        )
        ranked_ids = {int(summary["id"]) for summary in ranked}
        selected_rows = [summary for summary in summary_rows if summary.id in ranked_ids]
        if not selected_rows:
            selected_rows = summary_rows[-self.settings.context_summary_limit :]
        rendered = [
            f"[{normalize_timestamp(summary.start_at).date().isoformat()} to "
            f"{normalize_timestamp(summary.end_at).date().isoformat()}] {summary.content}"
            for summary in selected_rows
        ]
        source_ids: list[str] = []
        for summary in selected_rows:
            if summary.source_start_msg_id:
                source_ids.append(summary.source_start_msg_id)
            if summary.source_end_msg_id:
                source_ids.append(summary.source_end_msg_id)
        return rendered, source_ids

    def _select_relevant_history(
        self,
        *,
        request: GroupMemoryContextRequest,
        messages: MessageRepository,
        users: UserRepository,
        users_by_id: dict[int, object],
        recent_messages: list[object],
        history_detail: bool,
    ) -> tuple[list[str], list[object]]:
        if request.use_full_history:
            return [], []
        candidate_limit, selected_limit = history_recall_limits(
            self.settings.context_history_limit,
            history_detail=history_detail,
        )
        candidate_messages = messages.list_group_messages_matching_terms(
            group_id=request.group_id,
            terms=history_search_terms(request.query_text),
            exclude_platform_msg_ids={
                request.current_msg_id,
                *(message.platform_msg_id for message in recent_messages),
            },
            limit=candidate_limit,
        )
        ranked = retrieve_relevant_history(
            request.query_text,
            [
                {"id": message.id, "plain_text": message.plain_text}
                for message in candidate_messages
            ],
            limit=selected_limit,
        )
        selected_ids = {int(message["id"]) for message in ranked}
        selected_messages = [
            message for message in candidate_messages if message.id in selected_ids
        ]
        selected_messages.sort(key=lambda message: (message.timestamp, message.id))
        if selected_messages:
            users_by_id.update(
                users.get_users_by_ids([message.user_id for message in selected_messages])
            )
        rendered = [
            f"[{normalize_timestamp(message.timestamp).isoformat()}] "
            + format_message_line(
                user_id=message.user_id,
                plain_text=message.plain_text,
                users_by_id=users_by_id,
                bot_user_id=self.bot_user_id,
                bot_display_name=self.bot_display_name,
            )
            for message in selected_messages
        ]
        return rendered, selected_messages

    def _select_memories(
        self,
        *,
        request: GroupMemoryContextRequest,
        memories: MemoryRepository,
        history_detail: bool,
    ) -> tuple[list[str], list[str], int, int]:
        as_of = normalize_timestamp(request.current_timestamp)
        memory_rows = memories.list_current_group_memories(
            scope_id=str(request.group_id),
            limit=max(50, self.settings.context_history_limit * 8),
            as_of=as_of,
        )
        fts_rows = memories.search_group_memories_fts(
            scope_id=str(request.group_id),
            query=request.query_text,
            limit=max(12, self.settings.context_history_limit * 3),
            as_of=as_of,
        )
        vector_rows = memories.search_group_memories_vector(
            scope_id=str(request.group_id),
            query=request.query_text,
            limit=max(12, self.settings.context_history_limit * 3),
            as_of=as_of,
        )
        memory_by_id = {
            memory.id: memory for memory in [*fts_rows, *vector_rows, *memory_rows]
        }
        ranked = retrieve_relevant_memories(
            request.query_text,
            [
                {
                    "id": row.id,
                    "content": row.content,
                    "subject_id": row.subject_id,
                    "predicate": row.predicate,
                    "object_text": row.object_text,
                    "importance": row.importance,
                    "confidence": row.confidence,
                    "mention_count": row.mention_count,
                }
                for row in memory_by_id.values()
            ],
            limit=(
                self.settings.context_history_limit * 2
                if history_detail
                else self.settings.context_history_limit
            ),
        )
        selected_ids = {int(memory["id"]) for memory in ranked}
        source_ids = [
            memory_by_id[memory_id].source_msg_id
            for memory_id in selected_ids
            if memory_id in memory_by_id and memory_by_id[memory_id].source_msg_id
        ]
        return [memory["content"] for memory in ranked], source_ids, len(fts_rows), len(vector_rows)

    def _build_member_focus_lines(
        self,
        *,
        request: GroupMemoryContextRequest,
        messages: MessageRepository,
        memories: MemoryRepository,
        users: UserRepository,
    ) -> tuple[list[str], list[str]]:
        candidate_user_ids = messages.list_recent_group_user_ids(
            group_id=request.group_id,
            limit=200,
        )
        candidate_user_ids.extend([request.current_user_id, self.bot_user_id])
        users_by_id = users.get_users_by_ids(candidate_user_ids)
        referenced_user_id = self._resolve_referenced_member_id(
            query_text=request.query_text,
            users_by_id=users_by_id,
            exclude_user_ids={request.current_user_id},
        )
        if referenced_user_id is None:
            return [], []

        member_label = member_label_for_user(
            user_id=referenced_user_id,
            users_by_id=users_by_id,
            bot_user_id=self.bot_user_id,
            bot_display_name=self.bot_display_name,
        )
        member_lines = [f"Referenced member: {member_label}"]
        source_ids: list[str] = []
        member_messages = messages.list_recent_group_messages_for_user(
            group_id=request.group_id,
            user_id=referenced_user_id,
            limit=max(4, self.settings.context_history_limit),
        )
        if member_messages:
            source_ids.extend(message.platform_msg_id for message in member_messages)
            member_lines.append(
                "Recent messages from this member:\n"
                + "\n".join(
                    format_message_line(
                        user_id=message.user_id,
                        plain_text=message.plain_text,
                        users_by_id=users_by_id,
                        bot_user_id=self.bot_user_id,
                        bot_display_name=self.bot_display_name,
                    )
                    for message in member_messages
                )
            )
        member_memories = memories.list_group_memories_for_subject(
            scope_id=str(request.group_id),
            subject_id=str(referenced_user_id),
            limit=4,
        )
        if member_memories:
            source_ids.extend(
                memory.source_msg_id for memory in member_memories if memory.source_msg_id
            )
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
        return member_lines, source_ids

    def _resolve_referenced_member_id(
        self,
        *,
        query_text: str,
        users_by_id: dict[int, object],
        exclude_user_ids: set[int],
    ) -> int | None:
        normalized_query = normalize_lookup_text(query_text)
        if not normalized_query:
            return None
        best_match: tuple[int, int] | None = None
        for user_id, user in users_by_id.items():
            if user_id in exclude_user_ids or user_id == self.bot_user_id:
                continue
            group_card = str(getattr(user, "group_card", "")).strip()
            aliases = [group_card, str(getattr(user, "nickname", "")).strip()]
            for alias in aliases:
                normalized_alias = normalize_lookup_text(alias)
                if len(normalized_alias) < 2 or normalized_alias not in normalized_query:
                    continue
                score = len(normalized_alias) + (100 if alias == group_card else 0)
                if best_match is None or score > best_match[1]:
                    best_match = (user_id, score)
        return None if best_match is None else best_match[0]

    @staticmethod
    def _humanize_memory_content(*, content: str, user_id: int, member_label: str) -> str:
        numeric_prefix = f"{user_id} "
        if content.startswith(numeric_prefix):
            return member_label + content[len(str(user_id)) :]
        return content.replace(str(user_id), member_label)

    @staticmethod
    def _format_history_message_line(message: object) -> str:
        text = str(getattr(message, "plain_text", "") or "").strip().replace("\r", "").replace("\n", "\\n")
        has_image = str(getattr(message, "msg_type", "") or "").lower() in {"image", "mixed"}
        if not text:
            text = (
                "[image attachment; visual content not retained]"
                if has_image
                else "[non-text message; no text retained]"
            )
        elif has_image:
            text += " [image attachment not included]"
        return f"{getattr(message, 'user_id')}: {text}"

    @staticmethod
    def _history_member_label(message: object) -> str:
        raw_json = getattr(message, "raw_json", {})
        raw_json = raw_json if isinstance(raw_json, dict) else {}
        sender = raw_json.get("sender") if isinstance(raw_json.get("sender"), dict) else {}
        nickname = str(sender.get("nickname", "")).strip().replace("\r", " ").replace("\n", " ")
        group_card = str(sender.get("card", "")).strip().replace("\r", " ").replace("\n", " ")
        if group_card:
            return group_card[:80]
        if nickname:
            return nickname[:80]
        return str(getattr(message, "user_id"))

    def _format_full_history_lines(
        self,
        messages: list[object],
    ) -> tuple[list[str], list[str]]:
        if not messages:
            return [], []
        participant_labels: dict[int, str] = {}
        for message in messages:
            participant_labels[int(getattr(message, "user_id"))] = self._history_member_label(message)
        participants = "; ".join(
            f"{user_id}={label}" for user_id, label in participant_labels.items()
        )
        return (
            [
                "Participants (group-local display names; messages below remain in timestamp/id order): "
                + participants
            ],
            [self._format_history_message_line(message) for message in messages],
        )

    @staticmethod
    def _result(
        *,
        request: GroupMemoryContextRequest,
        context: LegacyMemoryPromptContext,
        selected_source_msg_ids: list[str],
        mode: str,
    ) -> MemoryContextResult:
        sections = [
            *context.recent_messages,
            *context.full_history_preamble,
            *context.full_history_messages,
            *context.member_focus_lines,
            *context.summaries,
            *context.relevant_history_messages,
            *context.memories,
        ]
        return MemoryContextResult(
            group_id=request.group_id,
            packed_context=context,
            selected_source_msg_ids=tuple(dict.fromkeys(selected_source_msg_ids)),
            estimated_tokens=ContextBuilder.estimate_prompt_tokens(sections),
            mode=mode,
        )
