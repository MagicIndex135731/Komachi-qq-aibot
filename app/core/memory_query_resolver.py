"""Deterministic, offline query parsing for group-memory retrieval.

The resolver deliberately has no repository dependency.  It only receives the
small recent-message snapshot owned by its caller, which keeps reference
resolution testable and prevents a rewrite provider from widening its scope.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import json
import re
from typing import Callable, Literal, Protocol, Sequence
from zoneinfo import ZoneInfo

from app.core.member_identity import (
    GroupMemberIdentity,
    classify_group_member_reference,
)


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A half-open UTC time range: ``start <= time < end``."""

    start: datetime | None
    end: datetime | None


class RecentMemoryMessage(Protocol):
    """Minimal recent-message contract used by deterministic parsing."""

    source_msg_id: str
    speaker: str
    content: str
    sent_at: datetime
    reply_to_msg_id: str | None
    blocked: bool
    user_id: int | str | None
    is_bot: bool


AnswerMode = Literal[
    "exact",
    "mention",
    "dated_history",
    "summary",
    "assessment",
    "current_fact",
    "general_history",
]
CoverageMode = Literal["relevance", "chronological", "time_buckets"]
SubjectBinding = Literal["explicit", "requester", "unbound"]


@dataclass(frozen=True, slots=True)
class ResolvedMemoryQuery:
    """The single typed contract shared by memory-query consumers.

    The legacy resolver fields remain first and retain defaults so existing
    V1/V2 callers can construct this result while the V3 constraints are wired
    through the runtime.
    """

    original_query: str
    retrieval_query: str
    entities: tuple[str, ...] = ()
    speaker_ids: tuple[str, ...] = ()
    subject_ids: tuple[str, ...] | None = None
    time_range: TimeRange | None = None
    reference_msg_ids: tuple[str, ...] = ()
    rewrite_used: bool = False
    retrieval_mode: Literal["hybrid", "exact_quote", "temporal"] = "hybrid"
    needs_history: bool = False
    needs_detail: bool = False
    confidence: float = 1.0
    group_id: int | None = None
    requester_id: str | None = None
    subject_binding: SubjectBinding = "unbound"
    answer_mode: AnswerMode = "general_history"
    coverage_mode: CoverageMode = "relevance"

    @property
    def resolved_query(self) -> str:
        return self.retrieval_query

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return self.entities

    @property
    def speaker(self) -> str | None:
        return self.speaker_ids[0] if len(self.speaker_ids) == 1 else None

    @property
    def parsed_query(self) -> str:
        """Compatibility-friendly name for the query passed to retrieval."""

        return self.retrieval_query

    @property
    def requester_uin(self) -> str | None:
        return self.requester_id

    @property
    def subject_uins(self) -> tuple[str, ...] | None:
        return self.subject_ids

    @property
    def start_at_utc(self) -> datetime | None:
        return self.time_range.start if self.time_range is not None else None

    @property
    def end_at_utc(self) -> datetime | None:
        return self.time_range.end if self.time_range is not None else None

    @property
    def coverage_strategy(self) -> CoverageMode:
        return self.coverage_mode


# V3 name for the same compatibility-preserving resolver result.
MemoryQueryPlan = ResolvedMemoryQuery


RewriteProvider = Callable[[str, tuple[RecentMemoryMessage, ...], float], str]
IdentityValidator = Callable[[str], bool]

_DATE_PATTERN = re.compile(r"(?<!\d)(?:(\d{4})[年\-/])?(\d{1,2})[月\-/](\d{1,2})日?")
_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,3})(?![\u4e00-\u9fff])")
_SPEECH_NAME_PATTERN = re.compile(r"([\u4e00-\u9fff]{2,3})(?=说|表示|提到|认为)")
_JOINED_NAME_PATTERN = re.compile(r"([\u4e00-\u9fff]{2})(?=和|、)|(?:和|、)([\u4e00-\u9fff]{2})(?=都|和|、|说|表示|提到|认为)")
_FOLLOW_UP_PATTERN = re.compile(r"详细讲讲|后来呢|之前那个|那个人|他说了什么|她说了什么|最后怎么样")
_HISTORY_PATTERN = re.compile(
    r"以前|曾经|过去|历史|之前|当时|那时|说过|发过|提过|聊过|发言"
)
_DETAIL_PATTERN = re.compile(r"详细|经过|后来|最后|怎么处理")
_FIRST_PERSON_SUBJECT_PATTERN = re.compile(
    r"(?:评价|点评|分析|总结|概括|说说|怎么看)\s*(?:一下)?我|"
    r"(?:我|我的)(?:最喜欢|喜欢什么|讨厌什么|不喜欢什么|过去|以前|历史)"
)
_ASSESSMENT_PATTERN = re.compile(r"评价|点评|印象|怎么看|性格|分析(?:一下)?(?:我|[\u4e00-\u9fffA-Za-z0-9_-]+)")
_SUMMARY_PATTERN = re.compile(r"总结|概括|汇总|发生了什么|聊了什么|都说了什么")
_MENTION_PATTERN = re.compile(
    r"(?:谁|哪些人|有人).*(?:提到|说到|叫|@)|"
    r"(?:提到|说到|叫|@).*(?:谁|哪些人)|"
    r"(?:他们|她们|大家|群里).*(?:叫|提到|说到|@)\s*我"
)
_REQUESTER_MENTION_PATTERN = re.compile(
    r"(?:谁|哪些人|有人|他们|她们|大家|群里).*(?:叫|提到|说到|@)\s*我"
)
_CURRENT_FACT_PATTERN = re.compile(r"最喜欢|喜欢什么|讨厌什么|不喜欢什么|还记得|记得")
_COMMON_WORDS = frozenset({"发布", "已经", "那个", "什么", "怎么", "后来", "之前", "最后", "结果", "消息", "延期", "完成", "服务", "迁移", "今天", "昨天", "前天"})
_PERSON_MEMORY_SUBJECT_PATTERN = re.compile(
    r"^\s*(?P<subject>[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}?)(?:最喜欢|喜欢什么|讨厌什么|不喜欢什么)"
)
_REMEMBER_PERSON_PATTERN = re.compile(
    r"^\s*(?:还)?记得\s*(?P<subject>[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}?)(?:吗|么|的|曾经|以前|喜欢|讨厌|[？?]|$)"
)
_PERSON_SPEECH_SUBJECT_PATTERN = re.compile(
    r"^\s*(?P<subject>[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}?)(?:最近|昨天|今天|以前|曾经|过去|当时)?"
    r"(?:说过|说了|发过|发言|提到|聊过)"
)
_PERSON_ASSESSMENT_SUBJECT_PATTERN = re.compile(
    r"^\s*(?:如何|怎么)?(?:评价|点评|分析|看待)\s*"
    r"(?P<subject>[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}?)(?:这个人|的|[？?]|$)"
)
_FIRST_PERSON_HISTORY_PATTERN = re.compile(
    r"^\s*(?:我|我的)(?:最近|昨天|今天|以前|曾经|过去|历史|当时)?.*"
    r"(?:说过|说了|发过|发言|提到|聊过|表现|最喜欢|喜欢什么|讨厌什么)"
)
_TEMPORAL_FIRST_PERSON_HISTORY_PATTERN = re.compile(
    r"^\s*(?:最近|昨天|今天|前天|以前|曾经|过去|当时)\s*(?:我|我的).*"
    r"(?:说过|说了|发过|发言|提到|聊过|表现|最喜欢|喜欢什么|讨厌什么)"
)
_NON_PERSON_MEMORY_SUBJECTS = frozenset(
    {
        "我",
        "你",
        "您",
        "他",
        "她",
        "它",
        "大家",
        "群里",
        "群友",
        "各位",
        "所有人",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "有人",
        "谁",
    }
)
_SUBJECTLESS_MEMORY_QUERY_PREFIXES = (
    "最喜欢什么",
    "喜欢什么",
    "讨厌什么",
    "不喜欢什么",
)
ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")


class MemoryQueryResolver:
    """Resolve time and conversational references before retrieval.

    ``rewrite_provider`` is an injected, bounded call contract.  The resolver
    passes its finite timeout value to it but never creates a network client or
    retries the call.  Any provider failure, malformed JSON, or schema
    violation returns the original query unchanged.
    """

    def __init__(
        self,
        rewrite_provider: RewriteProvider | None = None,
        *,
        rewrite_timeout_seconds: float = 0.75,
        recent_limit: int = 12,
        identity_validator: IdentityValidator | None = None,
    ) -> None:
        if rewrite_timeout_seconds <= 0:
            raise ValueError("rewrite_timeout_seconds must be positive")
        if recent_limit <= 0:
            raise ValueError("recent_limit must be positive")
        self._rewrite_provider = rewrite_provider
        self._rewrite_timeout_seconds = rewrite_timeout_seconds
        self._recent_limit = recent_limit
        self._identity_validator = identity_validator
        self._rewrite_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-query-rewrite")
            if rewrite_provider is not None
            else None
        )

    def resolve(
        self,
        query: str,
        *,
        recent_messages: Sequence[RecentMemoryMessage],
        quoted_message: RecentMemoryMessage | None = None,
        now: datetime | None = None,
        group_members: Sequence[GroupMemberIdentity] = (),
        excluded_member_ids: set[int] | frozenset[int] = frozenset(),
        group_id: int | None = None,
        requester_id: int | str | None = None,
        requester_uin: int | str | None = None,
    ) -> ResolvedMemoryQuery:
        """Return a typed retrieval query without reading persistence.

        Deterministic references take precedence over a rewrite.  A rewrite is
        only considered for a remaining ambiguous follow-up question, and its
        context excludes blocked messages and blocked quoted content.
        """

        original = query.strip()
        normalized_group_id = self._normalize_group_id(group_id)
        normalized_requester_id = self._normalize_requester_id(requester_id, requester_uin)
        current_time = self._as_shanghai_time(now or datetime.now(ASIA_SHANGHAI))
        recent = tuple(recent_messages[-self._recent_limit :])
        time_range = self._parse_time_range(original, current_time)
        needs_detail = bool(_DETAIL_PATTERN.search(original))
        answer_mode = self._answer_mode(original, time_range, quoted_message)
        coverage_mode = self._coverage_mode(answer_mode)
        needs_history = bool(
            time_range
            or _FOLLOW_UP_PATTERN.search(original)
            or _HISTORY_PATTERN.search(original)
            or answer_mode in {"mention", "summary", "assessment"}
        )

        direct_reference = classify_group_member_reference(
            original,
            group_members,
            match_mode="contained",
            exclude_user_ids=excluded_member_ids,
        )
        if direct_reference.status == "resolved":
            direct_member = direct_reference.member
            if direct_member is None:
                raise RuntimeError("resolved group member reference is missing its member")
            direct_subject_ids = (str(direct_member.user_id),)
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=original,
                entities=(direct_member.matched_alias,),
                speaker_ids=direct_subject_ids,
                subject_ids=direct_subject_ids,
                time_range=time_range,
                retrieval_mode="temporal" if time_range else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
                group_id=normalized_group_id,
                requester_id=normalized_requester_id,
                subject_binding="explicit",
                answer_mode=answer_mode,
                coverage_mode=coverage_mode,
            )
        if direct_reference.status == "ambiguous":
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=original,
                subject_ids=(),
                time_range=time_range,
                retrieval_mode="temporal" if time_range else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
                group_id=normalized_group_id,
                requester_id=normalized_requester_id,
                subject_binding="explicit",
                answer_mode=answer_mode,
                coverage_mode=coverage_mode,
            )

        if normalized_requester_id is not None and (
            self._is_first_person_subject(original)
            or self._is_requester_mention_query(original)
        ):
            requester_subject = (normalized_requester_id,)
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=original,
                speaker_ids=requester_subject,
                subject_ids=requester_subject,
                time_range=time_range,
                retrieval_mode="temporal" if time_range else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
                group_id=normalized_group_id,
                requester_id=normalized_requester_id,
                subject_binding="requester",
                answer_mode=answer_mode,
                coverage_mode=coverage_mode,
            )

        if answer_mode == "mention":
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=original,
                subject_ids=(),
                time_range=time_range,
                retrieval_mode="temporal" if time_range else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
                group_id=normalized_group_id,
                requester_id=normalized_requester_id,
                subject_binding="requester",
                answer_mode=answer_mode,
                coverage_mode=coverage_mode,
            )

        deterministic = self._resolve_reference(original, recent, quoted_message)
        if deterministic is not None:
            retrieval_query, entities, speaker_ids, source_ids = deterministic
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=retrieval_query,
                entities=entities,
                speaker_ids=speaker_ids,
                subject_ids=(
                    None if quoted_message is not None else speaker_ids or None
                ),
                time_range=time_range,
                reference_msg_ids=source_ids,
                retrieval_mode="exact_quote" if quoted_message is not None else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
                group_id=normalized_group_id,
                requester_id=normalized_requester_id,
                subject_binding="explicit" if speaker_ids else "unbound",
                answer_mode=answer_mode,
                coverage_mode=coverage_mode,
            )

        if self._is_person_memory_query(original):
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=original,
                subject_ids=(),
                time_range=time_range,
                retrieval_mode="temporal" if time_range else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
                group_id=normalized_group_id,
                requester_id=normalized_requester_id,
                subject_binding="explicit",
                answer_mode=answer_mode,
                coverage_mode=coverage_mode,
            )

        if self._rewrite_provider is not None and _FOLLOW_UP_PATTERN.search(original):
            rewritten = self._try_rewrite(original, recent, current_time)
            if rewritten is not None:
                return replace(
                    rewritten,
                    needs_history=True,
                    needs_detail=needs_detail,
                    time_range=time_range or rewritten.time_range,
                    retrieval_mode=(
                        "temporal"
                        if time_range is not None
                        else rewritten.retrieval_mode
                    ),
                    group_id=normalized_group_id,
                    requester_id=normalized_requester_id,
                    answer_mode=answer_mode,
                    coverage_mode=coverage_mode,
                )

        return ResolvedMemoryQuery(
            original,
            original,
            time_range=time_range,
            retrieval_mode="temporal" if time_range else "hybrid",
            needs_history=needs_history,
            needs_detail=needs_detail,
            group_id=normalized_group_id,
            requester_id=normalized_requester_id,
            answer_mode=answer_mode,
            coverage_mode=coverage_mode,
        )

    @staticmethod
    def _is_person_memory_query(query: str) -> bool:
        if query.lstrip().startswith(_SUBJECTLESS_MEMORY_QUERY_PREFIXES):
            return False
        for pattern in (
            _PERSON_MEMORY_SUBJECT_PATTERN,
            _REMEMBER_PERSON_PATTERN,
            _PERSON_SPEECH_SUBJECT_PATTERN,
            _PERSON_ASSESSMENT_SUBJECT_PATTERN,
        ):
            match = pattern.search(query)
            if match is not None and match.group("subject") not in _NON_PERSON_MEMORY_SUBJECTS:
                return True
        return False

    @staticmethod
    def _is_first_person_subject(query: str) -> bool:
        return bool(
            _FIRST_PERSON_SUBJECT_PATTERN.search(query)
            or _FIRST_PERSON_HISTORY_PATTERN.search(query)
            or _TEMPORAL_FIRST_PERSON_HISTORY_PATTERN.search(query)
        )

    @staticmethod
    def _is_requester_mention_query(query: str) -> bool:
        return bool(_REQUESTER_MENTION_PATTERN.search(query))

    @staticmethod
    def _answer_mode(
        query: str,
        time_range: TimeRange | None,
        quoted_message: RecentMemoryMessage | None,
    ) -> AnswerMode:
        if quoted_message is not None:
            return "exact"
        if _MENTION_PATTERN.search(query):
            return "mention"
        if _ASSESSMENT_PATTERN.search(query):
            return "assessment"
        if _SUMMARY_PATTERN.search(query):
            return "summary"
        if _CURRENT_FACT_PATTERN.search(query):
            return "current_fact"
        if time_range is not None:
            return "dated_history"
        return "general_history"

    @staticmethod
    def _coverage_mode(answer_mode: AnswerMode) -> CoverageMode:
        if answer_mode in {"summary", "assessment"}:
            return "time_buckets"
        if answer_mode == "dated_history":
            return "chronological"
        return "relevance"

    @staticmethod
    def _normalize_group_id(group_id: int | None) -> int | None:
        if group_id is None:
            return None
        if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id <= 0:
            raise ValueError("group_id must be a positive integer")
        return group_id

    @staticmethod
    def _normalize_requester_id(
        requester_id: int | str | None,
        requester_uin: int | str | None,
    ) -> str | None:
        normalized = {
            str(value).strip()
            for value in (requester_id, requester_uin)
            if value is not None and not isinstance(value, bool) and str(value).strip()
        }
        if any(isinstance(value, bool) for value in (requester_id, requester_uin)):
            raise ValueError("requester ID must not be boolean")
        if len(normalized) > 1:
            raise ValueError("requester_id and requester_uin must identify the same user")
        return next(iter(normalized), None)

    @staticmethod
    def _as_shanghai_time(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=ASIA_SHANGHAI)
        return value.astimezone(ASIA_SHANGHAI)

    def _resolve_reference(
        self,
        query: str,
        recent: tuple[RecentMemoryMessage, ...],
        quoted_message: RecentMemoryMessage | None,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
        if not _FOLLOW_UP_PATTERN.search(query):
            return None

        if quoted_message is not None and not quoted_message.blocked and quoted_message.content.strip():
            quoted_source_id = self._source_id(quoted_message)
            if bool(getattr(quoted_message, "is_bot", False)) and quoted_message.reply_to_msg_id:
                upstream = next(
                    (
                        item
                        for item in reversed(recent)
                        if not item.blocked
                        and self._source_id(item) == quoted_message.reply_to_msg_id
                        and item.content.strip()
                    ),
                    None,
                )
                if upstream is not None:
                    upstream_source_id = self._source_id(upstream)
                    speaker_id = self._canonical_user_id(upstream)
                    return (
                        upstream.content.strip(),
                        (),
                        ((speaker_id,) if speaker_id else ()),
                        tuple(dict.fromkeys((upstream_source_id, quoted_source_id))),
                    )
            entity = self._unique_entity((quoted_message,))
            speaker_id = self._canonical_user_id(quoted_message)
            return (
                quoted_message.content.strip(),
                ((entity,) if entity else ()),
                ((speaker_id,) if speaker_id else ()),
                (quoted_source_id,),
            )

        matching_speakers = {
            message.speaker.strip()
            for message in recent
            if not message.blocked and message.speaker.strip() and message.speaker in query
        }
        if len(matching_speakers) == 1:
            speaker = next(iter(matching_speakers))
            source = next(
                (item for item in reversed(recent) if not item.blocked and item.speaker == speaker and item.content.strip()),
                None,
            )
            if source is not None:
                speaker_id = self._canonical_user_id(source)
                return (
                    f"{speaker} {source.content.strip()}",
                    (speaker,),
                    ((speaker_id,) if speaker_id else ()),
                    (self._source_id(source),),
                )

        entity = self._unique_entity(recent)
        if entity is None:
            return None
        source = next((item for item in reversed(recent) if entity in item.content and not item.blocked), None)
        if source is None:
            return None
        return f"{entity} {source.content.strip()}", (entity,), (), (self._source_id(source),)

    @staticmethod
    def _source_id(message: RecentMemoryMessage) -> str:
        """Accept legacy snapshots during the V1 → V2 transition."""

        source_id = getattr(message, "source_msg_id", None)
        message_id = getattr(message, "message_id", None)
        identifier = source_id if source_id is not None else message_id
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("recent message is missing a source message ID")
        return identifier

    @staticmethod
    def _canonical_user_id(message: RecentMemoryMessage) -> str | None:
        raw_user_id = getattr(message, "user_id", None)
        if raw_user_id is None or isinstance(raw_user_id, bool):
            return None
        normalized = str(raw_user_id).strip()
        return normalized or None

    @staticmethod
    def _unique_entity(messages: Sequence[RecentMemoryMessage]) -> str | None:
        candidates: list[str] = []
        for message in messages:
            if message.blocked:
                continue
            # Chinese prose has no word boundaries.  Speech predicates are the
            # reliable deterministic form (e.g. “张三说…”); the boundary form
            # remains useful for nicknames surrounded by punctuation/spaces.
            joined_names = [name for pair in _JOINED_NAME_PATTERN.findall(message.content) for name in pair if name]
            names = [*joined_names, *_SPEECH_NAME_PATTERN.findall(message.content), *_NAME_PATTERN.findall(message.content)]
            for candidate in names:
                if candidate not in _COMMON_WORDS and "都" not in candidate and candidate not in candidates:
                    candidates.append(candidate)
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _parse_time_range(query: str, now: datetime) -> TimeRange | None:
        local_now = MemoryQueryResolver._as_shanghai_time(now)
        local_day = datetime(
            local_now.year,
            local_now.month,
            local_now.day,
            tzinfo=ASIA_SHANGHAI,
        )
        relative_days = {"今天": 0, "昨天": 1, "前天": 2}
        for word, offset in relative_days.items():
            if word in query:
                start = local_day - timedelta(days=offset)
                return TimeRange(start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC))
        if "上周" in query:
            start = local_day - timedelta(days=local_day.weekday() + 7)
            return TimeRange(start.astimezone(UTC), (start + timedelta(days=7)).astimezone(UTC))

        matches = tuple(_DATE_PATTERN.finditer(query))
        if not matches:
            return None
        start = MemoryQueryResolver._date_match_start(matches[0], local_now.year)
        if start is None:
            return None
        if len(matches) >= 2:
            bridge = query[matches[0].end() : matches[1].start()]
            if re.search(r"到|至|~|～|—|–", bridge):
                end_day = MemoryQueryResolver._date_match_start(matches[1], start.year)
                if end_day is None or end_day < start:
                    return None
                return TimeRange(
                    start.astimezone(UTC),
                    (end_day + timedelta(days=1)).astimezone(UTC),
                )
        return TimeRange(start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC))

    @staticmethod
    def _date_match_start(match: re.Match[str], default_year: int) -> datetime | None:
        year_text, month_text, day_text = match.groups()
        try:
            return datetime(
                int(year_text or default_year),
                int(month_text),
                int(day_text),
                tzinfo=ASIA_SHANGHAI,
            )
        except ValueError:
            return None

    def _try_rewrite(
        self,
        original: str,
        recent: tuple[RecentMemoryMessage, ...],
        now: datetime,
    ) -> ResolvedMemoryQuery | None:
        safe_recent = tuple(message for message in recent if not message.blocked)
        if self._rewrite_executor is None:
            return None
        future = self._rewrite_executor.submit(
            self._rewrite_provider,
            original,
            safe_recent,
            self._rewrite_timeout_seconds,
        )
        try:
            response = future.result(timeout=self._rewrite_timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            return None
        except Exception:
            return None
        return self._parse_rewrite_response(original, response, now)

    def _parse_rewrite_response(
        self,
        original: str,
        response: str,
        now: datetime,
    ) -> ResolvedMemoryQuery | None:
        try:
            payload = json.loads(response)
        except (TypeError, ValueError):
            return None
        allowed_fields = {
            "resolved_query",
            "retrieval_query",
            "entity_ids",
            "entities",
            "speaker_ids",
            "time_range",
            "confidence",
        }
        if not isinstance(payload, dict) or set(payload) - allowed_fields:
            return None
        retrieval_query = payload.get("resolved_query", payload.get("retrieval_query"))
        if not isinstance(retrieval_query, str) or not retrieval_query.strip():
            return None
        raw_entities = payload.get("entity_ids", payload.get("entities", []))
        if not isinstance(raw_entities, list) or any(not isinstance(value, str) or not value.strip() for value in raw_entities):
            return None
        raw_speakers = payload.get("speaker_ids", [])
        if not isinstance(raw_speakers, list) or any(
            not isinstance(value, str) or not value.strip() for value in raw_speakers
        ):
            return None
        normalized_entities = tuple(dict.fromkeys(value.strip() for value in raw_entities))
        normalized_speakers = tuple(dict.fromkeys(value.strip() for value in raw_speakers))
        if not self._identities_are_valid((*normalized_entities, *normalized_speakers)):
            return None
        confidence = payload.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            return None
        time_range = MemoryQueryResolver._parse_rewrite_time_range(payload.get("time_range"), now)
        if payload.get("time_range") is not None and time_range is None:
            return None
        return ResolvedMemoryQuery(
            original_query=original,
            retrieval_query=retrieval_query.strip(),
            entities=normalized_entities,
            speaker_ids=normalized_speakers,
            subject_ids=normalized_speakers or None,
            subject_binding="explicit" if normalized_speakers else "unbound",
            time_range=time_range,
            rewrite_used=True,
            confidence=float(confidence),
        )

    def _identities_are_valid(self, identities: Sequence[str]) -> bool:
        if self._identity_validator is None:
            return True
        try:
            return all(self._identity_validator(identity) for identity in identities)
        except Exception:
            return False

    @staticmethod
    def _parse_rewrite_time_range(value: object, now: datetime) -> TimeRange | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"start", "end"}:
            return None
        start_value, end_value = value["start"], value["end"]
        if not isinstance(start_value, str) or not isinstance(end_value, str):
            return None
        try:
            start = datetime.fromisoformat(start_value)
            end = datetime.fromisoformat(end_value)
        except ValueError:
            return None
        if start.tzinfo is None:
            start = start.replace(tzinfo=ASIA_SHANGHAI)
        if end.tzinfo is None:
            end = end.replace(tzinfo=ASIA_SHANGHAI)
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        return TimeRange(start_utc, end_utc) if start_utc < end_utc else None
