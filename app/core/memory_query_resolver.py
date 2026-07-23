"""Deterministic, offline query parsing for group-memory retrieval.

The resolver deliberately has no repository dependency.  It only receives the
small recent-message snapshot owned by its caller, which keeps reference
resolution testable and prevents a rewrite provider from widening its scope.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import json
import re
from typing import Callable, Literal, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A local-calendar half-open time range: ``start <= time < end``."""

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


@dataclass(frozen=True, slots=True)
class ResolvedMemoryQuery:
    original_query: str
    retrieval_query: str
    entities: tuple[str, ...] = ()
    speaker_ids: tuple[str, ...] = ()
    time_range: TimeRange | None = None
    reference_msg_ids: tuple[str, ...] = ()
    rewrite_used: bool = False
    retrieval_mode: Literal["hybrid", "exact_quote", "temporal"] = "hybrid"
    needs_history: bool = False
    needs_detail: bool = False
    confidence: float = 1.0

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


RewriteProvider = Callable[[str, tuple[RecentMemoryMessage, ...], float], str]
IdentityValidator = Callable[[str], bool]

_DATE_PATTERN = re.compile(r"(?<!\d)(?:(\d{4})[年\-/])?(\d{1,2})[月\-/](\d{1,2})日?")
_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,3})(?![\u4e00-\u9fff])")
_SPEECH_NAME_PATTERN = re.compile(r"([\u4e00-\u9fff]{2,3})(?=说|表示|提到|认为)")
_JOINED_NAME_PATTERN = re.compile(r"([\u4e00-\u9fff]{2})(?=和|、)|(?:和|、)([\u4e00-\u9fff]{2})(?=都|和|、|说|表示|提到|认为)")
_FOLLOW_UP_PATTERN = re.compile(r"详细讲讲|后来呢|之前那个|那个人|他说了什么|她说了什么|最后怎么样")
_DETAIL_PATTERN = re.compile(r"详细|经过|后来|最后|怎么处理")
_COMMON_WORDS = frozenset({"发布", "已经", "那个", "什么", "怎么", "后来", "之前", "最后", "结果", "消息", "延期", "完成", "服务", "迁移", "今天", "昨天", "前天"})


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
    ) -> ResolvedMemoryQuery:
        """Return a typed retrieval query without reading persistence.

        Deterministic references take precedence over a rewrite.  A rewrite is
        only considered for a remaining ambiguous follow-up question, and its
        context excludes blocked messages and blocked quoted content.
        """

        original = query.strip()
        current_time = now or datetime.now()
        recent = tuple(recent_messages[-self._recent_limit :])
        time_range = self._parse_time_range(original, current_time)
        needs_detail = bool(_DETAIL_PATTERN.search(original))
        needs_history = bool(time_range or _FOLLOW_UP_PATTERN.search(original) or "历史" in original)

        deterministic = self._resolve_reference(original, recent, quoted_message)
        if deterministic is not None:
            retrieval_query, entities, speaker_ids, source_ids = deterministic
            return ResolvedMemoryQuery(
                original_query=original,
                retrieval_query=retrieval_query,
                entities=entities,
                speaker_ids=speaker_ids,
                time_range=time_range,
                reference_msg_ids=source_ids,
                retrieval_mode="exact_quote" if quoted_message is not None else "hybrid",
                needs_history=needs_history,
                needs_detail=needs_detail,
            )

        if self._rewrite_provider is not None and _FOLLOW_UP_PATTERN.search(original):
            rewritten = self._try_rewrite(original, recent, current_time)
            if rewritten is not None:
                return replace(rewritten, needs_history=True, needs_detail=needs_detail)

        return ResolvedMemoryQuery(
            original,
            original,
            time_range=time_range,
            retrieval_mode="temporal" if time_range else "hybrid",
            needs_history=needs_history,
            needs_detail=needs_detail,
        )

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
        local_day = datetime(now.year, now.month, now.day, tzinfo=now.tzinfo)
        relative_days = {"今天": 0, "昨天": 1, "前天": 2}
        for word, offset in relative_days.items():
            if word in query:
                start = local_day - timedelta(days=offset)
                return TimeRange(start, start + timedelta(days=1))
        if "上周" in query:
            start = local_day - timedelta(days=local_day.weekday() + 7)
            return TimeRange(start, start + timedelta(days=7))

        match = _DATE_PATTERN.search(query)
        if match is None:
            return None
        year_text, month_text, day_text = match.groups()
        try:
            start = datetime(int(year_text or now.year), int(month_text), int(day_text), tzinfo=now.tzinfo)
        except ValueError:
            return None
        return TimeRange(start, start + timedelta(days=1))

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
        if start.tzinfo is None and now.tzinfo is not None:
            start = start.replace(tzinfo=now.tzinfo)
            end = end.replace(tzinfo=now.tzinfo)
        return TimeRange(start, end) if start < end else None
