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
    GroupMemberReferenceResolution,
    ResolvedGroupMember,
    classify_group_member_reference,
    normalize_member_alias,
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
TopicExtraction = Literal["none", "deterministic", "fallback"]


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
    topic_query: str | None = None
    topic_terms: tuple[str, ...] = ()
    topic_extraction: TopicExtraction = "fallback"
    subject_aliases_removed: tuple[str, ...] = ()

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
    r"以前|曾经|过去|历史|之前|当时|那时|说过|发过|提过|聊过|发言|"
    r"自称|哪条|哪句话|哪一句|什么时候|哪一次|原话"
)
_DETAIL_PATTERN = re.compile(
    r"详细|经过|后来|最后|怎么处理|原话|哪条|哪句话|哪一句|具体什么时候|哪一次"
)
_FIRST_PERSON_SUBJECT_PATTERN = re.compile(
    r"(?:评价|点评|分析|总结|概括|说说|怎么看)\s*(?:一下)?我|"
    r"(?:我|我的)(?:平时|一般|通常)?"
    r"(?:最?喜欢|爱|想)(?:看|听|玩|用|吃|喝|读|追)?什么|"
    r"(?:我|我的)(?:讨厌什么|不喜欢什么|过去|以前|历史)"
)
_ASSESSMENT_PATTERN = re.compile(
    r"评价|点评|印象|怎么看|性格|分析(?:一下)?(?:我|[\u4e00-\u9fffA-Za-z0-9_-]+)"
    r"|(?:觉得|感觉|认为|以为|看来|听起来|看起来|咋看|啥看法)"
    r".{0,16}?(?:怎么样|如何|咋样|怎样|评价|印象|看法)"
    r"|(?:对|给)[^，。？?！!\n]{1,16}?(?:什么看法|啥看法|什么印象|看法如何|印象如何)"
)
_FOLLOW_UP_PRONOUN_PATTERN = re.compile(r"他|她|那位|这个人|那家伙|这位")
_PERSON_SHAPE_SUFFIX_PATTERN = re.compile(
    r"(?:人|猫|哥|姐|老师|同学|君|酱|娃|叔|婶|妈|爸|兄|弟|妹|生|总|董|员)$"
)
_SUMMARY_PATTERN = re.compile(
    r"总结|概括|汇总|发生了什么|"
    r"(?:今天|昨天|前天|本周|这周|上周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?).*?"
    r"(?:都|分别)(?:说了|说过|发了|发过|讲了|聊了)什么|"
    r"(?:说了|说过|发了|发过|讲了|聊了)(?:哪些(?:话|内容)?|几条)|"
    r"(?:有|有哪些|有什么)(?:发言|内容|消息)|(?:哪些|所有)(?:发言|内容|消息)"
)
_MENTION_PATTERN = re.compile(
    r"(?:谁|哪些人|有人).*(?:提到|说到|叫|@)|"
    r"(?:提到|说到|叫|@).*(?:谁|哪些人)|"
    r"(?:他们|她们|大家|群里).*(?:叫|提到|说到|@)\s*我"
)
_REQUESTER_MENTION_PATTERN = re.compile(
    r"(?:谁|哪些人|有人|他们|她们|大家|群里).*(?:叫|提到|说到|@)\s*我"
)
_CURRENT_FACT_PATTERN = re.compile(
    r"最喜欢|(?:最?喜欢|爱|想)(?:看|听|玩|用|吃|喝|读|追)?什么|"
    r"讨厌什么|不喜欢什么|还记得|记得"
)
_TOPIC_PUNCTUATION_PATTERN = re.compile(r"^[\s，。！？、,.!?：:；;]+|[\s，。！？、,.!?：:；;]+$")
_TOPIC_TERM_SPLIT_PATTERN = re.compile(r"[\s，。！？、,.!?：:；;]+")
_TOPIC_CATEGORY_SUFFIXES = (
    "动画电影",
    "动画片",
    "纪录片",
    "电视剧",
    "电影",
    "动画",
    "影片",
    "片子",
    "作品",
    "游戏",
    "小说",
)
_ASSESSMENT_SCAFFOLD_PATTERN = re.compile(
    r"如何(?:评价|点评|分析|看待)|怎么(?:评价|点评|分析|看待)|"
    r"(?:评价|点评|分析|看待)(?:一下)?|怎么看"
)
_CURRENT_FACT_SCAFFOLD_PATTERN = re.compile(
    r"(?:平时|一般|通常)?(?:最?喜欢|爱|想)(?:看|听|玩|用|吃|喝|读|追)?什么|"
    r"讨厌什么|不喜欢什么|还记得|记得"
)
_HISTORY_SCAFFOLD_PATTERN = re.compile(
    r"说过什么|说了什么|发过什么|发了什么|提过什么|聊过什么|"
    r"说过|说了|发过|发了|提过|聊过"
)
_SUMMARY_SCAFFOLD_PATTERN = re.compile(
    r"总结|概括|汇总|发生了什么|"
    r"(?:都|分别)?(?:说了|说过|发了|发过|讲了|聊了)"
    r"(?:哪些(?:话|内容)?|几条|什么)?|"
    r"(?:有|有哪些|有什么|哪些|所有)(?:发言|内容|消息)"
)
_COMMON_WORDS = frozenset({"发布", "已经", "那个", "什么", "怎么", "后来", "之前", "最后", "结果", "消息", "延期", "完成", "服务", "迁移", "今天", "昨天", "前天"})
_RELATIVE_DAY_WORDS = frozenset({"今天", "昨天", "前天"})
_MEMBER_JOINER_PATTERN = r"(?:再加上|并且|还有|以及|和|与|跟|及|、)"
_JOINED_RELATIVE_DAY_QUERY_PATTERN = re.compile(
    rf"^\s*(?:今天|昨天|前天)"
    rf"(?:\s*{_MEMBER_JOINER_PATTERN}\s*(?:今天|昨天|前天))+"
    r"\s*(?:发生|群里|聊了|都发生|都聊)"
)
_SUBJECTLESS_RELATIVE_EVENT_PATTERN = re.compile(
    r"^\s*(?:今天|昨天|前天).*(?:发生了什么|群里发生|都发生)"
)
_PERSON_MEMORY_SUBJECT_PATTERN = re.compile(
    r"^\s*(?P<subject>[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}?)"
    r"(?:最?喜欢|爱|想)(?:看|听|玩|用|吃|喝|读|追)?什么|"
    r"讨厌什么|不喜欢什么"
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
_PERSON_OPINION_SUBJECT_PATTERN = re.compile(
    r"^\s*(?P<subject>[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}?)"
    r"(?:觉得|感觉|认为|以为|看来|听起来|看起来|咋看|啥看法)\s*"
    r"(?P<topic>[\u4e00-\u9fffA-Za-z0-9]{1,16}?)"
    r"(?:怎么样|如何|咋样|怎样|什么看法|啥看法|什么印象|印象如何|如何评价)"
)
_FIRST_PERSON_HISTORY_PATTERN = re.compile(
    r"^\s*(?:我|我的)(?:最近|昨天|今天|以前|曾经|过去|历史|当时)?.*"
    r"(?:说过|说了|发过|发言|提到|聊过|表现|最喜欢|"
    r"喜欢(?:看|听|玩|用|吃|喝|读|追)?什么|讨厌什么)"
)
_TEMPORAL_FIRST_PERSON_HISTORY_PATTERN = re.compile(
    r"^\s*(?:最近|昨天|今天|前天|以前|曾经|过去|当时)\s*(?:我|我的).*"
    r"(?:说过|说了|发过|发言|提到|聊过|表现|最喜欢|"
    r"喜欢(?:看|听|玩|用|吃|喝|读|追)?什么|讨厌什么)"
)
_FIRST_PERSON_CLAIM_QUOTE_PATTERN = re.compile(
    r"^\s*(?:我|我的).{0,32}?"
    r"(?:自称|说过|说的|发过|发的|提过|提到|讲过|表示过).{0,32}?"
    r"(?:是哪条|哪句话|哪一句|什么时候|哪一次|哪个时候|原话)"
)
_FIRST_PERSON_CLAIM_TIME_PATTERN = re.compile(
    r"^\s*(?:我|我的).{0,16}?"
    r"(?:什么时候|哪一次|哪个时候|哪条|哪句话|哪一句)"
    r".{0,24}?(?:自称|说过|说的|发过|发的|提过|提到|讲过|表示过)"
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
        coverage_mode = self._coverage_mode(answer_mode, time_range)
        needs_history = bool(
            time_range
            or _FOLLOW_UP_PATTERN.search(original)
            or _HISTORY_PATTERN.search(original)
            or answer_mode in {"mention", "summary", "assessment"}
        )

        direct_reference = self._classify_direct_member_reference(
            original,
            group_members,
            exclude_user_ids=excluded_member_ids,
            has_time_range=time_range is not None,
        )
        if direct_reference.status == "resolved":
            direct_member = direct_reference.member
            if direct_member is None:
                raise RuntimeError("resolved group member reference is missing its member")
            direct_subject_ids = (str(direct_member.user_id),)
            return self._with_topic_query(ResolvedMemoryQuery(
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
            ), aliases=(direct_member.matched_alias,))
        if direct_reference.status == "ambiguous":
            if (
                self._rewrite_provider is not None
                and needs_history
                and self._looks_like_member_or_history_query(original)
            ):
                rewritten = self._try_rewrite(original, recent, current_time)
                if rewritten is not None:
                    constrained = self._constrain_rewritten_subject(
                        rewritten,
                        group_members=group_members,
                        original=original,
                        recent=recent,
                    )
                    if constrained is not None:
                        return replace(
                            constrained,
                            needs_history=True,
                            needs_detail=needs_detail,
                            time_range=time_range or constrained.time_range,
                            retrieval_mode=(
                                "temporal"
                                if time_range is not None
                                else constrained.retrieval_mode
                            ),
                            group_id=normalized_group_id,
                            requester_id=normalized_requester_id,
                            answer_mode=answer_mode,
                            coverage_mode=coverage_mode,
                        )
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
            return self._with_topic_query(ResolvedMemoryQuery(
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
            ), aliases=("我的", "我"))

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
            plan = ResolvedMemoryQuery(
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
            if quoted_message is None and speaker_ids:
                return self._with_topic_query(plan, aliases=entities)
            return plan

        if (
            self._rewrite_provider is not None
            and needs_history
            and self._looks_like_member_or_history_query(original)
        ):
            rewritten = self._try_rewrite(original, recent, current_time)
            if rewritten is not None:
                constrained = self._constrain_rewritten_subject(
                    rewritten,
                    group_members=group_members,
                    original=original,
                    recent=recent,
                )
                if constrained is not None:
                    return replace(
                        constrained,
                        needs_history=True,
                        needs_detail=needs_detail,
                        time_range=time_range or constrained.time_range,
                        retrieval_mode=(
                            "temporal"
                            if time_range is not None
                            else constrained.retrieval_mode
                        ),
                        group_id=normalized_group_id,
                        requester_id=normalized_requester_id,
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
    def _classify_direct_member_reference(
        query: str,
        group_members: Sequence[GroupMemberIdentity],
        *,
        exclude_user_ids: set[int] | frozenset[int],
        has_time_range: bool,
    ) -> GroupMemberReferenceResolution:
        if (
            _JOINED_RELATIVE_DAY_QUERY_PATTERN.search(query)
            or _SUBJECTLESS_RELATIVE_EVENT_PATTERN.search(query)
        ):
            return GroupMemberReferenceResolution("unbound")
        allowed_members = tuple(
            member
            for member in group_members
            if member.in_scope and int(member.user_id) not in exclude_user_ids
        )
        alias_resolution = classify_group_member_reference(
            query,
            allowed_members,
            match_mode="contained",
        )
        strong_resolution = MemoryQueryResolver._classify_prefix_member_alias(
            query,
            group_members,
            has_time_range=has_time_range,
            exclude_user_ids=exclude_user_ids,
        )
        if strong_resolution.status == "ambiguous":
            return strong_resolution
        if strong_resolution.status == "resolved":
            if strong_resolution.member is None:
                raise RuntimeError("resolved strong member reference is missing its member")
            if MemoryQueryResolver._has_unsafe_member_suffix(
                query,
                member=strong_resolution.member,
                group_members=group_members,
                has_time_range=has_time_range,
                exclude_user_ids=exclude_user_ids,
            ):
                return GroupMemberReferenceResolution("ambiguous")
            alias_resolution = strong_resolution
        if strong_resolution.status != "resolved":
            alias_resolution = MemoryQueryResolver._deprioritize_weak_aliases(
                query,
                allowed_members,
                resolution=alias_resolution,
                exclude_user_ids=exclude_user_ids,
            )
        if (
            alias_resolution.status == "resolved"
            and alias_resolution.member is not None
            and MemoryQueryResolver._has_restricted_alias_shadow(
                query,
                member=alias_resolution.member,
                group_members=group_members,
                exclude_user_ids=exclude_user_ids,
            )
        ):
            return GroupMemberReferenceResolution("ambiguous")
        for identity in allowed_members:
            for alias in dict.fromkeys((identity.group_card, identity.nickname)):
                normalized_alias = normalize_member_alias(alias)
                if len(normalized_alias) < 2:
                    continue
                if (
                    normalized_alias
                    in {
                        normalize_member_alias(value)
                        for value in _COMMON_WORDS
                    }
                    and not MemoryQueryResolver._weak_alias_is_explicit_member(
                        query,
                        group_members,
                    )
                ):
                    continue
                if normalized_alias not in normalize_member_alias(query):
                    continue
                if (
                    alias_resolution.status == "resolved"
                    and alias_resolution.member is not None
                    and int(identity.user_id) != int(alias_resolution.member.user_id)
                    and MemoryQueryResolver._alias_is_media_topic(
                        query,
                        alias=alias,
                    )
                ):
                    continue
                if MemoryQueryResolver._has_unsafe_member_suffix(
                    query,
                    member=ResolvedGroupMember(
                        user_id=int(identity.user_id),
                        matched_alias=alias,
                    ),
                    group_members=group_members,
                    has_time_range=has_time_range,
                    exclude_user_ids=exclude_user_ids,
                ):
                    return GroupMemberReferenceResolution("ambiguous")
        explicit_resolution = MemoryQueryResolver._classify_explicit_member_ids(
            query,
            group_members,
            exclude_user_ids=exclude_user_ids,
        )
        if explicit_resolution.status == "resolved":
            if explicit_resolution.member is None:
                raise RuntimeError("resolved explicit member reference is missing its member")
            remainder_resolution = MemoryQueryResolver._classify_prefix_member_alias(
                re.sub(
                    r"^\s*(?:@\s*|QQ\s*(?:号\s*)?[:：]?\s*)\d{5,12}\s*",
                    "",
                    query,
                    count=1,
                    flags=re.IGNORECASE,
                ),
                group_members,
                has_time_range=has_time_range,
                exclude_user_ids=exclude_user_ids,
            )
            if (
                remainder_resolution.status == "resolved"
                and remainder_resolution.member is not None
                and remainder_resolution.member.user_id
                != explicit_resolution.member.user_id
            ):
                return GroupMemberReferenceResolution("ambiguous")
            if (
                remainder_resolution.status == "resolved"
                and remainder_resolution.member is not None
                and remainder_resolution.member.user_id
                == explicit_resolution.member.user_id
            ):
                alias_resolution = remainder_resolution
            if MemoryQueryResolver._has_unsafe_member_suffix(
                query,
                member=explicit_resolution.member,
                group_members=group_members,
                has_time_range=has_time_range,
                exclude_user_ids=exclude_user_ids,
            ):
                return GroupMemberReferenceResolution("ambiguous")
        if MemoryQueryResolver._has_unknown_joined_member(
            query,
            group_members,
            exclude_user_ids=exclude_user_ids,
        ):
            return GroupMemberReferenceResolution("ambiguous")
        if explicit_resolution.status == "unbound":
            return alias_resolution
        if explicit_resolution.status == "ambiguous":
            return explicit_resolution
        if alias_resolution.status == "unbound":
            return explicit_resolution
        if alias_resolution.status == "ambiguous":
            return alias_resolution
        if explicit_resolution.member is None or alias_resolution.member is None:
            raise RuntimeError("resolved member reference is missing its member")
        if (
            normalize_member_alias(alias_resolution.member.matched_alias)
            in {
                normalize_member_alias(value)
                for value in _COMMON_WORDS
            }
            and not MemoryQueryResolver._weak_alias_is_explicit_member(
                query,
                group_members,
            )
        ):
            return explicit_resolution
        if explicit_resolution.member.user_id != alias_resolution.member.user_id:
            return GroupMemberReferenceResolution("ambiguous")
        return alias_resolution

    @staticmethod
    def _alias_is_media_topic(query: str, *, alias: str) -> bool:
        normalized_query = normalize_member_alias(query)
        normalized_alias = normalize_member_alias(alias)
        if not normalized_alias or not _ASSESSMENT_PATTERN.search(query):
            return False
        return any(
            f"{normalized_alias}{normalize_member_alias(suffix)}" in normalized_query
            for suffix in _TOPIC_CATEGORY_SUFFIXES
        )

    @staticmethod
    def _has_restricted_alias_shadow(
        query: str,
        *,
        member: ResolvedGroupMember,
        group_members: Sequence[GroupMemberIdentity],
        exclude_user_ids: set[int] | frozenset[int],
    ) -> bool:
        """Reject a target alias embedded inside a longer unavailable alias.

        The check is position-based instead of relying on a finite list of
        polite prefixes, so cross-group and excluded aliases cannot be made to
        bind a shorter target alias by adding arbitrary text before them.
        """

        normalized_query = normalize_member_alias(query)
        target_aliases = {
            normalize_member_alias(alias)
            for identity in group_members
            if identity.in_scope and int(identity.user_id) == int(member.user_id)
            for alias in (identity.group_card, identity.nickname)
            if normalize_member_alias(alias)
            and normalize_member_alias(alias) in normalized_query
        }
        if not target_aliases:
            return False
        target_spans = tuple(
            (target_alias, target_match.start(), target_match.end())
            for target_alias in target_aliases
            for target_match in re.finditer(re.escape(target_alias), normalized_query)
        )
        for identity in group_members:
            if identity.in_scope and int(identity.user_id) not in exclude_user_ids:
                continue
            for alias in dict.fromkeys((identity.group_card, identity.nickname)):
                restricted_alias = normalize_member_alias(alias)
                if not restricted_alias:
                    continue
                for restricted_match in re.finditer(
                    re.escape(restricted_alias),
                    normalized_query,
                ):
                    restricted_start, restricted_end = restricted_match.span()
                    if restricted_alias in target_aliases:
                        continue
                    if any(
                        len(target_alias) > len(restricted_alias)
                        and target_start <= restricted_start
                        and restricted_end <= target_end
                        for target_alias, target_start, target_end in target_spans
                    ):
                        # A shorter external prefix inside the complete target
                        # alias is only a weak collision, not a second person.
                        continue
                    return True
        return False

    @staticmethod
    def _classify_prefix_member_alias(
        query: str,
        group_members: Sequence[GroupMemberIdentity],
        *,
        has_time_range: bool,
        exclude_user_ids: set[int] | frozenset[int] = frozenset(),
    ) -> GroupMemberReferenceResolution:
        if query.lstrip().startswith("@") and not re.match(
            r"^\s*@\s*[^\d]",
            query,
        ):
            return GroupMemberReferenceResolution("unbound")
        normalized = normalize_member_alias(query)
        for prefix in (
            "如何评价",
            "怎么评价",
            "评价",
            "点评",
            "分析",
            "怎么看待",
            "怎么看",
            "说说",
            "请问",
            "关于",
            "帮我看看",
            "麻烦说说",
            "想问",
        ):
            normalized_prefix = normalize_member_alias(prefix)
            if normalized.startswith(normalized_prefix):
                normalized = normalized[len(normalized_prefix) :]
                break
        if has_time_range:
            for value in _RELATIVE_DAY_WORDS:
                normalized_day = normalize_member_alias(value)
                if normalized.startswith(normalized_day):
                    normalized = normalized[len(normalized_day) :]
                    break
        matches: list[tuple[GroupMemberIdentity, str, str]] = []
        for member in group_members:
            for alias in dict.fromkeys((member.group_card, member.nickname)):
                normalized_alias = normalize_member_alias(alias)
                if len(normalized_alias) >= 2 and normalized.startswith(normalized_alias):
                    matches.append((member, alias, normalized_alias))
        if not matches:
            return GroupMemberReferenceResolution("unbound")
        longest_length = max(len(item[2]) for item in matches)
        longest_matches = tuple(
            item for item in matches if len(item[2]) == longest_length
        )
        allowed_matches = tuple(
            item
            for item in longest_matches
            if item[0].in_scope and int(item[0].user_id) not in exclude_user_ids
        )
        if not allowed_matches:
            # A longer excluded or cross-group alias must shadow every shorter
            # target-group substring.  Otherwise "王小明..." could bind "小明".
            if any(not item[0].in_scope for item in longest_matches):
                return GroupMemberReferenceResolution("ambiguous")
            return GroupMemberReferenceResolution("unbound")
        matched_ids = {int(item[0].user_id) for item in allowed_matches}
        if len(matched_ids) != 1:
            return GroupMemberReferenceResolution("ambiguous")
        user_id = next(iter(matched_ids))
        aliases = [item[1] for item in allowed_matches]
        return GroupMemberReferenceResolution(
            "resolved",
            ResolvedGroupMember(
                user_id=user_id,
                matched_alias=max(aliases, key=lambda value: len(normalize_member_alias(value))),
            ),
        )

    @staticmethod
    @staticmethod
    def _has_unsafe_member_suffix(
        query: str,
        *,
        member: ResolvedGroupMember,
        group_members: Sequence[GroupMemberIdentity],
        has_time_range: bool,
        exclude_user_ids: set[int] | frozenset[int],
    ) -> bool:
        """Reject only on evidence, never on an unknown continuation.

        Default is single-subject (allow). Ambiguity requires either the alias
        embedded inside another member's longer alias, or a second known member /
        person pronoun after a joiner.
        """
        del has_time_range
        if MemoryQueryResolver._has_restricted_alias_shadow(
            query,
            member=member,
            group_members=group_members,
            exclude_user_ids=exclude_user_ids,
        ):
            return True
        return MemoryQueryResolver._has_unknown_joined_member(
            query,
            group_members,
            exclude_user_ids=exclude_user_ids,
        )

    def _classify_explicit_member_ids(
        query: str,
        group_members: Sequence[GroupMemberIdentity],
        *,
        exclude_user_ids: set[int] | frozenset[int],
    ) -> GroupMemberReferenceResolution:
        member_ids = {
            int(member.user_id)
            for member in group_members
            if member.in_scope
        }
        prefixed_ids = tuple(
            int(value)
            for value in re.findall(
                r"(?:@\s*|QQ\s*(?:号\s*)?[:：]?\s*)(\d{5,12})(?!\d)",
                query,
                flags=re.IGNORECASE,
            )
        )
        bare_ids = tuple(
            int(value)
            for value in re.findall(r"(?<![@\d])(\d{5,12})(?!\d)", query)
        )
        joined_bare_ids = bool(
            re.search(
                rf"\d{{5,12}}\s*{_MEMBER_JOINER_PATTERN}\s*\d{{5,12}}",
                query,
            )
        )
        explicit_ids = (
            *prefixed_ids,
            *(
                bare_ids
                if joined_bare_ids and any(value in member_ids for value in bare_ids)
                else tuple(value for value in bare_ids if value in member_ids)
            ),
        )
        if any(
            user_id not in member_ids or user_id in exclude_user_ids
            for user_id in explicit_ids
        ):
            return GroupMemberReferenceResolution("ambiguous")
        matched_ids = set(explicit_ids)
        excluded_mention_seen = False
        for mention in re.findall(r"@\s*([^\s@]{1,64})", query):
            if re.match(r"\d{5,12}(?!\d)", mention):
                continue
            normalized_mention = normalize_member_alias(mention)
            if normalized_mention in {
                normalize_member_alias(value)
                for value in _NON_PERSON_MEMORY_SUBJECTS
            }:
                continue
            mention_matches = tuple(
                (int(member.user_id), bool(member.in_scope), normalize_member_alias(alias))
                for member in group_members
                for alias in (member.group_card, member.nickname)
                if normalize_member_alias(alias)
                and normalized_mention.startswith(normalize_member_alias(alias))
            )
            if not mention_matches:
                return GroupMemberReferenceResolution("ambiguous")
            longest_length = max(len(item[2]) for item in mention_matches)
            mentioned_identities = {
                (user_id, in_scope)
                for user_id, in_scope, alias in mention_matches
                if len(alias) == longest_length
            }
            mentioned_members = {
                user_id
                for user_id, in_scope in mentioned_identities
                if in_scope
            }
            # An exact longest target-group alias wins over an equally named
            # cross-group alias, matching the bare-alias contract.  A longer
            # external alias already wins the longest-match filter above and
            # therefore leaves mentioned_members empty.
            if not mentioned_members:
                return GroupMemberReferenceResolution("ambiguous")
            if mentioned_members & set(exclude_user_ids):
                excluded_mention_seen = True
            allowed_mentioned = mentioned_members - set(exclude_user_ids)
            if allowed_mentioned and allowed_mentioned != mentioned_members:
                return GroupMemberReferenceResolution("ambiguous")
            matched_ids.update(allowed_mentioned)
        if excluded_mention_seen and matched_ids:
            return GroupMemberReferenceResolution("ambiguous")
        if not matched_ids:
            return GroupMemberReferenceResolution("unbound")
        if any(
            user_id not in member_ids or user_id in exclude_user_ids
            for user_id in matched_ids
        ):
            return GroupMemberReferenceResolution("ambiguous")
        matched_ids = {
            int(user_id)
            for user_id in matched_ids
        }
        if len(matched_ids) != 1:
            return GroupMemberReferenceResolution("ambiguous")
        user_id = next(iter(matched_ids))
        return GroupMemberReferenceResolution(
            "resolved",
            ResolvedGroupMember(user_id=user_id, matched_alias=str(user_id)),
        )

    @staticmethod
    def _deprioritize_weak_aliases(
        query: str,
        group_members: Sequence[GroupMemberIdentity],
        *,
        resolution: GroupMemberReferenceResolution,
        exclude_user_ids: set[int] | frozenset[int],
    ) -> GroupMemberReferenceResolution:
        if resolution.status == "unbound":
            return resolution

        # Historical QQ cards occasionally equal ordinary query vocabulary.
        # Such contained aliases are weak unless the syntax explicitly places
        # them in a member slot; unique prefix aliases are handled earlier.
        if MemoryQueryResolver._weak_alias_is_explicit_member(query, group_members):
            return resolution
        common_aliases = {
            normalize_member_alias(value)
            for value in _COMMON_WORDS
        }
        strong_members = tuple(
            GroupMemberIdentity(
                user_id=member.user_id,
                nickname=(
                    ""
                    if normalize_member_alias(member.nickname) in common_aliases
                    else member.nickname
                ),
                group_card=(
                    ""
                    if normalize_member_alias(member.group_card) in common_aliases
                    else member.group_card
                ),
            )
            for member in group_members
        )
        strong_resolution = classify_group_member_reference(
            query,
            strong_members,
            match_mode="contained",
            exclude_user_ids=exclude_user_ids,
        )
        if resolution.status == "ambiguous":
            return (
                strong_resolution
                if strong_resolution.status in {"resolved", "unbound"}
                else resolution
            )
        if resolution.member is None:
            raise RuntimeError("resolved member reference is missing its member")
        if normalize_member_alias(resolution.member.matched_alias) not in common_aliases:
            return resolution
        unfiltered_strong = classify_group_member_reference(
            query,
            strong_members,
            match_mode="contained",
        )
        if unfiltered_strong.status != "unbound":
            return GroupMemberReferenceResolution("ambiguous")
        # A bare weak vocabulary alias is not enough evidence to bind a person.
        # Let the normal person-query classifier decide whether another
        # unknown subject should fail closed.
        return GroupMemberReferenceResolution("unbound")

    @staticmethod
    def _weak_alias_is_explicit_member(
        query: str,
        group_members: Sequence[GroupMemberIdentity],
    ) -> bool:
        weak_words = {
            normalize_member_alias(value)
            for value in _COMMON_WORDS
        }
        if any(
            re.search(rf"@\s*{re.escape(value)}", query)
            for value in _COMMON_WORDS
        ):
            return True
        parts = re.split(rf"\s*{_MEMBER_JOINER_PATTERN}\s*", query)
        if len(parts) < 2:
            return False
        member_aliases = {
            normalize_member_alias(alias)
            for member in group_members
            for alias in (member.group_card, member.nickname)
            if normalize_member_alias(alias)
        }
        part_aliases = tuple(
            {
                alias
                for alias in member_aliases
                if alias and alias in normalize_member_alias(part)
            }
            for part in parts
        )
        has_weak_alias = any(aliases & weak_words for aliases in part_aliases)
        has_strong_alias = any(aliases - weak_words for aliases in part_aliases)
        return has_weak_alias and has_strong_alias

    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    def _has_unknown_joined_member(
        query: str,
        group_members: Sequence[GroupMemberIdentity],
        *,
        exclude_user_ids: set[int] | frozenset[int],
    ) -> bool:
        """Evidence-based second-person detection; no continuation whitelist.

        Rejects only when the question demonstrably involves a second person:
        a joiner followed by another member/pronoun/person-shaped token, a
        possessive relation noun, a speech target that is person-shaped, or two
        distinct member aliases used as people (media-topic uses such as
        "八仙动画" do not count).
        """
        all_aliases = sorted(
            {
                normalize_member_alias(alias)
                for member in group_members
                for alias in (member.group_card, member.nickname, str(member.user_id))
                if normalize_member_alias(alias)
            },
            key=len,
            reverse=True,
        )
        excluded_aliases = sorted(
            {
                normalize_member_alias(alias)
                for member in group_members
                if int(member.user_id) in exclude_user_ids or not member.in_scope
                for alias in (member.group_card, member.nickname, str(member.user_id))
                if normalize_member_alias(alias)
            },
            key=len,
            reverse=True,
        )
        common_aliases = {
            normalize_member_alias(value) for value in _COMMON_WORDS
        }
        common_aliases.update(
            normalize_member_alias(value)
            for value in (
                "发言",
                "动画",
                "电影",
                "漫画",
                "游戏",
                "小说",
                "综艺",
                "节目",
                "项目",
                "内容",
                "消息",
                "事情",
                "话题",
                "服务",
                "结果",
                "大家",
            )
        )
        person_pronouns = (
            "他们",
            "她们",
            "别人",
            "对方",
            "那位",
            "那家伙",
            "这位",
            "他",
            "她",
        )
        relation_placeholders = ("谁", "什么人", "哪个人", "哪位", "什么")

        def is_person_shaped(value: str) -> bool:
            if not value:
                return False
            if re.fullmatch(r"\d{5,12}", value):
                return True
            if not re.fullmatch(r"[\u4e00-\u9fff]{2,6}", value):
                return False
            return bool(
                _PERSON_SHAPE_SUFFIX_PATTERN.search(value)
                or re.match(r"^(?:小|老)[\u4e00-\u9fff]{1,5}$", value)
            )

        def looks_like_placeholder(value: str) -> bool:
            normalized = normalize_member_alias(value)
            return any(
                normalized == placeholder or normalized.startswith(placeholder)
                for placeholder in relation_placeholders
            )

        # 1) joiner followed by a second person.
        parts = re.split(
            r"\s*(?:和|与|跟|以及|还有|或|、|外加|再加上|并且|又和|也和|也与|也跟|再)\s*",
            query,
        )
        if len(parts) >= 2:
            for part in parts[1:]:
                stripped = re.sub(r"^[\s@]+", "", part)
                if not stripped or looks_like_placeholder(stripped):
                    continue
                normalized = normalize_member_alias(stripped)
                if any(normalized.startswith(alias) for alias in all_aliases):
                    return True
                if any(normalized.startswith(alias) for alias in excluded_aliases):
                    return True
                if stripped.startswith(person_pronouns):
                    return True
                head = re.match(r"(?:[\u4e00-\u9fff]{2,6}?(?=昨天|今天|前天|最近|以前|曾经|过去|当时|说|发|提|聊|做|发生|怎么|都|也|了|[，。？?！!\s]|$)|\d{5,12})", stripped)
                if head is not None and is_person_shaped(head.group()):
                    return True

        # 2) possessive relation noun followed by a person/statement target.
        if re.search(
            r"的(?:朋友|同学|室友|老师|兄弟|姐妹|同事|邻居|家人|对象|好友|"
            r"猫主人|原主人|学长|学姐|师傅|老板|客户|女友|男友|老婆|老公)"
            r"[^，。？?！!\n]{0,24}"
            r"(?:说|发|提|聊|消息|发言|昨天|今天|最近|之前|做|发生|都|"
            r"[，。？?！!\s]|$)",
            query,
        ):
            return True

        # 3) speech/mention target that is another member or person-shaped.
        known_or_pronoun = "|".join(re.escape(alias) for alias in all_aliases)
        if known_or_pronoun:
            speech = re.search(
                r"(?:说|提到|问到|提起)(?P<target>(?:" + known_or_pronoun + r")"
                r"|他|她|他们|她们|[\u4e00-\u9fff]{2,6}?(?=昨天|今天|前天|最近|以前|说|发|提|聊|做|发生|都|[，。？?！!\s]|$)|\d{5,12})",
                query,
            )
            if speech is not None:
                target = speech.group("target")
                if (
                    not looks_like_placeholder(target)
                    and not any(
                        normalize_member_alias(target) == alias
                        and normalize_member_alias(alias) in common_aliases
                        for alias in all_aliases
                    )
                ):
                    if (
                        any(
                            normalize_member_alias(target).startswith(alias)
                            for alias in all_aliases
                        )
                        or target.startswith(person_pronouns)
                        or is_person_shaped(target)
                    ):
                        return True

        # 4) two distinct member aliases used as people (not media topics).
        normalized_query = normalize_member_alias(query)
        present_by_id: dict[int, set[str]] = {}
        for member in group_members:
            if not member.in_scope or int(member.user_id) in exclude_user_ids:
                continue
            user_id = int(member.user_id)
            for alias in dict.fromkeys((member.group_card, member.nickname)):
                normalized_alias = normalize_member_alias(alias)
                if (
                    not normalized_alias
                    or normalized_alias in common_aliases
                    or normalized_alias not in normalized_query
                ):
                    continue
                person_use = False
                for match in re.finditer(re.escape(normalized_alias), normalized_query):
                    following = normalized_query[match.end() :]
                    if any(
                        following.startswith(normalize_member_alias(suffix))
                        for suffix in _TOPIC_CATEGORY_SUFFIXES
                    ):
                        continue
                    person_use = True
                    break
                if person_use:
                    present_by_id.setdefault(user_id, set()).add(normalized_alias)
        if len(present_by_id) >= 2:
            return True
        return False

    @staticmethod
    def _looks_like_member_or_history_query(query: str) -> bool:
        return bool(
            _HISTORY_PATTERN.search(query)
            or _FOLLOW_UP_PATTERN.search(query)
            or _ASSESSMENT_PATTERN.search(query)
            or MemoryQueryResolver._is_person_memory_query(query)
        )

    def _constrain_rewritten_subject(
        self,
        rewritten: ResolvedMemoryQuery,
        *,
        group_members: Sequence[GroupMemberIdentity],
        original: str,
        recent: tuple[RecentMemoryMessage, ...],
    ) -> ResolvedMemoryQuery | None:
        """Keep rewrite subject strictly inside the group and the question.

        A rewrite may never invent a personal subject. It must either mention
        the member by alias/QQ in the question or resolve a follow-up pronoun
        to a recent speaker in this group; otherwise the subject is rejected.
        """
        subject_ids = rewritten.subject_ids
        if not subject_ids:
            return rewritten
        allowed_ids: set[str] = set()
        aliases_by_id: dict[str, set[str]] = {}
        for member in group_members:
            if not member.in_scope:
                continue
            user_id = str(member.user_id)
            allowed_ids.add(user_id)
            aliases_by_id.setdefault(user_id, set()).update(
                normalize_member_alias(alias)
                for alias in (member.nickname, member.group_card)
                if normalize_member_alias(alias)
            )
        if allowed_ids and not set(subject_ids) <= allowed_ids:
            return None
        normalized_original = normalize_member_alias(original)
        for subject_id in subject_ids:
            if str(subject_id) in original:
                continue
            aliases = aliases_by_id.get(str(subject_id), ())
            if any(alias and alias in normalized_original for alias in aliases):
                continue
            if _FOLLOW_UP_PRONOUN_PATTERN.search(original):
                recent_ids = [
                    str(message.user_id)
                    for message in reversed(recent)
                    if not message.blocked
                    and message.user_id is not None
                    and not isinstance(message.user_id, bool)
                ]
                if str(subject_id) in recent_ids:
                    continue
            return None
        return rewritten

    @staticmethod
    def _is_person_memory_query(query: str) -> bool:
        if query.lstrip().startswith(_SUBJECTLESS_MEMORY_QUERY_PREFIXES):
            return False
        if _JOINED_RELATIVE_DAY_QUERY_PATTERN.search(query):
            return False
        for pattern in (
            _PERSON_MEMORY_SUBJECT_PATTERN,
            _REMEMBER_PERSON_PATTERN,
            _PERSON_SPEECH_SUBJECT_PATTERN,
            _PERSON_ASSESSMENT_SUBJECT_PATTERN,
            _PERSON_OPINION_SUBJECT_PATTERN,
        ):
            match = pattern.search(query)
            if match is not None and match.group("subject") not in _NON_PERSON_MEMORY_SUBJECTS:
                return True
        return False

    @staticmethod
    def _is_first_person_subject(query: str) -> bool:
        claim_match = (
            _FIRST_PERSON_CLAIM_QUOTE_PATTERN.search(query)
            or _FIRST_PERSON_CLAIM_TIME_PATTERN.search(query)
        )
        if claim_match is not None:
            start, end = claim_match.span()
            if re.search(r"[你您他她]", query[start:end]):
                return False
            return True
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
    def _coverage_mode(
        answer_mode: AnswerMode,
        time_range: TimeRange | None,
    ) -> CoverageMode:
        if answer_mode == "summary" or (
            answer_mode == "assessment" and time_range is not None
        ):
            return "time_buckets"
        if answer_mode == "dated_history":
            return "chronological"
        return "relevance"

    @staticmethod
    def _with_topic_query(
        plan: ResolvedMemoryQuery,
        *,
        aliases: Sequence[str],
    ) -> ResolvedMemoryQuery:
        topic = plan.original_query
        removed: list[str] = []
        for alias in sorted(
            {value.strip() for value in aliases if value and value.strip()},
            key=len,
            reverse=True,
        ):
            updated, count = re.subn(re.escape(alias), " ", topic, flags=re.IGNORECASE)
            if count:
                topic = updated
                removed.append(alias)

        scaffold = {
            "assessment": _ASSESSMENT_SCAFFOLD_PATTERN,
            "current_fact": _CURRENT_FACT_SCAFFOLD_PATTERN,
            "general_history": _HISTORY_SCAFFOLD_PATTERN,
            "summary": _SUMMARY_SCAFFOLD_PATTERN,
        }.get(plan.answer_mode)
        if scaffold is not None:
            topic = scaffold.sub(" ", topic)
        topic = re.sub(r"^\s*(?:请|麻烦|帮忙|帮我|能否|可以)?\s*", "", topic)
        topic = re.sub(r"^的|的$", "", topic.strip())
        topic = re.sub(r"^(?:对|关于)\s*", "", topic).strip()
        topic = _TOPIC_PUNCTUATION_PATTERN.sub("", topic).strip()
        topic = re.sub(r"\s+", " ", topic)
        topic_query = topic or None
        topic_terms: list[str] = []
        if topic_query is not None:
            for term in _TOPIC_TERM_SPLIT_PATTERN.split(topic_query):
                if not term:
                    continue
                topic_terms.append(term)
                for suffix in _TOPIC_CATEGORY_SUFFIXES:
                    if not term.endswith(suffix):
                        continue
                    core = term[: -len(suffix)].strip()
                    if len(core) >= 2:
                        topic_terms.append(core)
                    break
        return replace(
            plan,
            retrieval_query=topic_query or plan.original_query,
            topic_query=topic_query,
            topic_terms=tuple(dict.fromkeys(topic_terms)),
            topic_extraction="deterministic" if topic_query is not None else "none",
            subject_aliases_removed=tuple(removed),
        )

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

        # An explicit QQ quote is itself a deterministic reference. Textual
        # follow-up markers are only required when inferring a reference from
        # the recent-message window without transport-level quote metadata.
        if not _FOLLOW_UP_PATTERN.search(query):
            return None

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
            topic_query=retrieval_query.strip(),
            topic_terms=(retrieval_query.strip(),),
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
