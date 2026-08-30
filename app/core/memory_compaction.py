"""Structured, source-backed compaction helpers for group memory.

This module deliberately has no storage or LLM dependency.  Callers provide
the model response and the source message ids that were actually supplied to
the model; the parser then keeps only facts that remain attributable to those
messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


_ALLOWED_KINDS = frozenset(
    {"fact", "preference", "taboo", "plan", "decision", "profile", "relationship", "event", "running_joke", "current", "expired"}
)

KIND_SEMANTIC_GUIDANCE_EN = (
    "Kind semantics - use exactly one of:\n"
    "- decision: a user's final/confirmed choice (bought X, cancelled Y, quit Z). "
    "Never addressing rules, opinions, or chit-chat.\n"
    "- preference: likes/dislikes, opinions, and behavior rules toward the bot; "
    "addressing rules ('call me master from now on') belong here with predicate starting with 'addressing rule'.\n"
    "- plan: intended future action. current: what someone is doing now. event: what happened. "
    "profile: explicit, relatively stable identity or biographical attributes only; profile is not a fallback category. "
    "Likes, dislikes, opinions, temporary activity, purchases, and one-off experiences belong in preference, taboo, current, event, or fact as appropriate. "
    "taboo: must-not topics. relationship: who is who to whom. "
    "running_joke: recurring in-group joke. fact: durable plain fact.\n"
    "When a user explicitly says they are currently watching, following, or catching up on a work, "
    "record it as current even if the sentence describes watching it with someone. "
    "Merely discussing a plot, character, or season is not enough to infer current viewing or preference.\n"
    "Addressing rules may only be recorded when the requester is the person the rule applies to; "
    "never record one member changing how the bot addresses another member.\n"
)

KIND_SEMANTIC_GUIDANCE_ZH = (
    "kind 语义（只能选一个）：\n"
    "- decision：用户已拍板的决定（买了X/取消了Y/不玩Z）；不能放称呼规则、观点或闲聊。\n"
    "- preference：喜欢/讨厌/观点，以及针对机器人的行为约定；称呼规则放这里，predicate 以“称呼规则”开头。\n"
    "- plan：打算做的未来事项；current：正在做的事；event：发生过的事；"
    "profile：仅限用户明确表达、相对稳定的身份或履历属性，profile 不是兜底分类；"
    "喜好、厌恶、观点、临时活动、购买行为和单次经历应分别归入 preference、taboo、current、event 或 fact；"
    "taboo：禁区；relationship：人物关系；running_joke：群内固定梗；fact：普通持久事实。\n"
    "用户明确说正在看、追或补某部作品时，应记录为 current；即使句子同时说和某人一起观看，也仍是 current。"
    "只讨论剧情、角色或季度不足以推断其正在观看或喜欢该作品。\n"
    "称呼规则只能记录提出者本人适用的（subject=提出者=被称呼对象）；禁止记录帮别人改称呼的规则。\n"
)

_ADDRESSING_RULE_MARKERS = re.compile(r"(?:称呼|统一改为|以后|回复时|请叫我|叫我)", re.IGNORECASE)
_ADDRESSING_TARGET_QQ_PATTERN = re.compile(r"(?:对用户|对|针对)\s*[“\"']?\s*(\d{5,12})")

_MAX_SUMMARY_CHARS = 2_000
_MAX_FIELD_CHARS = 600
_MAX_PERSON_ACTION_FACT_CHARS = 320
_MAX_FACTS = 64
_MAX_INVALIDATIONS = 32
_ROLLING_PREFIX = re.compile(
    r"^\s*(?:(?:rolling group memory|structured memory(?: digest)?|memory digest|summary)\s*:\s*|(?:滚动群记忆|结构化记忆|记忆摘要|摘要)\s*[：:]\s*)",
    re.IGNORECASE,
)
_DIGEST_SUMMARY = re.compile(
    r"^\s*(?:memory digest|structured memory(?: digest)?)\s*:\s*\r?\n\s*summary\s*:\s*(.*?)(?:\r?\n\s*facts\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE = re.compile(r"\s+")
_COLLECTIVE_PATTERN = re.compile(r"(?:大家|我们|群里|群内|全员|\bwe\b|\bour\b|\bgroup\b|\beveryone\b)", re.IGNORECASE)
_ENUMERATION_MARKER = re.compile(
    r"(?:^|\s)(?:\d{1,3}\s*[.、:：)）]|[一二三四五六七八九十]{1,3}\s*[.、:：)）])"
)
_SINGLE_VALUE_PROFILE_ATTRIBUTE_ALIASES = {
    "age": frozenset({"age", "年龄", "岁数"}),
    "nationality": frozenset({"nationality", "国籍"}),
    "origin": frozenset({"hometown", "origin", "籍贯", "家乡", "老家", "出生地"}),
    "location": frozenset({"residence", "location", "所在地", "居住地", "常住地"}),
}
_AGE_VALUE_PATTERN = re.compile(r"(?P<age>\d{1,3})\s*岁")
_FUTURE_AGE_PATTERN = re.compile(
    r"(?:等到?|到了?|到|等)\s*(?:\d{1,3}|[零〇一二两三四五六七八九十]{1,3})\s*岁|"
    r"(?:\d{1,3}|[零〇一二两三四五六七八九十]{1,3})\s*岁.{0,8}?(?:再|以后|之后|的时候)"
)
_EXPLICIT_DENIAL_PATTERN = re.compile(
    r"不是|并非|才不是|没有|没看过|没读过|不看|不喜欢|说错|错误|不对|假的|作废|别再说|别乱说|哪有|什么时候.{0,12}(?:岁|\d)"
)
_CJK_TERM_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,}")
_CHINESE_AGE_PATTERN = re.compile(
    r"(?P<age>[零〇一二两三四五六七八九十]{1,3})\s*岁"
)
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


@dataclass(frozen=True, slots=True)
class MemoryFact:
    """One compact fact with only fields safe to persist or render."""

    kind: str
    subject_id: str
    predicate: str
    object_text: str
    content: str
    importance: int
    confidence: float
    source_msg_ids: tuple[str, ...]
    valid_until: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryInvalidation:
    """A source-backed request to retire one exact active canonical fact."""

    target_canonical_key: str
    source_msg_ids: tuple[str, ...]
    reason: str = "explicit_denial"
    valid_until: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryCompaction:
    """Validated model result. Facts are source-backed and de-duplicated."""

    summary: str
    facts: tuple[MemoryFact, ...] = ()
    invalidations: tuple[MemoryInvalidation, ...] = ()
    rejected_fact_count: int = 0
    rejected_invalidation_count: int = 0


def is_addressing_rule(*values: object) -> bool:
    """Return whether persisted fact fields describe a bot-addressing rule."""

    return _ADDRESSING_RULE_MARKERS.search(
        " ".join(str(value or "") for value in values)
    ) is not None


def is_single_value_profile_attribute(predicate: object) -> bool:
    """Whether a newer profile value should replace the older same attribute."""

    return bool(single_value_profile_attribute_predicates(predicate))


def single_value_profile_attribute_predicates(predicate: object) -> tuple[str, ...]:
    """Return all exact predicate spellings that share one profile slot."""

    normalized = str(predicate or "").strip().casefold()
    for aliases in _SINGLE_VALUE_PROFILE_ATTRIBUTE_ALIASES.values():
        if normalized in {value.casefold() for value in aliases}:
            return tuple(sorted(aliases))
    return ()


def _canonical_profile_predicate(predicate: object) -> str:
    normalized = str(predicate or "").strip().casefold()
    for slot, aliases in _SINGLE_VALUE_PROFILE_ATTRIBUTE_ALIASES.items():
        if normalized in {value.casefold() for value in aliases}:
            return slot
    return str(predicate or "")


def derive_explicit_memory_invalidations(
    *,
    messages: Sequence[Mapping[str, Any]],
    active_correction_targets: Sequence[Mapping[str, Any]],
) -> tuple[MemoryInvalidation, ...]:
    """Derive only direct, same-subject denials of an exact catalog object."""

    sources_by_target: dict[str, set[str]] = {}
    for message in messages:
        source_id = _clean_text(
            message.get("source_msg_id") or message.get("platform_msg_id"),
            limit=128,
        )
        subject_id = _clean_text(message.get("user_id"), limit=128)
        text = _clean_text(
            message.get("plain_text") or message.get("text") or message.get("content"),
            limit=_MAX_SUMMARY_CHARS,
        )
        if not source_id or not subject_id or not _EXPLICIT_DENIAL_PATTERN.search(text):
            continue
        for target in active_correction_targets:
            if str(target.get("subject_id") or "") != subject_id:
                continue
            if str(target.get("memory_kind") or "") not in {"profile", "preference"}:
                continue
            key = _clean_text(target.get("target_canonical_key"), limit=255)
            object_text = _clean_text(target.get("object_text"), limit=_MAX_FIELD_CHARS)
            if not key or not object_text:
                continue
            if _message_explicitly_denies_target(text=text, target=target):
                sources_by_target.setdefault(key, set()).add(source_id)
    return tuple(
        MemoryInvalidation(
            target_canonical_key=key,
            source_msg_ids=tuple(sorted(source_ids)),
        )
        for key, source_ids in sorted(sources_by_target.items())
    )


def _message_explicitly_denies_target(
    *, text: str, target: Mapping[str, Any]
) -> bool:
    """Match a denial locally to its target; fail closed on negated negations."""

    object_text = _clean_text(target.get("object_text"), limit=_MAX_FIELD_CHARS)
    predicate = _clean_text(target.get("predicate"), limit=_MAX_FIELD_CHARS)
    memory_kind = str(target.get("memory_kind") or "")
    if not object_text or memory_kind not in {"profile", "preference"}:
        return False
    if re.search(
        r"(?:没有|没|不是)\s*(?:说过?|表示|承认)?\s*(?:不喜欢|不看|不读|不是)|"
        r"(?:其实|仍然|还是|就是|确实)\s*(?:喜欢|在看|是)",
        text,
    ):
        return False

    age_values = _extract_age_values(object_text)
    if age_values and (
        memory_kind == "profile"
        or re.search(r"age|年龄|岁数|年龄阶段", predicate, re.IGNORECASE)
    ):
        return any(_directly_denies_age(text, age) for age in age_values)

    terms = _target_lexical_terms(object_text)
    if not terms:
        return False
    for term in terms:
        escaped = re.escape(term)
        if re.search(
            rf"(?:我|本人)?\s*(?:不是|并非|才不是|没看过|没读过|不看|不读|不喜欢|不爱)"
            rf".{{0,10}}{escaped}|{escaped}.{{0,10}}(?:不是真的|是假的|不对|错误|说错|作废)",
            text,
            re.IGNORECASE,
        ):
            return True
    return False


def _directly_denies_age(text: str, age: int) -> bool:
    aliases = {str(age), _format_chinese_number(age)}
    alias_pattern = "(?:" + "|".join(re.escape(value) for value in aliases) + ")"
    if re.search(
        rf"(?:我|本人)\s*(?:今年)?\s*(?:就是|确实是|是)\s*{alias_pattern}\s*岁",
        text,
    ):
        return False
    return re.search(
        rf"(?:我|本人)?\s*(?:不是|并非|才不是|没到|不到|不满)\s*{alias_pattern}\s*岁|"
        rf"(?:我|本人).{{0,6}}(?:哪有|什么时候).{{0,6}}{alias_pattern}(?:\s*岁|了|啊|呀|呢|[？?！!]|$)|"
        rf"{alias_pattern}\s*岁.{{0,6}}(?:不对|错误|假的|说错)|"
        rf"(?:不对|错误|说错).{{0,8}}(?:年近|接近)?\s*{alias_pattern}\s*(?:岁)?\s*(?:哪来的|从哪来)",
        text,
    ) is not None


def _target_lexical_terms(object_text: str) -> tuple[str, ...]:
    normalized = _canonical_part(object_text)
    terms: set[str] = {normalized} if len(normalized) >= 2 else set()
    for raw_term in _CJK_TERM_PATTERN.findall(object_text):
        term = raw_term.casefold()
        if len(term) >= 2:
            terms.add(term)
        if len(term) >= 4:
            terms.update(term[index : index + 4] for index in range(len(term) - 3))
    return tuple(sorted(terms, key=lambda value: (-len(value), value)))


def _format_chinese_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value < 20:
        return "十" + (digits[value % 10] if value % 10 else "")
    if value < 100:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    return str(value)


def _extract_age_values(text: object) -> set[int]:
    """Extract comparable Arabic/Chinese ages without broad semantic guessing."""

    value = str(text or "")
    ages = {
        int(match)
        for match in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", value)
        if 0 < int(match) < 130
    }
    for match in _CHINESE_AGE_PATTERN.finditer(value):
        parsed = _parse_chinese_number(match.group("age"))
        if parsed is not None and 0 < parsed < 130:
            ages.add(parsed)
    return ages


def _parse_chinese_number(value: str) -> int | None:
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = _CHINESE_DIGITS.get(left, 1) if left else 1
        units = _CHINESE_DIGITS.get(right, 0) if right else 0
        return tens * 10 + units
    digits = [_CHINESE_DIGITS.get(char) for char in value]
    if not digits or any(digit is None for digit in digits):
        return None
    return int("".join(str(digit) for digit in digits))


def canonical_key(kind: str, subject_id: str, predicate: str, object_text: str) -> str:
    """Return a stable fact identity insensitive to case, spacing and Unicode form."""

    if str(kind or "").strip().casefold() == "profile":
        predicate = _canonical_profile_predicate(predicate)
    return "|".join(_canonical_part(part) for part in (kind, subject_id, predicate, object_text))


def parse_memory_compaction_response(
    raw: str | bytes | Mapping[str, Any] | None,
    *,
    allowed_source_msg_ids: Iterable[str] | None = None,
    allowed_subject_ids: Iterable[str] | None = None,
    source_subject_ids: Mapping[str, str] | None = None,
    source_contents: Mapping[str, str] | None = None,
    allowed_invalidation_targets: Mapping[str, Mapping[str, str]] | None = None,
    fallback_text: str = "",
    strict: bool = False,
) -> MemoryCompaction:
    """Parse a model result without trusting schema extensions or source citations.

    Passing ``allowed_source_msg_ids`` makes source validation strict: a fact
    citing even one unknown message id is discarded. Invalid JSON and invalid
    top-level shapes return a summary-only fallback instead of raising.
    """

    fallback = _clean_text(fallback_text, limit=_MAX_SUMMARY_CHARS)
    payload = _load_json_object(raw)
    if payload is None:
        if strict:
            raise ValueError("memory compaction response must be a JSON object")
        return MemoryCompaction(summary=fallback)
    if strict and (not isinstance(payload.get("summary"), str) or not isinstance(payload.get("facts"), list)):
        raise ValueError("memory compaction response has an invalid schema")
    if strict and not _clean_text(payload.get("summary"), limit=_MAX_SUMMARY_CHARS):
        raise ValueError("memory compaction response summary must not be blank")

    summary = _clean_text(payload.get("summary"), limit=_MAX_SUMMARY_CHARS) or fallback
    allowed_sources = None
    if allowed_source_msg_ids is not None:
        allowed_sources = {source for item in allowed_source_msg_ids if (source := _clean_text(item, limit=128))}
    allowed_subjects = None
    if allowed_subject_ids is not None:
        allowed_subjects = {subject for item in allowed_subject_ids if (subject := _clean_text(item, limit=128))}

    parsed: list[MemoryFact] = []
    rejected_fact_count = 0
    candidate_facts = payload.get("facts")
    if isinstance(candidate_facts, list):
        for candidate in candidate_facts[:_MAX_FACTS]:
            candidate = _normalize_fact_candidate(candidate)
            fact = _parse_fact(
                candidate,
                allowed_sources=allowed_sources,
                allowed_subjects=allowed_subjects,
                source_subject_ids=source_subject_ids,
                source_contents=source_contents,
            )
            if fact is None:
                rejected_fact_count += 1
                continue
            parsed.append(fact)

    invalidations: list[MemoryInvalidation] = []
    rejected_invalidation_count = 0
    candidate_invalidations = payload.get("invalidations", [])
    if isinstance(candidate_invalidations, list):
        for candidate in candidate_invalidations[:_MAX_INVALIDATIONS]:
            invalidation = _parse_invalidation(
                candidate,
                allowed_sources=allowed_sources,
                source_subject_ids=source_subject_ids,
                allowed_targets=allowed_invalidation_targets,
            )
            if invalidation is None:
                rejected_invalidation_count += 1
                continue
            invalidations.append(invalidation)
    elif candidate_invalidations is not None:
        rejected_invalidation_count = 1

    deduped_invalidations = {
        item.target_canonical_key: item for item in invalidations
    }
    return MemoryCompaction(
        summary=summary,
        facts=_dedupe_facts(parsed),
        invalidations=tuple(deduped_invalidations.values()),
        rejected_fact_count=rejected_fact_count,
        rejected_invalidation_count=rejected_invalidation_count,
    )


def _normalize_fact_candidate(candidate: Any) -> Any:
    if not isinstance(candidate, Mapping):
        return candidate
    normalized = dict(candidate)
    valid_until = normalized.get("valid_until")
    if valid_until is not None and _parse_valid_until(valid_until) is None:
        normalized["valid_until"] = None
    importance = normalized.get("importance")
    if isinstance(importance, (int, float)) and not isinstance(importance, bool):
        normalized["importance"] = max(1, min(5, int(round(importance))))
    confidence = normalized.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        normalized["confidence"] = max(0.0, min(1.0, float(confidence)))
    sources = normalized.get("source_msg_ids")
    if isinstance(sources, list):
        normalized["source_msg_ids"] = [
            str(source) if isinstance(source, (int, float)) and not isinstance(source, bool) else source
            for source in sources
        ]
    if not _clean_text(normalized.get("content"), limit=_MAX_FIELD_CHARS):
        subject = _clean_text(normalized.get("subject_id"), limit=128)
        predicate = _clean_text(normalized.get("predicate"), limit=128)
        object_text = _clean_text(normalized.get("object_text"), limit=_MAX_FIELD_CHARS)
        if subject and predicate and object_text:
            normalized["content"] = f"{subject}: {predicate} {object_text}"
    return normalized


def build_memory_compaction_prompt(
    *,
    messages: Sequence[Mapping[str, Any]],
    previous_digest: str = "",
    language: str = "zh",
    active_correction_targets: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Build a bounded, bilingual-capable prompt for one compact JSON result."""

    normalized_language = language.lower().strip()
    if normalized_language not in {"zh", "en"}:
        raise ValueError("language must be 'zh' or 'en'")

    message_lines = _format_prompt_messages(messages)
    previous = _strip_rolling_prefix(previous_digest)
    if normalized_language == "zh":
        instructions = (
            "Compact the chat into auditable structured memory and write summary and fact content in Chinese. "
            "Output exactly one compact JSON object with no Markdown or explanation.\n"
            "The object must contain summary (a non-empty string), facts (an array), and invalidations (an array). Each fact may contain only kind, "
            "subject_id, predicate, object_text, content, importance, confidence, source_msg_ids, valid_until.\n"
            "For a user fact, subject_id must be the numeric user_id of the author and every cited source must be written by that user. "
            "Use subject_id=group only for explicitly collective facts; cite at least two authors unless the source explicitly says everyone, the group, or we.\n"
            "Use only fact, preference, taboo, plan, decision, profile, relationship, event, running_joke, current, or expired as kind. "
            "Every fact needs at least one exact source_msg_id from the messages below. Never invent a source. "
            "If any field is uncertain, omit that fact instead of guessing. Return facts=[] when there is no durable fact.\n"
            "For profile facts, require a direct declarative self-statement by that user. Never infer age from a future or hypothetical phrase such as '40岁再看', and never turn a question, joke, quotation, bot reply, another member's claim, or a denied claim into an active profile fact.\n"
            "When the user explicitly denies or retracts an older fact repeated in this window, emit kind=expired with the same subject_id, predicate, and object_text as the rejected fact and cite the user's denial message. If the rejected target is not unique, omit the expiration. When the user also supplies a corrected stable value, emit a new profile fact for that value as well.\n"
            "Prefer invalidations over kind=expired when the exact target is listed in Active correction targets below. Each invalidation may contain only target_canonical_key, source_msg_ids, reason, valid_until; copy target_canonical_key exactly, use reason=explicit_denial, and cite only the target user's direct denial. Never invent or approximately match a target key.\n"
            "importance must be an integer from 1 to 5, confidence a number from 0 to 1, and valid_until an ISO date/time or null. "
            "The previous digest is context only and is never evidence."
        )
        lines = [instructions]
        lines.append(KIND_SEMANTIC_GUIDANCE_ZH)
        if previous:
            lines.extend(("Previous digest (context only, not evidence):", previous))
        if active_correction_targets:
            lines.append("Active correction targets (catalog only, not evidence):")
            lines.extend(
                json.dumps(dict(target), ensure_ascii=False, sort_keys=True)
                for target in active_correction_targets
            )
        lines.append("Citable messages:")
        lines.extend(message_lines or ["(none)"])
        return "\n".join(lines)
    if normalized_language == "zh":
        instructions = (
            "将聊天记录压缩为可审计的结构化记忆。只输出一个紧凑 JSON 对象，不要 Markdown 或解释。\n"
            "JSON 只能包含 summary 和 facts；每个 fact 只能包含 kind、subject_id、predicate、object_text、content、"
            "importance、confidence、source_msg_ids、valid_until。subject_id 必须使用消息中给出的 user_id 数字，群级事实使用 group。\n"
            "facts 按 kind + subject_id + predicate + object_text 去重。每个 fact 必须保留至少一个下方给出的 source_msg_ids，"
            "不得编造来源。当前事实使用语义 kind（fact、preference、taboo、plan、decision、profile）；已失效或被替代的事实"
            "使用 kind=expired，并在已知时填写 valid_until。不要把旧摘要当作新证据。\n"
            "importance 为 1 到 5 的整数，confidence 为 0 到 1 的数字，valid_until 为 ISO 日期/时间或 null。"
        )
        history_label = "既有摘要（仅供压缩上下文，不是证据）"
        messages_label = "可引用消息"
    else:
        instructions = (
            "Compact the chat into auditable structured memory. Output exactly one compact JSON object, with no Markdown or explanation.\n"
            "The object may contain only summary and facts. Each fact may contain only kind, subject_id, predicate, object_text, content, "
            "importance, confidence, source_msg_ids, valid_until.\n"
            "Deduplicate facts by kind + subject_id + predicate + object_text. Every fact must retain at least one source_msg_ids value from the "
            "messages below; never invent sources. Use semantic kinds (fact, preference, taboo, plan, decision, profile) for current facts. "
            "Use kind=expired for facts that are no longer current and set valid_until when known. Do not treat the previous digest as new evidence.\n"
            "importance is an integer from 1 to 5, confidence is a number from 0 to 1, and valid_until is an ISO date/time or null."
        )
        history_label = "Previous digest (context only, not evidence)"
        messages_label = "Citable messages"

    lines = [instructions]
    lines.append(
        KIND_SEMANTIC_GUIDANCE_ZH
        if normalized_language == "zh"
        else KIND_SEMANTIC_GUIDANCE_EN
    )
    lines.append(
        "重要：不要从机器人（小町/助手）自己的发言中提取用户个人事实；"
        "个人事实的 subject 必须是真实群成员，且该成员是引用来源的作者。"
        "机器人发言最多作为群级事实的辅助证据，不能作为个人事实的来源。"
    )
    if previous:
        lines.extend((f"{history_label}:", previous))
    lines.append(f"{messages_label}:")
    lines.extend(message_lines or ["(none)"])
    return "\n".join(lines)


def structured_digest(text: str = "", facts: Iterable[MemoryFact] = ()) -> str:
    """Render a deterministic digest without recursively embedding rolling labels."""

    summary = _strip_rolling_prefix(text)
    lines = ["Memory digest:", f"summary: {summary or '(empty)'}", "facts:"]
    normalized_facts = sorted(_dedupe_facts(fact for fact in facts if isinstance(fact, MemoryFact)), key=_fact_sort_key)
    if not normalized_facts:
        lines.append("- (none)")
        return "\n".join(lines)

    for fact in normalized_facts:
        sources = ",".join(sorted(set(fact.source_msg_ids)))
        until = fact.valid_until or "null"
        lines.append(
            f"- {fact.kind} | {fact.subject_id} | {fact.predicate} | {fact.object_text} | "
            f"{fact.content} | sources={sources} | valid_until={until}"
        )
    return "\n".join(lines)


def _load_json_object(raw: str | bytes | Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _parse_fact(
    candidate: Any,
    *,
    allowed_sources: set[str] | None,
    allowed_subjects: set[str] | None,
    source_subject_ids: Mapping[str, str] | None,
    source_contents: Mapping[str, str] | None,
) -> MemoryFact | None:
    if not isinstance(candidate, Mapping):
        return None
    kind = _clean_text(candidate.get("kind"), limit=32).lower()
    subject_id = _clean_text(candidate.get("subject_id"), limit=128)
    predicate = _clean_text(candidate.get("predicate"), limit=128).lower()
    object_text = _clean_text(candidate.get("object_text"), limit=_MAX_FIELD_CHARS)
    content = _clean_text(candidate.get("content"), limit=_MAX_FIELD_CHARS)
    importance = candidate.get("importance")
    confidence = candidate.get("confidence")
    sources_raw = candidate.get("source_msg_ids")
    valid_until = _parse_valid_until(candidate.get("valid_until"))

    addressing_rule = is_addressing_rule(content, object_text, predicate)
    if addressing_rule and kind == "decision":
        # 称呼/行为规则是 preference，不是 decision；模型误分类时纠正。
        kind = "preference"
    if addressing_rule and kind == "preference" and subject_id != "group":
        target_match = _ADDRESSING_TARGET_QQ_PATTERN.search(content)
        if target_match is not None and target_match.group(1) != subject_id:
            # 规则目标明确是另一个 QQ：fail-closed，禁止“替别人改称呼”。
            return None

    if (
        kind not in _ALLOWED_KINDS
        or not subject_id
        or not predicate
        or not object_text
        or not content
        or isinstance(importance, bool)
        or not isinstance(importance, int)
        or not 1 <= importance <= 5
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
        or not isinstance(sources_raw, list)
        or (allowed_subjects is not None and subject_id not in allowed_subjects)
    ):
        return None
    if kind in {"preference", "taboo", "profile"} and (
        len(object_text) < 2 or len(content) < 6
    ):
        # Quality gate: preference/profile fragments such as "likes 你" or
        # "玩调式的" lack a meaningful object and must not be persisted as
        # long-term profile evidence.
        return None

    cleaned_sources: list[str] = []
    for source in sources_raw:
        cleaned_source = _clean_text(source, limit=128)
        if not cleaned_source:
            return None
        cleaned_sources.append(cleaned_source)
    source_ids = tuple(sorted(set(cleaned_sources)))
    if not source_ids or (allowed_sources is not None and any(source not in allowed_sources for source in source_ids)):
        return None
    if source_subject_ids is not None and subject_id != "group":
        if any(str(source_subject_ids.get(source, "")) != subject_id for source in source_ids):
            return None
    if source_subject_ids is not None and subject_id == "group":
        source_authors = {str(source_subject_ids.get(source, "")) for source in source_ids}
        if len(source_authors) < 2 and not _COLLECTIVE_PATTERN.search(f"{content} {object_text}"):
            return None
    if kind == "profile" and source_contents is not None:
        source_texts = tuple(
            str(source_contents.get(source, "") or "").strip()
            for source in source_ids
        )
        if not profile_sources_directly_support(
            predicate=predicate,
            object_text=object_text,
            content=content,
            source_texts=source_texts,
        ):
            return None
    if kind in {"plan", "decision"} and subject_id != "group":
        # Long enumerations and article-sized prose are usually forwarded or
        # pasted material, not one member's own plan/decision.  Fail closed
        # here instead of letting a truncated paragraph pollute the profile.
        supporting_text = (
            " ".join(
                str(source_contents.get(source, "") or "")
                for source in source_ids
            )
            if source_contents is not None
            else ""
        )
        if (
            len(content) > _MAX_PERSON_ACTION_FACT_CHARS
            or len(object_text) > _MAX_PERSON_ACTION_FACT_CHARS
            or (
                len(supporting_text) > 1_200
                and len(_ENUMERATION_MARKER.findall(supporting_text)) >= 4
            )
        ):
            return None
    if candidate.get("valid_until") is not None and valid_until is None:
        return None
    return MemoryFact(
        kind=kind,
        subject_id=subject_id,
        predicate=predicate,
        object_text=object_text,
        content=content,
        importance=importance,
        confidence=float(confidence),
        source_msg_ids=source_ids,
        valid_until=valid_until,
    )


def _parse_invalidation(
    candidate: Any,
    *,
    allowed_sources: set[str] | None,
    source_subject_ids: Mapping[str, str] | None,
    allowed_targets: Mapping[str, Mapping[str, str]] | None,
) -> MemoryInvalidation | None:
    if not isinstance(candidate, Mapping) or allowed_targets is None:
        return None
    target_key = _clean_text(candidate.get("target_canonical_key"), limit=255)
    reason = _clean_text(candidate.get("reason"), limit=32)
    sources_raw = candidate.get("source_msg_ids")
    valid_until = _parse_valid_until(candidate.get("valid_until"))
    target = allowed_targets.get(target_key)
    if (
        not target_key
        or target is None
        or str(target.get("memory_kind") or "") not in {"profile", "preference"}
        or reason != "explicit_denial"
        or not isinstance(sources_raw, list)
    ):
        return None
    source_ids = tuple(
        sorted(
            {
                cleaned
                for source in sources_raw
                if (cleaned := _clean_text(source, limit=128))
            }
        )
    )
    if (
        not source_ids
        or len(source_ids) != len(sources_raw)
        or (allowed_sources is not None and any(source not in allowed_sources for source in source_ids))
    ):
        return None
    subject_id = str(target.get("subject_id") or "")
    if not subject_id or source_subject_ids is None or any(
        str(source_subject_ids.get(source, "")) != subject_id
        for source in source_ids
    ):
        return None
    if candidate.get("valid_until") is not None and valid_until is None:
        return None
    return MemoryInvalidation(
        target_canonical_key=target_key,
        source_msg_ids=source_ids,
        reason=reason,
        valid_until=valid_until,
    )


def profile_sources_directly_support(
    *,
    predicate: str,
    object_text: str,
    content: str,
    source_texts: Sequence[str],
) -> bool:
    """Fail closed for high-risk profile inferences from self-authored chatter."""

    texts = tuple(text for text in source_texts if text)
    if not texts:
        return False
    combined = "\n".join(texts)
    if any(_profile_source_is_non_declarative(text) for text in texts):
        return False
    if _FUTURE_AGE_PATTERN.search(combined):
        return False
    if not _profile_content_matches_object(content=content, object_text=object_text):
        return False
    escaped_object = re.escape(str(object_text or "").strip())
    if escaped_object and re.search(
        rf"(?:不是|并非|才不是|没有|没说过|别再说).{{0,12}}{escaped_object}|"
        rf"{escaped_object}.{{0,12}}(?:不是真的|是假的|不对|作废)",
        combined,
        re.IGNORECASE,
    ):
        return False
    age_values = _extract_age_values(object_text)
    if age_values and re.search(
        r"age|年龄|岁数|年龄阶段|岁", f"{predicate} {object_text}", re.IGNORECASE
    ):
        for age in age_values:
            aliases = {str(age), _format_chinese_number(age)}
            alias_pattern = "(?:" + "|".join(
                re.escape(value) for value in aliases
            ) + ")"
            if any(
                _FUTURE_AGE_PATTERN.search(text) is None
                and re.search(
                    rf"(?:我|本人).{{0,8}}(?:今年|已经|都|才|刚满|就是|是)?\s*"
                    rf"{alias_pattern}\s*岁",
                    text,
                )
                is not None
                for text in texts
            ):
                return True
        return False
    if re.search(r"nationality|国籍", predicate, re.IGNORECASE):
        return any(
            re.search(
                rf"(?:我|本人)\s*(?:是|来自)\s*.{{0,8}}{escaped_object}",
                text,
                re.IGNORECASE,
            )
            is not None
            for text in texts
        )
    terms = _target_lexical_terms(object_text)
    for text in texts:
        if re.search(r"[？?]|(?:如果|假如|等到?|以后).{0,12}(?:我|本人)", text):
            continue
        for term in terms:
            escaped_term = re.escape(term)
            if re.search(
                rf"(?:我|本人)(?:的)?[^。！？\n]{{0,24}}(?:是|叫|在|住在|来自|出生于|属于|有|为)"
                rf"[^。！？\n]{{0,24}}{escaped_term}",
                text,
                re.IGNORECASE,
            ):
                return True
    return False


def _profile_source_is_non_declarative(text: str) -> bool:
    """Reject questions and attributed claims before profile-specific parsing."""

    return re.search(
        r"[？?]|"
        r"(?:谁|哪有|怎么|为何|为什么|是不是|难道|凭什么).{0,18}(?:我|本人)|"
        r"(?:你|他|她|它|别人|人家|大家|他们|她们|群里).{0,12}"
        r"(?:说|讲|觉得|认为|猜|听说|传|称).{0,18}(?:我|本人)|"
        r"(?:我|本人).{0,18}(?:听说|被说成|被称为)",
        text,
        re.IGNORECASE,
    ) is not None


def _profile_content_matches_object(*, content: str, object_text: str) -> bool:
    """Ensure the persisted prose cannot silently contradict its structured value."""

    object_ages = _extract_age_values(object_text)
    if object_ages:
        return bool(object_ages.intersection(_extract_age_values(content)))
    normalized_content = _canonical_part(content)
    return any(
        _canonical_part(term) in normalized_content
        for term in _target_lexical_terms(object_text)
    )


def _dedupe_facts(facts: Iterable[MemoryFact]) -> tuple[MemoryFact, ...]:
    deduped: dict[str, MemoryFact] = {}
    for fact in facts:
        key = canonical_key(fact.kind, fact.subject_id, fact.predicate, fact.object_text)
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = fact
            continue
        source_ids = tuple(sorted(set(previous.source_msg_ids).union(fact.source_msg_ids)))
        winner = max(
            (previous, fact),
            key=lambda item: (
                item.confidence,
                item.importance,
                _canonical_part(item.content),
                item.content,
                _canonical_part(item.subject_id),
                item.subject_id,
                _canonical_part(item.kind),
                item.kind,
                _canonical_part(item.predicate),
                item.predicate,
                _canonical_part(item.object_text),
                item.object_text,
                item.valid_until or "",
            ),
        )
        deduped[key] = MemoryFact(
            kind=winner.kind,
            subject_id=winner.subject_id,
            predicate=winner.predicate,
            object_text=winner.object_text,
            content=winner.content,
            importance=max(previous.importance, fact.importance),
            confidence=max(previous.confidence, fact.confidence),
            source_msg_ids=source_ids,
            valid_until=winner.valid_until,
        )
    return tuple(deduped.values())


def _format_prompt_messages(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        source_id = _clean_text(
            message.get("source_msg_id") or message.get("platform_msg_id") or message.get("message_id") or message.get("id"),
            limit=128,
        )
        content = _clean_text(message.get("content") or message.get("plain_text") or message.get("text"), limit=_MAX_SUMMARY_CHARS)
        if source_id and content:
            lines.append(f"[{source_id}] {content}")
    return lines


def _parse_valid_until(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _canonical_part(value: Any) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(value or "")).casefold()).strip()


def _clean_text(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()[:limit]


def _strip_rolling_prefix(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest_match = _DIGEST_SUMMARY.match(value)
    text = _clean_text(digest_match.group(1) if digest_match is not None else value, limit=_MAX_SUMMARY_CHARS)
    while text:
        stripped = _ROLLING_PREFIX.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return text


def _fact_sort_key(fact: MemoryFact) -> tuple[str, str]:
    return (canonical_key(fact.kind, fact.subject_id, fact.predicate, fact.object_text), fact.content)
