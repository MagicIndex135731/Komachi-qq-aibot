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
_MAX_SUMMARY_CHARS = 2_000
_MAX_FIELD_CHARS = 600
_MAX_FACTS = 64
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
class MemoryCompaction:
    """Validated model result. Facts are source-backed and de-duplicated."""

    summary: str
    facts: tuple[MemoryFact, ...] = ()
    rejected_fact_count: int = 0


def canonical_key(kind: str, subject_id: str, predicate: str, object_text: str) -> str:
    """Return a stable fact identity insensitive to case, spacing and Unicode form."""

    return "|".join(_canonical_part(part) for part in (kind, subject_id, predicate, object_text))


def parse_memory_compaction_response(
    raw: str | bytes | Mapping[str, Any] | None,
    *,
    allowed_source_msg_ids: Iterable[str] | None = None,
    allowed_subject_ids: Iterable[str] | None = None,
    source_subject_ids: Mapping[str, str] | None = None,
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
            )
            if fact is None:
                rejected_fact_count += 1
                continue
            parsed.append(fact)

    return MemoryCompaction(
        summary=summary,
        facts=_dedupe_facts(parsed),
        rejected_fact_count=rejected_fact_count,
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
            "The object must contain summary (a non-empty string) and facts (an array). Each fact may contain only kind, "
            "subject_id, predicate, object_text, content, importance, confidence, source_msg_ids, valid_until.\n"
            "For a user fact, subject_id must be the numeric user_id of the author and every cited source must be written by that user. "
            "Use subject_id=group only for explicitly collective facts; cite at least two authors unless the source explicitly says everyone, the group, or we.\n"
            "Use only fact, preference, taboo, plan, decision, profile, relationship, event, running_joke, current, or expired as kind. "
            "Every fact needs at least one exact source_msg_id from the messages below. Never invent a source. "
            "If any field is uncertain, omit that fact instead of guessing. Return facts=[] when there is no durable fact.\n"
            "importance must be an integer from 1 to 5, confidence a number from 0 to 1, and valid_until an ISO date/time or null. "
            "The previous digest is context only and is never evidence."
        )
        lines = [instructions]
        if previous:
            lines.extend(("Previous digest (context only, not evidence):", previous))
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
