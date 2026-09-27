"""Pure, bounded linking for short clarification answers in group memory."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from app.core.memory_context_packer import EvidenceMessage


_QUESTION_SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("company", re.compile(r"哪家|哪个公司|哪间")),
    ("item", re.compile(r"哪个|哪些|什么|啥")),
    ("place", re.compile(r"哪里|哪儿|何处")),
    ("person", re.compile(r"谁|哪位")),
    ("count", re.compile(r"多少|几个|几次|几部")),
    ("reason", re.compile(r"为什么|为啥|怎么会")),
)
_QUESTION_MARKER = re.compile(r"[?？]|(?:哪家|哪个|哪些|什么|啥|哪里|哪儿|谁|哪位|多少|几个|为什么|为啥)")
_INELIGIBLE_DELIVERY_STATES = frozenset({"reserved", "blocked", "uncertain", "deleted"})
_TRIVIAL_TEXT = re.compile(r"^[\s\W_]*$", re.UNICODE)
_ANCHOR_CUE_FILLERS = re.compile(
    r"请问|想问|问下|你(?:刚才)?说的|他(?:刚才)?说的|那个|这个|"
    r"到底|究竟|(?:是|有|了|的|呢|啊|呀|吗|嘛)+"
)
_ANCHOR_CUE_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")


@dataclass(frozen=True, slots=True)
class ClarificationThread:
    source_msg_ids: tuple[str, ...]
    anchor_source_id: str
    question_source_id: str
    answer_source_id: str
    subject_id: str
    slot_kind: str


def resolve_clarification_threads(
    messages: Sequence[EvidenceMessage],
    *,
    anchor_source_ids: Sequence[str],
    max_turns: int = 12,
    max_gap_seconds: int = 900,
    max_answer_turns: int = 4,
) -> tuple[ClarificationThread, ...]:
    """Link a target statement, another member's question, and target short answer.

    The function is intentionally conservative.  It consumes already scoped,
    same-episode rows and rejects unsafe delivery states, bots, missing speaker
    identities, ambiguous competing answers, and unbounded gaps.
    """

    if min(max_turns, max_gap_seconds, max_answer_turns) < 1:
        raise ValueError("clarification thread bounds must be positive")
    ordered = tuple(
        sorted(
            (message for message in messages if _safe(message)),
            key=lambda message: (message.sent_at, message.source_msg_id),
        )
    )
    hit_ids = {str(value) for value in anchor_source_ids if str(value)}
    if not ordered or not hit_ids:
        return ()

    threads: list[ClarificationThread] = []
    seen: set[tuple[str, str, str]] = set()
    for question_index, question in enumerate(ordered):
        slot_kind = _question_slot(question.content)
        if slot_kind is None:
            continue
        for answer_index in range(
            question_index + 1,
            min(len(ordered), question_index + 1 + max_answer_turns),
        ):
            answer = ordered[answer_index]
            if answer.user_id == question.user_id or not _short_answer(answer.content):
                continue
            if (answer.sent_at - question.sent_at).total_seconds() > max_gap_seconds:
                break
            anchor = _find_anchor(
                ordered,
                question_index=question_index,
                question=question,
                answer=answer,
                max_turns=max_turns,
                max_gap_seconds=max_gap_seconds,
            )
            if anchor is None:
                continue
            intervening = ordered[question_index + 1 : answer_index]
            if not _intervening_is_safe(intervening, answer=answer):
                continue
            group = (
                anchor.source_msg_id,
                question.source_msg_id,
                answer.source_msg_id,
            )
            if not hit_ids.intersection(group):
                continue
            identity = tuple(group)
            if identity in seen:
                continue
            seen.add(identity)
            threads.append(
                ClarificationThread(
                    source_msg_ids=group,
                    anchor_source_id=anchor.source_msg_id,
                    question_source_id=question.source_msg_id,
                    answer_source_id=answer.source_msg_id,
                    subject_id=str(answer.user_id),
                    slot_kind=slot_kind,
                )
            )
            break
    return tuple(threads)


def _find_anchor(
    messages: Sequence[EvidenceMessage],
    *,
    question_index: int,
    question: EvidenceMessage,
    answer: EvidenceMessage,
    max_turns: int,
    max_gap_seconds: int,
) -> EvidenceMessage | None:
    candidates: list[EvidenceMessage] = []
    start = max(0, question_index - max_turns)
    for candidate in reversed(messages[start:question_index]):
        if candidate.user_id != answer.user_id:
            continue
        if (question.sent_at - candidate.sent_at).total_seconds() > max_gap_seconds:
            continue
        if not _substantive_anchor(candidate.content):
            continue
        if question.reply_to_msg_id and question.reply_to_msg_id != candidate.source_msg_id:
            continue
        candidates.append(candidate)
        # An explicit reply edge identifies one anchor.  Without that edge,
        # retain every plausible statement so multiple candidates fail closed
        # instead of silently choosing the nearest one.
        if question.reply_to_msg_id:
            break
    if len(candidates) == 1:
        return candidates[0]
    cues = _question_anchor_cues(question.content)
    if not cues:
        return None
    matched = [
        candidate
        for candidate in candidates
        if any(cue in _normalize_short(candidate.content) for cue in cues)
    ]
    return matched[0] if len(matched) == 1 else None


def _intervening_is_safe(
    messages: Sequence[EvidenceMessage],
    *,
    answer: EvidenceMessage,
) -> bool:
    normalized_answer = _normalize_short(answer.content)
    for message in messages:
        if message.user_id == answer.user_id or _question_slot(message.content) is not None:
            continue
        # A third party may guess the same terse value and the target can then
        # confirm it.  A different answer makes the linkage ambiguous.
        if _short_answer(message.content) and _normalize_short(message.content) == normalized_answer:
            continue
        return False
    return True


def _safe(message: EvidenceMessage) -> bool:
    return bool(
        message.source_msg_id
        and message.user_id is not None
        and not message.is_bot
        and not message.blocked
        and str(message.delivery_state or "").casefold() not in _INELIGIBLE_DELIVERY_STATES
        and str(message.content or "").strip()
    )


def _question_slot(text: str) -> str | None:
    normalized = str(text or "").strip()
    if not _QUESTION_MARKER.search(normalized):
        return None
    for slot_kind, pattern in _QUESTION_SLOT_PATTERNS:
        if pattern.search(normalized):
            return slot_kind
    return None


def _short_answer(text: str) -> bool:
    normalized = str(text or "").strip()
    return bool(
        1 <= len(normalized) <= 32
        and not _QUESTION_MARKER.search(normalized)
        and not _TRIVIAL_TEXT.fullmatch(normalized)
    )


def _substantive_anchor(text: str) -> bool:
    normalized = str(text or "").strip()
    return bool(
        len(normalized) >= 4
        and not _QUESTION_MARKER.search(normalized)
        and not _TRIVIAL_TEXT.fullmatch(normalized)
    )


def _normalize_short(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?：:；;]+", "", str(text or "")).casefold()


def _question_anchor_cues(text: str) -> tuple[str, ...]:
    """Extract non-slot wording that can disambiguate a prior statement."""

    residual = str(text or "")
    for _, pattern in _QUESTION_SLOT_PATTERNS:
        residual = pattern.sub("", residual)
    residual = _ANCHOR_CUE_FILLERS.sub("", residual)
    tokens = [
        _normalize_short(token)
        for token in _ANCHOR_CUE_TOKEN.findall(residual)
        if len(_normalize_short(token)) >= 2
    ]
    cues: list[str] = []
    for token in tokens:
        cues.append(token)
        if len(token) > 4:
            cues.extend(
                token[index:index + width]
                for width in (4, 3, 2)
                for index in range(len(token) - width + 1)
            )
    return tuple(dict.fromkeys(cues))
