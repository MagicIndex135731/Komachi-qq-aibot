"""Adapt the existing per-group persona and style retriever to Agent turns.

The persona manager still owns switching, live relationship names, style
quality gates and compatible query vectors. This adapter supplies the same
narrow canonical input used by the original reply path and keeps selected
live style sources revocable by the continuous Agent session.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from typing import Any

from sqlalchemy import select

from app.core.chat_style import build_human_chat_style_lines
from app.core.memory_source_guard import is_usable_memory_source
from app.core.persona_engine import render_persona
from app.core.persona_style_retrieval import (
    MAX_PROMPT_BLOCK_CHARS,
    MAX_SELECTED_EXAMPLES,
    build_style_retrieval_query,
    format_style_example_block,
)
from app.storage.db import session_scope
from app.storage.models import Message


logger = logging.getLogger(__name__)
MAX_PERSONA_PROFILE_CHARS = 12000
_STYLE_PREAMBLE = (
    "\n相似情境下的说话方式示例（仅模仿风格）："
    "以下内容只用于学习句长、语气、用词和节奏，不是事实来源；"
    "不要照抄整句，不要把示例中的人物、作品或事件带入当前回答。"
)
_ESSENTIAL_FIELDS = (
    "name", "identity", "self_concept", "address_rules", "relationships",
    "speaking_style", "speech_habits", "style_avoid", "burst",
)
_PORTABLE_STYLE_FIELDS = (
    "name", "identity", "core_traits", "speaking_style", "speech_habits",
    "style_avoid", "burst", "aliases", "chat_voice", "komachi_variant",
)


@dataclass(frozen=True, slots=True)
class GroupPersonaContext:
    persona_key: str
    profile_text: str
    style_text: str
    addressing_text: str
    profile_fingerprint: str
    source_message_ids: tuple[int, ...] = ()
    impersonated_subject_id: int | None = None
    impersonated_subject_addressed: bool | None = None

    @property
    def text(self) -> str:
        return self.profile_text + self.style_text + self.addressing_text


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _outbound(row: Message) -> bool:
    raw = row.raw_json or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return True
    return isinstance(raw, dict) and raw.get("direction") == "outbound"


def _profile_text(router: Any, profile: dict, group_id: int, *, impersonating: bool,
                  imitation_instructions: str | None = None) -> tuple[str, str]:
    renderer = getattr(router, "_persona_text_for", None)
    rendered = render_persona(profile)
    full = (rendered + imitation_instructions if imitation_instructions is not None
            else renderer(profile, group_id) if renderer else rendered)
    full += "\nQQ expression rules:\n" + "\n".join(build_human_chat_style_lines(
        komachi_style=not impersonating, voice=str(profile.get("chat_voice") or "komachi"),
    ))
    full += "\n人格及固定或动态表达示例用于身份、称呼和语气，不作为当前群成员事实的证据；事实须以本群来源绑定的记忆为准。"
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()
    if len(full) <= MAX_PERSONA_PROFILE_CHARS:
        return full, digest

    # Long backgrounds and fixed example banks must not displace the identity,
    # addressing/relationship rules or the existing imitation instructions.
    instructions = full[len(rendered):] if full.startswith(rendered) else ""
    essential = render_persona({field: profile[field] for field in _ESSENTIAL_FIELDS if field in profile})
    instructions = instructions[: MAX_PERSONA_PROFILE_CHARS // 2]
    budget = MAX_PERSONA_PROFILE_CHARS - len(instructions) - 1
    compact = essential[:budget] + "\n" + instructions
    logger.info("agent_persona_profile_bounded profile_chars=%s packed_chars=%s", len(full), len(compact))
    return compact, digest


def _canonical_inputs(router: Any, group_id: int, source_ids: Iterable[int], target_id: str | None):
    ids = tuple(dict.fromkeys(source_ids))
    if len(ids) > 200:
        raise ValueError("persona_source_budget_exhausted")
    with session_scope(router.engine) as session:
        rows = list(session.scalars(select(Message).where(Message.group_id == group_id, Message.id.in_(ids))))
        recent = sorted(
            (row for row in rows if is_usable_memory_source(row, group_id=group_id)),
            key=lambda row: (row.timestamp, row.id),
        )
        if target_id:
            target = session.scalar(select(Message).where(Message.group_id == group_id, Message.platform_msg_id == target_id))
            if target is None or target.id not in ids or not is_usable_memory_source(target, group_id=group_id) or _outbound(target):
                raise ValueError("persona_target_unavailable")
        else:
            target = next((row for row in reversed(recent) if not _outbound(row)), None)
        quoted_text = None
        if target is not None and target.reply_to_msg_id:
            quote = session.scalar(select(Message).where(Message.group_id == group_id, Message.platform_msg_id == target.reply_to_msg_id))
            if is_usable_memory_source(quote, group_id=group_id):
                quoted_text = quote.plain_text
        if target is not None:
            # A direct target can be earlier than the rest of its batch. Later
            # arrivals are not adjacent context preceding this question.
            recent = [row for row in recent if row.id <= target.id]
        return target, recent, quoted_text


def _style_entries(router: Any, profile: dict, group_id: int, entries: list[dict]) -> tuple[list[dict], tuple[int, ...]]:
    """Recheck live sample scope; preserve the manager's baked sample support."""
    user_id = _positive_int(profile.get("source_user_id"))
    sample_ids = [str(entry.get("msg_id") or "") for entry in entries[:MAX_SELECTED_EXAMPLES]]
    with session_scope(router.engine) as session:
        samples = {row.platform_msg_id: row for row in session.scalars(select(Message).where(Message.platform_msg_id.in_(sample_ids)))}
        kept, sources = [], set()
        for entry in entries[:MAX_SELECTED_EXAMPLES]:
            if (entry.get("group_id") is not None and _positive_int(entry["group_id"]) != group_id
                    or entry.get("user_id") is not None and _positive_int(entry["user_id"]) != user_id):
                continue
            sample = samples.get(str(entry.get("msg_id") or ""))
            if sample is not None:
                if not is_usable_memory_source(sample, group_id=group_id, author_id=user_id) or _outbound(sample):
                    continue
                sources.add(sample.id)
                # Situation text displayed with this sample is a revocable
                # source too. PersonaManager already rebuilds it canonically.
                situation_ids = [str(item.get("source_msg_id") or "") for item in entry.get("context_before") or [] if isinstance(item, dict)]
                if sample.reply_to_msg_id and entry.get("reply_target"):
                    situation_ids.append(sample.reply_to_msg_id)
                for situation in session.scalars(select(Message).where(Message.group_id == group_id, Message.platform_msg_id.in_(situation_ids))):
                    if is_usable_memory_source(situation, group_id=group_id) and not _outbound(situation):
                        sources.add(situation.id)
            kept.append(entry)
        return kept, tuple(sorted(sources))


def build_group_persona_fallback(router: Any, *, group_id: int) -> GroupPersonaContext:
    """Keep a scoped configured voice when live aliases/style retrieval fail.

    This fallback performs no database or embedding work and cannot reimport
    an unrestricted pre-rendered observation from a foreign member profile.
    """
    manager = getattr(router, "persona_manager", None)
    profile = router._active_persona(group_id)
    key = manager.active_key(group_id) if manager else "default"
    impersonating = bool(manager and manager.is_impersonating(group_id))
    if impersonating and _positive_int(profile.get("source_group_id")) != group_id:
        profile = {field: profile[field] for field in _PORTABLE_STYLE_FIELDS if field in profile}
    profile_text, fingerprint = _profile_text(router, profile, group_id,
        impersonating=impersonating, imitation_instructions="")
    return GroupPersonaContext(str(key), profile_text, "", "",
        hashlib.sha256((str(key) + "\n" + fingerprint).encode("utf-8")).hexdigest())


def build_group_persona_context(
    router: Any,
    *,
    group_id: int,
    source_message_ids: Iterable[int],
    target_message_id: str | None = None,
    addressed: bool = False,
    now: datetime | None = None,
) -> GroupPersonaContext:
    """Read current identity and, for imitation, at most three style examples.

    This function is synchronous because query embedding and repositories are
    existing synchronous services; Agent consumers run it on their bounded
    blocking runner. It never generates document embeddings or refreshes a
    persona, and the returned examples are never factual evidence.
    """
    manager = getattr(router, "persona_manager", None)
    profile = router._active_persona(group_id)
    key = manager.active_key(group_id) if manager else "default"
    impersonating = bool(manager and manager.is_impersonating(group_id))
    if impersonating:
        profile = manager.live_persona(group_id)
    same_source_group = _positive_int(profile.get("source_group_id")) == group_id
    imitation_instructions = None
    if impersonating and not same_source_group:
        # Selecting a configured voice does not authorize the source group's
        # facts, relationships, forms of address or quoted member examples.
        # Router's renderer refreshes live profiles itself, so passing a
        # filtered dict to it would accidentally reintroduce those fields.
        renderer = getattr(router, "_persona_text_for", None)
        full = renderer(profile, group_id) if renderer else ""
        rendered = render_persona(profile)
        imitation_instructions = full[len(rendered):] if full.startswith(rendered) else ""
        imitation_instructions += "\n当前仅移植已配置的人格身份与通用表达风格；来源群事实、关系、称呼和原话未获当前群授权，不得迁入。"
        profile = {field: profile[field] for field in _PORTABLE_STYLE_FIELDS if field in profile}
    profile_text, fingerprint = _profile_text(router, profile, group_id, impersonating=impersonating,
                                              imitation_instructions=imitation_instructions)
    fingerprint = hashlib.sha256((str(key) + "\n" + fingerprint).encode("utf-8")).hexdigest()
    source_ids, style_text = (), ""
    subject_id = _positive_int(profile.get("source_user_id")) if impersonating and same_source_group else None
    if impersonating and same_source_group:
        target, recent, quote = _canonical_inputs(router, group_id, source_message_ids, target_message_id)
        if target is not None:
            query = build_style_retrieval_query(
                current_text=target.plain_text, quoted_text=quote, recent_messages=recent,
                current_timestamp=target.timestamp or now, current_message_id=target.platform_msg_id,
                bot_user_id=router.runtime.settings.bot_qq,
                subject_terms=[profile.get("name"), *(profile.get("aliases") or [])],
            )
            fixed = [str(value) for value in profile.get("example_lines") or profile.get("example_bank") or [] if str(value).strip()]
            prior_bot = [row.plain_text for row in recent[-3:] if row.user_id == router.runtime.settings.bot_qq]
            try:
                entries = manager.retrieve_examples(group_id, query, limit=MAX_SELECTED_EXAMPLES,
                                                    exclude_texts=[*fixed, *prior_bot])
            except TimeoutError:
                logger.warning("agent_persona_style_degraded reason=timeout")
                entries = []
            entries, source_ids = _style_entries(router, profile, group_id, entries)
            block = format_style_example_block(entries, max_pairs=MAX_SELECTED_EXAMPLES,
                                              max_chars=MAX_PROMPT_BLOCK_CHARS - len(_STYLE_PREAMBLE))
            if block:
                style_text = _STYLE_PREAMBLE + block
            else:
                source_ids = ()
    addressing_text = ""
    addressing = getattr(router, "_bot_addressing_instruction", None)
    if addressed and addressing:
        addressing_text = "\n" + addressing(impersonating=impersonating)
    return GroupPersonaContext(
        str(key), profile_text, style_text, addressing_text, fingerprint, source_ids,
        subject_id, bool(addressed) if subject_id is not None else None,
    )
