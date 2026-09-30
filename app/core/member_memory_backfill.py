"""Per-member retrospective fact extraction and scheduled incremental refresh."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Mapping, Sequence

import yaml
from zoneinfo import ZoneInfo

from app.core.message_mentions import (
    bot_text_mention_names,
    collect_bot_display_names,
    message_mentions_bot,
)
from app.core.memory_clarification_threads import resolve_clarification_threads
from app.core.memory_context_packer import EvidenceMessage
from app.core.worker_status import write_worker_status
from app.providers.llm_client import LlmClient
from app.storage.db import session_scope
from app.storage.models import MemberFactRefreshState, MemoryItem
from app.storage.repositories import MemoryRepository, MessageRepository


logger = logging.getLogger(__name__)


_TEMPORAL_PROCESS_EVALUATION = re.compile(
    r"(?:目前|现在|当前|眼下|手上).{0,48}"
    r"(?:流程|候选|方向|方案|选择|机会|公司|岗位|项目).{0,48}"
    r"(?:最好|更好|优先|倾向|更可能|概率|保底)|"
    r"(?:流程|候选|方向|方案|选择|机会|公司|岗位|项目).{0,48}"
    r"(?:最好|更好|优先|倾向|更可能|概率|保底).{0,24}"
    r"(?:目前|现在|当前|眼下|手上)"
)


def _normalize_extracted_fact_kind(
    kind: str,
    *,
    evidence: str,
    fact_text: str,
) -> str:
    """Correct provider drift for a time-scoped comparison of alternatives."""

    normalized = str(kind or "fact").strip()
    if normalized != "preference":
        return normalized
    if _TEMPORAL_PROCESS_EVALUATION.search(f"{evidence} {fact_text}"):
        return "decision"
    return normalized


_FACT_PROMPT = (
    "你是成员事实蒸馏专家。请从下面群成员的真实发言提取有依据、可追溯的事实，"
    '只输出一个 ```json 代码块：{"facts": [{"kind": "current/event/plan/decision/preference/taboo/profile/relationship/fact",'
    ' "category": "游戏/体育/动漫/工作/生活/观点/人际关系/外部人物/其他",'
    ' "fact": "第三人称具体事实", "evidence": "逐字引用目标成员的一句话",'
    ' "context_evidence": ["仅在用于消歧时逐字引用相邻上下文"]}]}。'
    "要求：只提取能直接推断的事实；明确陈述正在/最近在看、玩、做、学或所处状态时用 current，"
    "明确陈述一次已发生的活动用 event，打算/准备/计划和已作决定分别用 plan/decision；"
    "preference 只用于相对稳定的喜好；如果发言是在比较当前流程、候选、方向、方案或机会，"
    "并表达现阶段哪个更好、更可能、优先或更倾向，应使用 decision（必要时另提 current），"
    "不能把这种带时点的过程判断只写成 durable preference；"
    "同一句目标发言可能同时表达多个独立切面，例如当前阶段、已经发生的里程碑、"
    "关系和明确倾向；必须逐个评估并可输出多条不同 kind 的事实，不能因为先识别出"
    "relationship 就丢掉同句里的 current/event/decision；每条仍只能写证据直接支持的内容；"
    "不要仅因讨论作品剧情、角色、地点或某个计划，就推断目标成员正在进行、身处其中或已有该计划；"
    "不要从玩笑、反讽、虚构故事或'又失忆了'这类梗里反推事实；"
    "不要把'评价/排行低于某对象'写成'讨厌某对象'，讨厌类事实必须有明确的讨厌/不喜欢表述；"
    "目标发言才能支持事实，相邻上下文只能用来消歧缩写/指代，不能把他人讨论推断成目标成员的活动。"
    "不确定的不写；他反复转发/维护/玩梗的对象（虚拟主播、球星、up主等外部人物）要作为事实列出。"
    "\n语料：\n"
)


def build_slices(
    lines: list[str],
    *,
    slice_chars: int = 16000,
    overlap_lines: int = 2,
) -> list[list[str]]:
    slices: list[list[str]] = [[]]
    used = 0
    for text in lines:
        text = str(text).strip()
        if not text:
            continue
        if used + len(text) > slice_chars and slices[-1]:
            tail = list(slices[-1][-max(0, overlap_lines) :])
            slices.append([])
            used = 0
            if tail:
                slices[-1].extend(tail)
                used = sum(len(item) for item in tail)
        slices[-1].append(text)
        used += len(text)
    if not slices[0]:
        return []
    return slices


def extract_facts_from_lines(
    settings,
    lines: list[str],
    *,
    slice_chars: int = 16000,
    overlap_lines: int = 2,
    context_lines: list[str] | None = None,
    source_records: Sequence[Mapping[str, object]] | None = None,
    context_records: Sequence[Mapping[str, object]] | None = None,
) -> list[dict]:
    slices = build_slices(
        lines,
        slice_chars=slice_chars,
        overlap_lines=overlap_lines,
    )
    if not slices:
        return []

    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_only=True,
        responses_model=settings.llm_model,
        max_output_tokens=8000,
        timeout_seconds=300.0,
        reasoning_effort="low",
    )
    facts: list[dict] = []
    for index, lines_slice in enumerate(slices):
        source_text = "\n".join(lines_slice)
        source_line_set = frozenset(lines_slice)
        normalized_source_records = [
            dict(record)
            for record in (source_records or ())
            if str(record.get("content") or "").strip() in source_line_set
        ]
        if normalized_source_records:
            prompt = _FACT_PROMPT + "\n".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                for record in normalized_source_records
            )
        else:
            prompt = _FACT_PROMPT + source_text
        normalized_context_records = [
            dict(record)
            for record in (context_records or ())
            if str(record.get("content") or "").strip()
        ]
        bounded_context = [
            str(record.get("content") or "").strip()
            for record in normalized_context_records
        ] or [
            str(line).strip()
            for line in (context_lines or [])
            if str(line).strip()
        ]
        if bounded_context:
            if normalized_context_records:
                rendered_context = "\n".join(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    for record in normalized_context_records[:400]
                )
            else:
                rendered_context = "\n".join(bounded_context[:400])
            prompt += "\n\n相邻上下文（仅用于消歧）：\n" + rendered_context
        generated = client.generate_text([prompt])
        if not str(generated or "").strip():
            raise ValueError("member fact provider returned empty text")
        match = re.search(r"```(?:json|yaml|yml)?\s*(.*?)```", generated, re.DOTALL)
        raw = match.group(1) if match else generated
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = yaml.safe_load(raw) or {}
            except yaml.YAMLError as exc:
                raise ValueError("member fact provider returned malformed JSON/YAML") from exc
        candidates = data.get("facts") if isinstance(data, dict) else None
        if not isinstance(candidates, list):
            raise ValueError("member fact provider response has no facts list")
        context_evidence_set = frozenset(bounded_context)
        for fact in candidates:
            if not isinstance(fact, dict):
                continue
            fact_text = str(fact.get("fact") or "").strip()
            evidence = str(fact.get("evidence") or "").strip()
            if not fact_text or not evidence:
                continue
            if evidence not in source_text:
                continue
            kind = _normalize_extracted_fact_kind(
                str(fact.get("kind") or "fact").strip(),
                evidence=evidence,
                fact_text=fact_text,
            )
            if kind not in {
                "current", "event", "plan", "decision", "preference",
                "taboo", "profile", "relationship", "fact",
            }:
                continue
            raw_context_evidence = fact.get("context_evidence") or []
            if not isinstance(raw_context_evidence, list):
                continue
            context_evidence = [
                str(item).strip()
                for item in raw_context_evidence
                if str(item).strip() and str(item).strip() in context_evidence_set
            ]
            if len(context_evidence) != len(
                [item for item in raw_context_evidence if str(item).strip()]
            ):
                continue
            facts.append(
                {
                    "kind": kind,
                    "category": str(fact.get("category") or "其他"),
                    "fact": fact_text,
                    "evidence": evidence,
                    "context_evidence": context_evidence,
                }
            )
        logger.info(
            "member_fact_extract_slice index=%s/%s facts=%s",
            index + 1,
            len(slices),
            len(facts),
        )
    return facts


def upsert_member_facts(
    engine,
    *,
    group_id: int,
    user_id: int,
    facts: list[dict],
    source_message_ids: set[int] | None = None,
    context_source_message_ids: set[int] | None = None,
) -> int:
    imported = 0
    with session_scope(engine) as session:
        repo = MemoryRepository(session)
        seen: set[tuple[str, str]] = set()
        for fact in facts:
            fact_text = str(fact.get("fact") or "").strip()
            kind = str(fact.get("kind") or "fact")
            seen_key = (kind, fact_text)
            if not fact_text or seen_key in seen:
                continue
            seen.add(seen_key)
            evidence = str(fact.get("evidence") or "").strip()
            source = _find_source_message(
                session,
                group_id=group_id,
                user_id=user_id,
                evidence=evidence,
                allowed_message_ids=source_message_ids,
            )
            if source is None:
                raise ValueError("member fact evidence source could not be resolved")
            source_ids = [str(source.platform_msg_id)]
            for context_evidence in fact.get("context_evidence") or []:
                context_source = _find_context_source_message(
                    session,
                    group_id=group_id,
                    evidence=str(context_evidence),
                    anchor_message_id=int(source.id),
                    allowed_message_ids=context_source_message_ids,
                )
                if context_source is None:
                    raise ValueError("member fact context source could not be resolved")
                source_ids.append(str(context_source.platform_msg_id))
            observed_at = source.timestamp
            category = str(fact.get("category") or "fact")
            canonical_key = _member_fact_canonical_key(
                group_id=group_id,
                user_id=user_id,
                kind=kind,
                source_msg_id=str(source.platform_msg_id),
            )
            memory = repo.upsert_canonical_memory(
                scope_type="group",
                scope_id=str(group_id),
                subject_type="user",
                subject_id=str(user_id),
                memory_kind=kind,
                canonical_key=canonical_key,
                predicate=category,
                object_text="",
                content=fact_text,
                importance=3,
                confidence=0.75,
                source_msg_ids=list(dict.fromkeys(source_ids)),
                valid_from=observed_at,
                valid_until=(
                    observed_at + timedelta(days=14)
                    if kind == "current"
                    else None
                ),
            )
            # Migrate facts created before source-stable keys were introduced.
            # A replay may phrase the same evidence differently, so text-based
            # canonical keys are not idempotent.  Retire same-source/same-kind
            # legacy rows after the deterministic upsert.
            legacy_candidates = list(
                session.query(MemoryItem).filter(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == str(group_id),
                    MemoryItem.subject_id == str(user_id),
                    MemoryItem.memory_kind == kind,
                    MemoryItem.status == "active",
                    MemoryItem.id != int(memory.id),
                )
            )
            for duplicate in legacy_candidates:
                duplicate_sources = {
                    str(value)
                    for value in (duplicate.source_msg_ids or [])
                    if str(value).strip()
                }
                if (
                    str(source.platform_msg_id) in duplicate_sources
                ):
                    repo.mark_superseded(
                        memory_id=int(duplicate.id),
                        superseded_by_id=int(memory.id),
                        valid_until=observed_at,
                    )
            imported += 1
    return imported


def _member_fact_canonical_key(
    *,
    group_id: int,
    user_id: int,
    kind: str,
    source_msg_id: str,
) -> str:
    identity = "\n".join(
        (
            str(int(group_id)),
            str(int(user_id)),
            str(kind).strip().casefold(),
            str(source_msg_id).strip(),
        )
    )
    return "member-fact|" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def review_facts(settings, facts: list[dict]) -> list[dict]:
    """Second-pass semantic review; drop joke/irony/misread facts."""

    if not facts:
        return []
    block = "\n".join(
        f"- fact: {fact.get('fact')}\n  evidence: {fact.get('evidence')}"
        for fact in facts
    )
    prompt = (
        "你是记忆事实审核员。下面是从群成员聊天记录里抽取的候选事实，每条附逐字证据。"
        "候选既可能是长期事实，也可能是有时效的当前状态、事件、计划或决定；"
        "判断每条是否由证据直接支持，还是从玩笑、反讽、虚构故事、断章取义里反推出的错误事实。"
        "不要仅因事实是短期状态或计划就丢弃它。"
        "特别注意：'评价/排行低于某对象'不是'讨厌'，这类事实要丢弃；讨厌类事实必须有明确的讨厌/不喜欢表述。"
        "只输出一个 ```json 代码块："
        '{"drop": ["要丢弃的事实原文"], "reasons": {"事实原文": "一句话理由"}}。'
        "只把有明确依据判定为玩笑/反讽/错误的放 drop；证据不足或不确定的一律保留。\n候选：\n"
        + block
    )
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_only=True,
        responses_model=settings.llm_model,
        max_output_tokens=4000,
        timeout_seconds=300.0,
        reasoning_effort="low",
    )
    generated = client.generate_text([prompt])
    if not str(generated or "").strip():
        raise ValueError("member fact review provider returned empty text")
    drop_set = parse_review_output(generated)
    return [fact for fact in facts if str(fact.get("fact") or "") not in drop_set]


def parse_review_output(text: str) -> set[str]:
    if not str(text or "").strip():
        raise ValueError("member fact review provider returned empty text")
    match = re.search(r"```(?:json|yaml|yml)?\s*(.*?)```", str(text or ""), re.DOTALL)
    raw = match.group(1) if match else text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError("member fact review provider returned malformed JSON/YAML") from exc
    if not isinstance(data, dict) or not isinstance(data.get("drop"), list):
        raise ValueError("member fact review provider response has no drop list")
    drop = data["drop"]
    return {str(item).strip() for item in (drop or []) if str(item).strip()}


def _find_source_message(
    session,
    *,
    group_id: int,
    user_id: int,
    evidence: str,
    allowed_message_ids: set[int] | None = None,
):
    from sqlalchemy import select

    from app.storage.models import Message

    text = str(evidence or "").strip()
    if not text:
        return None
    filters = [
        Message.group_id == int(group_id),
        Message.user_id == int(user_id),
    ]
    if allowed_message_ids is not None:
        if not allowed_message_ids:
            return None
        filters.append(Message.id.in_(sorted(int(value) for value in allowed_message_ids)))
    row = session.scalars(
        select(Message)
        .where(*filters, Message.plain_text == text)
        .order_by(Message.id.desc())
    ).first()
    if row is not None:
        return row
    row = session.scalars(
        select(Message)
        .where(*filters, Message.plain_text.like(f"%{text[:40]}%"))
        .order_by(Message.id.desc())
    ).first()
    return row


def _find_context_source_message(
    session,
    *,
    group_id: int,
    evidence: str,
    anchor_message_id: int | None,
    allowed_message_ids: set[int] | None = None,
):
    from sqlalchemy import select

    from app.storage.models import Message

    text = str(evidence or "").strip()
    if not text:
        return None
    filters = [
        Message.group_id == int(group_id),
        Message.plain_text == text,
    ]
    if allowed_message_ids is not None:
        if not allowed_message_ids:
            return None
        filters.append(Message.id.in_(sorted(int(value) for value in allowed_message_ids)))
    elif anchor_message_id is not None:
        filters.extend(
            (
                Message.id >= max(1, int(anchor_message_id) - 2),
                Message.id <= int(anchor_message_id) + 2,
            )
        )
    stmt = select(Message).where(*filters)
    if anchor_message_id is not None:
        from sqlalchemy import func

        stmt = stmt.order_by(func.abs(Message.id - int(anchor_message_id)), Message.id.desc())
    else:
        stmt = stmt.order_by(Message.id.desc())
    return session.scalars(stmt).first()


class MemberFactRefreshService:
    """Daily fact refresh for maintained members, with semantic review.

    ``member_allowlist`` restricts refresh to the personas that are actually
    maintained (live_refresh-enabled). When None, every active member is
    refreshed (legacy behavior).
    """

    def __init__(
        self,
        *,
        engine,
        settings,
        group_ids: set[int],
        bot_qq: int,
        bot_name: str = "",
        member_allowlist: set[int] | None = None,
        interval_seconds: float = 21600.0,
        threshold: int = 50,
        cooldown_seconds: float = 86400.0,
        min_member_messages: int = 300,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.group_ids = set(int(value) for value in group_ids)
        self.bot_qq = int(bot_qq)
        self.bot_name = str(bot_name or "").strip()
        self.bot_qqs: set[int] = {int(bot_qq)}
        self.bot_text_names: set[str] = set()
        self.member_allowlist = (
            set(int(value) for value in member_allowlist)
            if member_allowlist is not None
            else None
        )
        self.interval_seconds = max(1800.0, float(interval_seconds))
        self.threshold = max(10, int(threshold))
        self.cooldown_seconds = max(3600.0, float(cooldown_seconds))
        self.min_member_messages = max(50, int(min_member_messages))

    async def run(self) -> None:
        while True:
            write_worker_status(
                self.settings.log_dir, "member_facts", "running", self.interval_seconds
            )
            try:
                await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("member_fact_refresh_tick_failed")
                write_worker_status(
                    self.settings.log_dir, "member_facts", "error", self.interval_seconds
                )
            else:
                write_worker_status(
                    self.settings.log_dir, "member_facts", "idle", self.interval_seconds
                )
            await asyncio.sleep(self.interval_seconds)

    def replay_member_window(
        self,
        *,
        group_id: int,
        user_id: int,
        start_message_id: int,
        end_message_id: int | None = None,
        max_messages: int = 200,
        dry_run: bool = True,
    ) -> dict[str, int | bool]:
        """Re-extract one bounded member window without moving live watermarks.

        Canonical fact upserts make an applied replay idempotent.  Keeping the
        live watermark untouched also means a provider/parser failure cannot
        skip new messages or corrupt normal refresh scheduling.
        """

        if self.member_allowlist is not None and int(user_id) not in self.member_allowlist:
            raise ValueError("member is not enabled for live refresh")
        if start_message_id <= 0 or max_messages <= 0:
            raise ValueError("replay bounds must be positive")
        if end_message_id is not None and end_message_id < start_message_id:
            raise ValueError("replay end precedes start")
        from sqlalchemy import select

        from app.storage.models import Message

        with session_scope(self.engine) as session:
            stmt = (
                select(Message)
                .where(
                    Message.group_id == int(group_id),
                    Message.user_id == int(user_id),
                    Message.id >= int(start_message_id),
                    Message.plain_text != "",
                )
                .order_by(Message.id)
                .limit(int(max_messages) + 1)
            )
            if end_message_id is not None:
                stmt = stmt.where(Message.id <= int(end_message_id))
            rows = list(session.scalars(stmt))
        if len(rows) > max_messages:
            raise ValueError("replay window exceeds max_messages")
        eligible = [
            row
            for row in rows
            if not message_mentions_bot(
                getattr(row, "raw_json", None),
                bot_qqs=self.bot_qqs,
                bot_text_names=self.bot_text_names,
            )
        ]
        report: dict[str, int | bool] = {
            "dry_run": bool(dry_run),
            "scanned_messages": len(rows),
            "eligible_messages": len(eligible),
            "facts": 0,
            "imported": 0,
        }
        if dry_run or not eligible:
            return report
        context_rows = self._fact_context_rows(
            group_id=group_id,
            target_rows=eligible,
        )
        facts = extract_facts_from_lines(
            self.settings,
            [str(row.plain_text) for row in eligible],
            source_records=self._fact_prompt_records(eligible, role="target"),
            context_records=self._fact_prompt_records(context_rows, role="context"),
        )
        facts = review_facts(self.settings, facts)
        facts.extend(
            self._clarification_facts_for_targets(
                group_id=group_id,
                target_rows=eligible,
            )
        )
        imported = upsert_member_facts(
            self.engine,
            group_id=group_id,
            user_id=user_id,
            facts=facts,
            source_message_ids={int(row.id) for row in eligible},
            context_source_message_ids={
                int(row.id) for row in (*eligible, *context_rows)
            },
        )
        report["facts"] = len(facts)
        report["imported"] = int(imported)
        return report

    def _tick(self) -> None:
        self._refresh_bot_names()
        for group_id in self.group_ids:
            members = _active_members(
                self.engine,
                group_id=group_id,
                bot_qq=self.bot_qq,
                min_messages=self.min_member_messages,
            )
            if self.member_allowlist is not None:
                members = [
                    user_id
                    for user_id in members
                    if int(user_id) in self.member_allowlist
                ]
            due: list[tuple[int, int]] = []
            now = datetime.now(UTC)
            with session_scope(self.engine) as session:
                for user_id in members:
                    state = session.get(
                        MemberFactRefreshState,
                        (int(group_id), int(user_id)),
                    )
                    watermark = int(state.last_msg_id or 0) if state else 0
                    _, eligible_lines = self._pending_member_lines(
                        session,
                        group_id=group_id,
                        user_id=user_id,
                        watermark=watermark,
                    )
                    new_count = len(eligible_lines)
                    last_refresh = state.last_refresh_at if state else None
                    overdue = False
                    due_today = True
                    if last_refresh is not None:
                        if last_refresh.tzinfo is None:
                            last_refresh = last_refresh.replace(tzinfo=UTC)
                        overdue = (
                            now - last_refresh
                        ).total_seconds() >= self.cooldown_seconds
                        last_local = last_refresh.astimezone(ZoneInfo("Asia/Shanghai"))
                        due_today = last_local.date() < now.astimezone(
                            ZoneInfo("Asia/Shanghai")
                        ).date()
                    if new_count >= self.threshold or overdue or due_today:
                        due.append((new_count, user_id))
            due.sort(reverse=True)
            for _, user_id in due:
                self._refresh_member(group_id, user_id)

    def _refresh_member(self, group_id: int, user_id: int) -> None:
        self._refresh_bot_names()
        with session_scope(self.engine) as session:
            state = session.get(
                MemberFactRefreshState,
                (int(group_id), int(user_id)),
            )
            watermark = int(state.last_msg_id or 0) if state is not None else 0
            all_new_lines, new_lines = self._pending_member_lines(
                session,
                group_id=group_id,
                user_id=user_id,
                watermark=watermark,
            )
            last_id = watermark
            if all_new_lines:
                last_id = max(int(row.id) for row in all_new_lines)
        if not new_lines:
            self._commit_refresh_state(
                group_id=group_id,
                user_id=user_id,
                last_id=last_id,
            )
            return

        context_rows = self._fact_context_rows(
            group_id=group_id,
            target_rows=new_lines,
        )
        facts = extract_facts_from_lines(
            self.settings,
            [str(row.plain_text) for row in new_lines],
            source_records=self._fact_prompt_records(new_lines, role="target"),
            context_records=self._fact_prompt_records(context_rows, role="context"),
        )
        facts = review_facts(self.settings, facts)
        facts.extend(
            self._clarification_facts_for_targets(
                group_id=group_id,
                target_rows=new_lines,
            )
        )
        imported = upsert_member_facts(
            self.engine,
            group_id=group_id,
            user_id=user_id,
            facts=facts,
            source_message_ids={int(row.id) for row in new_lines},
            context_source_message_ids={
                int(row.id) for row in (*new_lines, *context_rows)
            },
        )
        self._commit_refresh_state(
            group_id=group_id,
            user_id=user_id,
            last_id=last_id,
        )
        logger.info(
            "member_fact_refresh group_id=%s user_id=%s facts=%s imported=%s",
            group_id,
            user_id,
            len(facts),
            imported,
        )

    def _neighbor_context_lines(self, *, group_id: int, target_rows) -> list[str]:
        """Load a bounded same-group window used only to disambiguate targets."""

        return [
            str(row.plain_text)
            for row in self._neighbor_context_rows(
                group_id=group_id,
                target_rows=target_rows,
            )
        ]

    def _fact_context_rows(self, *, group_id: int, target_rows) -> list:
        """Combine local neighbors with verified same-episode clarification questions."""

        rows = self._neighbor_context_rows(
            group_id=group_id,
            target_rows=target_rows,
        )
        rows.extend(
            self._clarification_context_rows(
                group_id=group_id,
                target_rows=target_rows,
            )
        )
        deduped = {int(row.id): row for row in rows}
        return [deduped[key] for key in sorted(deduped)]

    def _clarification_facts_for_targets(self, *, group_id: int, target_rows) -> list[dict]:
        """Materialize verified short answers as source-stable structured facts.

        A terse answer such as ``哇为`` can be meaningful only together with
        the bounded question and its member-authored anchor.  Preserve that
        atomic provenance deterministically so offline refresh does not depend
        on an extractor deciding whether a typo is a company name.
        """

        if not target_rows:
            return []
        facts: list[dict] = []
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            for target in target_rows:
                anchor_id = str(target.platform_msg_id)
                rows = messages.list_bounded_same_episode_message_context(
                    group_id=int(group_id),
                    anchor_platform_msg_ids=[anchor_id],
                    per_anchor_limit=12,
                    max_gap_seconds=900,
                    excluded_user_ids=self.bot_qqs,
                )
                evidence = tuple(
                    EvidenceMessage(
                        source_msg_id=str(row.platform_msg_id),
                        speaker=str(row.user_id),
                        content=str(row.plain_text or ""),
                        sent_at=row.timestamp,
                        blocked=messages.is_qq_blocked_outbound(row),
                        group_id=(
                            int(row.group_id) if row.group_id is not None else None
                        ),
                        reply_to_msg_id=row.reply_to_msg_id,
                        is_bot=int(row.user_id) in self.bot_qqs,
                        user_id=int(row.user_id),
                        delivery_state=(
                            str(row.raw_json.get("delivery_state") or "")
                            .strip()
                            .casefold()
                            if isinstance(row.raw_json, dict)
                            else ""
                        ),
                    )
                    for row in rows
                )
                threads = resolve_clarification_threads(
                    evidence,
                    anchor_source_ids=(anchor_id,),
                    max_turns=12,
                    max_gap_seconds=900,
                )
                row_by_source = {str(row.platform_msg_id): row for row in rows}
                for thread in threads:
                    if thread.answer_source_id != anchor_id:
                        continue
                    question = row_by_source.get(thread.question_source_id)
                    answer = row_by_source.get(thread.answer_source_id)
                    if question is None or answer is None:
                        continue
                    slot_kind = str(thread.slot_kind or "item")
                    kind = (
                        "decision"
                        if slot_kind in {"company", "item", "place", "person", "count"}
                        else "fact"
                    )
                    facts.append(
                        {
                            "kind": kind,
                            "category": "澄清回答",
                            "fact": (
                                f"该成员在被问到“{str(question.plain_text or '').strip()}”时，"
                                f"回答为“{str(answer.plain_text or '').strip()}”。"
                            ),
                            "evidence": str(answer.plain_text or "").strip(),
                            "context_evidence": [
                                str(question.plain_text or "").strip(),
                                str(row_by_source[thread.anchor_source_id].plain_text or "").strip(),
                            ],
                        }
                    )
        return facts

    def _neighbor_context_rows(self, *, group_id: int, target_rows) -> list:
        """Load the existing narrow ID-neighbor context without widening it."""

        if not target_rows:
            return []
        from sqlalchemy import select

        from app.storage.models import Message

        target_ids = {int(row.id) for row in target_rows}
        neighbor_ids = {
            candidate_id
            for target_id in target_ids
            for candidate_id in range(max(1, target_id - 2), target_id + 3)
        }
        with session_scope(self.engine) as session:
            rows = list(
                session.scalars(
                    select(Message)
                    .where(
                        Message.group_id == int(group_id),
                        Message.id.in_(sorted(neighbor_ids)),
                        Message.plain_text != "",
                    )
                    .order_by(Message.id)
                )
            )
        return [
            row
            for row in rows
            if int(row.id) not in target_ids
            and int(row.user_id) not in self.bot_qqs
            and not message_mentions_bot(
                getattr(row, "raw_json", None),
                bot_qqs=self.bot_qqs,
                bot_text_names=self.bot_text_names,
            )
        ]

    def _clarification_context_rows(self, *, group_id: int, target_rows) -> list:
        """Return non-target rows from per-anchor verified clarification threads."""

        if not target_rows:
            return []
        target_user_ids = {int(row.user_id) for row in target_rows}
        if len(target_user_ids) != 1:
            return []
        target_user_id = next(iter(target_user_ids))
        target_ids = {int(row.id) for row in target_rows}
        selected: dict[int, object] = {}
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            for target in target_rows:
                anchor_id = str(target.platform_msg_id)
                rows = messages.list_bounded_same_episode_message_context(
                    group_id=int(group_id),
                    anchor_platform_msg_ids=[anchor_id],
                    per_anchor_limit=12,
                    max_gap_seconds=900,
                    excluded_user_ids=self.bot_qqs,
                )
                evidence = tuple(
                    EvidenceMessage(
                        source_msg_id=str(row.platform_msg_id),
                        speaker=str(row.user_id),
                        content=str(row.plain_text or ""),
                        sent_at=row.timestamp,
                        blocked=messages.is_qq_blocked_outbound(row),
                        group_id=(
                            int(row.group_id) if row.group_id is not None else None
                        ),
                        reply_to_msg_id=row.reply_to_msg_id,
                        is_bot=int(row.user_id) in self.bot_qqs,
                        user_id=int(row.user_id),
                        delivery_state=(
                            str(row.raw_json.get("delivery_state") or "")
                            .strip()
                            .casefold()
                            if isinstance(row.raw_json, dict)
                            else ""
                        ),
                    )
                    for row in rows
                )
                threads = resolve_clarification_threads(
                    evidence,
                    anchor_source_ids=(anchor_id,),
                    max_turns=12,
                    max_gap_seconds=900,
                )
                source_ids = {
                    source_id
                    for thread in threads
                    if thread.subject_id == str(target_user_id)
                    for source_id in thread.source_msg_ids
                }
                for row in rows:
                    if (
                        int(row.id) not in target_ids
                        and str(row.platform_msg_id) in source_ids
                    ):
                        selected[int(row.id)] = row
        return [selected[key] for key in sorted(selected)]

    @staticmethod
    def _fact_prompt_records(rows, *, role: str) -> list[dict[str, object]]:
        """Render source identity and chronology without changing evidence text."""

        return [
            {
                "role": str(role),
                "source_msg_id": str(row.platform_msg_id),
                "user_id": str(row.user_id),
                "sent_at": row.timestamp.isoformat(),
                "content": str(row.plain_text or ""),
            }
            for row in rows
        ]

    def _pending_member_lines(
        self,
        session,
        *,
        group_id: int,
        user_id: int,
        watermark: int,
    ):
        all_new_lines = _new_member_lines(
            session,
            group_id=group_id,
            user_id=user_id,
            watermark=watermark,
        )
        eligible_lines = [
            row
            for row in all_new_lines
            if not message_mentions_bot(
                getattr(row, "raw_json", None),
                bot_qqs=self.bot_qqs,
                bot_text_names=self.bot_text_names,
            )
        ]
        return all_new_lines, eligible_lines

    def _commit_refresh_state(
        self,
        *,
        group_id: int,
        user_id: int,
        last_id: int,
    ) -> None:
        with session_scope(self.engine) as session:
            session.merge(
                MemberFactRefreshState(
                    group_id=int(group_id),
                    user_id=int(user_id),
                    last_msg_id=str(last_id),
                    last_refresh_at=datetime.now(UTC),
                )
            )

    def _refresh_bot_names(self) -> None:
        from sqlalchemy import bindparam, text

        with session_scope(self.engine) as session:
            user_rows = session.execute(
                text("SELECT user_id, nickname, group_card FROM users")
            ).fetchall()
            bot_ids = {int(self.bot_qq)}
            for user_id, nickname, card in user_rows:
                label = str(card or "").strip() or str(nickname or "").strip()
                if "小町" in label:
                    bot_ids.add(int(user_id))
            ids_param = list(bot_ids)
            bot_rows = session.execute(
                text(
                    "SELECT raw_json FROM messages WHERE user_id = :bot_qq "
                    "AND raw_json IS NOT NULL ORDER BY id DESC LIMIT 3000"
                ),
                {"bot_qq": ids_param[0]},
            ).fetchall()
            if len(ids_param) > 1:
                extra_rows = session.execute(
                    text(
                        "SELECT raw_json FROM messages WHERE user_id IN :bot_ids "
                        "AND raw_json IS NOT NULL ORDER BY id DESC LIMIT 3000"
                    ).bindparams(bindparam("bot_ids", expanding=True)),
                    {"bot_ids": ids_param[1:]},
                ).fetchall()
                bot_rows = [*bot_rows, *extra_rows]
            member_rows = session.execute(
                text(
                    "SELECT DISTINCT json_extract(raw_json, '$.sender.card') AS card, "
                    "json_extract(raw_json, '$.sender.nickname') AS nickname "
                    "FROM messages WHERE raw_json IS NOT NULL AND user_id NOT IN :bot_ids"
                ).bindparams(bindparam("bot_ids", expanding=True)),
                {"bot_ids": ids_param},
            ).fetchall()
        bot_display = collect_bot_display_names(row[0] for row in bot_rows)
        member_display: set[str] = set()
        for card, nickname in member_rows:
            for value in (card, nickname):
                cleaned = str(value or "").strip()
                if cleaned:
                    member_display.add(cleaned)
        self.bot_qqs = bot_ids
        self.bot_text_names = bot_text_mention_names(
            bot_qqs=bot_ids,
            default_name=self.bot_name,
            bot_display_names=bot_display,
            member_display_names=member_display,
        )


def _active_members(engine, *, group_id: int, bot_qq: int, min_messages: int) -> list[int]:
    from sqlalchemy import text

    with session_scope(engine) as session:
        rows = session.execute(
            text(
                "SELECT user_id FROM messages WHERE group_id = :group_id "
                "AND user_id <> :bot AND plain_text <> '' "
                "GROUP BY user_id HAVING COUNT(*) >= :min_messages"
            ),
            {
                "group_id": int(group_id),
                "bot": int(bot_qq),
                "min_messages": int(min_messages),
            },
        ).fetchall()
    return [int(row[0]) for row in rows]


def _new_member_lines(session, *, group_id: int, user_id: int, watermark: int):
    from sqlalchemy import select

    from app.storage.models import Message

    return list(
        session.scalars(
            select(Message)
            .where(
                Message.group_id == int(group_id),
                Message.user_id == int(user_id),
                Message.id > int(watermark),
                Message.plain_text != "",
            )
            .order_by(Message.id)
        )
    )
