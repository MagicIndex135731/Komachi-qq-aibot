"""Per-member retrospective fact extraction and scheduled incremental refresh."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta

import yaml
from zoneinfo import ZoneInfo

from app.core.message_mentions import (
    bot_text_mention_names,
    collect_bot_display_names,
    message_mentions_bot,
)
from app.providers.llm_client import LlmClient
from app.storage.db import session_scope
from app.storage.models import MemberFactRefreshState, MemoryItem
from app.storage.repositories import MemoryRepository


logger = logging.getLogger(__name__)


_FACT_PROMPT = (
    "你是成员事实蒸馏专家。请从下面群成员的真实发言提取有依据、可追溯的事实，"
    '只输出一个 ```json 代码块：{"facts": [{"kind": "current/event/plan/decision/preference/taboo/profile/relationship/fact",'
    ' "category": "游戏/体育/动漫/工作/生活/观点/人际关系/外部人物/其他",'
    ' "fact": "第三人称具体事实", "evidence": "逐字引用目标成员的一句话",'
    ' "context_evidence": ["仅在用于消歧时逐字引用相邻上下文"]}]}。'
    "要求：只提取能直接推断的事实；明确陈述正在/最近在看、玩、做、学或所处状态时用 current，"
    "明确陈述一次已发生的活动用 event，打算/准备/计划和已作决定分别用 plan/decision；"
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
        prompt = _FACT_PROMPT + "\n".join(lines_slice)
        bounded_context = [str(line).strip() for line in (context_lines or []) if str(line).strip()]
        if bounded_context:
            prompt += "\n\n相邻上下文（仅用于消歧）：\n" + "\n".join(bounded_context[:400])
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
        source_text = "\n".join(lines_slice)
        context_text = "\n".join(bounded_context)
        for fact in candidates:
            if not isinstance(fact, dict):
                continue
            fact_text = str(fact.get("fact") or "").strip()
            evidence = str(fact.get("evidence") or "").strip()
            if not fact_text or not evidence:
                continue
            if evidence not in source_text:
                continue
            kind = str(fact.get("kind") or "fact").strip()
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
                if str(item).strip() and str(item).strip() in context_text
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
) -> int:
    imported = 0
    with session_scope(engine) as session:
        repo = MemoryRepository(session)
        seen: set[str] = set()
        for fact in facts:
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text or fact_text in seen:
                continue
            seen.add(fact_text)
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
                )
                if context_source is None:
                    raise ValueError("member fact context source could not be resolved")
                source_ids.append(str(context_source.platform_msg_id))
            observed_at = source.timestamp
            kind = str(fact.get("kind") or "fact")
            category = str(fact.get("category") or "fact")
            canonical_key = _member_fact_canonical_key(
                group_id=group_id,
                user_id=user_id,
                kind=kind,
                category=category,
                evidence=evidence,
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
                    and str(duplicate.predicate or "").strip().casefold()
                    == category.strip().casefold()
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
    category: str,
    evidence: str,
) -> str:
    identity = "\n".join(
        (
            str(int(group_id)),
            str(int(user_id)),
            str(kind).strip().casefold(),
            str(category).strip().casefold(),
            " ".join(str(evidence).split()).casefold(),
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
    if anchor_message_id is not None:
        filters.extend(
            (
                Message.id >= max(1, int(anchor_message_id) - 2),
                Message.id <= int(anchor_message_id) + 2,
            )
        )
    return session.scalars(
        select(Message).where(*filters).order_by(Message.id.desc())
    ).first()


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
            try:
                await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("member_fact_refresh_tick_failed")
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
        facts = extract_facts_from_lines(
            self.settings,
            [str(row.plain_text) for row in eligible],
            context_lines=self._neighbor_context_lines(
                group_id=group_id,
                target_rows=eligible,
            ),
        )
        facts = review_facts(self.settings, facts)
        imported = upsert_member_facts(
            self.engine,
            group_id=group_id,
            user_id=user_id,
            facts=facts,
            source_message_ids={int(row.id) for row in eligible},
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

        facts = extract_facts_from_lines(
            self.settings,
            [str(row.plain_text) for row in new_lines],
            context_lines=self._neighbor_context_lines(
                group_id=group_id,
                target_rows=new_lines,
            ),
        )
        facts = review_facts(self.settings, facts)
        imported = upsert_member_facts(
            self.engine,
            group_id=group_id,
            user_id=user_id,
            facts=facts,
            source_message_ids={int(row.id) for row in new_lines},
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
            str(row.plain_text)
            for row in rows
            if int(row.id) not in target_ids
            and int(row.user_id) not in self.bot_qqs
            and not message_mentions_bot(
                getattr(row, "raw_json", None),
                bot_qqs=self.bot_qqs,
                bot_text_names=self.bot_text_names,
            )
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
