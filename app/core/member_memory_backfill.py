"""Per-member retrospective fact extraction and scheduled incremental refresh."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime

import yaml

from app.providers.llm_client import LlmClient
from app.storage.db import session_scope
from app.storage.repositories import (
    MemoryRepository,
    PersonaStyleSyncStateRepository,
)


logger = logging.getLogger(__name__)


_FACT_PROMPT = (
    "你是人设蒸馏专家。请从下面群成员的真实发言提取有依据的持久事实，"
    '只输出一个 ```json 代码块：{"facts": [{"category": "游戏/体育/动漫/工作/生活/观点/人际关系/外部人物/其他",'
    ' "fact": "第三人称具体事实", "evidence": "逐字引用他的一句话"}]}。'
    "要求：只提取能直接推断的持久事实；不要从玩笑、反讽、虚构故事或'又失忆了'这类梗里反推事实；"
    "不确定的不写；他反复转发/维护/玩梗的对象（虚拟主播、球星、up主等外部人物）要作为事实列出。"
    "\n语料：\n"
)


def extract_facts_from_lines(
    settings,
    lines: list[str],
    *,
    slice_chars: int = 16000,
) -> list[dict]:
    slices: list[list[str]] = [[]]
    used = 0
    for text in lines:
        text = str(text).strip()
        if not text:
            continue
        if used + len(text) > slice_chars and slices[-1]:
            slices.append([])
            used = 0
        slices[-1].append(text)
        used += len(text)
    if not slices[0]:
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
        generated = client.generate_text([_FACT_PROMPT + "\n".join(lines_slice)])
        match = re.search(r"```(?:json|yaml|yml)?\s*(.*?)```", generated, re.DOTALL)
        raw = match.group(1) if match else generated
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = yaml.safe_load(raw) or {}
        candidates = data.get("facts") if isinstance(data, dict) else None
        if not isinstance(candidates, list):
            continue
        source_text = "\n".join(lines_slice)
        for fact in candidates:
            if not isinstance(fact, dict):
                continue
            fact_text = str(fact.get("fact") or "").strip()
            evidence = str(fact.get("evidence") or "").strip()
            if not fact_text or not evidence:
                continue
            if evidence not in source_text:
                continue
            facts.append(
                {
                    "category": str(fact.get("category") or "其他"),
                    "fact": fact_text,
                    "evidence": evidence,
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
            source_id = _find_source_id(
                session, group_id=group_id, user_id=user_id, evidence=evidence
            )
            repo.upsert_canonical_memory(
                scope_type="group",
                scope_id=str(group_id),
                subject_type="user",
                subject_id=str(user_id),
                memory_kind="fact",
                canonical_key=fact_text,
                predicate=str(fact.get("category") or "fact"),
                object_text="",
                content=fact_text,
                importance=3,
                confidence=0.75,
                source_msg_ids=[source_id] if source_id else [],
            )
            imported += 1
    return imported


def _find_source_id(session, *, group_id: int, user_id: int, evidence: str) -> str | None:
    from sqlalchemy import select

    from app.storage.models import Message

    text = str(evidence or "").strip()
    if not text:
        return None
    row = session.scalars(
        select(Message).where(
            Message.group_id == int(group_id),
            Message.user_id == int(user_id),
            Message.plain_text == text,
        )
    ).first()
    if row is not None:
        return str(row.platform_msg_id)
    row = session.scalars(
        select(Message).where(
            Message.group_id == int(group_id),
            Message.user_id == int(user_id),
            Message.plain_text.like(f"%{text[:40]}%"),
        )
    ).first()
    return str(row.platform_msg_id) if row is not None else None


class MemberFactRefreshService:
    """Scheduled incremental fact refresh for all active members, with a quota."""

    def __init__(
        self,
        *,
        engine,
        settings,
        group_ids: set[int],
        bot_qq: int,
        interval_seconds: float = 21600.0,
        daily_quota: int = 2,
        threshold: int = 50,
        cooldown_seconds: float = 86400.0,
        min_member_messages: int = 300,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.group_ids = set(int(value) for value in group_ids)
        self.bot_qq = int(bot_qq)
        self.interval_seconds = max(1800.0, float(interval_seconds))
        self.daily_quota = max(1, int(daily_quota))
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

    def _tick(self) -> None:
        for group_id in self.group_ids:
            members = _active_members(
                self.engine,
                group_id=group_id,
                bot_qq=self.bot_qq,
                min_messages=self.min_member_messages,
            )
            due: list[tuple[int, int]] = []
            now = datetime.now(UTC)
            with session_scope(self.engine) as session:
                state_repo = PersonaStyleSyncStateRepository(session)
                for user_id in members:
                    state = state_repo.get(group_id=group_id, user_id=user_id)
                    new_count = int(state.new_since_refresh or 0) if state else 0
                    last_refresh = state.last_refresh_at if state else None
                    overdue = False
                    if last_refresh is not None:
                        if last_refresh.tzinfo is None:
                            last_refresh = last_refresh.replace(tzinfo=UTC)
                        overdue = (
                            now - last_refresh
                        ).total_seconds() >= self.cooldown_seconds
                    if new_count >= self.threshold or overdue:
                        due.append((new_count, user_id))
            due.sort(reverse=True)
            for _, user_id in due[: self.daily_quota]:
                self._refresh_member(group_id, user_id)

    def _refresh_member(self, group_id: int, user_id: int) -> None:
        with session_scope(self.engine) as session:
            state_repo = PersonaStyleSyncStateRepository(session)
            state = state_repo.get(group_id=group_id, user_id=user_id)
            watermark = int(state.last_msg_id or 0) if state is not None else 0
            new_lines = _new_member_lines(
                session,
                group_id=group_id,
                user_id=user_id,
                watermark=watermark,
            )
            last_id = watermark
            if new_lines:
                last_id = max(int(row.id) for row in new_lines)
            state_repo.set_watermark(
                group_id=group_id,
                user_id=user_id,
                last_msg_id=str(last_id),
                new_count=0,
            )
        if not new_lines:
            return
        facts = extract_facts_from_lines(
            self.settings,
            [str(row.plain_text) for row in new_lines],
        )
        imported = upsert_member_facts(
            self.engine,
            group_id=group_id,
            user_id=user_id,
            facts=facts,
        )
        with session_scope(self.engine) as session:
            PersonaStyleSyncStateRepository(session).mark_refreshed(
                group_id=group_id,
                user_id=user_id,
                when=datetime.now(UTC),
            )
        logger.info(
            "member_fact_refresh group_id=%s user_id=%s facts=%s imported=%s",
            group_id,
            user_id,
            len(facts),
            imported,
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
