from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import app.core.member_memory_backfill as member_memory_backfill
from app.core.member_memory_backfill import (
    MemberFactRefreshService,
    build_slices,
    parse_review_output,
)
from app.storage.db import session_scope
from app.storage.models import Group, MemberFactRefreshState, Message, User


def test_parse_review_output_extracts_drop_set() -> None:
    text = (
        "```json\n"
        '{"drop": ["他的外公保有记忆"], '
        '"reasons": {"他的外公保有记忆": "从失忆梗反推"}}\n'
        "```"
    )

    assert parse_review_output(text) == {"他的外公保有记忆"}


def test_parse_review_output_tolerates_missing_fence() -> None:
    assert parse_review_output('{"drop": ["x", "y"]}') == {"x", "y"}


def test_build_slices_overlaps_boundaries() -> None:
    slices = build_slices(
        ["一一一一一一", "二二二二二二二二", "三三三三三三三三", "四四四四四四四四"],
        slice_chars=8,
        overlap_lines=2,
    )

    assert slices[-1][:2] == ["二二二二二二二二", "三三三三三三三三"]
    assert "四四四四四四四四" in slices[-1]


def _seed_member_message(
    engine,
    *,
    platform_msg_id: str,
    plain_text: str,
    mentions_bot: bool = False,
) -> int:
    raw_message = (
        [{"type": "at", "data": {"qq": "900000999"}}]
        if mentions_bot
        else [{"type": "text", "data": {"text": plain_text}}]
    )
    with session_scope(engine) as session:
        row = Message(
            platform_msg_id=platform_msg_id,
            group_id=900000001,
            user_id=900000101,
            timestamp=datetime.now(UTC),
            raw_json={"message": raw_message},
            plain_text=plain_text,
            msg_type="text",
            mentioned_bot=mentions_bot,
        )
        session.add(row)
        session.flush()
        return int(row.id)


def _fact_refresh_service(engine, *, threshold: int = 10) -> MemberFactRefreshService:
    return MemberFactRefreshService(
        engine=engine,
        settings=SimpleNamespace(),
        group_ids={900000001},
        bot_qq=900000999,
        threshold=threshold,
        cooldown_seconds=86400.0,
        min_member_messages=50,
    )


@pytest.fixture
def fact_refresh_database(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        session.add(Group(group_id=900000001, group_name="test"))
        session.add(User(user_id=900000101, nickname="member", group_card="member"))
    return sqlite_engine


def test_refresh_failure_keeps_watermark_and_retries_same_messages(
    fact_refresh_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_id = _seed_member_message(
        fact_refresh_database,
        platform_msg_id="member-1",
        plain_text="我喜欢火锅",
    )
    with session_scope(fact_refresh_database) as session:
        session.add(
            MemberFactRefreshState(
                group_id=900000001,
                user_id=900000101,
                last_msg_id="0",
                last_refresh_at=datetime.now(UTC),
            )
        )

    service = _fact_refresh_service(fact_refresh_database)
    monkeypatch.setattr(service, "_refresh_bot_names", lambda: None)
    attempts: list[list[str]] = []

    def fail_extract(_settings, lines, **_kwargs):
        attempts.append(list(lines))
        raise RuntimeError("provider response was malformed")

    monkeypatch.setattr(member_memory_backfill, "extract_facts_from_lines", fail_extract)
    with pytest.raises(RuntimeError, match="malformed"):
        service._refresh_member(900000001, 900000101)

    with session_scope(fact_refresh_database) as session:
        state = session.get(MemberFactRefreshState, (900000001, 900000101))
        assert state is not None
        assert state.last_msg_id == "0"

    monkeypatch.setattr(
        member_memory_backfill,
        "extract_facts_from_lines",
        lambda _settings, lines, **_kwargs: attempts.append(list(lines)) or [],
    )
    monkeypatch.setattr(member_memory_backfill, "review_facts", lambda _settings, facts: facts)
    monkeypatch.setattr(member_memory_backfill, "upsert_member_facts", lambda *_args, **_kwargs: 0)
    service._refresh_member(900000001, 900000101)

    assert attempts == [["我喜欢火锅"], ["我喜欢火锅"]]
    with session_scope(fact_refresh_database) as session:
        state = session.get(MemberFactRefreshState, (900000001, 900000101))
        assert state is not None
        assert state.last_msg_id == str(message_id)


def test_tick_uses_pending_eligible_message_count_and_excludes_bot_mentions(
    fact_refresh_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_scope(fact_refresh_database) as session:
        session.add(
            MemberFactRefreshState(
                group_id=900000001,
                user_id=900000101,
                last_msg_id="0",
                last_refresh_at=datetime.now(UTC),
            )
        )
    for index in range(10):
        _seed_member_message(
            fact_refresh_database,
            platform_msg_id=f"mention-{index}",
            plain_text="@小町 测试",
            mentions_bot=True,
        )
    for index in range(9):
        _seed_member_message(
            fact_refresh_database,
            platform_msg_id=f"eligible-{index}",
            plain_text=f"普通发言 {index}",
        )

    service = _fact_refresh_service(fact_refresh_database, threshold=10)
    monkeypatch.setattr(service, "_refresh_bot_names", lambda: None)
    monkeypatch.setattr(
        member_memory_backfill,
        "_active_members",
        lambda *_args, **_kwargs: [900000101],
    )
    refreshed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        service,
        "_refresh_member",
        lambda group_id, user_id: refreshed.append((group_id, user_id)),
    )

    service._tick()
    assert refreshed == []

    _seed_member_message(
        fact_refresh_database,
        platform_msg_id="eligible-9",
        plain_text="达到阈值的普通发言",
    )
    service._tick()

    assert refreshed == [(900000001, 900000101)]
