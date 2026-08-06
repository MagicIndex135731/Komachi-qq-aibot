from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from sqlalchemy import text

import scripts.backfill_structured_memory as backfill_module
from app.config import AppSettings
from app.core.memory_background_service import (
    DerivedFact,
    EpisodeDerivation,
)
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    UserRepository,
)


NOW = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)


def _fake_settings() -> AppSettings:
    return AppSettings.model_construct(
        napcat_ws_url="ws://127.0.0.1:3001",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="test-key",
        llm_model="gpt-5.4",
        bot_qq=123456789,
        owner_qq=987654321,
        memory_episode_max_messages=50,
        memory_episode_max_tokens=8000,
        memory_chunk_max_tokens=1800,
        memory_chunk_overlap_messages=5,
        memory_episode_idle_minutes=30,
        memory_compaction_max_facts=5,
        memory_compaction_retry_limit=3,
        memory_embedding_provider="disabled",
        memory_embedding_model="",
        memory_embedding_dimensions=8,
        memory_embedding_cache_dir=Path("models"),
        memory_embedding_base_url="",
        memory_embedding_api_key="",
        memory_embedding_version="test-v1",
        memory_orchestration_v2_enabled=True,
        memory_orchestration_shadow_mode=False,
        data_dir=Path("data"),
    )


class FakeDeriver:
    def __init__(self, *, llm_client=None, max_facts: int = 5) -> None:
        del llm_client
        self.max_facts = max_facts

    def derive(self, *, episode, messages, windows) -> EpisodeDerivation:
        del episode
        platform_ids = tuple(
            str(message.platform_msg_id)
            for message in messages
            if str(getattr(message, "plain_text", "") or "").strip()
        )
        return EpisodeDerivation(
            summary="回填摘要",
            facts=(
                DerivedFact(
                    content="回填事实",
                    source_msg_ids=platform_ids[:1],
                    kind="fact",
                    subject_id="group",
                ),
            ),
            events=(),
            windows=windows,
        )


def _seed_messages(engine) -> None:
    with session_scope(engine) as session:
        for group_id in (10001, 10002):
            GroupRepository(session).upsert_group(
                group_id=group_id,
                group_name=f"group-{group_id}",
                enabled=True,
                speak_enabled=True,
            )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        messages = MessageRepository(session)
        for index in range(3):
            messages.add_group_message(
                platform_msg_id=f"bf-10001-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=NOW + timedelta(minutes=index),
                plain_text=f"消息 {index}",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        messages.add_group_message(
            platform_msg_id="bf-10002-0",
            group_id=10002,
            user_id=20001,
            timestamp=NOW,
            plain_text="另一个群的消息",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )


def _database_path(sqlite_engine) -> str:
    return str(sqlite_engine.url.database)


def _run_main(monkeypatch, capsys, argv: list[str]) -> dict:
    monkeypatch.setattr(backfill_module, "AppSettings", _fake_settings)
    monkeypatch.setattr(backfill_module, "CompactionEpisodeDeriver", FakeDeriver)
    assert backfill_module.main(argv) == 0
    output = capsys.readouterr().out.strip()
    return json.loads(output.splitlines()[-1])


def test_plan_is_read_only_inventory(monkeypatch, capsys, sqlite_engine) -> None:
    _seed_messages(sqlite_engine)
    report = _run_main(
        monkeypatch,
        capsys,
        ["plan", "--database", _database_path(sqlite_engine)],
    )
    assert report["command"] == "plan"
    assert {group["group_id"] for group in report["groups"]} == {10001, 10002}
    assert report["estimated_llm_calls"] >= 1
    assert all(group["messages"] == group["eligible_messages"] for group in report["groups"])
    with sqlite_engine.connect() as connection:
        runs = connection.execute(
            text("SELECT COUNT(*) FROM memory_backfill_runs")
        ).scalar_one()
    assert runs == 0


def test_run_completes_idempotently_and_writes_structured_layers(
    monkeypatch,
    capsys,
    sqlite_engine,
) -> None:
    _seed_messages(sqlite_engine)
    database = _database_path(sqlite_engine)
    first = _run_main(
        monkeypatch,
        capsys,
        ["run", "--database", database, "--run-key", "test-run", "--finalize"],
    )
    assert first["status"] == "completed"
    assert first["failed_jobs"] == 0
    assert first["pending_jobs"] == first["running_jobs"] == 0
    assert all(
        group["assigned_messages"] == group["eligible_messages"]
        for group in first["groups"]
    )

    with session_scope(sqlite_engine) as session:
        summaries = session.execute(
            text(
                "SELECT COUNT(*) FROM summaries "
                "WHERE scope_type='group' AND scope_id IN ('10001','10002')"
            )
        ).scalar_one()
        memory_items = session.execute(
            text(
                "SELECT COUNT(*) FROM memory_items "
                "WHERE scope_type='group' AND scope_id IN ('10001','10002')"
            )
        ).scalar_one()
    assert summaries >= 1
    assert memory_items >= 1

    rerun = _run_main(
        monkeypatch,
        capsys,
        ["run", "--database", database, "--run-key", "test-run", "--finalize"],
    )
    assert rerun["status"] == "completed"
    with session_scope(sqlite_engine) as session:
        summaries_after = session.execute(
            text(
                "SELECT COUNT(*) FROM summaries "
                "WHERE scope_type='group' AND scope_id IN ('10001','10002')"
            )
        ).scalar_one()
        memory_items_after = session.execute(
            text(
                "SELECT COUNT(*) FROM memory_items "
                "WHERE scope_type='group' AND scope_id IN ('10001','10002')"
            )
        ).scalar_one()
    assert summaries_after == summaries
    assert memory_items_after == memory_items


def test_status_and_finalize_report_existing_run(
    monkeypatch,
    capsys,
    sqlite_engine,
) -> None:
    _seed_messages(sqlite_engine)
    database = _database_path(sqlite_engine)
    _run_main(
        monkeypatch,
        capsys,
        ["run", "--database", database, "--run-key", "status-run"],
    )
    status = _run_main(
        monkeypatch,
        capsys,
        ["status", "--database", database, "--run-key", "status-run"],
    )
    assert status["status"] in {"running", "completed"}
    finalized = _run_main(
        monkeypatch,
        capsys,
        ["finalize", "--database", database, "--run-key", "status-run"],
    )
    assert finalized["status"] == "completed"
