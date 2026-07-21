from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
from threading import Barrier

import pytest

from app.core.memory_compaction_service import MemoryCompactionService
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, JobRepository, MemoryRepository, MessageRepository, SummaryRepository, UserRepository


class FakeCompactionLlm:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.prompts: list[list[str]] = []

    def generate_text(self, prompt_lines: list[str]) -> str:
        self.prompts.append(prompt_lines)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _seed_messages(engine, *, count: int = 10) -> None:
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=10001, group_name="test", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=42, nickname="Alice", group_card="Alice")
        messages = MessageRepository(session)
        for index in range(count):
            messages.add_group_message(
                platform_msg_id=f"m-{index}",
                group_id=10001,
                user_id=42,
                timestamp=datetime(2026, 7, 16, 1, index, tzinfo=UTC),
                plain_text="我喜欢火锅" if index in {0, 3} else f"普通聊天 {index}",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )


@pytest.mark.asyncio
async def test_service_compacts_window_and_persists_canonical_fact(sqlite_engine) -> None:
    _seed_messages(sqlite_engine)
    response = json.dumps(
        {
            "summary": "Alice多次表示喜欢火锅。",
            "facts": [
                {
                    "kind": "preference",
                    "subject_id": "42",
                    "predicate": "喜欢",
                    "object_text": "火锅",
                    "content": "Alice喜欢火锅。",
                    "importance": 4,
                    "confidence": 0.9,
                    "source_msg_ids": ["m-0", "m-3"],
                    "valid_until": None,
                }
            ],
        },
        ensure_ascii=False,
    )
    service = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm(response),
        batch_size=10,
        backfill_windows=1,
    )

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session).list_current_group_memories(scope_id="10001", limit=10)
        summaries = SummaryRepository(session).list_group_summaries(
            scope_id="10001", limit=10, summary_levels=["semantic_window", "semantic_daily"]
        )
        jobs = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["completed"])
    assert len(memories) == 1
    assert memories[0].canonical_key == "preference|42|喜欢|火锅"
    assert memories[0].source_msg_ids == ["m-0", "m-3"]
    assert memories[0].mention_count == 2
    assert {summary.summary_level for summary in summaries} == {"semantic_window", "semantic_daily"}
    assert len(jobs) == 1


@pytest.mark.asyncio
async def test_service_requeues_transient_llm_failure(sqlite_engine) -> None:
    _seed_messages(sqlite_engine)
    service = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm(RuntimeError("temporary")),
        batch_size=10,
        retry_limit=3,
        backfill_windows=1,
    )

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        queued = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["queued"])
    assert len(queued) == 1
    assert queued[0].payload_json["attempts"] == 1
    assert "temporary" in queued[0].payload_json["last_error"]


@pytest.mark.asyncio
async def test_service_waits_for_future_queued_job_without_timezone_error(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        JobRepository(session).add_job(
            job_type="memory_compaction",
            job_key="future-memory-job",
            payload_json={"group_id": 10001, "start_id": 1, "end_id": 2, "attempts": 0},
            run_at=datetime.now(UTC) + timedelta(minutes=5),
        )
    service = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm("{}"),
        backfill_windows=0,
    )

    await service.start()
    await asyncio.sleep(0.05)
    assert service._worker_task is not None
    assert not service._worker_task.done()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        queued = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["queued"])
    assert [job.job_key for job in queued] == ["future-memory-job"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_response", ["not json", json.dumps({"summary": "   ", "facts": []})])
async def test_invalid_model_response_retries_without_overwriting_daily_digest(sqlite_engine, invalid_response) -> None:
    _seed_messages(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        SummaryRepository(session).upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="semantic_daily",
            summary_key="semantic-daily:2026-07-16",
            start_at=datetime(2026, 7, 16, 0, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 16, 0, 30, tzinfo=UTC),
            content="OLD IMPORTANT FACT",
            source_count=5,
        )
    service = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm(invalid_response),
        batch_size=10,
        retry_limit=3,
        backfill_windows=1,
    )

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        daily = SummaryRepository(session).list_group_summaries(
            scope_id="10001",
            limit=1,
            summary_levels=["semantic_daily"],
            summary_key="semantic-daily:2026-07-16",
        )
        queued = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["queued"])
    assert daily[0].content == "OLD IMPORTANT FACT"
    assert queued[0].payload_json["attempts"] == 1


@pytest.mark.asyncio
async def test_overlapping_partial_backfill_keeps_unique_daily_source_count(sqlite_engine) -> None:
    _seed_messages(sqlite_engine, count=9)
    response = json.dumps({"summary": "Daily digest.", "facts": []})
    first = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm(response),
        batch_size=10,
        backfill_windows=1,
    )
    await first.start()
    await first.stop()

    with session_scope(sqlite_engine) as session:
        MessageRepository(session).add_group_message(
            platform_msg_id="m-9",
            group_id=10001,
            user_id=42,
            timestamp=datetime(2026, 7, 16, 1, 9, tzinfo=UTC),
            plain_text="message 9",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
    second = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm(response),
        batch_size=10,
        backfill_windows=1,
    )
    await second.start()
    await second.stop()

    with session_scope(sqlite_engine) as session:
        daily = SummaryRepository(session).list_group_summaries(
            scope_id="10001",
            limit=1,
            summary_levels=["semantic_daily"],
            summary_key="semantic-daily:2026-07-16",
        )
    assert daily[0].source_count == 10


@pytest.mark.asyncio
async def test_semantically_rejected_fact_completes_without_overwriting_old_daily(sqlite_engine) -> None:
    _seed_messages(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        SummaryRepository(session).upsert_summary(
            scope_type="group", scope_id="10001", summary_level="semantic_daily",
            summary_key="semantic-daily:2026-07-16", start_at=datetime(2026, 7, 16, tzinfo=UTC),
            end_at=datetime(2026, 7, 16, 0, 30, tzinfo=UTC), content="OLD SAFE DAILY", source_count=5,
        )
    response = json.dumps(
        {
            "summary": "Alice made a personal decision.",
            "facts": [
                {
                    "kind": "decision", "subject_id": "group", "predicate": "decided",
                    "object_text": "resign", "content": "Alice decided to resign.",
                    "importance": 4, "confidence": 0.9, "source_msg_ids": ["m-0"],
                    "valid_until": None,
                }
            ],
        }
    )
    service = MemoryCompactionService(
        engine=sqlite_engine, llm_client=FakeCompactionLlm(response), batch_size=10, backfill_windows=1,
    )

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        daily = SummaryRepository(session).list_group_summaries(
            scope_id="10001", limit=1, summary_levels=["semantic_daily"],
            summary_key="semantic-daily:2026-07-16",
        )
        completed = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["completed"])
        memories = MemoryRepository(session).list_current_group_memories(scope_id="10001", limit=10)
    assert daily[0].content == "OLD SAFE DAILY"
    assert completed[0].payload_json["rejected_fact_count"] == 1
    assert memories == []


def test_concurrent_add_job_is_idempotent(sqlite_engine) -> None:
    barrier = Barrier(2)

    def add_once() -> int:
        barrier.wait()
        with session_scope(sqlite_engine) as session:
            job = JobRepository(session).add_job(
                job_type="memory_compaction",
                job_key="same-key",
                payload_json={"attempts": 0},
                run_at=datetime.now(UTC),
            )
            return job.id

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _index: add_once(), range(2)))

    assert ids[0] == ids[1]
    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["queued"])
    assert len(jobs) == 1


def test_fresh_running_job_is_not_requeued_before_lease_expires(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        jobs.add_job(
            job_type="memory_compaction",
            job_key="leased-job",
            payload_json={},
            run_at=datetime.now(UTC),
        )
        claimed = jobs.claim_oldest_queued_job(job_type="memory_compaction", now=datetime.now(UTC))
        assert claimed is not None

    service = MemoryCompactionService(
        engine=sqlite_engine,
        llm_client=FakeCompactionLlm("{}"),
        backfill_windows=0,
    )
    service._prepare_jobs()

    with session_scope(sqlite_engine) as session:
        running = JobRepository(session).list_jobs(job_type="memory_compaction", statuses=["running"])
    assert [job.job_key for job in running] == ["leased-job"]
