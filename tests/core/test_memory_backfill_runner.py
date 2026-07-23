from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.memory_backfill_runner import (
    collect_backfill_coverage,
    finalize_backfill_run,
    group_watermarks_from_manifest,
    run_memory_backfill,
)
from app.core.memory_background_service import (
    EpisodeDerivation,
    MemoryBackgroundService,
    SqlAlchemyMemoryBackgroundStore,
)
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    UserRepository,
)


NOW = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)


class SafeDeriver:
    def derive(self, *, episode, messages, windows) -> EpisodeDerivation:
        del episode, messages
        return EpisodeDerivation(
            summary="safe summary",
            facts=(),
            events=(),
            windows=windows,
        )


def _seed_messages(engine) -> list[int]:
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        messages = MessageRepository(session)
        rows = [
            messages.add_group_message(
                platform_msg_id="safe-1",
                group_id=10001,
                user_id=20001,
                timestamp=NOW,
                plain_text="first safe message",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            ),
            messages.add_group_message(
                platform_msg_id="safe-2",
                group_id=10001,
                user_id=20001,
                timestamp=NOW + timedelta(minutes=40),
                plain_text="second safe message",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            ),
        ]
        session.flush()
        return [int(row.id) for row in rows]


def _manifest(watermark: int) -> dict:
    return {
        "format_version": 1,
        "buckets": {
            "group:10001": {
                "group_id": 10001,
                "watermark": watermark,
                "count": 2,
                "sha256": "0" * 64,
            },
            "private": {
                "group_id": None,
                "watermark": 0,
                "count": 0,
                "sha256": "0" * 64,
            },
        },
    }


def _background(
    engine,
    *,
    segmentation_generation: str = "segment-v2",
    compaction_generation: str = "compact-v2",
) -> MemoryBackgroundService:
    return MemoryBackgroundService(
        store=SqlAlchemyMemoryBackgroundStore(engine),
        deriver=SafeDeriver(),
        worker_id="backfill-test",
        segmentation_generation=segmentation_generation,
        compaction_generation=compaction_generation,
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
        chunk_max_tokens=1800,
        chunk_overlap_messages=5,
    )


def test_group_watermarks_reject_invalid_or_empty_contract() -> None:
    assert group_watermarks_from_manifest(_manifest(12)) == {10001: 12}
    with pytest.raises(ValueError, match="no group watermarks"):
        group_watermarks_from_manifest({"buckets": {"private": {}}})
    with pytest.raises(ValueError, match="invalid group watermark"):
        group_watermarks_from_manifest(
            {"buckets": {"group:bad": {"group_id": True, "watermark": 1}}}
        )


def test_manifest_bounded_backfill_completes_and_resumes(sqlite_engine) -> None:
    message_ids = _seed_messages(sqlite_engine)
    manifest = _manifest(message_ids[-1])

    first = run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=manifest,
        run_key="verified-backup",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )

    assert first.status == "completed"
    assert first.pending_jobs == first.running_jobs == first.failed_jobs == 0
    assert first.orphan_episode_messages == first.orphan_document_sources == 0
    assert len(first.groups) == 1
    assert first.groups[0].messages == 2
    assert first.groups[0].eligible_messages == 2
    assert first.groups[0].assigned_messages == 2
    assert first.groups[0].retrieval_documents > 0
    assert first.groups[0].blocked_derived_documents == 0
    assert "plain_text" not in repr(first.as_safe_dict())

    resumed = run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=manifest,
        run_key="verified-backup",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )
    assert resumed == collect_backfill_coverage(
        engine=sqlite_engine,
        run_id=first.run_id,
        run_key="verified-backup",
        watermarks={10001: message_ids[-1]},
    )


def test_resume_recovers_an_expired_worker_lease(sqlite_engine) -> None:
    message_ids = _seed_messages(sqlite_engine)
    background = _background(sqlite_engine)
    stale_now = datetime.now(UTC) - timedelta(minutes=5)
    background.enqueue_message(
        group_id=10001,
        message_id=message_ids[-1],
        now=stale_now,
    )
    claimed = background.store.claim_next_job(
        worker_id="stopped-worker",
        now=stale_now,
        lease_seconds=1,
    )
    assert claimed is not None

    report = run_memory_backfill(
        engine=sqlite_engine,
        background_service=background,
        manifest=_manifest(message_ids[-1]),
        run_key="resume-expired-lease",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )

    assert report.status == "completed"
    assert report.pending_jobs == report.running_jobs == report.failed_jobs == 0


def test_run_key_cannot_resume_with_different_generation(sqlite_engine) -> None:
    message_ids = _seed_messages(sqlite_engine)
    manifest = _manifest(message_ids[-1])
    run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=manifest,
        run_key="generation-pinned",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )
    with pytest.raises(ValueError, match="different backfill contract"):
        run_memory_backfill(
            engine=sqlite_engine,
            background_service=_background(sqlite_engine),
            manifest=manifest,
            run_key="generation-pinned",
            segmentation_generation="segment-v3",
            compaction_generation="compact-v2",
            index_generation="fts-v2",
        )


def test_run_key_cannot_resume_with_different_manifest_at_same_watermark(
    sqlite_engine,
) -> None:
    message_ids = _seed_messages(sqlite_engine)
    manifest = _manifest(message_ids[-1])
    run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=manifest,
        run_key="manifest-pinned",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )
    changed_manifest = {**manifest, "backup_name": "different-snapshot.db"}

    with pytest.raises(ValueError, match="different backfill contract"):
        run_memory_backfill(
            engine=sqlite_engine,
            background_service=_background(sqlite_engine),
            manifest=changed_manifest,
            run_key="manifest-pinned",
            segmentation_generation="segment-v2",
            compaction_generation="compact-v2",
            index_generation="fts-v2",
        )


def test_production_backfill_is_not_completed_before_external_final_gates(
    sqlite_engine,
) -> None:
    message_ids = _seed_messages(sqlite_engine)
    report = run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=_manifest(message_ids[-1]),
        run_key="provisional-until-ledger",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
        finalize=False,
    )

    assert report.status == "running"
    finalize_backfill_run(engine=sqlite_engine, run_id=report.run_id)
    completed = collect_backfill_coverage(
        engine=sqlite_engine,
        run_id=report.run_id,
        run_key="provisional-until-ledger",
        watermarks={10001: message_ids[-1]},
    )
    assert completed.status == "completed"


def test_backfill_rejects_worker_generation_identity_mismatch(sqlite_engine) -> None:
    message_ids = _seed_messages(sqlite_engine)

    with pytest.raises(ValueError, match="worker generation identity"):
        run_memory_backfill(
            engine=sqlite_engine,
            background_service=_background(
                sqlite_engine,
                segmentation_generation="segment-v3",
            ),
            manifest=_manifest(message_ids[-1]),
            run_key="wrong-worker-generation",
            segmentation_generation="segment-v2",
            compaction_generation="compact-v2",
            index_generation="fts-v2",
        )


def test_old_episode_generation_cannot_complete_a_new_backfill_run(
    sqlite_engine,
) -> None:
    message_ids = _seed_messages(sqlite_engine)
    manifest = _manifest(message_ids[-1])
    run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=manifest,
        run_key="old-generation",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        run_memory_backfill(
            engine=sqlite_engine,
            background_service=_background(
                sqlite_engine,
                segmentation_generation="segment-v3",
                compaction_generation="compact-v3",
            ),
            manifest=manifest,
            run_key="new-generation",
            segmentation_generation="segment-v3",
            compaction_generation="compact-v3",
            index_generation="fts-v2",
        )


def test_reserved_outbound_is_counted_but_not_required_for_episode_coverage(
    sqlite_engine,
) -> None:
    _seed_messages(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        reserved = MessageRepository(session).add_group_message(
            platform_msg_id="reserved-outbound",
            group_id=10001,
            user_id=20001,
            timestamp=NOW + timedelta(minutes=41),
            plain_text="reserved placeholder",
            raw_json={"direction": "outbound", "delivery_state": "reserved"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        watermark = int(reserved.id)

    report = run_memory_backfill(
        engine=sqlite_engine,
        background_service=_background(sqlite_engine),
        manifest=_manifest(watermark),
        run_key="reserved-safe",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        index_generation="fts-v2",
    )

    assert report.status == "completed"
    assert report.groups[0].messages == 3
    assert report.groups[0].eligible_messages == 2
    assert report.groups[0].assigned_messages == 2
