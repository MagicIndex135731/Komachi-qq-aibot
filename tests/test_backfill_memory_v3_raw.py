from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text

from app.core.memory_backfill import create_verified_sqlite_backup
from app.storage.db import (
    activate_retrieval_vector_generation,
    build_engine,
    create_all,
    ensure_retrieval_vector_generation,
    refresh_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.models import RetrievalDocument
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)
import scripts.backfill_memory_v3_raw as backfill_v3


main = backfill_v3.main


def _write_activation_gate(
    path,
    *,
    prepared_report: dict,
    status: str = "passed",
) -> None:
    dataset_path = path.with_suffix(".dataset.jsonl")
    dataset_path.write_text('{"frozen":true}\n', encoding="utf-8")
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    fingerprint = "b" * 64
    quality_path = path.with_suffix(".quality.json")
    quality_path.write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256,
                "snapshot_manifest_sha256": prepared_report["manifest_sha256"],
                "retrieval_fingerprint_sha256": fingerprint,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    zero_metrics = {
        name: 0
        for name in (
            "group_leak_count",
            "subject_leak_count",
            "time_leak_count",
            "ineligible_source_count",
            "unresolved_source_count",
            "outside_snapshot_source_count",
            "forbidden_source_count",
            "plan_mismatch_count",
            "derived_evidence_count",
            "retrieval_over_150_count",
            "packet_over_150_count",
            "packet_over_24k_count",
            "recent_over_60_count",
            "citation_not_in_packet_count",
            "citation_forbidden_source_count",
            "citation_unresolved_source_count",
            "citation_group_leak_count",
            "citation_subject_leak_count",
            "citation_time_leak_count",
            "citation_ineligible_source_count",
        )
    }
    metrics = {
        **zero_metrics,
        "recall_at_150": 1.0,
        "recall_within_24k": 1.0,
        "time_bucket_coverage_rate": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "grounded_answer_accuracy": 1.0,
        "answer_accuracy": 1.0,
        "abstention_f1": 1.0,
        "index_visibility_p95_ms": 100.0,
        "ttft_p95_ms": 100.0,
        "retrieval_p95_ms": 100.0,
    }
    path.write_text(
        json.dumps(
            {
                "evaluation_schema_version": 3,
                "memory_path": "raw_message_v3",
                "dataset_sha256": dataset_sha256,
                "snapshot_manifest_sha256": prepared_report["manifest_sha256"],
                "retrieval_fingerprint_sha256": fingerprint,
                "case_count": 64,
                "vector_generation": prepared_report["vector_generation"],
                "quality_sidecar_present": True,
                "quality_sidecar_sha256": hashlib.sha256(
                    quality_path.read_bytes()
                ).hexdigest(),
                "metrics": metrics,
                "acceptance": {
                    "status": status,
                    "error_codes": [] if status == "passed" else ["quality"],
                },
            }
        ),
        encoding="utf-8",
    )


def _activation_artifact_args(gate_path) -> list[str]:
    return [
        "--dataset",
        str(gate_path.with_suffix(".dataset.jsonl")),
        "--quality-sidecar",
        str(gate_path.with_suffix(".quality.json")),
    ]


def test_activation_gate_binds_dataset_quality_and_retrieval_fingerprint(
    tmp_path,
) -> None:
    gate_path = tmp_path / "gate.json"
    prepared = {"manifest_sha256": "a" * 64, "vector_generation": 2}
    _write_activation_gate(gate_path, prepared_report=prepared)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    dataset_path = gate_path.with_suffix(".dataset.jsonl")
    quality_path = gate_path.with_suffix(".quality.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    backfill_v3._validate_activation_gate(
        gate,
        manifest_sha256=prepared["manifest_sha256"],
        generation=2,
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        quality_sidecar_sha256=hashlib.sha256(
            quality_path.read_bytes()
        ).hexdigest(),
        quality_sidecar=quality,
    )

    gate["dataset_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="dataset"):
        backfill_v3._validate_activation_gate(
            gate,
            manifest_sha256=prepared["manifest_sha256"],
            generation=2,
            dataset_sha256=hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest(),
            quality_sidecar_sha256=hashlib.sha256(
                quality_path.read_bytes()
            ).hexdigest(),
            quality_sidecar=quality,
        )


@pytest.mark.parametrize(
    ("metric_name", "invalid_value"),
    [
        ("answer_accuracy", float("nan")),
        ("citation_precision", float("inf")),
        ("recall_at_150", float("-inf")),
        ("ttft_p95_ms", True),
        ("retrieval_p95_ms", "100"),
    ],
)
def test_activation_gate_rejects_non_finite_or_non_numeric_metrics(
    tmp_path,
    metric_name,
    invalid_value,
) -> None:
    gate_path = tmp_path / "gate.json"
    prepared = {"manifest_sha256": "a" * 64, "vector_generation": 2}
    _write_activation_gate(gate_path, prepared_report=prepared)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    dataset_path = gate_path.with_suffix(".dataset.jsonl")
    quality_path = gate_path.with_suffix(".quality.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    gate["metrics"][metric_name] = invalid_value

    with pytest.raises((ValueError, RuntimeError)):
        backfill_v3._validate_activation_gate(
            gate,
            manifest_sha256=prepared["manifest_sha256"],
            generation=2,
            dataset_sha256=hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest(),
            quality_sidecar_sha256=hashlib.sha256(
                quality_path.read_bytes()
            ).hexdigest(),
            quality_sidecar=quality,
        )


def test_activation_gate_json_loader_rejects_nan(tmp_path) -> None:
    path = tmp_path / "non-standard.json"
    path.write_text('{"metrics":{"answer_accuracy":NaN}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        backfill_v3._load_strict_json(path)


def test_fts_only_raw_backfill_is_manifest_bounded_and_idempotent(tmp_path) -> None:
    database = tmp_path / "bot.db"
    engine = build_engine(database)
    create_all(engine)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        MessageRepository(session).add_group_message(
            platform_msg_id="raw-v3-source",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 29, tzinfo=UTC),
            plain_text="历史原文",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        MessageRepository(session).add_private_message(
            platform_msg_id="private-source",
            user_id=20001,
            timestamp=datetime(2026, 7, 29, tzinfo=UTC),
            plain_text="private rows must not enter the group index",
            raw_json={},
        )
    engine.dispose()
    backup = create_verified_sqlite_backup(
        database,
        tmp_path / "backups",
        backup_tag="raw-v3",
    )

    arguments = [
        "--phase",
        "prepare",
        "--database",
        str(database),
        "--manifest",
        str(backup.manifest_path),
        "--fts-only",
        "--batch-size",
        "1",
    ]
    assert main(arguments) == 0
    assert main(arguments) == 0

    engine = build_engine(database)
    with session_scope(engine) as session:
        documents = list(
            session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.document_kind == "raw_message_v3",
                    RetrievalDocument.status == "active",
                )
            )
        )
    engine.dispose()
    assert len(documents) == 1
    assert documents[0].content == "历史原文"


def test_vector_backfill_prepares_without_activation_then_checks_ledger_inside_explicit_activation(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "bot.db"
    engine = build_engine(database)
    create_all(engine)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        message = MessageRepository(session).add_group_message(
            platform_msg_id="raw-v3-vector-source",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 29, tzinfo=UTC),
            plain_text="历史向量原文",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        MessageRepository(session).add_private_message(
            platform_msg_id="private-vector-source",
            user_id=20001,
            timestamp=datetime(2026, 7, 29, tzinfo=UTC),
            plain_text="private vector rows must be ignored",
            raw_json={},
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        episode = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id="old-episode",
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="旧 episode 不应进入 V3 generation",
            metadata_json={},
            content_hash="old-episode",
            source_message_ids=[message.id],
            embedding_eligible=True,
            embedding_status="pending",
        )
        episode_id = int(episode.id)
    active_generation = ensure_retrieval_vector_generation(
        engine,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
    )
    assert active_generation is not None
    assert (
        write_retrieval_vector_embeddings(
            engine,
            generation=active_generation,
            rows=[(episode_id, 10001, [1.0, 0.0])],
        )
        == 1
    )
    assert (
        refresh_retrieval_vector_generation(
            engine,
            generation=active_generation,
            mark_ready=True,
        ).status
        == "ready"
    )
    assert activate_retrieval_vector_generation(
        engine,
        generation=active_generation,
        expected_active_generation=None,
    )
    engine.dispose()
    backup = create_verified_sqlite_backup(
        database,
        tmp_path / "backups",
        backup_tag="raw-v3-vector",
    )

    class FakeProvider:
        available = True
        identity = SimpleNamespace(
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
        )

        def embed_documents(self, texts):
            return [[0.0, 1.0] for _ in texts]

    settings = SimpleNamespace(
        memory_embedding_provider="fake",
        memory_embedding_device="cpu",
        memory_embedding_model="semantic",
        memory_embedding_dimensions=2,
        memory_embedding_cache_dir=None,
        memory_embedding_local_files_only=True,
        memory_embedding_version="v1",
        memory_embedding_base_url=None,
        memory_embedding_api_key=None,
        memory_embedding_timeout_seconds=5.0,
    )
    monkeypatch.setattr(backfill_v3, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        backfill_v3,
        "build_embedding_provider",
        lambda **_kwargs: FakeProvider(),
    )
    verification_count = 0
    real_verify = backfill_v3.verify_message_ledger_manifest
    real_activate = backfill_v3.activate_retrieval_vector_generation

    def verify(*args, **kwargs):
        nonlocal verification_count
        verification_count += 1
        return real_verify(*args, **kwargs)

    def activate(*args, **kwargs):
        assert verification_count == 2
        assert callable(kwargs["pre_activation_check"])
        return real_activate(*args, **kwargs)

    monkeypatch.setattr(backfill_v3, "verify_message_ledger_manifest", verify)
    monkeypatch.setattr(backfill_v3, "activate_retrieval_vector_generation", activate)
    prepared_output = tmp_path / "prepared-report.json"

    assert (
        main(
            [
                "--phase",
                "prepare",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--batch-size",
                "1",
                "--output",
                str(prepared_output),
            ]
        )
        == 0
    )
    assert verification_count == 2
    prepared_report = json.loads(prepared_output.read_text(encoding="utf-8"))
    failed_gate = tmp_path / "failed-gate.json"
    _write_activation_gate(
        failed_gate,
        prepared_report=prepared_report,
        status="failed",
    )
    with pytest.raises(RuntimeError, match="gate did not pass"):
        main(
            [
                "--phase",
                "activate",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--prepared-report",
                str(prepared_output),
                "--gate-report",
                str(failed_gate),
                *_activation_artifact_args(failed_gate),
            ]
        )

    engine = build_engine(database)
    with engine.connect() as connection:
        active = connection.execute(
            text(
                "SELECT generation FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).scalar_one()
    engine.dispose()
    assert int(active) == active_generation

    passed_gate = tmp_path / "passed-gate.json"
    _write_activation_gate(
        passed_gate,
        prepared_report=prepared_report,
    )
    assert (
        main(
            [
                "--phase",
                "activate",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--prepared-report",
                str(prepared_output),
                "--gate-report",
                str(passed_gate),
                *_activation_artifact_args(passed_gate),
                "--batch-size",
                "1",
            ]
        )
        == 0
    )
    assert verification_count == 3

    engine = build_engine(database)
    with engine.connect() as connection:
        state = connection.execute(
            text(
                "SELECT generation, physical_table, document_family, total_documents, "
                "indexed_documents FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).one()
        vector_count = connection.execute(
            text(f"SELECT count(*) FROM {state.physical_table}")
        ).scalar_one()
    engine.dispose()
    assert int(state.generation) != active_generation
    assert state.document_family == "raw_message_v3"
    assert int(state.total_documents) == 1
    assert int(state.indexed_documents) == 1
    assert int(vector_count) == 1

    engine = build_engine(database)
    with session_scope(engine) as session:
        recalled = MessageRepository(session).mark_group_message_deleted(
            group_id=10001,
            platform_msg_id="raw-v3-vector-source",
        )
        assert recalled is not None
    engine.dispose()

    assert (
        main(
            [
                "--phase",
                "rollback",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--prepared-report",
                str(prepared_output),
            ]
        )
        == 0
    )
    engine = build_engine(database)
    with engine.connect() as connection:
        rolled_back = connection.execute(
            text(
                "SELECT generation, document_family FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).one()
    engine.dispose()
    assert int(rolled_back.generation) == active_generation
    assert rolled_back.document_family == ""


def test_vector_backfill_catches_live_messages_around_activation_and_resumes(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "bot.db"
    engine = build_engine(database)
    create_all(engine)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        MessageRepository(session).add_group_message(
            platform_msg_id="manifest-message",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 29, 0, 0, tzinfo=UTC),
            plain_text="manifest evidence",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
    engine.dispose()
    backup = create_verified_sqlite_backup(
        database,
        tmp_path / "backups",
        backup_tag="raw-v3-live-catchup",
    )

    engine = build_engine(database)
    with session_scope(engine) as session:
        live_before = MessageRepository(session).add_group_message(
            platform_msg_id="live-before-activation",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
            plain_text="live before activation",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        projected = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=live_before.id,
            embedding_generation=None,
        )
        assert projected is not None
        live_before_id = int(live_before.id)
    engine.dispose()

    class FakeProvider:
        available = True
        identity = SimpleNamespace(
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
        )

        def embed_documents(self, texts):
            return [[0.0, 1.0] for _ in texts]

    settings = SimpleNamespace(
        memory_embedding_provider="fake",
        memory_embedding_device="cpu",
        memory_embedding_model="semantic",
        memory_embedding_dimensions=2,
        memory_embedding_cache_dir=None,
        memory_embedding_local_files_only=True,
        memory_embedding_version="v1",
        memory_embedding_base_url=None,
        memory_embedding_api_key=None,
        memory_embedding_timeout_seconds=5.0,
    )
    monkeypatch.setattr(backfill_v3, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        backfill_v3,
        "build_embedding_provider",
        lambda **_kwargs: FakeProvider(),
    )
    real_activate = backfill_v3.activate_retrieval_vector_generation
    inserted_after_activation = False
    live_after_id = 0

    def activate(*args, **kwargs):
        nonlocal inserted_after_activation, live_after_id
        activated = real_activate(*args, **kwargs)
        if activated and not inserted_after_activation:
            inserted_after_activation = True
            engine = build_engine(database)
            with session_scope(engine) as session:
                live_after = MessageRepository(session).add_group_message(
                    platform_msg_id="live-after-activation",
                    group_id=10001,
                    user_id=20001,
                    timestamp=datetime(2026, 7, 29, 0, 2, tzinfo=UTC),
                    plain_text="live after activation",
                    raw_json={},
                    msg_type="text",
                    reply_to_msg_id=None,
                    mentioned_bot=False,
                )
                session.flush()
                projected = RetrievalDocumentRepository(session).project_raw_message_v3(
                    group_id=10001,
                    message_id=live_after.id,
                    embedding_generation=None,
                )
                assert projected is not None
                live_after_id = int(live_after.id)
            engine.dispose()
        return activated

    monkeypatch.setattr(backfill_v3, "activate_retrieval_vector_generation", activate)
    prepared_output = tmp_path / "prepared-report.json"
    assert (
        main(
            [
                "--phase",
                "prepare",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--batch-size",
                "1",
                "--output",
                str(prepared_output),
            ]
        )
        == 0
    )
    prepared_report = json.loads(prepared_output.read_text(encoding="utf-8"))
    generation = int(prepared_report["vector_generation"])
    assert prepared_report["phase"] == "prepared"
    assert prepared_report["live_above_watermark"] == 1
    assert prepared_report["live_catchup_high_watermarks"] == {"10001": live_before_id}
    gate_output = tmp_path / "passed-gate.json"
    _write_activation_gate(
        gate_output,
        prepared_report=prepared_report,
    )

    activated_output = tmp_path / "activated-report.json"
    assert (
        main(
            [
                "--phase",
                "activate",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--prepared-report",
                str(prepared_output),
                "--gate-report",
                str(gate_output),
                *_activation_artifact_args(gate_output),
                "--batch-size",
                "1",
                "--output",
                str(activated_output),
            ]
        )
        == 0
    )
    activated_report = json.loads(activated_output.read_text(encoding="utf-8"))
    assert activated_report["phase"] == "activated"
    assert activated_report["live_above_watermark"] == 2
    assert activated_report["live_catchup_high_watermarks"] == {"10001": live_after_id}

    engine = build_engine(database)
    with session_scope(engine) as session:
        caught_up = list(
            session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.document_kind == "raw_message_v3",
                    RetrievalDocument.source_id.in_(
                        (str(live_before_id), str(live_after_id))
                    ),
                )
            )
        )
        assert len(caught_up) == 2
        assert {
            (document.embedding_generation, document.embedding_status)
            for document in caught_up
        } == {(generation, "ready")}
    with engine.connect() as connection:
        generation_count_before = int(
            connection.execute(
                text(
                    "SELECT count(*) FROM retrieval_index_state "
                    "WHERE channel = 'vector'"
                )
            ).scalar_one()
        )
    engine.dispose()

    with pytest.raises(
        RuntimeError,
        match="not resumable",
    ):
        main(
            [
                "--phase",
                "prepare",
                "--database",
                str(database),
                "--manifest",
                str(backup.manifest_path),
                "--batch-size",
                "1",
                "--resume-generation",
                str(generation),
            ]
        )

    engine = build_engine(database)
    with engine.connect() as connection:
        generation_count_after = int(
            connection.execute(
                text(
                    "SELECT count(*) FROM retrieval_index_state "
                    "WHERE channel = 'vector'"
                )
            ).scalar_one()
        )
    engine.dispose()
    assert generation_count_after == generation_count_before
