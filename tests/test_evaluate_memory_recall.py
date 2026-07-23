from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.evaluate_memory_recall import (
    EvaluationCase,
    EvaluationResult,
    compute_case_evidence_sha256,
    evaluate,
    load_evaluation_cases,
    main,
    validate_real_dataset,
    validate_real_dataset_review,
)
from scripts.run_memory_recall_eval import (
    AcceptanceGateError,
    _acceptance_failures,
    _benchmark,
    _validate_local_vector_runtime,
    _validate_rollout_state,
)
from app.storage.db import ensure_retrieval_vector_generation


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def test_evaluation_case_schema_is_strict_and_validates_categories_and_time_ranges(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "group_id": 100,
                "query": "后来怎么样了？",
                "recent_context_message_ids": ["recent-1"],
                "expected_evidence_message_ids": ["evidence-1"],
                "category": "temporal",
                "time_range": {"start_at": "2026-01-01T00:00:00Z", "end_at": "2026-01-02T00:00:00Z"},
            }
        ],
    )

    cases, dataset_sha256 = load_evaluation_cases(dataset)

    assert cases == (
        EvaluationCase(
            group_id=100,
            query="后来怎么样了？",
            recent_context_message_ids=("recent-1",),
            expected_evidence_message_ids=("evidence-1",),
            category="temporal",
            time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        ),
    )
    assert len(dataset_sha256) == 64

    write_jsonl(
        dataset,
        [
            {
                "group_id": 100,
                "query": "x",
                "recent_context_message_ids": [],
                "expected_evidence_message_ids": [],
                "category": "not-a-category",
            }
        ],
    )
    with pytest.raises(ValueError, match="category"):
        load_evaluation_cases(dataset)

    write_jsonl(
        dataset,
        [
            {
                "group_id": 100,
                "query": "x",
                "recent_context_message_ids": [],
                "expected_evidence_message_ids": [],
                "category": "exact",
                "extra": "forbidden",
            }
        ],
    )
    with pytest.raises(ValueError, match="unexpected fields"):
        load_evaluation_cases(dataset)


def test_evaluation_aggregates_v1_v2_and_category_metrics_without_reported_chat_content() -> None:
    cases = (
        EvaluationCase(100, "private query one", ("recent-1",), ("a", "b"), "exact"),
        EvaluationCase(200, "private query two", (), ("c",), "paraphrase"),
    )
    results = (
        EvaluationResult(0, "v1", ("a", "x", "b"), ("a",), 10, 2.0, False),
        EvaluationResult(1, "v1", ("x",), (), 30, 4.0, False),
        EvaluationResult(
            0, "v2", ("x", "b", "a"), ("b",), 20, 6.0, True,
            retrieved_evidence_units=(("x",), ("b",), ("a",)),
        ),
        EvaluationResult(
            1, "v2", ("c",), ("c",), 40, 8.0, True,
            retrieved_evidence_units=(("c",),),
        ),
    )

    report = evaluate(cases=cases, results=results, dataset_sha256="a" * 64, recall_k=2)
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["dataset_sha256"] == "a" * 64
    assert report["case_count"] == 2
    assert report["variants"]["v1"]["recall_at_k"] == pytest.approx(0.25)
    assert report["variants"]["v1"]["mrr"] == pytest.approx(0.5)
    assert report["variants"]["v1"]["ndcg"] == pytest.approx(0.25)
    assert report["variants"]["v1"]["packed_evidence_hit_rate"] == pytest.approx(0.5)
    assert report["variants"]["v2"]["recall_at_k"] == pytest.approx(0.75)
    assert report["variants"]["v2"]["rewrite_rate"] == pytest.approx(1.0)
    assert report["variants"]["v2"]["categories"]["paraphrase"]["mean_context_tokens"] == pytest.approx(40.0)
    assert report["variants"]["v2"]["groups"]["100"]["case_count"] == 1
    assert report["variants"]["v2"]["groups"]["200"]["recall_at_k"] == pytest.approx(1.0)
    assert "private query" not in serialized
    assert "recent-1" not in serialized
    assert '"a"' not in serialized


def test_recall_k_counts_ranked_retrieval_units_instead_of_flattened_window_messages() -> None:
    cases = (
        EvaluationCase(100, "query", ("recent",), ("target",), "exact"),
    )
    result = EvaluationResult(
        0,
        "v2",
        ("window-1", "window-2", "target"),
        ("target",),
        20,
        3.0,
        False,
        retrieved_evidence_units=(
            ("window-1", "window-2"),
            ("target", "window-3"),
        ),
    )

    report = evaluate(
        cases=cases,
        results=(
            EvaluationResult(0, "v1", (), (), 0, 1.0, False),
            result,
        ),
        dataset_sha256="b" * 64,
        recall_k=2,
    )

    assert report["variants"]["v2"]["recall_at_k"] == 1.0
    assert report["variants"]["v2"]["mrr"] == 0.5


def test_v2_recall_units_are_required_bounded_and_deduplicated() -> None:
    cases = (
        EvaluationCase(100, "query", (), ("target",), "exact"),
    )
    with pytest.raises(ValueError, match="retrieved_evidence_units"):
        evaluate(
            cases=cases,
            results=(
                EvaluationResult(0, "v1", (), (), 0, 1.0, False),
                EvaluationResult(0, "v2", ("target",), (), 0, 1.0, False),
            ),
            dataset_sha256="c" * 64,
        )

    units = tuple(("duplicate",) for _ in range(10)) + (("target",),)
    report = evaluate(
        cases=cases,
        results=(
            EvaluationResult(0, "v1", (), (), 0, 1.0, False),
            EvaluationResult(
                0, "v2", ("duplicate", "target"), (), 0, 1.0, False,
                retrieved_evidence_units=units,
            ),
        ),
        dataset_sha256="d" * 64,
        recall_k=10,
    )
    assert report["variants"]["v2"]["recall_at_k"] == 0.0


def test_ndcg_stays_bounded_when_one_window_contains_multiple_expected_messages() -> None:
    cases = (
        EvaluationCase(100, "query", (), ("a", "b"), "multi_hop"),
    )
    report = evaluate(
        cases=cases,
        results=(
            EvaluationResult(0, "v1", (), (), 0, 1.0, False),
            EvaluationResult(
                0, "v2", ("a", "b"), ("a", "b"), 1, 1.0, False,
                retrieved_evidence_units=(("a", "b"),),
            ),
        ),
        dataset_sha256="e" * 64,
    )
    assert report["variants"]["v2"]["ndcg"] == 1.0


def test_cli_accepts_fixture_results_jsonl_and_writes_a_safe_report(tmp_path, capsys) -> None:
    dataset = tmp_path / "dataset.jsonl"
    results = tmp_path / "results.jsonl"
    output = tmp_path / "report.json"
    write_jsonl(
        dataset,
        [
            {
                "group_id": 100,
                "query": "sensitive question",
                "recent_context_message_ids": ["recent-1"],
                "expected_evidence_message_ids": ["evidence-1"],
                "category": "exact",
            }
        ],
    )
    write_jsonl(
        results,
        [
            {
                "case_index": 0,
                "variant": "v1",
                "retrieved_evidence_message_ids": ["evidence-1"],
                "packed_evidence_message_ids": ["evidence-1"],
                "context_tokens": 12,
                "latency_ms": 4.5,
                "rewrite_used": False,
            },
            {
                "case_index": 0,
                "variant": "v2",
                "retrieved_evidence_message_ids": ["evidence-1"],
                "retrieved_evidence_units": [["evidence-1"]],
                "packed_evidence_message_ids": ["evidence-1"],
                "context_tokens": 10,
                "latency_ms": 3.0,
                "rewrite_used": True,
            },
        ],
    )

    assert main(["--dataset", str(dataset), "--results", str(results), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["variants"]["v2"]["rewrite_rate"] == 1.0
    assert "sensitive question" not in output.read_text(encoding="utf-8")
    assert "evidence-1" not in output.read_text(encoding="utf-8")
    assert "sensitive question" not in capsys.readouterr().out


def test_memory_evaluation_and_model_backup_artifacts_are_ignored() -> None:
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

    assert "data/memory_eval/" in gitignore
    assert "data/models/" in gitignore
    assert "data/backups/" in gitignore


def test_real_dataset_gate_enforces_size_and_per_category_quotas() -> None:
    quotas = {
        "exact": 10,
        "paraphrase": 10,
        "vague_reference": 10,
        "temporal": 10,
        "multi_hop": 8,
        "update": 8,
        "abstention": 8,
    }
    cases = tuple(
        EvaluationCase(
            100,
            f"q-{category}-{index}",
            (),
            () if category == "abstention" else (f"e-{category}-{index}",),
            category,
            quoted_context_message_id=(
                f"quoted-{index}" if category == "vague_reference" else None
            ),
        )
        for category, count in quotas.items()
        for index in range(count)
    )

    validate_real_dataset(cases)

    missing_abstention = cases[:-1] + (
        EvaluationCase(100, "extra-exact", (), ("e-extra",), "exact"),
    )
    with pytest.raises(ValueError, match="quota"):
        validate_real_dataset(missing_abstention)


def test_real_dataset_rejects_empty_or_recent_evidence() -> None:
    quotas = {
        "exact": 10, "paraphrase": 10, "vague_reference": 10,
        "temporal": 10, "multi_hop": 8, "update": 8, "abstention": 8,
    }
    cases = [
        EvaluationCase(
            100,
            f"q-{category}-{index}",
            (),
            () if category == "abstention" else (f"e-{category}-{index}",),
            category,
            quoted_context_message_id=(
                f"quoted-{index}" if category == "vague_reference" else None
            ),
        )
        for category, count in quotas.items()
        for index in range(count)
    ]
    cases[0] = EvaluationCase(100, "empty", (), (), "exact")
    with pytest.raises(ValueError, match="require evidence"):
        validate_real_dataset(cases)
    cases[0] = EvaluationCase(100, "overlap", ("same",), ("same",), "exact")
    with pytest.raises(ValueError, match="separate"):
        validate_real_dataset(cases)


def test_real_dataset_review_is_item_approved_and_bound_to_dataset_hash(tmp_path) -> None:
    quotas = {
        "exact": 10, "paraphrase": 10, "vague_reference": 10,
        "temporal": 10, "multi_hop": 8, "update": 8, "abstention": 8,
    }
    cases = tuple(
        EvaluationCase(
            100,
            f"q-{category}-{index}",
            (),
            () if category == "abstention" else (f"e-{category}-{index}",),
            category,
            quoted_context_message_id=(
                f"quoted-{index}" if category == "vague_reference" else None
            ),
        )
        for category, count in quotas.items()
        for index in range(count)
    )
    digest = "a" * 64
    database = tmp_path / "review.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE messages (platform_msg_id TEXT PRIMARY KEY, group_id INTEGER, plain_text TEXT)"
        )
        connection.executemany(
            "INSERT INTO messages(platform_msg_id, group_id, plain_text) VALUES (?, ?, ?)",
            [
                (source_id, case.group_id, f"evidence text {source_id}")
                for case in cases
                for source_id in case.expected_evidence_message_ids
            ],
        )
        connection.commit()
    finally:
        connection.close()
    review = {
        "review_version": 1,
        "dataset_sha256": digest,
        "cases": [
            {
                "case_index": index,
                "group_id": case.group_id,
                "category": case.category,
                "expected_evidence_message_ids": list(case.expected_evidence_message_ids),
                "evidence_sha256": compute_case_evidence_sha256(database, case),
                "approved": True,
                "reviewer": "release-reviewer",
                "reviewed_at": "2026-07-23T12:00:00Z",
            }
            for index, case in enumerate(cases)
        ],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    validate_real_dataset_review(
        cases,
        dataset_sha256=digest,
        review_path=review_path,
        database=database,
    )
    review["snapshot_manifest_sha256"] = "d" * 64
    review_path.write_text(json.dumps(review), encoding="utf-8")
    validate_real_dataset_review(
        cases,
        dataset_sha256=digest,
        review_path=review_path,
        database=database,
        snapshot_manifest_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="snapshot manifest"):
        validate_real_dataset_review(
            cases,
            dataset_sha256=digest,
            review_path=review_path,
            database=database,
            snapshot_manifest_sha256="e" * 64,
        )
    review["cases"][3]["approved"] = False
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        validate_real_dataset_review(
            cases,
            dataset_sha256=digest,
            review_path=review_path,
            database=database,
        )
    review["cases"][3]["approved"] = True
    review["dataset_sha256"] = "c" * 64
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_real_dataset_review(
            cases,
            dataset_sha256=digest,
            review_path=review_path,
            database=database,
        )
    review["dataset_sha256"] = digest
    review["cases"][0]["evidence_sha256"] = "b" * 64
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match database"):
        validate_real_dataset_review(
            cases,
            dataset_sha256=digest,
            review_path=review_path,
            database=database,
        )


def test_acceptance_gate_enforces_recall_and_local_p95() -> None:
    def category(recall: float, count: int = 10) -> dict[str, float | int]:
        return {"recall_at_k": recall, "case_count": count}
    report = {
        "variants": {
            "v1": {"categories": {"exact": category(.9), "paraphrase": category(.4), "vague_reference": category(.4)}},
            "v2": {"categories": {"exact": category(.94), "paraphrase": category(.6), "vague_reference": category(.6)}},
        }
    }
    assert _acceptance_failures(report=report, benchmark={"p95_latency_ms": 500.0}) == (
        "AC4_KEYWORD_RECALL", "AC5_VAGUE_REWRITE_RECALL", "AC7_LOCAL_P95",
    )
    report["variants"]["v2"]["categories"]["exact"]["recall_at_k"] = .95
    report["variants"]["v2"]["categories"]["paraphrase"]["recall_at_k"] = .7
    report["variants"]["v2"]["categories"]["vague_reference"]["recall_at_k"] = .7
    assert _acceptance_failures(report=report, benchmark={"p95_latency_ms": 499.9}) == ()


def test_benchmark_reports_explicitly_disabled_remote_paths() -> None:
    class Provider:
        def evaluate(self, request):
            return request
    benchmark = _benchmark(requests=(object(),), provider=Provider(), warmup=20, runs=250)
    assert benchmark["warmup_runs"] == 20
    assert benchmark["measured_runs"] == 250
    assert benchmark["rewrite_enabled"] is False
    assert benchmark["rerank_enabled"] is False
    assert benchmark["network_enabled"] is False


def test_benchmark_fails_closed_when_vector_query_fails() -> None:
    class FailedVectorProvider:
        def evaluate(self, _request):
            return SimpleNamespace(
                attempted_channels=("bm25", "vector"),
                failed_channels=("vector",),
            )

    with pytest.raises(AcceptanceGateError) as failed:
        _benchmark(
            requests=(object(),),
            provider=FailedVectorProvider(),
            warmup=20,
            runs=250,
            enforce_vector=True,
        )
    assert failed.value.codes == ("AC_VECTOR_QUERY_FAILED",)

    class SuccessfulVectorProvider:
        def evaluate(self, _request):
            return SimpleNamespace(
                attempted_channels=("bm25", "vector"),
                failed_channels=(),
                channel_candidate_counts=(("bm25", 1), ("vector", 1)),
            )

    benchmark = _benchmark(
        requests=(object(),),
        provider=SuccessfulVectorProvider(),
        warmup=20,
        runs=250,
        enforce_vector=True,
    )
    assert benchmark["vector_success_verified"] is True

    class EmptyVectorProvider:
        def evaluate(self, _request):
            return SimpleNamespace(
                attempted_channels=("bm25", "vector"),
                failed_channels=(),
                channel_candidate_counts=(("bm25", 1), ("vector", 0)),
            )

    with pytest.raises(AcceptanceGateError) as empty:
        _benchmark(
            requests=(object(),),
            provider=EmptyVectorProvider(),
            warmup=20,
            runs=250,
            enforce_vector=True,
        )
    assert empty.value.codes == ("AC_VECTOR_NOT_EXERCISED",)


def test_real_benchmark_rejects_disabled_or_network_embedding_provider() -> None:
    disabled = SimpleNamespace(
        embedding_provider=SimpleNamespace(
            available=False,
            identity=SimpleNamespace(provider="disabled"),
        ),
        embedding_generation=None,
    )
    with pytest.raises(AcceptanceGateError) as unavailable:
        _validate_local_vector_runtime(disabled, warm=False)
    assert unavailable.value.codes == ("AC_VECTOR_UNAVAILABLE",)

    remote = SimpleNamespace(
        embedding_provider=SimpleNamespace(
            available=True,
            identity=SimpleNamespace(provider="openai_compatible"),
        ),
        embedding_generation=7,
    )
    with pytest.raises(AcceptanceGateError) as network:
        _validate_local_vector_runtime(remote, warm=False)
    assert network.value.codes == ("AC7_NETWORK_PROVIDER_FORBIDDEN",)

    class LazyFailureProvider:
        available = True
        identity = SimpleNamespace(provider="local")

        def embed_query(self, _query: str):
            self.available = False
            return None

    lazy_failure = SimpleNamespace(
        embedding_provider=LazyFailureProvider(),
        embedding_generation=7,
    )
    with pytest.raises(AcceptanceGateError) as lazy:
        _validate_local_vector_runtime(lazy_failure, warm=True)
    assert lazy.value.codes == ("AC_VECTOR_UNAVAILABLE",)


def test_real_benchmark_rejects_building_or_inactive_vector_generation(
    sqlite_engine,
    tmp_path,
) -> None:
    generation = ensure_retrieval_vector_generation(
        sqlite_engine,
        provider="local",
        model="test-model",
        dimensions=3,
        version="test-v1",
    )
    assert generation is not None
    runtime = SimpleNamespace(
        embedding_provider=SimpleNamespace(
            available=True,
            identity=SimpleNamespace(provider="local"),
        ),
        embedding_generation=generation,
    )
    with pytest.raises(AcceptanceGateError) as inactive:
        _validate_rollout_state(
            engine=sqlite_engine,
            runtime=runtime,
            database=tmp_path / "unused.db",
            run_key="missing-run",
        )
    assert inactive.value.codes == ("AC_VECTOR_NOT_ACTIVE",)
