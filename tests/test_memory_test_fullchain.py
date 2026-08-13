import json

import pytest

from scripts import memory_test_fullchain as fullchain
from scripts.run_memory_v3_quality_replay import QualityReplayError


def test_stratify_limit_and_seed():
    cases = [{"category": f"c{i % 3}", "query": f"q{i}"} for i in range(30)]
    selected = fullchain._stratify(cases, limit=10, seed=1)
    assert len(selected) == 10
    categories = {case["category"] for case in selected}
    assert categories == {"c0", "c1", "c2"}
    again = fullchain._stratify(cases, limit=10, seed=1)
    assert [case["query"] for case in selected] == [case["query"] for case in again]


def test_cache_roundtrip(tmp_path):
    key = fullchain._sha256("hello")
    fullchain._cache_save(
        tmp_path,
        key,
        {"text": "x", "input_tokens": 1, "output_tokens": 2, "ttft_ms": 3.0, "model": "m"},
    )
    value = fullchain._cache_load(tmp_path, key)
    assert value is not None
    assert value["text"] == "x"
    assert fullchain._cache_load(tmp_path, "missing") is None


def test_dry_run_never_calls_transport(tmp_path):
    cases = [
        {"category": "preference", "kind": "preference", "query": "阿渣喜欢什么",
         "expected_evidence_message_ids": ["p1"], "group_id": 1001,
         "recent_context_message_ids": [], "requester_uin": "11"},
        {"category": "abstention", "kind": "abstention", "query": "晚上吃什么",
         "expected_evidence_message_ids": [], "group_id": 1001,
         "recent_context_message_ids": [], "requester_uin": "11"},
    ]
    rows, summary = fullchain.run_cases(
        None,
        cases,
        limit=10,
        seed=1,
        cache_dir=tmp_path,
        dry_run=True,
    )
    assert rows == []
    assert summary["mode"] == "dry-run"
    assert summary["cases"] == 2
    assert summary["estimated_cost_usd"] >= 0


def test_run_cases_orchestration_with_fake_case_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    cases = [
        {"category": f"c{i % 2}", "query": f"q{i}", "case_id": f"case{i}"}
        for i in range(20)
    ]
    calls: list[str] = []

    def fake_run_case(**kwargs):
        calls.append(str(kwargs["case_id"]))
        return {"case_id": str(kwargs["case_id"])}

    monkeypatch.setattr(fullchain, "_run_case", fake_run_case)
    injected = {
        "settings": fullchain.SimpleNamespace(bot_qq=123456789),
        "runtime": object(),
        "transport": object(),
    }
    rows, summary = fullchain.run_cases(
        None,
        cases,
        limit=6,
        seed=3,
        cache_dir=tmp_path,
        progress_path=tmp_path / "progress.jsonl",
        model="test-model",
        judge_model="test-model",
        **injected,
    )
    assert len(rows) == 6
    assert summary["executed"] == 6
    # Resume: all six are already in progress -> zero executed.
    rows2, summary2 = fullchain.run_cases(
        None,
        cases,
        limit=6,
        seed=3,
        cache_dir=tmp_path,
        resume=True,
        progress_path=tmp_path / "progress.jsonl",
        model="test-model",
        judge_model="test-model",
        **injected,
    )
    assert rows2 == []
    assert summary2["skipped_resumed"] == 6


def test_run_cases_detail_path_appends_rows_and_survives_resume(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    cases = [
        {"category": f"c{i % 2}", "query": f"q{i}", "case_id": f"case{i}"}
        for i in range(8)
    ]
    calls: list[str] = []

    def fake_run_case(**kwargs):
        calls.append(str(kwargs["case_id"]))
        return {"case_id": str(kwargs["case_id"])}

    monkeypatch.setattr(fullchain, "_run_case", fake_run_case)
    injected = {
        "settings": fullchain.SimpleNamespace(bot_qq=123456789),
        "runtime": object(),
        "transport": object(),
    }
    detail = tmp_path / "detail.jsonl"
    progress = tmp_path / "progress.jsonl"
    rows, _ = fullchain.run_cases(
        None,
        cases,
        limit=4,
        seed=1,
        cache_dir=tmp_path,
        progress_path=progress,
        detail_path=detail,
        model="m",
        judge_model="m",
        **injected,
    )
    assert len(rows) == 4
    first_lines = [
        json.loads(line) for line in detail.read_text(encoding="utf-8").splitlines()
    ]
    assert len(first_lines) == 4
    assert {row["case_id"] for row in first_lines} == {row["case_id"] for row in rows}
    # Resume executes the remaining cases and appends their rows; previously
    # checkpointed rows stay in the detail file.
    rows2, summary2 = fullchain.run_cases(
        None,
        cases,
        limit=8,
        seed=1,
        cache_dir=tmp_path,
        resume=True,
        progress_path=progress,
        detail_path=detail,
        model="m",
        judge_model="m",
        **injected,
    )
    assert summary2["skipped_resumed"] == 4
    assert len(rows2) == 4
    all_lines = [
        json.loads(line) for line in detail.read_text(encoding="utf-8").splitlines()
    ]
    assert len(all_lines) == 8


def test_generate_with_retries_recovers_and_gives_up():
    class FlakyTransport:
        def __init__(self, failures: int):
            self.failures = failures
            self.calls = 0

        def generate(self, prompt_lines, *, model):
            self.calls += 1
            if self.calls <= self.failures:
                raise QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED")
            return {"text": "ok"}

    flaky = FlakyTransport(2)
    result = fullchain._generate_with_retries(
        flaky, ["p"], model="m", attempts=3, backoff=0
    )
    assert flaky.calls == 3
    assert result["text"] == "ok"

    dead = FlakyTransport(99)
    with pytest.raises(QualityReplayError):
        fullchain._generate_with_retries(dead, ["p"], model="m", attempts=2, backoff=0)
    assert dead.calls == 2


def test_parse_answer_loose_structural_only():
    parsed = fullchain._parse_answer_loose(
        json.dumps(
            {
                "answer": "答案",
                "cited_source_message_ids": ["a", "b", "c", "a"],
                "abstained": False,
            }
        )
    )
    assert parsed.answer == "答案"
    assert parsed.cited_source_message_ids == ("a", "b", "c")
    assert parsed.abstained is False
    # No hard citation cap or citation-missing rule: judgment belongs to the model.
    no_citation = fullchain._parse_answer_loose(
        json.dumps(
            {"answer": "答案", "cited_source_message_ids": [], "abstained": False}
        )
    )
    assert no_citation.cited_source_message_ids == ()
    with pytest.raises(ValueError):
        fullchain._parse_answer_loose('{"answer": "", "cited_source_message_ids": [], "abstained": false}')
    with pytest.raises(ValueError):
        fullchain._parse_answer_loose('{"answer": "x", "cited_source_message_ids": "bad", "abstained": false}')


def test_merge_results_accumulates_by_case_id(tmp_path):
    path = tmp_path / "results.jsonl"
    fullchain._merge_results(path, [{"case_id": "a", "v": 1}])
    fullchain._merge_results(path, [{"case_id": "b", "v": 2}, {"case_id": "a", "v": 3}])
    rows = fullchain._load_cases(path)
    by_id = {row["case_id"]: row for row in rows}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"]["v"] == 3


def test_build_eval_clients_luna_medium_answer_and_luna_low_aux():
    class FakeSettings:
        llm_base_url = "https://api.example.test/v1"
        llm_api_key = "test-key"
        llm_max_output_tokens = 4096
        llm_timeout_seconds = 120.0

    answer_client, aux_client = fullchain._build_eval_clients(
        FakeSettings(),
        answer_model="gpt-5.6-luna",
        answer_effort="medium",
        aux_model="gpt-5.6-luna",
        aux_effort="low",
    )
    assert answer_client.responses_model == "gpt-5.6-luna"
    assert answer_client.reasoning_effort == "medium"
    assert aux_client.responses_model == "gpt-5.6-luna"
    assert aux_client.reasoning_effort == "low"

def test_build_answer_prompt_requires_citation_per_claim_and_recommendation_abstention():
    case = {
        "query": "阿渣最近有什么计划",
        "gold_text": "参考",
        "expected_evidence_message_ids": ["p1"],
    }
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        source_msg_ids=("p1",),
    )
    prompt = fullchain.build_answer_prompt(case, packet)
    text = "\n".join(prompt)
    assert "Every substantive factual claim in answer must trace to at least" in text
    assert "a partial but supported answer is required" in text
    assert "never infer" in text
    assert "recommendation, opinion" in text


def test_build_judge_prompt_allows_open_ended_partial_answers():
    case = {
        "query": "阿渣最近有什么计划",
        "gold_text": "参考",
        "expected_evidence_message_ids": ["p1"],
    }
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        source_msg_ids=("p1",),
    )
    prompt = fullchain.build_judge_prompt(case, "answer", ("p1",), False, packet)
    text = "\n".join(prompt)
    assert "open-ended questions" in text
    assert "Do not mark reference_mismatch merely because" in text
    assert "supported_alternative" in text
