import json

from scripts import memory_test_fullchain as fullchain


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
