import json
from datetime import UTC, datetime

import pytest

from app.core.memory_context_packer import EvidenceMessage, MemoryFact
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


def test_stratify_sixty_is_exact_prefix_of_three_hundred():
    cases = [
        {"category": f"c{i % 10}", "query": f"q{i}", "case_id": f"case-{i}"}
        for i in range(400)
    ]
    sixty = fullchain._stratify(cases, limit=60, seed=20260811)
    three_hundred = fullchain._stratify(cases, limit=300, seed=20260811)

    assert [row["case_id"] for row in sixty] == [
        row["case_id"] for row in three_hundred[:60]
    ]


def test_filter_categories_applies_allowlist_before_stratification():
    cases = [
        {"category": "running_joke", "query": "q1"},
        {"category": "raw_history", "query": "q2"},
        {"category": "profile", "query": "q3"},
    ]
    selected = fullchain._filter_categories(cases, " running_joke, profile,missing ")
    assert [case["category"] for case in selected] == ["running_joke", "profile"]
    assert fullchain._filter_categories(cases, "") == cases


def test_filter_case_ids_is_exact_and_fails_closed():
    cases = [
        {"case_id": "event-1", "category": "event", "query": "q1"},
        {"case_id": "joke-2", "category": "running_joke", "query": "q2"},
    ]

    selected = fullchain._filter_case_ids(cases, " joke-2,event-1 ")

    assert [case["case_id"] for case in selected] == ["event-1", "joke-2"]
    assert fullchain._filter_case_ids(cases, "") == cases
    with pytest.raises(ValueError, match="missing"):
        fullchain._filter_case_ids(cases, "missing")


def test_fullchain_parser_defaults_to_production_channel_timeout():
    args = fullchain.build_argument_parser().parse_args(
        [
            "--database",
            "snapshot.db",
            "--cases",
            "cases.jsonl",
            "--output-detail",
            "detail.jsonl",
        ]
    )
    assert args.channel_timeout == 4.0
    assert args.prewarm_embedding is False
    assert args.categories == ""
    assert args.judge_packet_mode == "full"
    assert args.answer_packet_mode == "full"


def test_fullchain_parser_accepts_focused_answer_packet_mode():
    args = fullchain.build_argument_parser().parse_args(
        [
            "--database",
            "snapshot.db",
            "--cases",
            "cases.jsonl",
            "--output-detail",
            "detail.jsonl",
            "--answer-packet-mode",
            "focused",
        ]
    )
    assert args.answer_packet_mode == "focused"


def test_answer_focused_packet_text_trims_low_ranked_segments(monkeypatch):
    segment = fullchain.SimpleNamespace(
        episode_id="e",
        fused_score=1.0,
        messages=(),
        hit_source_msg_ids=(),
        document_id=None,
        atomic_source_groups=(),
        pinned=False,
        blocked_output_present=False,
    )
    packed = fullchain.SimpleNamespace(
        blocked_output_present=False,
        grounding_policy="grounding contract",
        facts=(),
        summaries=(),
        evidence_segments=tuple(segment for _ in range(15)),
        recent_messages=(),
        text="",
    )
    rendered: list[object] = []

    def fake_render_segment(segment):
        rendered.append(segment)
        return "SEG"

    monkeypatch.setattr(
        fullchain.MemoryContextPacker,
        "_render_segment",
        staticmethod(fake_render_segment),
    )
    focused = fullchain._answer_focused_packet_text(packed)
    assert len(rendered) == fullchain.ANSWER_FOCUSED_RAW_SEGMENT_LIMIT
    assert "grounding contract" in focused
    assert focused.count("SEG") == fullchain.ANSWER_FOCUSED_RAW_SEGMENT_LIMIT


def test_run_cases_prewarms_actual_runtime_embedding_provider(monkeypatch, tmp_path):
    class FakeEmbeddingProvider:
        available = True
        active_accelerator = "cuda"
        identity = fullchain.SimpleNamespace(provider="local", model="test-model")

        def __init__(self):
            self.queries: list[str] = []

        def embed_query(self, query: str):
            self.queries.append(query)
            return [0.1, 0.2]

    provider = FakeEmbeddingProvider()
    runtime = fullchain.SimpleNamespace(embedding_provider=provider)
    monkeypatch.setattr(
        fullchain,
        "_run_case",
        lambda **kwargs: {"case_id": str(kwargs["case_id"])},
    )
    rows, summary = fullchain.run_cases(
        None,
        [{"category": "fact", "query": "q", "case_id": "case-1"}],
        limit=1,
        seed=1,
        cache_dir=tmp_path,
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        runtime=runtime,
        transport=object(),
        prewarm_embedding=True,
    )

    assert [row["case_id"] for row in rows] == ["case-1"]
    assert len(rows[0]["case_input_signature"]) == 64
    assert provider.queries == [fullchain.EMBEDDING_PREWARM_QUERY]
    assert summary["embedding_prewarm"] == {
        "provider": "local",
        "model": "test-model",
        "accelerator": "cuda",
        "dimensions": 2,
    }


def test_embedding_prewarm_fails_before_cases_when_provider_is_unavailable():
    with pytest.raises(RuntimeError, match="embedding provider is unavailable"):
        fullchain._prewarm_embedding_runtime(fullchain.SimpleNamespace())


def test_cache_roundtrip(tmp_path):
    key = fullchain._sha256("hello")
    fullchain._cache_save(
        tmp_path,
        key,
        {
            "text": "x",
            "input_tokens": 1,
            "output_tokens": 2,
            "ttft_ms": 3.0,
            "model": "m",
        },
    )
    value = fullchain._cache_load(tmp_path, key)
    assert value is not None
    assert value["text"] == "x"
    assert fullchain._cache_load(tmp_path, "missing") is None


def test_extract_shadow_envelope_keeps_answer_json_intact():
    value = (
        '{"answer":"hi","cited_source_message_ids":[],"abstained":false}\n'
        'SHADOW_ENVELOPE: {"decision":"answer","claims":[{"text":"hi",'
        '"evidence_ids":["e1"],"source_ids":["m1"]}],"answer":"hi",'
        '"expansion_request":null}'
    )
    clean, envelope, error = fullchain._extract_shadow_envelope(value)
    assert error is None
    assert envelope is not None
    assert envelope["decision"] == "answer"
    assert json.loads(clean)["answer"] == "hi"
    assert "SHADOW_ENVELOPE" not in clean


def test_extract_shadow_envelope_records_invalid_decision_without_breaking_answer():
    value = (
        '{"answer":"hi","cited_source_message_ids":[],"abstained":false}\n'
        'SHADOW_ENVELOPE: {"decision":"guess"}'
    )
    clean, envelope, error = fullchain._extract_shadow_envelope(value)
    assert envelope is None
    assert error is not None
    assert json.loads(clean)["answer"] == "hi"


def test_extract_shadow_envelope_reads_inline_top_level_field():
    value = json.dumps(
        {
            "answer": "hi",
            "cited_source_message_ids": [],
            "abstained": False,
            "decision_envelope": {
                "decision": "answer",
                "claims": [
                    {"text": "hi", "evidence_ids": ["e1"], "source_ids": ["m1"]}
                ],
                "answer": "hi",
                "expansion_request": None,
            },
        },
        ensure_ascii=False,
    )
    clean, envelope, error = fullchain._extract_shadow_envelope(value)
    assert error is None
    assert envelope is not None
    assert envelope["decision"] == "answer"
    assert clean == value


def test_extract_shadow_envelope_reports_invalid_inline_field():
    value = (
        '{"answer":"hi","cited_source_message_ids":[],"abstained":false,'
        '"decision_envelope":{"decision":"guess"}}'
    )
    clean, envelope, error = fullchain._extract_shadow_envelope(value)
    assert envelope is None
    assert error is not None
    assert json.loads(clean)["answer"] == "hi"


def test_answer_prompt_shadow_contract_includes_envelope_field():
    case = {"category": "fact", "query": "q"}
    packed = fullchain.SimpleNamespace(text="packet")
    plain = fullchain.build_answer_prompt(case, packed)
    shadow = fullchain.build_answer_prompt(case, packed, decision_envelope_shadow=True)
    plain_text = "\n".join(plain)
    shadow_text = "\n".join(shadow)
    assert "decision_envelope" not in plain_text
    assert "decision_envelope" in shadow_text
    assert (
        "fields answer, cited_source_message_ids, abstained, decision_envelope"
        in shadow_text
    )
    plain_without_contract = [
        message
        for message in plain
        if "Evaluation-only output contract" not in message
    ]
    shadow_without_contract = [
        message
        for message in shadow
        if "Evaluation-only output contract" not in message
    ]
    assert plain_without_contract == shadow_without_contract


def test_run_cases_forwards_decision_envelope_shadow(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run_case(**kwargs):
        captured.update(kwargs)
        return {"case_id": str(kwargs["case_id"])}

    monkeypatch.setattr(fullchain, "_run_case", fake_run_case)
    rows, summary = fullchain.run_cases(
        None,
        [{"category": "fact", "query": "q", "case_id": "case-1"}],
        limit=1,
        seed=1,
        cache_dir=tmp_path,
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        runtime=fullchain.SimpleNamespace(),
        transport=object(),
        decision_envelope_shadow=True,
    )

    assert captured["decision_envelope_shadow"] is True
    assert summary["decision_envelope_shadow"] is True
    assert [row["case_id"] for row in rows] == ["case-1"]


def test_run_case_sends_shadow_prompt_and_records_envelope(monkeypatch, tmp_path):
    packed = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=(),
        text="packet",
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    trace = fullchain.SimpleNamespace(
        result=fullchain.SimpleNamespace(packed_context=packed),
        resolved_query=fullchain.SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
        channel_candidate_counts=(),
    )
    runtime = fullchain.SimpleNamespace(
        v2_provider=fullchain.SimpleNamespace(evaluate=lambda request: trace)
    )
    captured_prompts: list[list[str]] = []

    def fake_generate_with_retries(transport, prompt_lines, **kwargs):
        del transport, kwargs
        captured_prompts.append(list(prompt_lines))
        text = "\n".join(prompt_lines)
        if "Evaluation-only output contract" in text:
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": "没有足够的记忆素材回答这个问题。",
                        "cited_source_message_ids": [],
                        "abstained": True,
                        "decision_envelope": {
                            "decision": "abstain",
                            "claims": [],
                            "answer": "没有足够的记忆素材回答这个问题。",
                            "expansion_request": None,
                        },
                    },
                    ensure_ascii=False,
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model="m",
                attempt_count=1,
                no_event_attempts=0,
            )
        return fullchain.SimpleNamespace(
            text=json.dumps(
                {
                    "answer_grounded": True,
                    "answer_correct": True,
                    "abstained": True,
                    "reason_code": "abstained_no_evidence",
                },
                ensure_ascii=False,
            ),
            input_tokens=1,
            output_tokens=1,
            ttft_ms=1.0,
            model="m",
            attempt_count=1,
            no_event_attempts=0,
        )

    monkeypatch.setattr(fullchain, "_generate_with_retries", fake_generate_with_retries)
    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=object(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": [],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        decision_envelope_shadow=True,
    )

    assert "decision_envelope" in "\n".join(captured_prompts[0])
    assert row["decision_envelope_shadow"] is not None
    assert row["decision_envelope_shadow"]["decision"] == "abstain"
    assert row["decision_envelope_shadow_error"] is None
    assert row["abstained"] is True


def _enforce_envelope_fixture(
    *,
    answers: list[dict],
    judge: dict | None = None,
):
    packed = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(
            MemoryFact(
                text="valid fact",
                source_msg_ids=("m1",),
                memory_kind="fact",
            ),
        ),
        summaries=(),
        recent_messages=(),
        source_msg_ids=("m1",),
        text="packet",
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    trace = fullchain.SimpleNamespace(
        result=fullchain.SimpleNamespace(packed_context=packed),
        resolved_query=fullchain.SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
        channel_candidate_counts=(),
    )
    runtime = fullchain.SimpleNamespace(
        v2_provider=fullchain.SimpleNamespace(evaluate=lambda request: trace)
    )
    captured_prompts: list[list[str]] = []
    answer_index = 0

    def fake_generate_with_retries(transport, prompt_lines, **kwargs):
        del transport, kwargs
        captured_prompts.append(list(prompt_lines))
        text = "\n".join(prompt_lines)
        if "Evaluation-only output contract" in text:
            nonlocal answer_index
            answer = answers[answer_index]
            answer_index += 1
            return fullchain.SimpleNamespace(
                text=json.dumps(answer, ensure_ascii=False),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model="m",
                attempt_count=1,
                no_event_attempts=0,
            )
        decision = judge or {
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "reason_code": "supported_evidence",
        }
        return fullchain.SimpleNamespace(
            text=json.dumps(decision, ensure_ascii=False),
            input_tokens=1,
            output_tokens=1,
            ttft_ms=1.0,
            model="m",
            attempt_count=1,
            no_event_attempts=0,
        )

    return packed, trace, runtime, captured_prompts, fake_generate_with_retries


def test_run_case_enforce_envelope_reanswers_on_invalid_references(
    monkeypatch,
    tmp_path,
):
    (
        _packed,
        _trace,
        runtime,
        captured_prompts,
        fake_generate_with_retries,
    ) = _enforce_envelope_fixture(
        answers=[
            {
                "answer": "坏引用",
                "cited_source_message_ids": [],
                "abstained": True,
                "decision_envelope": {
                    "decision": "abstain",
                    "claims": [{"text": "坏", "evidence_ids": [], "source_ids": []}],
                    "answer": "坏引用",
                    "expansion_request": None,
                },
            },
            {
                "answer": "没有足够的记忆素材回答这个问题。",
                "cited_source_message_ids": [],
                "abstained": True,
                "decision_envelope": {
                    "decision": "abstain",
                    "claims": [],
                    "answer": "没有足够的记忆素材回答这个问题。",
                    "expansion_request": None,
                },
            },
        ],
    )
    monkeypatch.setattr(fullchain, "_generate_with_retries", fake_generate_with_retries)
    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=object(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": [],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        decision_envelope=True,
    )

    assert row["decision_envelope_reanswered"] is True
    assert row["decision_envelope_validation"] == {"ok": True, "failures": []}
    assert row["decision_envelope_shadow"]["decision"] == "abstain"
    assert len(captured_prompts) == 3
    assert "failed structural validation" in "\n".join(captured_prompts[1])


def test_run_case_enforce_envelope_accepts_valid_references(monkeypatch, tmp_path):
    (
        _packed,
        _trace,
        runtime,
        _captured_prompts,
        fake_generate_with_retries,
    ) = _enforce_envelope_fixture(
        answers=[
            {
                "answer": "他有一个有效事实。",
                "cited_source_message_ids": ["m1"],
                "abstained": False,
                "decision_envelope": {
                    "decision": "answer",
                    "claims": [
                        {
                            "text": "他有一个有效事实。",
                            "evidence_ids": ["m1"],
                            "source_ids": ["m1"],
                        }
                    ],
                    "answer": "他有一个有效事实。",
                    "expansion_request": None,
                },
            }
        ],
    )
    monkeypatch.setattr(fullchain, "_generate_with_retries", fake_generate_with_retries)
    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=object(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m1"],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        decision_envelope=True,
    )

    assert row["decision_envelope_reanswered"] is False
    assert row["decision_envelope_validation"] == {"ok": True, "failures": []}
    assert row["decision_envelope_expanded"] is False
    assert row["answer_correct"] is True


def test_run_case_enforce_envelope_records_expansion(monkeypatch, tmp_path):
    (
        _packed,
        _trace,
        runtime,
        _captured_prompts,
        fake_generate_with_retries,
    ) = _enforce_envelope_fixture(
        answers=[
            {
                "answer": "他有一个有效事实。",
                "cited_source_message_ids": ["m1"],
                "abstained": False,
                "decision_envelope": {
                    "decision": "expand",
                    "claims": [
                        {
                            "text": "他有一个有效事实。",
                            "evidence_ids": ["m1"],
                            "source_ids": ["m1"],
                        }
                    ],
                    "answer": "他有一个有效事实。",
                    "expansion_request": {
                        "facets": ["时间"],
                        "layers": ["raw"],
                    },
                },
            }
        ],
    )
    monkeypatch.setattr(fullchain, "_generate_with_retries", fake_generate_with_retries)
    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=object(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m1"],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        decision_envelope=True,
    )

    assert row["decision_envelope_validation"] == {"ok": True, "failures": []}
    assert row["decision_envelope_expanded"] is True
    assert row["decision_envelope_reanswered"] is False


def test_run_case_enforce_envelope_skips_local_citation_repair(
    monkeypatch,
    tmp_path,
):
    (
        _packed,
        _trace,
        runtime,
        _captured_prompts,
        fake_generate_with_retries,
    ) = _enforce_envelope_fixture(
        answers=[
            {
                "answer": "他有一个有效事实。",
                "cited_source_message_ids": ["outside"],
                "abstained": False,
                "decision_envelope": {
                    "decision": "answer",
                    "claims": [
                        {
                            "text": "他有一个有效事实。",
                            "evidence_ids": ["m1"],
                            "source_ids": ["m1"],
                        }
                    ],
                    "answer": "他有一个有效事实。",
                    "expansion_request": None,
                },
            }
        ],
    )
    repair_calls: list[tuple] = []

    def fake_repair(*args, **kwargs):
        repair_calls.append((args, kwargs))
        raise AssertionError("local citation repair must not run in enforce mode")

    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        fake_repair,
    )
    monkeypatch.setattr(fullchain, "_generate_with_retries", fake_generate_with_retries)
    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=object(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m1"],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        decision_envelope=True,
    )

    assert repair_calls == []
    assert row["decision_envelope_validation"] == {"ok": True, "failures": []}
    assert row["decision_envelope_reanswered"] is False


def test_run_case_enforce_envelope_reanswers_abstain_on_must_answer(
    monkeypatch,
    tmp_path,
):
    (
        _packed,
        _trace,
        runtime,
        captured_prompts,
        fake_generate_with_retries,
    ) = _enforce_envelope_fixture(
        answers=[
            {
                "answer": "没有足够的记忆素材回答这个问题。",
                "cited_source_message_ids": [],
                "abstained": True,
                "decision_envelope": {
                    "decision": "abstain",
                    "claims": [],
                    "answer": "没有足够的记忆素材回答这个问题。",
                    "expansion_request": None,
                },
            },
            {
                "answer": "他有一个有效事实。",
                "cited_source_message_ids": ["m1"],
                "abstained": False,
                "decision_envelope": {
                    "decision": "answer",
                    "claims": [
                        {
                            "text": "他有一个有效事实。",
                            "evidence_ids": ["m1"],
                            "source_ids": ["m1"],
                        }
                    ],
                    "answer": "他有一个有效事实。",
                    "expansion_request": None,
                },
            },
        ],
    )
    monkeypatch.setattr(fullchain, "_generate_with_retries", fake_generate_with_retries)
    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=object(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m1"],
            "answer_expectation": "must_answer",
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        decision_envelope=True,
    )

    assert row["decision_envelope_reanswered"] is True
    assert row["decision_envelope_validation"] == {"ok": True, "failures": []}
    assert row["decision_envelope_shadow"]["decision"] == "answer"
    assert "requires an answer" in "\n".join(captured_prompts[1])


def test_sanitize_citation_ids_drops_invalid_extras_without_inventing_ids():
    assert fullchain._sanitize_citation_ids(("valid", "bad", "valid-2"), ("valid", "valid-2")) == (
        ("valid", "valid-2"),
        True,
    )
    assert fullchain._sanitize_citation_ids(("bad",), ("valid",)) == (
        ("bad",),
        False,
    )
    assert fullchain._sanitize_citation_ids((), ("valid",)) == ((), False)


def test_resume_settings_projection_excludes_credentials():
    settings = fullchain.SimpleNamespace(
        memory_query_rewrite_enabled=False,
        memory_embedding_api_key="embedding-secret-sentinel",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="secret-sentinel",
        context_recent_limit=60,
        context_summary_limit=3,
        bot_qq=123456789,
    )

    projection = fullchain._settings_resume_projection(settings)

    assert "llm_api_key" not in projection
    assert "memory_embedding_api_key" not in projection
    assert "secret-sentinel" not in fullchain._canonical_json(projection)
    assert "embedding-secret-sentinel" not in fullchain._canonical_json(projection)
    assert projection["context_recent_limit"] == 60
    assert projection["context_summary_limit"] == 3


def test_dry_run_never_calls_transport(tmp_path):
    cases = [
        {
            "category": "preference",
            "kind": "preference",
            "query": "阿渣喜欢什么",
            "expected_evidence_message_ids": ["p1"],
            "group_id": 1001,
            "recent_context_message_ids": [],
            "requester_uin": "11",
        },
        {
            "category": "abstention",
            "kind": "abstention",
            "query": "晚上吃什么",
            "expected_evidence_message_ids": [],
            "group_id": 1001,
            "recent_context_message_ids": [],
            "requester_uin": "11",
        },
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


def test_resume_reexecutes_legacy_unsigned_progress(monkeypatch, tmp_path):
    cases = [{"category": "fact", "query": "q", "case_id": "case-1"}]
    progress = tmp_path / "progress.jsonl"
    progress.write_text('{"case_id":"case-1","ok":true}\n', encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        fullchain,
        "_run_case",
        lambda **kwargs: calls.append(str(kwargs["case_id"]))
        or {"case_id": str(kwargs["case_id"])},
    )

    rows, summary = fullchain.run_cases(
        None,
        cases,
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        resume=True,
        progress_path=progress,
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        runtime=object(),
        transport=object(),
    )

    assert calls == ["case-1"]
    assert len(rows) == 1
    assert summary["invalidated_resumed"] == 1
    latest = json.loads(progress.read_text(encoding="utf-8").splitlines()[-1])
    assert latest["case_input_signature"] == rows[0]["case_input_signature"]


@pytest.mark.parametrize(
    ("changed_case", "changed_effort", "changed_packet_mode"),
    [
        (
            {"category": "fact", "query": "changed", "case_id": "case-1"},
            "high",
            "full",
        ),
        (
            {"category": "fact", "query": "q", "case_id": "case-1"},
            "medium",
            "full",
        ),
        (
            {"category": "fact", "query": "q", "case_id": "case-1"},
            "high",
            "citation-focused",
        ),
    ],
)
def test_resume_reexecutes_when_case_or_configuration_changes(
    monkeypatch, tmp_path, changed_case, changed_effort, changed_packet_mode
):
    original = {"category": "fact", "query": "q", "case_id": "case-1"}
    progress = tmp_path / "progress.jsonl"
    calls: list[str] = []
    monkeypatch.setattr(
        fullchain,
        "_run_case",
        lambda **kwargs: calls.append(str(kwargs["case_id"]))
        or {"case_id": str(kwargs["case_id"])},
    )
    injected = {
        "settings": fullchain.SimpleNamespace(bot_qq=123456789),
        "runtime": object(),
        "transport": object(),
    }
    fullchain.run_cases(
        None,
        [original],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        progress_path=progress,
        answer_effort="high",
        **injected,
    )
    calls.clear()

    rows, summary = fullchain.run_cases(
        None,
        [changed_case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        resume=True,
        progress_path=progress,
        answer_effort=changed_effort,
        judge_packet_mode=changed_packet_mode,
        **injected,
    )

    assert calls == ["case-1"]
    assert len(rows) == 1
    assert summary["invalidated_resumed"] == 1


def test_resume_uses_latest_progress_state(monkeypatch, tmp_path):
    case = {"category": "fact", "query": "q", "case_id": "case-1"}
    progress = tmp_path / "progress.jsonl"
    calls: list[str] = []
    monkeypatch.setattr(
        fullchain,
        "_run_case",
        lambda **kwargs: calls.append(str(kwargs["case_id"]))
        or {"case_id": str(kwargs["case_id"])},
    )
    injected = {
        "settings": fullchain.SimpleNamespace(bot_qq=123456789),
        "runtime": object(),
        "transport": object(),
    }
    rows, _ = fullchain.run_cases(
        None,
        [case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        progress_path=progress,
        **injected,
    )
    signature = rows[0]["case_input_signature"]
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "case_id": "case-1",
                    "ok": False,
                    "case_input_signature": signature,
                }
            )
            + "\n"
        )
    calls.clear()

    resumed, summary = fullchain.run_cases(
        None,
        [case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        resume=True,
        progress_path=progress,
        **injected,
    )

    assert calls == ["case-1"]
    assert len(resumed) == 1
    assert summary["skipped_resumed"] == 0


def test_resume_reexecutes_protocol_failure(monkeypatch, tmp_path):
    case = {"category": "fact", "query": "q", "case_id": "case-1"}
    progress = tmp_path / "progress.jsonl"
    calls: list[str] = []

    def fake_run_case(**kwargs):
        calls.append(str(kwargs["case_id"]))
        return {
            "case_id": str(kwargs["case_id"]),
            "protocol_failure_codes": (
                ["answer_json_invalid"] if len(calls) == 1 else []
            ),
        }

    monkeypatch.setattr(fullchain, "_run_case", fake_run_case)
    injected = {
        "settings": fullchain.SimpleNamespace(bot_qq=123456789),
        "runtime": object(),
        "transport": object(),
    }
    fullchain.run_cases(
        None,
        [case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        progress_path=progress,
        **injected,
    )
    rows, summary = fullchain.run_cases(
        None,
        [case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        resume=True,
        progress_path=progress,
        **injected,
    )

    assert calls == ["case-1", "case-1"]
    assert len(rows) == 1
    assert summary["skipped_resumed"] == 0


def test_run_case_propagates_prejudge_repair_protocol_failure(monkeypatch, tmp_path):
    packed = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        source_msg_ids=(),
    )
    trace = fullchain.SimpleNamespace(
        result=fullchain.SimpleNamespace(packed_context=packed),
        resolved_query=fullchain.SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
        channel_candidate_counts=(),
    )
    runtime = fullchain.SimpleNamespace(
        v2_provider=fullchain.SimpleNamespace(evaluate=lambda request: trace)
    )

    class AnswerTransport:
        def generate(self, prompt_lines, *, model):
            del prompt_lines
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": "有内容但没有引用",
                        "cited_source_message_ids": [],
                        "abstained": False,
                    },
                    ensure_ascii=False,
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        lambda *args, **kwargs: fullchain.SimpleNamespace(
            answer=fullchain.GeneratedAnswer(
                answer="有内容但没有引用",
                cited_source_message_ids=(),
                abstained=False,
            ),
            protocol_failure_codes=("citation_repair_invalid",),
            observation=fullchain.SimpleNamespace(no_event_attempts=0),
        ),
    )

    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=AnswerTransport(),
        aux_transport=object(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": [],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
    )

    assert row["protocol_failure_codes"] == ["citation_repair_invalid"]
    assert row["answer_grounded"] is False
    assert row["answer_correct"] is False

    def fail_repair(*args, **kwargs):
        raise QualityReplayError(
            "QUALITY_REPLAY_PROVIDER_NO_EVENT",
            retryable=False,
            failure_kind="provider_no_event",
        )

    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        fail_repair,
    )
    provider_failed_row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=AnswerTransport(),
        aux_transport=object(),
        case={
            "case_id": "case-2",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": [],
        },
        case_id="case-2",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "provider-failed-cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
    )

    assert provider_failed_row["protocol_failure_codes"] == ["provider_failed"]
    assert (
        provider_failed_row["citation_repair_prejudge_error"]
        == "QUALITY_REPLAY_PROVIDER_NO_EVENT"
    )
    assert provider_failed_row["answer_grounded"] is False
    assert provider_failed_row["answer_correct"] is False


def test_run_case_accepts_canonical_abstention_repair_for_empty_allowlist(
    monkeypatch, tmp_path
):
    packed = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=(),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    trace = fullchain.SimpleNamespace(
        result=fullchain.SimpleNamespace(packed_context=packed),
        resolved_query=fullchain.SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
        channel_candidate_counts=(),
    )
    runtime = fullchain.SimpleNamespace(
        v2_provider=fullchain.SimpleNamespace(evaluate=lambda request: trace)
    )

    class AnswerTransport:
        def generate(self, prompt_lines, *, model):
            del prompt_lines
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": "recent-only unsupported draft",
                        "cited_source_message_ids": [],
                        "abstained": False,
                    }
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    class JudgeTransport:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt_lines, *, model):
            self.calls += 1
            prompt_text = "\n".join(prompt_lines)
            assert fullchain.FIXED_ABSTENTION_ANSWER in prompt_text
            assert "Generated abstained flag:\ntrue" in prompt_text
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer_grounded": True,
                        "answer_correct": True,
                        "abstained": True,
                        "reason_code": "supported_abstention",
                    }
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    judge_transport = JudgeTransport()
    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        lambda *args, **kwargs: fullchain.SimpleNamespace(
            answer=fullchain.GeneratedAnswer(
                answer=fullchain.FIXED_ABSTENTION_ANSWER,
                cited_source_message_ids=(),
                abstained=True,
            ),
            protocol_failure_codes=(),
            observation=fullchain.SimpleNamespace(no_event_attempts=0),
        ),
    )

    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=AnswerTransport(),
        aux_transport=judge_transport,
        case={
            "case_id": "empty-allowlist",
            "group_id": 1001,
            "query": "测试问题",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": [],
            "answer_expectation": "must_abstain",
        },
        case_id="empty-allowlist",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
    )

    assert judge_transport.calls == 1
    assert row["protocol_failure_codes"] == []
    assert row["generated_abstained"] is True
    assert row["abstained"] is True
    assert row["answer"] == fullchain.FIXED_ABSTENTION_ANSWER
    assert row["cited_source_message_ids"] == []
    assert row["repaired"] is True
    assert row["judge_attempt_count"] == 1


def test_run_case_sanitizes_mixed_citations_without_model_repair(monkeypatch, tmp_path):
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    segment = fullchain.SimpleNamespace(
        episode_id="episode",
        document_id="document",
        fused_score=1.0,
        hit_source_msg_ids=("m1",),
        atomic_source_groups=(),
        pinned=False,
        blocked_output_present=False,
        messages=(
            EvidenceMessage(
                source_msg_id="m1",
                speaker="sender",
                content="SUPPORTED_SENTINEL",
                sent_at=observed_at,
            ),
        ),
    )
    packed = fullchain.SimpleNamespace(
        text="",
        evidence_segments=(segment,),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=("m1",),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    packed.text = fullchain._packet_text(packed)
    trace = fullchain.SimpleNamespace(
        result=fullchain.SimpleNamespace(packed_context=packed),
        resolved_query=fullchain.SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
        channel_candidate_counts=(),
    )
    runtime = fullchain.SimpleNamespace(
        v2_provider=fullchain.SimpleNamespace(evaluate=lambda request: trace)
    )

    class AnswerTransport:
        def generate(self, prompt_lines, *, model):
            del prompt_lines
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": "supported answer",
                        "cited_source_message_ids": ["m1", "outside"],
                        "abstained": False,
                    }
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    class JudgeTransport:
        def generate(self, prompt_lines, *, model):
            assert 'Generated citation IDs:\n["m1"]' in "\n".join(prompt_lines)
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer_grounded": True,
                        "answer_correct": True,
                        "abstained": False,
                        "reason_code": "supported",
                    }
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        lambda *args, **kwargs: pytest.fail("model repair must not run"),
    )

    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=AnswerTransport(),
        aux_transport=JudgeTransport(),
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "test question",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m1"],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        judge_packet_mode="citation-focused",
    )

    assert row["protocol_failure_codes"] == []
    assert row["cited_source_message_ids"] == ["m1"]
    assert row["citation_sanitized"] is True
    assert row["repaired"] is False
    assert row["answer_grounded"] is True
    assert row["answer_correct"] is True
    assert row["answer_expectation"] == "must_answer"
    assert row["expected_abstention"] is False
    assert row["legacy_expected_abstention"] is False
    assert row["allowed_citation_ids"] == ["m1"]
    assert row["judge_prompt_full"]
    assert row["judge_prompt_full_chars"] == sum(
        len(line) for line in row["judge_prompt_full"]
    )


def test_run_case_records_the_actual_prompt_used_after_rejudge_repair(
    monkeypatch, tmp_path
):
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)

    def segment(source_id: str, sentinel: str):
        return fullchain.SimpleNamespace(
            episode_id=source_id,
            document_id=f"doc-{source_id}",
            fused_score=1.0,
            hit_source_msg_ids=(source_id,),
            atomic_source_groups=(),
            pinned=False,
            blocked_output_present=False,
            messages=(
                EvidenceMessage(
                    source_msg_id=source_id,
                    speaker="sender",
                    content=sentinel,
                    sent_at=observed_at,
                ),
            ),
        )

    packed = fullchain.SimpleNamespace(
        text="",
        evidence_segments=(
            segment("m1", "INITIAL_SEGMENT_SENTINEL"),
            segment("m2", "FINAL_SEGMENT_SENTINEL"),
        ),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=("m1", "m2"),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    packed.text = fullchain._packet_text(packed)
    trace = fullchain.SimpleNamespace(
        result=fullchain.SimpleNamespace(packed_context=packed),
        resolved_query=fullchain.SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
        channel_candidate_counts=(),
    )
    runtime = fullchain.SimpleNamespace(
        v2_provider=fullchain.SimpleNamespace(evaluate=lambda request: trace)
    )

    class AnswerTransport:
        def generate(self, prompt_lines, *, model):
            del prompt_lines
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": "supported answer",
                        "cited_source_message_ids": ["m1"],
                        "abstained": False,
                    }
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    class JudgeTransport:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt_lines, *, model):
            self.prompts.append(list(prompt_lines))
            final = len(self.prompts) == 2
            return fullchain.SimpleNamespace(
                text=json.dumps(
                    {
                        "answer_grounded": final,
                        "answer_correct": final,
                        "abstained": False,
                        "reason_code": (
                            "supported" if final else "citation_insufficient"
                        ),
                    }
                ),
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model=model,
                attempt_count=1,
                no_event_attempts=0,
            )

    judge_transport = JudgeTransport()
    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        lambda *args, **kwargs: fullchain.SimpleNamespace(
            answer=fullchain.GeneratedAnswer(
                answer="supported answer",
                cited_source_message_ids=("m2",),
                abstained=False,
            ),
            protocol_failure_codes=(),
            observation=fullchain.SimpleNamespace(no_event_attempts=0),
        ),
    )

    row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=AnswerTransport(),
        aux_transport=judge_transport,
        case={
            "case_id": "case-1",
            "group_id": 1001,
            "query": "test question",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m2"],
        },
        case_id="case-1",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        judge_packet_mode="citation-focused",
    )

    final_prompt_text = "\n".join(row["judge_prompt"])
    assert len(judge_transport.prompts) == 2
    assert row["judge_prompt"] == judge_transport.prompts[-1]
    assert row["judge_prompt_chars"] == sum(
        len(line) for line in judge_transport.prompts[-1]
    )
    assert row["cited_source_message_ids"] == ["m2"]
    assert "FINAL_SEGMENT_SENTINEL" in final_prompt_text
    assert "INITIAL_SEGMENT_SENTINEL" in final_prompt_text
    assert row["judge_cached"] is False

    def fail_repair(*args, **kwargs):
        raise QualityReplayError(
            "QUALITY_REPLAY_PROVIDER_NO_EVENT",
            retryable=False,
            failure_kind="provider_no_event",
        )

    monkeypatch.setattr(
        fullchain,
        "_generate_citation_repair_with_retry",
        fail_repair,
    )
    failed_judge_transport = JudgeTransport()
    provider_failed_row = fullchain._run_case(
        engine=None,
        runtime=runtime,
        transport=AnswerTransport(),
        aux_transport=failed_judge_transport,
        case={
            "case_id": "case-2",
            "group_id": 1001,
            "query": "test question",
            "requester_uin": "11",
            "recent_context_message_ids": [],
            "expected_evidence_message_ids": ["m2"],
        },
        case_id="case-2",
        model="m",
        judge_model="m",
        answer_effort="high",
        aux_effort="medium",
        cache_dir=tmp_path / "provider-failed-cache",
        settings=fullchain.SimpleNamespace(bot_qq=123456789),
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=1,
        provider_backoff=0,
        judge_packet_mode="citation-focused",
    )

    assert len(failed_judge_transport.prompts) == 1
    assert provider_failed_row["protocol_failure_codes"] == ["provider_failed"]
    assert (
        provider_failed_row["citation_repair_rejudge_error"]
        == "QUALITY_REPLAY_PROVIDER_NO_EVENT"
    )
    assert provider_failed_row["answer_grounded"] is False
    assert provider_failed_row["answer_correct"] is False


def test_resume_reexecutes_when_context_limits_change(monkeypatch, tmp_path):
    case = {"category": "fact", "query": "q", "case_id": "case-1"}
    progress = tmp_path / "progress.jsonl"
    calls: list[str] = []
    monkeypatch.setattr(
        fullchain,
        "_run_case",
        lambda **kwargs: calls.append(str(kwargs["case_id"]))
        or {"case_id": str(kwargs["case_id"])},
    )
    common = {
        "runtime": object(),
        "transport": object(),
        "limit": 1,
        "seed": 1,
        "cache_dir": tmp_path / "cache",
        "progress_path": progress,
    }
    fullchain.run_cases(
        None,
        [case],
        settings=fullchain.SimpleNamespace(
            bot_qq=123456789, context_recent_limit=60, context_summary_limit=3
        ),
        **common,
    )
    calls.clear()

    rows, summary = fullchain.run_cases(
        None,
        [case],
        resume=True,
        settings=fullchain.SimpleNamespace(
            bot_qq=123456789, context_recent_limit=40, context_summary_limit=3
        ),
        **common,
    )

    assert calls == ["case-1"]
    assert len(rows) == 1
    assert summary["invalidated_resumed"] == 1


def test_resume_reexecutes_when_database_snapshot_changes(monkeypatch, tmp_path):
    case = {"category": "fact", "query": "q", "case_id": "case-1"}
    progress = tmp_path / "progress.jsonl"
    database = tmp_path / "snapshot.db"
    database.write_bytes(b"snapshot-v1")
    engine = fullchain.SimpleNamespace(
        url=fullchain.SimpleNamespace(database=str(database))
    )
    calls: list[str] = []
    monkeypatch.setattr(
        fullchain,
        "_run_case",
        lambda **kwargs: calls.append(str(kwargs["case_id"]))
        or {"case_id": str(kwargs["case_id"])},
    )
    injected = {
        "settings": fullchain.SimpleNamespace(bot_qq=123456789),
        "runtime": object(),
        "transport": object(),
    }
    fullchain.run_cases(
        engine,
        [case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        progress_path=progress,
        **injected,
    )
    calls.clear()
    database.write_bytes(b"snapshot-v2")

    rows, summary = fullchain.run_cases(
        engine,
        [case],
        limit=1,
        seed=1,
        cache_dir=tmp_path / "cache",
        resume=True,
        progress_path=progress,
        **injected,
    )

    assert calls == ["case-1"]
    assert len(rows) == 1
    assert summary["invalidated_resumed"] == 1


def test_run_cases_detail_path_appends_rows_and_survives_resume(monkeypatch, tmp_path):
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


def test_main_passes_output_detail_for_incremental_checkpointing(monkeypatch, tmp_path):
    database = tmp_path / "snapshot.db"
    cases_path = tmp_path / "cases.jsonl"
    detail_path = tmp_path / "detail.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(fullchain, "_build_engine", lambda path, **kwargs: object())
    monkeypatch.setattr(fullchain, "_load_cases", lambda path: [{"query": "q"}])

    def fake_run_cases(engine, cases, **kwargs):
        del engine, cases
        captured.update(kwargs)
        return [], {"requested": 1, "executed": 0, "skipped_resumed": 0}

    monkeypatch.setattr(fullchain, "run_cases", fake_run_cases)

    assert (
        fullchain.main(
            [
                "--database",
                str(database),
                "--cases",
                str(cases_path),
                "--output-detail",
                str(detail_path),
                "--progress",
                str(progress_path),
            ]
        )
        == 0
    )
    assert captured["detail_path"] == detail_path
    assert captured["progress_path"] == progress_path


def test_provider_preflight_summary_requires_ten_clean_rows():
    clean_rows = [
        {
            "case_id": f"case-{index}",
            "protocol_failure_codes": [],
            "answer_prompt_chars": 1000 + index * 100,
            "judge_prompt_chars": 2000 + index * 100,
        }
        for index in range(10)
    ]
    passed = fullchain._provider_preflight_summary(clean_rows, expected_cases=10)
    assert passed["passed"] is True
    assert passed["completed"] == 10
    assert passed["provider_failed"] == 0
    assert passed["provider_no_event"] == 0
    assert passed["provider_no_event_attempts"] == 0
    assert passed["answer_prompt_chars"] == {"min": 1000, "median": 1500, "max": 1900}
    assert passed["judge_prompt_chars"] == {"min": 2000, "median": 2500, "max": 2900}

    failed_rows = [dict(row) for row in clean_rows]
    failed_rows[-1]["protocol_failure_codes"] = ["provider_failed"]
    failed_rows[-1]["judge_provider_error"] = "QUALITY_REPLAY_PROVIDER_NO_EVENT"
    failed_rows[-1]["answer_no_event_attempts"] = 1
    failed = fullchain._provider_preflight_summary(failed_rows, expected_cases=10)
    assert failed["passed"] is False
    assert failed["provider_failed"] == 1
    assert failed["provider_no_event"] == 1
    assert failed["provider_no_event_attempts"] == 1


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


def test_generate_with_retries_does_not_repeat_non_retryable_transport_failure():
    class NonRetryableTransport:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt_lines, *, model):
            del prompt_lines, model
            self.calls += 1
            raise QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED", retryable=False)

    transport = NonRetryableTransport()
    with pytest.raises(QualityReplayError):
        fullchain._generate_with_retries(
            transport,
            ["p"],
            model="m",
            attempts=5,
            backoff=0,
        )
    assert transport.calls == 1


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
        fullchain._parse_answer_loose(
            '{"answer": "", "cited_source_message_ids": [], "abstained": false}'
        )
    with pytest.raises(ValueError):
        fullchain._parse_answer_loose(
            '{"answer": "x", "cited_source_message_ids": "bad", "abstained": false}'
        )


def test_time_bucket_citation_recall_requires_coverage_not_every_source():
    assert fullchain._citation_recall_score(
        gold={"m1", "m2", "m3", "m4"},
        citations={"m3"},
        coverage_strategy="time_buckets",
        minimum_time_bucket_count=1,
    ) == 1.0
    assert fullchain._citation_recall_score(
        gold={"m1", "m2", "m3", "m4"},
        citations={"m3"},
        coverage_strategy="relevance",
        minimum_time_bucket_count=1,
    ) == 0.25


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


def test_build_answer_prompt_includes_production_grounding_policy() -> None:
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        source_msg_ids=(),
        grounding_policy="production memory grounding contract",
    )

    prompt = fullchain.build_answer_prompt(
        {"query": "昨天群里聊了什么"},
        packet,
    )

    assert "production memory grounding contract" in "\n".join(prompt)
    assert "today's news, weather, or time" in prompt[-1]


def test_answer_expectation_supports_explicit_and_legacy_tri_state():
    assert fullchain._answer_expectation({"answer_expectation": "either"}) == "either"
    assert (
        fullchain._answer_expectation(
            {"category": "mention", "expected_evidence_message_ids": []}
        )
        == "either"
    )
    assert (
        fullchain._answer_expectation(
            {"category": "distractor", "expected_evidence_message_ids": []}
        )
        == "must_abstain"
    )
    assert (
        fullchain._answer_expectation(
            {"category": "fact", "expected_evidence_message_ids": ["m1"]}
        )
        == "must_answer"
    )
    with pytest.raises(ValueError, match="unknown answer expectation"):
        fullchain._answer_expectation({"answer_expectation": "sometimes"})


def test_mention_either_judge_contract_accepts_grounded_answer_or_abstention():
    packet = fullchain.SimpleNamespace(
        text="grounding contract",
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=(),
    )

    prompt = fullchain.build_judge_prompt(
        {"category": "mention", "query": "question", "answer_expectation": "either"},
        "grounded answer",
        (),
        False,
        packet,
    )
    text = "\n".join(prompt)

    assert "Answer expectation: either" in text
    assert "grounded answer or genuine abstention accepted" in text
    assert (
        "For answer_expectation=either, a genuine no-claim abstention "
        "remains correct even when relevant packet evidence exists"
    ) in text
    assert "For answer_expectation=must_answer only" in text
    assert "[expected abstention: no reference evidence]" not in text


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
    assert "raw history, running jokes, summaries" in text
    assert "unsupported inferred attribute" in text


def test_build_judge_prompt_keeps_small_packet_canonical():
    now = fullchain.datetime.now(fullchain.UTC)
    cited_segment = fullchain.SimpleNamespace(
        episode_id="cited",
        document_id="doc-cited",
        hit_source_msg_ids=("m1",),
        messages=(
            fullchain.EvidenceMessage(
                source_msg_id="m1",
                speaker="speaker",
                content="CITED_SEGMENT_SENTINEL",
                sent_at=now,
            ),
        ),
    )
    unrelated_segment = fullchain.SimpleNamespace(
        episode_id="unrelated",
        document_id="doc-unrelated",
        hit_source_msg_ids=("m2",),
        messages=(
            fullchain.EvidenceMessage(
                source_msg_id="m2",
                speaker="speaker",
                content="UNRELATED_SEGMENT_SENTINEL",
                sent_at=now,
            ),
        ),
    )
    packet = fullchain.SimpleNamespace(
        evidence_segments=(cited_segment, unrelated_segment),
        facts=(
            fullchain.SimpleNamespace(
                kind="fact", source_msg_ids=("m1",), text="CITED_FACT_SENTINEL"
            ),
            fullchain.SimpleNamespace(
                kind="fact", source_msg_ids=("m2",), text="UNRELATED_FACT_SENTINEL"
            ),
        ),
        summaries=(),
        source_msg_ids=("m1", "m2"),
        grounding_policy="grounding contract",
    )

    prompt = fullchain.build_judge_prompt(
        {"query": "question", "gold_text": "reference"},
        "answer",
        ("m1",),
        False,
        packet,
        judge_packet_mode="citation-focused",
    )
    text = "\n".join(prompt)

    assert "CITED_SEGMENT_SENTINEL" in text
    assert "CITED_FACT_SENTINEL" in text
    assert "UNRELATED_SEGMENT_SENTINEL" in text
    assert "UNRELATED_FACT_SENTINEL" in text


def test_packet_text_prefers_canonical_runtime_render_with_recent_context():
    packet = fullchain.SimpleNamespace(
        text=(
            "Memory fact (kind: current; observed_at: 2026-08-20 13:45 +08; "
            "sources: m1): CURRENT_FACT_SENTINEL\n\n"
            "Recent message [2026-08-20 13:46 +08] sender "
            "(uin: 1; source: r1; reply_to: none): RECENT_SENTINEL"
        ),
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(),
    )

    rendered = fullchain._packet_text(packet)

    assert "kind: current" in rendered
    assert "observed_at: 2026-08-20 13:45 +08" in rendered
    assert "RECENT_SENTINEL" in rendered


def test_focused_packet_preserves_runtime_fact_metadata_and_recent_context():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    cited_segment = fullchain.SimpleNamespace(
        episode_id="cited",
        document_id="doc-cited",
        hit_source_msg_ids=("m1",),
        messages=(
            EvidenceMessage(
                source_msg_id="m1",
                speaker="sender",
                content="RAW_SENTINEL",
                sent_at=observed_at,
            ),
        ),
    )
    packet = fullchain.SimpleNamespace(
        evidence_segments=(cited_segment,),
        facts=(
            MemoryFact(
                text="CURRENT_FACT_SENTINEL",
                source_msg_ids=("m1",),
                memory_kind="current",
                observed_at=observed_at,
            ),
        ),
        summaries=(),
        recent_messages=(
            EvidenceMessage(
                source_msg_id="r1",
                speaker="sender",
                content="RECENT_SENTINEL",
                sent_at=observed_at,
            ),
        ),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )

    rendered = fullchain._citation_focused_packet_text(packet, ("m1",))

    assert rendered is not None
    assert "Memory fact (kind: current; observed_at: 2026-08-20 13:45 +08;" in rendered
    assert "CURRENT_FACT_SENTINEL" in rendered
    assert "RECENT_SENTINEL" in rendered


def test_focused_packet_rejects_derived_only_citation():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(
            MemoryFact(
                text="FACT_SENTINEL",
                source_msg_ids=("m1",),
                memory_kind="fact",
                observed_at=observed_at,
            ),
        ),
        summaries=(),
        recent_messages=(),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )

    assert fullchain._citation_focused_packet_text(packet, ("m1",)) is None


def test_focused_packet_rejects_fact_only_citation_with_unrelated_raw_segments():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    unrelated_segment = fullchain.SimpleNamespace(
        episode_id="unrelated",
        document_id="doc-unrelated",
        hit_source_msg_ids=("m2",),
        messages=(
            EvidenceMessage(
                source_msg_id="m2",
                speaker="sender",
                content="UNRELATED_RAW_SENTINEL",
                sent_at=observed_at,
            ),
        ),
    )
    packet = fullchain.SimpleNamespace(
        evidence_segments=(unrelated_segment,),
        facts=(
            MemoryFact(
                text="FACT_SENTINEL",
                source_msg_ids=("m1",),
                memory_kind="fact",
                observed_at=observed_at,
            ),
        ),
        summaries=(),
        recent_messages=(),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )

    assert fullchain._citation_focused_packet_text(packet, ("m1",)) is None


def test_focused_packet_bounds_large_raw_neighborhood_for_fact_only_citation():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)

    def segment(index: int):
        source_id = f"raw-{index}"
        return fullchain.SimpleNamespace(
            episode_id=source_id,
            document_id=f"doc-{source_id}",
            hit_source_msg_ids=(source_id,),
            messages=(
                EvidenceMessage(
                    source_msg_id=source_id,
                    speaker="sender",
                    content=f"RAW_SENTINEL_{index}",
                    sent_at=observed_at,
                ),
            ),
        )

    segments = tuple(
        segment(index)
        for index in range(fullchain.CITATION_FOCUSED_RAW_SEGMENT_LIMIT + 1)
    )
    packet = fullchain.SimpleNamespace(
        text="",
        evidence_segments=segments,
        facts=(
            MemoryFact(
                text="FACT_SENTINEL",
                source_msg_ids=("fact-source",),
                memory_kind="fact",
                observed_at=observed_at,
            ),
        ),
        summaries=(),
        recent_messages=(),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    packet.text = fullchain._packet_text(packet)

    rendered = fullchain._citation_focused_packet_text(packet, ("fact-source",))

    assert rendered is not None
    assert rendered.count("Evidence - quoted chat data") == 10
    assert "RAW_SENTINEL_9" in rendered
    assert "RAW_SENTINEL_10" not in rendered
    assert rendered == packet.text.removesuffix(
        "\n\n" + fullchain.MemoryContextPacker._render_segment(segments[-1])
    )


def test_focused_packet_keeps_late_raw_hit_with_derived_citation():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)

    def segment(index: int):
        source_id = f"raw-{index}"
        return fullchain.SimpleNamespace(
            episode_id=source_id,
            document_id=f"doc-{source_id}",
            hit_source_msg_ids=(source_id,),
            messages=(
                EvidenceMessage(
                    source_msg_id=source_id,
                    speaker="sender",
                    content=f"RAW_SENTINEL_{index}",
                    sent_at=observed_at,
                ),
            ),
        )

    segments = tuple(segment(index) for index in range(12))
    packet = fullchain.SimpleNamespace(
        text="",
        evidence_segments=segments,
        facts=(
            MemoryFact(
                text="FACT_SENTINEL",
                source_msg_ids=("fact-source",),
                memory_kind="fact",
                observed_at=observed_at,
            ),
        ),
        summaries=(),
        recent_messages=(),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    packet.text = fullchain._packet_text(packet)

    rendered = fullchain._citation_focused_packet_text(
        packet, ("fact-source", "raw-11")
    )

    assert rendered is not None
    assert rendered.count("Evidence - quoted chat data") == 11
    assert "RAW_SENTINEL_9" in rendered
    assert "RAW_SENTINEL_10" not in rendered
    assert "RAW_SENTINEL_11" in rendered
    removed = fullchain.MemoryContextPacker._render_segment(segments[10])
    expected_blocks = packet.text.split("\n\n")
    expected_blocks.remove(removed)
    assert rendered == "\n\n".join(expected_blocks)


def test_focused_packet_only_deletes_unmatched_segments_from_canonical_order():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    cited_segment = fullchain.SimpleNamespace(
        episode_id="cited",
        document_id="doc-cited",
        hit_source_msg_ids=("m1",),
        messages=(
            EvidenceMessage(
                source_msg_id="m1",
                speaker="speaker",
                content="CITED_SEGMENT_SENTINEL",
                sent_at=observed_at,
            ),
        ),
    )
    unrelated_segment = fullchain.SimpleNamespace(
        episode_id="unrelated",
        document_id="doc-unrelated",
        hit_source_msg_ids=("m2",),
        messages=(
            EvidenceMessage(
                source_msg_id="m2",
                speaker="speaker",
                content="UNRELATED_SEGMENT_SENTINEL",
                sent_at=observed_at,
            ),
        ),
    )
    fact = MemoryFact(
        text="FACT_SENTINEL",
        source_msg_ids=("m1",),
        memory_kind="fact",
        observed_at=observed_at,
    )
    recent = EvidenceMessage(
        source_msg_id="r1",
        speaker="sender",
        content="RECENT_SENTINEL",
        sent_at=observed_at,
    )
    policy = "grounding contract"
    fact_block = fullchain._render_fact_for_evaluation(fact)
    cited_block = fullchain.MemoryContextPacker._render_segment(cited_segment)
    unrelated_block = fullchain.MemoryContextPacker._render_segment(unrelated_segment)
    recent_block = fullchain.MemoryContextPacker._render_recent(recent)
    canonical = "\n\n".join(
        (policy, fact_block, cited_block, unrelated_block, recent_block)
    )
    packet = fullchain.SimpleNamespace(
        text=canonical,
        evidence_segments=(cited_segment, unrelated_segment),
        facts=(fact,),
        summaries=(),
        recent_messages=(recent,),
        grounding_policy=policy,
        blocked_output_present=False,
    )

    rendered = fullchain._citation_focused_packet_text(packet, ("m1",))

    assert rendered == canonical


def test_focused_packet_keeps_bounded_ranked_context_for_raw_citation():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)

    def segment(index: int):
        source_id = f"raw-{index}"
        return fullchain.SimpleNamespace(
            episode_id=source_id,
            document_id=f"doc-{source_id}",
            hit_source_msg_ids=(source_id,),
            messages=(
                EvidenceMessage(
                    source_msg_id=source_id,
                    speaker="sender",
                    content=f"RAW_SENTINEL_{index}",
                    sent_at=observed_at,
                ),
            ),
        )

    segments = tuple(segment(index) for index in range(12))
    packet = fullchain.SimpleNamespace(
        text="",
        evidence_segments=segments,
        facts=(),
        summaries=(),
        recent_messages=(),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )
    packet.text = fullchain._packet_text(packet)

    rendered = fullchain._citation_focused_packet_text(packet, ("raw-11",))

    assert rendered is not None
    assert rendered.count("Evidence - quoted chat data") == 11
    assert "RAW_SENTINEL_9" in rendered
    assert "RAW_SENTINEL_10" not in rendered
    assert "RAW_SENTINEL_11" in rendered


def test_focused_packet_rejects_recent_only_citation():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(
            EvidenceMessage(
                source_msg_id="r1",
                speaker="sender",
                content="RECENT_SENTINEL",
                sent_at=observed_at,
            ),
        ),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )

    rendered = fullchain._citation_focused_packet_text(packet, ("r1",))

    assert rendered is None


def test_focused_packet_rejects_mixed_valid_and_recent_citations():
    observed_at = datetime(2026, 8, 20, 5, 45, tzinfo=UTC)
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(
            MemoryFact(
                text="FACT_SENTINEL",
                source_msg_ids=("m1",),
                memory_kind="fact",
                observed_at=observed_at,
            ),
        ),
        summaries=(),
        recent_messages=(
            EvidenceMessage(
                source_msg_id="r1",
                speaker="sender",
                content="RECENT_SENTINEL",
                sent_at=observed_at,
            ),
        ),
        grounding_policy="grounding contract",
        blocked_output_present=False,
    )

    assert fullchain._citation_focused_packet_text(packet, ("m1", "r1")) is None


@pytest.mark.parametrize(
    ("cited_ids", "abstained"),
    [
        ((), True),
        (("missing",), False),
    ],
)
def test_build_judge_prompt_falls_back_to_full_packet_when_focus_is_unsafe(
    cited_ids, abstained
):
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(
            fullchain.SimpleNamespace(
                kind="fact", source_msg_ids=("m1",), text="FULL_PACKET_SENTINEL"
            ),
        ),
        summaries=(),
        source_msg_ids=("m1",),
    )

    prompt = fullchain.build_judge_prompt(
        {"query": "question", "gold_text": "reference"},
        "answer",
        cited_ids,
        abstained,
        packet,
    )

    assert "FULL_PACKET_SENTINEL" in "\n".join(prompt)


def test_unknown_citation_focused_judge_prompt_equals_full_prompt():
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(
            fullchain.SimpleNamespace(
                kind="fact", source_msg_ids=("m1",), text="FULL_PACKET_SENTINEL"
            ),
        ),
        summaries=(),
        recent_messages=(),
        source_msg_ids=("m1",),
        grounding_policy="grounding contract",
    )
    case = {"query": "question", "gold_text": "reference"}

    full_prompt = fullchain.build_judge_prompt(
        case, "answer", ("unknown",), False, packet, judge_packet_mode="full"
    )
    focused_prompt = fullchain.build_judge_prompt(
        case,
        "answer",
        ("unknown",),
        False,
        packet,
        judge_packet_mode="citation-focused",
    )

    assert focused_prompt == full_prompt


@pytest.mark.parametrize(
    ("category", "sentinel"),
    [
        ("profile", "Category contract (profile)"),
        ("current", "Category contract (current)"),
        ("relationship", "Category contract (relationship)"),
        ("decision", "Category contract (decision)"),
        ("preference", "Category contract (preference)"),
        ("first_person", "Category contract (first_person)"),
        ("identity_audit", "Category contract (identity_audit)"),
        ("running_joke", "Category contract (running_joke)"),
        ("raw_history", "Category contract (raw_history)"),
        ("event", "Category contract (event)"),
        ("summary", "Category contract (summary)"),
    ],
)
def test_build_answer_prompt_includes_category_contract(category, sentinel):
    packet = fullchain.SimpleNamespace(
        evidence_segments=(), facts=(), summaries=(), source_msg_ids=()
    )

    prompt = fullchain.build_answer_prompt(
        {"query": "测试问题", "category": category}, packet
    )

    assert sentinel in "\n".join(prompt)
    assert sentinel in prompt[-1]


def test_identity_audit_contract_requires_partial_answer_and_prefers_corrections():
    packet = fullchain.SimpleNamespace(
        evidence_segments=(), facts=(), summaries=(), source_msg_ids=()
    )

    prompt = fullchain.build_answer_prompt(
        {"query": "我是谁", "category": "identity_audit"},
        packet,
    )

    reminder = prompt[-1]
    assert "One such attribute is enough" in reminder
    assert "not as a demand for a legal name" in reminder
    assert "sender display name is direct identity evidence" in reminder
    assert "cite only that query source for the name" in reminder
    assert "set abstained=false" in reminder
    assert "never abstain merely because a complete profile is unavailable" in reminder
    assert "newer direct self-denial or correction" in reminder


def test_identity_audit_prompt_pins_exact_target_sender_metadata() -> None:
    packet = fullchain.SimpleNamespace(
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=("target-source",),
    )
    case = {
        "query": "我是谁",
        "category": "identity_audit",
        "requester_uin": "42",
        "_requester_display_name": "Maple",
        "_requester_source_msg_id": "target-source",
        "answer_expectation": "must_answer",
    }

    answer_prompt = fullchain.build_answer_prompt(case, packet)
    judge_prompt = fullchain.build_judge_prompt(
        case,
        "你是群里的 Maple。",
        ("target-source",),
        False,
        packet,
    )

    assert any('"requester_uin":"42"' in line for line in answer_prompt)
    assert any('"display_name":"Maple"' in line for line in answer_prompt)
    assert any('"target_source_msg_id":"target-source"' in line for line in judge_prompt)
    assert any(
        'Allowed citation IDs JSON list: ["target-source"]' in line
        for line in answer_prompt
    )


def test_fullchain_message_loader_preserves_historical_sender_card() -> None:
    row = (
        1,
        "source-1",
        42,
        "2026-08-24T19:00:00+08:00",
        "我是谁",
        None,
        10001,
        '{"sender":{"card":"当时群名片","nickname":"昵称"}}',
    )

    message = fullchain._row_to_message(row)
    evidence = fullchain._evidence(message, bot_user_id=99)

    assert message["speaker"] == "当时群名片"
    assert evidence.speaker == "当时群名片"


def test_current_contract_drops_clauses_without_one_direct_citation_each():
    packet = fullchain.SimpleNamespace(
        evidence_segments=(), facts=(), summaries=(), source_msg_ids=()
    )

    reminder = fullchain.build_answer_prompt(
        {"query": "最近在做什么", "category": "current"}, packet
    )[-1]

    assert "one valid direct citation for every clause" in reminder
    assert "only the newest single supported activity" in reminder
    assert "Do not turn a quoted opinion about media" in reminder
    assert "recently mentioned it" in reminder
    assert "Match the requested current activity exactly" in reminder


def test_relationship_contract_answers_one_explicit_edge_without_abstaining():
    packet = fullchain.SimpleNamespace(
        evidence_segments=(), facts=(), summaries=(), source_msg_ids=()
    )

    reminder = fullchain.build_answer_prompt(
        {"query": "他和谁是什么关系", "category": "relationship"}, packet
    )[-1]

    assert "One explicit relationship fact" in reminder
    assert "must be answered rather than abstained" in reminder
    assert "one quoted source explicitly names the relationship edge" in reminder
    assert "Do not require the counterpart to have a proper name" in reminder
    assert "whose kind is relationship" in reminder


def test_raw_history_contract_accepts_exact_substring_hit() -> None:
    packet = fullchain.SimpleNamespace(
        evidence_segments=(), facts=(), summaries=(), source_msg_ids=()
    )

    reminder = fullchain.build_answer_prompt(
        {"query": "以前提到勤是不时说了什么", "category": "raw_history"}, packet
    )[-1]

    assert "substring inside a longer word sequence" in reminder
    assert "do not reassess whether surrounding context is complete" in reminder


def test_build_answer_prompt_keeps_external_current_abstention_rule_last():
    packet = fullchain.SimpleNamespace(
        text="Historical news item from an earlier date",
        evidence_segments=(),
        facts=(),
        summaries=(),
        recent_messages=(),
        source_msg_ids=(),
    )

    prompt = fullchain.build_answer_prompt(
        {"query": "今天有什么新闻", "category": "abstention"}, packet
    )

    assert "Retrieved memory packet" in prompt[-2]
    assert "today's news" in prompt[-1]
    assert "without a current external source, abstain" in prompt[-1]
