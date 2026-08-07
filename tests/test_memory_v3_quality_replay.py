from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from scripts.evaluate_memory_recall import EvaluationCase
from scripts.run_memory_v3_quality_replay import (
    AnswerContractError,
    CitationLimitError,
    CitationContractDecision,
    FIXED_ABSTENTION_ANSWER,
    GeneratedAnswer,
    JudgeDecision,
    ObservedGeneration,
    ObservedResponsesTransport,
    QualityReplayError,
    _sqlite_readonly_backup,
    _write_json,
    _generate_answer_with_retry,
    _generate_citation_repair_with_retry,
    _generate_valid_json,
    allowed_citation_ids_from_packed_context,
    apply_fail_closed_judgment,
    build_answer_prompt,
    build_answer_repair_prompt,
    build_citation_contract_prompt,
    build_judge_prompt,
    finalize_replay_case_judgment,
    build_public_sidecar,
    generated_citation_failure_codes,
    parse_generated_answer,
    parse_citation_contract_decision,
    parse_judge_decision,
)
from app.core.memory_context_packer import (
    EvidenceMessage,
    EvidenceSegment,
    MemoryContextPacker,
    PackedMemoryContext,
)
from app.providers.llm_client import LlmClient


class _TimedSseStream(httpx.SyncByteStream):
    def __init__(self, *, include_delta: bool = True, include_usage: bool = True) -> None:
        self.include_delta = include_delta
        self.include_usage = include_usage

    def __iter__(self):
        if self.include_delta:
            yield b'data: {"type":"response.output_text.delta","delta":"{\\"answer\\":\\"ok\\","}\n\n'
            time.sleep(0.04)
            yield b'data: {"type":"response.output_text.delta","delta":"\\"cited_source_message_ids\\":[\\"m1\\"],\\"abstained\\":false}"}\n\n'
        completed = {
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 123, "output_tokens": 17}
                if self.include_usage
                else None
            },
        }
        yield ("data: " + json.dumps(completed) + "\n\n").encode()
        yield b"data: [DONE]\n\n"


def _client(stream: httpx.SyncByteStream) -> LlmClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/responses"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    return LlmClient(
        base_url="https://provider.invalid",
        api_key="secret",
        model="answer-model",
        responses_model="answer-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _case(*, evidence=("m1",)) -> EvaluationCase:
    return EvaluationCase(
        group_id=1,
        query="private query marker",
        recent_context_message_ids=("recent",),
        expected_evidence_message_ids=tuple(evidence),
        category="exact" if evidence else "abstention",
        schema_version=3,
        requester_uin="42",
        allowed_subject_user_ids=("42",),
        allowed_evidence_user_ids=("42",),
        expected_answer_mode="exact",
        expected_coverage_strategy="relevance",
        minimum_time_bucket_count=0,
        forbidden_evidence_message_ids=(),
        gate_tags=("source_resolution",),
        contract_fields_complete=True,
    )


def test_observed_transport_measures_first_delta_not_full_response() -> None:
    client = _client(_TimedSseStream())
    try:
        started = time.perf_counter()
        result = ObservedResponsesTransport(client).generate(
            ["Target message: test"], model="answer-model"
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        client.http_client.close()

    assert parse_generated_answer(result.text).answer == "ok"
    assert result.input_tokens == 123
    assert result.output_tokens == 17
    assert result.ttft_ms >= 0
    assert result.ttft_ms < elapsed_ms - 20


@pytest.mark.parametrize(
    ("stream", "error"),
    [
        (_TimedSseStream(include_delta=False), "TTFT_UNOBSERVABLE"),
        (_TimedSseStream(include_usage=False), "USAGE_MISSING"),
    ],
)
def test_observed_transport_rejects_unmeasurable_metrics(
    stream: httpx.SyncByteStream, error: str
) -> None:
    client = _client(stream)
    try:
        with pytest.raises(QualityReplayError, match=error):
            ObservedResponsesTransport(client).generate(["test"], model="answer-model")
    finally:
        client.http_client.close()


def test_observed_transport_retries_xbai_403_without_mutating_payload() -> None:
    request_bodies: list[bytes] = []
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        request_bodies.append(request.content)
        if attempts < 3:
            return httpx.Response(403, request=request, json={"error": "gateway policy"})
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=_TimedSseStream(),
        )

    client = LlmClient(
        base_url="https://api.xbai.top/v1",
        api_key="secret",
        model="answer-model",
        responses_model="answer-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    sleep_attempts: list[int] = []
    client._sleep_before_retry = lambda *, attempt, max_attempts: sleep_attempts.append(attempt)
    try:
        result = ObservedResponsesTransport(client).generate(
            [
                "System persona: native instruction marker",
                "Target message: private input marker",
            ],
            model="answer-model",
        )
    finally:
        client.http_client.close()

    assert parse_generated_answer(result.text).answer == "ok"
    assert attempts == 3
    assert sleep_attempts == [1, 2]
    assert request_bodies[0] == request_bodies[1] == request_bodies[2]
    payload = json.loads(request_bodies[0])
    assert payload["instructions"] == "System persona: native instruction marker"


def test_observed_transport_exhausts_xbai_403_and_logs_only_safe_metadata(caplog) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            403,
            request=request,
            json={"error": "private response body marker"},
        )

    client = LlmClient(
        base_url="https://api.xbai.top/v1",
        api_key="private-api-key-marker",
        model="answer-model",
        responses_model="answer-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client._sleep_before_retry = lambda **kwargs: None
    try:
        with caplog.at_level(logging.WARNING, logger="scripts.run_memory_v3_quality_replay"):
            with pytest.raises(QualityReplayError, match="^QUALITY_REPLAY_PROVIDER_FAILED$") as exc_info:
                ObservedResponsesTransport(client).generate(
                    [
                        "System persona: private instruction marker",
                        "Target message: private query marker",
                    ],
                    model="answer-model",
                )
    finally:
        client.http_client.close()

    assert exc_info.value.retryable is False
    assert attempts == client.REQUEST_MAX_ATTEMPTS
    log_text = caplog.text
    assert "status=403" in log_text
    assert "attempt=3" in log_text
    assert "prompt_chars=" in log_text
    assert "instructions_chars=" in log_text
    assert "private instruction marker" not in log_text
    assert "private query marker" not in log_text
    assert "private response body marker" not in log_text
    assert "private-api-key-marker" not in log_text


def test_observed_transport_does_not_retry_invalid_provider_url() -> None:
    client = LlmClient(
        base_url="missing-protocol.invalid",
        api_key="secret",
        model="answer-model",
        responses_model="answer-model",
    )
    sleep_attempts: list[int] = []
    client._sleep_before_retry = lambda *, attempt, max_attempts: sleep_attempts.append(attempt)
    try:
        with pytest.raises(QualityReplayError, match="^QUALITY_REPLAY_PROVIDER_FAILED$") as exc_info:
            ObservedResponsesTransport(client).generate(
                ["Target message: private query marker"],
                model="answer-model",
            )
    finally:
        client.http_client.close()

    assert exc_info.value.retryable is False
    assert sleep_attempts == []


def test_strict_answer_and_judge_parsers_reject_extensions_and_nonstandard_json() -> None:
    answer = parse_generated_answer(
        '{"answer":"ok","cited_source_message_ids":["m1"],"abstained":false}'
    )
    assert answer.cited_source_message_ids == ("m1",)
    judge = parse_judge_decision(
        '{"answer_grounded":true,"answer_correct":true,"abstained":false,"reason_code":"supported"}'
    )
    assert judge.answer_correct
    citation_contract = parse_citation_contract_decision(
        '{"citations_minimal":true,"reason_code":"minimal"}'
    )
    assert citation_contract == CitationContractDecision(True, "minimal")

    with pytest.raises(ValueError, match="fields"):
        parse_generated_answer(
            '{"answer":"ok","cited_source_message_ids":[],"abstained":false,"extra":1}'
        )
    with pytest.raises(ValueError, match="constant"):
        parse_judge_decision(
            '{"answer_grounded":true,"answer_correct":true,"abstained":false,"reason_code":NaN}'
        )
    with pytest.raises(ValueError, match="citations"):
        parse_generated_answer(
            '{"answer":"ok","cited_source_message_ids":["m1","m1"],"abstained":false}'
        )
    with pytest.raises(AnswerContractError) as missing_citation:
        parse_generated_answer(
            '{"answer":"unsupported","cited_source_message_ids":[],"abstained":false}'
        )
    assert missing_citation.value.protocol_failure_codes == ("citation_missing",)
    with pytest.raises(CitationLimitError) as exc_info:
        parse_generated_answer(
            '{"answer":"too many","cited_source_message_ids":["m1","m2","m3"],"abstained":false}'
        )
    assert exc_info.value.answer.cited_source_message_ids == ("m1", "m2", "m3")


def test_abstention_parser_requires_fixed_text_empty_citations_and_true_flag() -> None:
    valid = parse_generated_answer(
        json.dumps(
            {
                "answer": FIXED_ABSTENTION_ANSWER,
                "cited_source_message_ids": [],
                "abstained": True,
            },
            ensure_ascii=False,
        )
    )
    assert valid.answer == FIXED_ABSTENTION_ANSWER

    with pytest.raises(AnswerContractError) as wording_error:
        parse_generated_answer(
            '{"answer":"换一种自由措辞拒答","cited_source_message_ids":[],"abstained":true}'
        )
    assert wording_error.value.protocol_failure_codes == (
        "abstention_text_mismatch",
    )

    with pytest.raises(AnswerContractError) as citation_error:
        parse_generated_answer(
            json.dumps(
                {
                    "answer": FIXED_ABSTENTION_ANSWER,
                    "cited_source_message_ids": ["m1"],
                    "abstained": True,
                },
                ensure_ascii=False,
            )
        )
    assert citation_error.value.protocol_failure_codes == (
        "abstention_citations_nonempty",
    )


def test_answer_repair_prompt_exposes_failure_but_not_gold_ids() -> None:
    prompt = build_answer_repair_prompt(
        original_prompt=["original packet"],
        answer=GeneratedAnswer("draft", ("m1", "m2"), False),
        protocol_failure_codes=("citation_not_minimal",),
    )

    rendered = "\n".join(prompt)
    assert "citation_not_minimal" in rendered
    assert '"cited_source_message_ids":["m1","m2"]' in rendered
    assert "No reference answer or gold evidence is available" in rendered
    assert "expected_evidence_message_ids" not in rendered


def test_citation_contract_prompt_is_gold_free_and_repair_cannot_change_answer() -> None:
    case = _case(evidence=("m1",))
    original = GeneratedAnswer("draft", ("m1", "m2"), False)
    contract = "\n".join(
        build_citation_contract_prompt(
            case=case,
            answer=original,
            packet_text="packet source m1",
        )
    )
    assert "reference evidence" not in contract.lower()
    assert "expected_evidence_message_ids" not in contract

    transport = _SequenceGenerationTransport(
        [
            '{"answer":"changed","cited_source_message_ids":["m1"],"abstained":false}',
            '{"answer":"draft","cited_source_message_ids":["m1"],"abstained":false}',
        ]
    )
    outcome = _generate_citation_repair_with_retry(
        transport,
        ["repair"],
        model="generator",
        attempts=2,
        original_answer=original,
        allowed_citation_ids=("m1", "m2"),
    )

    assert transport.call_count == 2
    assert outcome.answer.answer == "draft"
    assert outcome.answer.cited_source_message_ids == ("m1",)


def test_citation_repair_exhaustion_keeps_original_answer_fail_closed() -> None:
    original = GeneratedAnswer("draft", ("m1", "m2"), False)
    transport = _SequenceGenerationTransport(
        [
            '{"answer":"changed-one","cited_source_message_ids":["m1"],"abstained":false}',
            '{"answer":"changed-two","cited_source_message_ids":["m1"],"abstained":false}',
        ]
    )

    outcome = _generate_citation_repair_with_retry(
        transport,
        ["repair"],
        model="generator",
        attempts=2,
        original_answer=original,
        allowed_citation_ids=("m1", "m2"),
    )

    assert outcome.answer == original
    assert outcome.protocol_failure_codes == ("citation_repair_invalid",)


def test_fail_closed_judgment_does_not_upgrade_unsupported_answers() -> None:
    raw = JudgeDecision(True, True, False, "supported")
    answer = GeneratedAnswer("ok", ("m1",), False)
    accepted = apply_fail_closed_judgment(
        case=_case(), answer=answer, decision=raw, packet_source_ids=("m1",)
    )
    assert accepted.answer_grounded and accepted.answer_correct

    abstention = apply_fail_closed_judgment(
        case=_case(evidence=()),
        answer=GeneratedAnswer("没有足够素材", (), True),
        decision=JudgeDecision(True, True, True, "abstained"),
        packet_source_ids=(),
    )
    assert abstention.answer_correct and abstention.abstained

    failed = apply_fail_closed_judgment(
        case=_case(),
        answer=GeneratedAnswer("ok", ("not-in-packet",), False),
        decision=raw,
        packet_source_ids=("m1",),
        known_source_ids=("m1", "not-in-packet"),
    )
    assert not failed.answer_grounded
    assert not failed.answer_correct
    assert failed.reason_code == "citation_outside_packet"


def test_bad_citations_are_recorded_per_case_without_cleaning_or_aborting() -> None:
    raw = JudgeDecision(True, True, False, "supported")
    answers = (
        GeneratedAnswer("bad unresolved", ("unknown",), False),
        GeneratedAnswer("bad forbidden", ("blocked",), False),
        GeneratedAnswer("good", ("m1",), False),
    )
    decisions = []
    for answer in answers:
        decisions.append(
            apply_fail_closed_judgment(
                case=_case(),
                answer=answer,
                decision=raw,
                packet_source_ids=("m1",),
                forbidden_source_ids=("blocked",),
                known_source_ids=("m1", "blocked"),
                ineligible_source_ids=("blocked",),
            )
        )

    assert decisions[0].reason_code == "citation_unresolved+citation_outside_packet"
    assert decisions[1].reason_code == (
        "citation_forbidden+citation_ineligible+citation_outside_packet"
    )
    assert not decisions[0].answer_correct
    assert not decisions[1].answer_correct
    assert decisions[2].answer_correct
    # The original invalid IDs remain available for the public source audit;
    # they were not silently replaced with the packet's valid source.
    assert answers[0].cited_source_message_ids == ("unknown",)
    assert answers[1].cited_source_message_ids == ("blocked",)


def test_generated_citation_failure_codes_are_stable_and_complete() -> None:
    failures = generated_citation_failure_codes(
        answer=GeneratedAnswer("bad", ("unknown", "blocked"), False),
        packet_source_ids=("m1",),
        forbidden_source_ids=("blocked",),
        known_source_ids=("m1", "blocked"),
        ineligible_source_ids=("blocked",),
    )
    assert failures == (
        "citation_unresolved",
        "citation_forbidden",
        "citation_ineligible",
        "citation_outside_packet",
    )


def test_answer_prompt_exposes_only_stably_sorted_authorized_evidence_ids() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    recent = EvidenceMessage(
        source_msg_id="recent-only",
        speaker="requester",
        content="recent context",
        sent_at=now,
        user_id="42",
        group_id=1,
    )
    evidence = (
        EvidenceMessage(
            source_msg_id="z-source",
            speaker="member",
            content="z evidence",
            sent_at=now,
            user_id="42",
            group_id=1,
        ),
        EvidenceMessage(
            source_msg_id="a-source",
            speaker="member",
            content="a evidence",
            sent_at=now,
            user_id="42",
            group_id=1,
        ),
    )
    segment = EvidenceSegment(episode_id="e1", fused_score=1.0, messages=evidence)
    packed = MemoryContextPacker().pack(
        "normal",
        available_input=32_000,
        target_message_id=None,
        recent_messages=(recent,),
        evidence_segments=(segment,),
    )
    trace = SimpleNamespace(
        result=SimpleNamespace(packed_context=packed),
        resolved_query=SimpleNamespace(
            answer_mode="dated_history",
            coverage_mode="chronological",
        ),
    )
    runtime_config = SimpleNamespace(persona={}, safety={})

    assert allowed_citation_ids_from_packed_context(packed) == (
        "a-source",
        "z-source",
    )
    prompt = build_answer_prompt(
        case=_case(evidence=("a-source", "z-source")),
        trace=trace,
        runtime_config=runtime_config,
    )
    contract_line = next(line for line in prompt if "Allowed citation IDs JSON list" in line)
    assert '["a-source","z-source"]' in contract_line
    assert "smallest set of messages" in contract_line
    assert "A one-fact answer must cite exactly one source ID" in contract_line
    assert "not an adjacent message, reaction, reply" in contract_line
    assert "two distinct factual clauses" in contract_line
    assert "more than two citation IDs" in contract_line
    assert "never copy the whole allowlist" in contract_line
    assert "Abstain only when no message in the allowlist" in contract_line
    assert "answer only that supported part" in contract_line
    assert "set abstained to false" in contract_line
    assert "colloquial, slang, brief" in contract_line
    assert "set cited_source_message_ids to []" in contract_line
    assert any("single-event dated-history lookup" in line for line in prompt)
    assert "set abstained to true" in contract_line
    assert FIXED_ABSTENTION_ANSWER in contract_line
    assert "Whenever abstained is true" in contract_line
    assert "Do not infer missing facts" in contract_line
    assert "evaluations, jokes, embellishment" in contract_line
    allowed_fragment = contract_line.split("Allowed citation IDs JSON list: ", 1)[1].split(
        ". IDs shown elsewhere", 1
    )[0]
    assert "recent-only" not in allowed_fragment


def test_answer_prompt_uses_empty_allowlist_when_there_is_no_evidence() -> None:
    packed = PackedMemoryContext(
        mode="normal",
        budget=32_000,
        estimated_tokens=0,
        text="",
        source_msg_ids=("orphan-source",),
    )
    trace = SimpleNamespace(result=SimpleNamespace(packed_context=packed))
    prompt = build_answer_prompt(
        case=_case(evidence=()),
        trace=trace,
        runtime_config=SimpleNamespace(persona={}, safety={}),
    )
    contract_line = next(line for line in prompt if "Allowed citation IDs JSON list" in line)
    assert "Allowed citation IDs JSON list: []" in contract_line
    assert "orphan-source" not in contract_line


def test_interval_answer_prompt_binds_each_citation_to_a_separate_clause() -> None:
    packed = PackedMemoryContext(
        mode="normal",
        budget=32_000,
        estimated_tokens=0,
        text="",
    )
    trace = SimpleNamespace(
        result=SimpleNamespace(packed_context=packed),
        resolved_query=SimpleNamespace(
            answer_mode="summary",
            coverage_mode="time_buckets",
        ),
    )

    prompt = build_answer_prompt(
        case=_case(evidence=()),
        trace=trace,
        runtime_config=SimpleNamespace(persona={}, safety={}),
    )

    assert any("Never attach two citations to one blended claim" in line for line in prompt)


def test_answer_prompt_does_not_inject_expected_evidence_or_abstention_label() -> None:
    packed = PackedMemoryContext(
        mode="normal",
        budget=32_000,
        estimated_tokens=0,
        text="",
        source_msg_ids=(),
    )
    trace = SimpleNamespace(result=SimpleNamespace(packed_context=packed))
    case = _case(evidence=("secret-expected-source",))
    prompt = build_answer_prompt(
        case=case,
        trace=trace,
        runtime_config=SimpleNamespace(persona={}, safety={}),
    )
    rendered = "\n".join(prompt)
    assert "secret-expected-source" not in rendered
    assert "expected evidence" not in rendered.casefold()


class _SequenceGenerationTransport:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.call_count = 0

    def generate(self, prompt, *, model):
        del prompt
        self.call_count += 1
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return ObservedGeneration(
            text=str(output),
            input_tokens=100,
            output_tokens=20,
            ttft_ms=10.0,
            model=model,
        )


def test_answer_generation_retries_citation_cap_then_accepts_valid_answer() -> None:
    transport = _SequenceGenerationTransport(
        [
            '{"answer":"over","cited_source_message_ids":["m1","m2","m3"],"abstained":false}',
            '{"answer":"valid","cited_source_message_ids":["m1"],"abstained":false}',
        ]
    )
    outcome = _generate_answer_with_retry(
        transport,
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=("m1", "m2", "m3"),
    )
    assert transport.call_count == 2
    assert outcome.answer.answer == "valid"
    assert outcome.protocol_failure_codes == ()


def test_answer_generation_retries_noncanonical_abstention_then_accepts_fixed_text() -> None:
    fixed = json.dumps(
        {
            "answer": FIXED_ABSTENTION_ANSWER,
            "cited_source_message_ids": [],
            "abstained": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    transport = _SequenceGenerationTransport(
        [
            '{"answer":"自由措辞","cited_source_message_ids":[],"abstained":true}',
            fixed,
        ]
    )
    outcome = _generate_answer_with_retry(
        transport,
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=(),
    )
    assert transport.call_count == 2
    assert outcome.answer.answer == FIXED_ABSTENTION_ANSWER
    assert outcome.protocol_failure_codes == ()


def test_final_noncanonical_abstention_is_preserved_and_failed() -> None:
    raw = '{"answer":"自由措辞","cited_source_message_ids":[],"abstained":true}'
    transport = _SequenceGenerationTransport([raw, raw])
    outcome = _generate_answer_with_retry(
        transport,
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=(),
    )
    assert outcome.answer.answer == "自由措辞"
    assert outcome.protocol_failure_codes == ("abstention_text_mismatch",)
    failed, _ = finalize_replay_case_judgment(
        case=_case(evidence=()),
        answer_outcome=outcome,
        raw_decision=JudgeDecision(True, True, True, "abstained"),
        packet_source_ids=(),
    )
    assert not failed.answer_grounded
    assert not failed.answer_correct


def test_answer_generation_retries_outside_allowlist_without_cleaning() -> None:
    outside = (
        '{"answer":"outside","cited_source_message_ids":["not-allowed"],'
        '"abstained":false}'
    )
    valid = (
        '{"answer":"valid","cited_source_message_ids":["allowed"],'
        '"abstained":false}'
    )
    retried = _generate_answer_with_retry(
        _SequenceGenerationTransport([outside, valid]),
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=("allowed",),
    )
    assert retried.answer.cited_source_message_ids == ("allowed",)
    assert retried.protocol_failure_codes == ()

    exhausted = _generate_answer_with_retry(
        _SequenceGenerationTransport([outside, outside]),
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=("allowed",),
    )
    assert exhausted.answer.cited_source_message_ids == ("not-allowed",)
    assert exhausted.protocol_failure_codes == ("citation_outside_allowlist",)


def test_final_citation_cap_failure_is_preserved_as_per_case_failure() -> None:
    over_limit = (
        '{"answer":"over","cited_source_message_ids":["m1","m2","m3"],'
        '"abstained":false}'
    )
    transport = _SequenceGenerationTransport([over_limit, over_limit])
    outcome = _generate_answer_with_retry(
        transport,
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=("m1", "m2", "m3"),
    )
    assert transport.call_count == 2
    assert outcome.answer.cited_source_message_ids == ("m1", "m2", "m3")
    assert outcome.protocol_failure_codes == ("citation_count_over_limit",)
    failed, citation_failures = finalize_replay_case_judgment(
        case=_case(),
        answer_outcome=outcome,
        raw_decision=JudgeDecision(True, True, False, "supported"),
        packet_source_ids=("m1", "m2", "m3"),
        known_source_ids=("m1", "m2", "m3"),
    )
    assert citation_failures == ()
    assert not failed.answer_grounded
    assert not failed.answer_correct
    assert failed.reason_code == "citation_count_over_limit"


def test_answer_generation_still_aborts_after_repeated_invalid_json() -> None:
    transport = _SequenceGenerationTransport(["not-json", "still-not-json"])
    with pytest.raises(QualityReplayError, match="MODEL_JSON_INVALID"):
        _generate_answer_with_retry(
            transport,
            ["prompt"],
            model="model",
            attempts=2,
            allowed_citation_ids=(),
        )
    assert transport.call_count == 2


def test_answer_generation_retries_exact_provider_failure_then_uses_success() -> None:
    transport = _SequenceGenerationTransport(
        [
            QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED"),
            '{"answer":"valid","cited_source_message_ids":["m1"],"abstained":false}',
        ]
    )
    outcome = _generate_answer_with_retry(
        transport,
        ["prompt"],
        model="model",
        attempts=2,
        allowed_citation_ids=("m1",),
    )
    assert transport.call_count == 2
    assert outcome.answer.answer == "valid"
    assert outcome.observation.ttft_ms == 10.0


def test_judge_generation_retries_exact_provider_failure_then_uses_success() -> None:
    transport = _SequenceGenerationTransport(
        [
            QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED"),
            (
                '{"answer_grounded":true,"answer_correct":true,'
                '"abstained":false,"reason_code":"supported"}'
            ),
        ]
    )
    observed, decision = _generate_valid_json(
        transport,
        ["judge prompt"],
        model="judge-model",
        attempts=2,
        parser=parse_judge_decision,
    )
    assert transport.call_count == 2
    assert observed.ttft_ms == 10.0
    assert decision.answer_correct


def test_continuous_provider_failures_exhaust_budget_and_abort() -> None:
    transport = _SequenceGenerationTransport(
        [
            QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED"),
            QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED"),
        ]
    )
    with pytest.raises(QualityReplayError, match="^QUALITY_REPLAY_PROVIDER_FAILED$"):
        _generate_answer_with_retry(
            transport,
            ["prompt"],
            model="model",
            attempts=2,
            allowed_citation_ids=(),
        )
    assert transport.call_count == 2


def test_provider_failure_then_invalid_json_reports_json_failure() -> None:
    transport = _SequenceGenerationTransport(
        [
            QualityReplayError("QUALITY_REPLAY_PROVIDER_FAILED"),
            "not-json",
        ]
    )
    with pytest.raises(QualityReplayError, match="MODEL_JSON_INVALID"):
        _generate_answer_with_retry(
            transport,
            ["prompt"],
            model="model",
            attempts=2,
            allowed_citation_ids=(),
        )
    assert transport.call_count == 2


def test_non_provider_protocol_error_is_not_retried() -> None:
    transport = _SequenceGenerationTransport(
        [
            QualityReplayError("QUALITY_REPLAY_TTFT_UNOBSERVABLE"),
            '{"answer":"must not run","cited_source_message_ids":[],"abstained":true}',
        ]
    )
    with pytest.raises(QualityReplayError, match="TTFT_UNOBSERVABLE"):
        _generate_answer_with_retry(
            transport,
            ["prompt"],
            model="model",
            attempts=2,
            allowed_citation_ids=(),
        )
    assert transport.call_count == 1


def test_judge_prompt_marks_clean_expected_abstention_correct() -> None:
    prompt = build_judge_prompt(
        case=_case(evidence=()),
        answer=GeneratedAnswer(FIXED_ABSTENTION_ANSWER, (), True),
        packet_text="",
        gold_text="",
    )
    rendered = "\n".join(prompt)
    assert "expected abstention" in rendered
    assert "has no citations" in rendered
    assert "no factual assertion" in rendered
    assert "answer_grounded=true and answer_correct=true" in rendered
    assert "makes any factual assertion" in rendered
    assert "must both be false" in rendered
    assert FIXED_ABSTENTION_ANSWER in rendered
    assert "is a protocol marker, not a factual assertion" in rendered
    assert "generated abstained=true" in rendered


def test_public_sidecar_binds_private_artifacts_without_content(tmp_path: Path) -> None:
    private_payload = {
        "query": "private query marker",
        "answer": "private answer marker",
    }
    private_path = tmp_path / "private.json"
    private_sha = _write_json(private_path, private_payload, private=True)
    visibility_path = tmp_path / "visibility.json"
    visibility_sha = _write_json(
        visibility_path,
        {"measurement_mode": "disposable_sqlite_online_backup_clone"},
        private=False,
    )
    row = {
        "case_index": 0,
        "cited_source_message_ids": ["m1"],
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "answer_protocol_failure_codes": [],
        "total_prompt_tokens": 123,
        "ttft_ms": 45.0,
    }
    sidecar = build_public_sidecar(
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
        retrieval_fingerprint="c" * 64,
        generator_model="generator",
        judge_model="judge",
        private_artifact_sha256=private_sha,
        visibility_artifact_sha256=visibility_sha,
        visibility_ms=[100.0] * 20,
        case_rows=[row],
        evaluated_at="2026-08-01T00:00:00Z",
        context_profile="adaptive",
    )
    rendered = json.dumps(sidecar, ensure_ascii=False)

    assert "private query marker" not in rendered
    assert "private answer marker" not in rendered
    assert sidecar["private_replay_sha256"] == private_sha
    assert sidecar["visibility_artifact_sha256"] == visibility_sha
    assert len(sidecar["prompt_contract_sha256"]) == 64
    assert sidecar["index_visibility_ms"] == [100.0] * 20
    assert sidecar["context_profile"] == "adaptive"


def test_sqlite_backup_reads_source_without_modifying_it(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    connection = sqlite3.connect(source)
    try:
        connection.execute("CREATE TABLE sample(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample(value) VALUES ('original')")
        connection.commit()
    finally:
        connection.close()
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    clone = tmp_path / "clone.db"
    _sqlite_readonly_backup(source, clone)
    clone_connection = sqlite3.connect(clone)
    try:
        clone_connection.execute("INSERT INTO sample(value) VALUES ('probe')")
        clone_connection.commit()
    finally:
        clone_connection.close()

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    source_connection = sqlite3.connect(source)
    try:
        assert source_connection.execute("SELECT count(*) FROM sample").fetchone()[0] == 1
    finally:
        source_connection.close()
