import json
from types import SimpleNamespace

import pytest

from scripts import memory_test_confidence as confidence


CASE_SIGNATURE = "a" * 64
BASE_SIGNATURE = "b" * 64


def _prompt(answer="old", citations=("m1",), abstained=False):
    return [
        "judge contract",
        "Question:\nq\n"
        f"Generated answer:\n{answer}\n"
        "Generated citation IDs:\n"
        + json.dumps(list(citations))
        + "\nGenerated abstained flag:\n"
        + json.dumps(abstained)
        + "\nRetrieved packet:\npacket sentinel\n"
        "Human-reviewed reference evidence:\ngold sentinel",
    ]


def test_replace_judge_answer_preserves_frozen_packet_and_gold():
    result = confidence._replace_judge_answer(
        _prompt(), answer="new", cited_ids=("m2",), abstained=True
    )
    rendered = "\n".join(result)
    assert "Generated answer:\nnew" in rendered
    assert 'Generated citation IDs:\n["m2"]' in rendered
    assert "Generated abstained flag:\ntrue" in rendered
    assert "packet sentinel" in rendered
    assert "gold sentinel" in rendered
    assert "Generated answer:\nold" not in rendered


def test_extract_judge_answer_distinguishes_generator_from_judge_abstention():
    answer, citations, abstained = confidence._extract_judge_answer(
        _prompt(answer="kept", citations=("m1", "m2"), abstained=True)
    )

    assert answer == "kept"
    assert citations == ("m1", "m2")
    assert abstained is True


def test_replace_judge_answer_rejects_public_or_incomplete_prompt():
    with pytest.raises(ValueError, match="replaceable"):
        confidence._replace_judge_answer(
            ["aggregate only"], answer="x", cited_ids=(), abstained=False
        )


def test_public_summary_reports_judge_and_answer_flips_without_private_text():
    good = {
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "reason_code": "ok",
    }
    bad = {
        "answer_grounded": True,
        "answer_correct": False,
        "abstained": False,
        "reason_code": "mismatch",
    }
    private = [
        {
            "case_id": "private-case-id",
            "category": "raw_history",
            "answer_prompt": ["private query sentinel"],
            "answer_samples": [
                {
                    "answer": "private answer sentinel",
                    "judge_samples": [
                        {"decision": good},
                        {"decision": bad},
                        {"decision": good},
                    ],
                    "judge_majority": {
                        "grounded_correct": True,
                        "valid_samples": 3,
                    },
                },
                {
                    "answer": "different private answer",
                    "judge_samples": [{"decision": bad}],
                    "judge_majority": {
                        "grounded_correct": False,
                        "valid_samples": 1,
                    },
                },
            ],
        }
    ]
    public = confidence.build_public_summary(private)
    assert public["overall"]["judge_flip_cases"] == 1
    assert public["overall"]["answer_outcome_flip_cases"] == 1
    assert public["overall"]["judge_majority_grounded_correct"] == 1
    rendered = json.dumps(public, ensure_ascii=False)
    assert "private-case-id" not in rendered
    assert "private query sentinel" not in rendered
    assert "private answer sentinel" not in rendered


def test_run_confidence_replay_requires_odd_repeat_counts():
    with pytest.raises(ValueError, match="answer_repeats"):
        confidence.run_confidence_replay(
            [],
            answer_transport=object(),
            judge_transport=object(),
            cache_dir=None,
            answer_repeats=2,
        )
    with pytest.raises(ValueError, match="judge_repeats"):
        confidence.run_confidence_replay(
            [],
            answer_transport=object(),
            judge_transport=object(),
            cache_dir=None,
            judge_repeats=2,
        )


def test_run_confidence_replay_checkpoints_each_completed_case(tmp_path, monkeypatch):
    decision = {
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "reason_code": "ok",
    }
    monkeypatch.setattr(
        confidence,
        "_judge_samples",
        lambda **kwargs: [{"decision": decision}] * 3,
    )
    checkpoint = tmp_path / "private.jsonl"
    rows = [
        {
            "case_id": case_id,
            "case_input_signature": CASE_SIGNATURE,
            "resume_base_signature": BASE_SIGNATURE,
            "category": "raw_history",
            "answer_prompt": ["answer prompt"],
            "judge_prompt": _prompt(),
            "answer": "old",
            "cited_source_message_ids": ["m1"],
            "packed_source_ids": ["m1"],
            "generated_abstained": False,
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "judge_reason_code": "ok",
            "protocol_failure_codes": [],
        }
        for case_id in ("case-1", "case-2")
    ]
    private_rows, _public = confidence.run_confidence_replay(
        rows,
        answer_transport=object(),
        judge_transport=object(),
        cache_dir=tmp_path / "cache",
        answer_repeats=1,
        judge_repeats=3,
        provider_backoff=0,
        checkpoint_path=checkpoint,
    )
    checkpoint_rows = [
        json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in checkpoint_rows] == ["case-1", "case-2"]
    assert checkpoint_rows == private_rows
    assert {row["case_input_signature"] for row in private_rows} == {CASE_SIGNATURE}
    assert {row["resume_base_signature"] for row in private_rows} == {BASE_SIGNATURE}


def test_load_latest_rows_rejects_unsigned_or_mixed_base(tmp_path):
    unsigned = tmp_path / "unsigned.jsonl"
    unsigned.write_text('{"case_id":"case-1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="signed fullchain"):
        confidence._load_latest_rows(unsigned)

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-1",
                        "case_input_signature": "a" * 64,
                        "resume_base_signature": "b" * 64,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-2",
                        "case_input_signature": "c" * 64,
                        "resume_base_signature": "d" * 64,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="share one signed fullchain base"):
        confidence._load_latest_rows(mixed)


def test_load_latest_rows_uses_latest_signed_duplicate(tmp_path):
    path = tmp_path / "detail.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-1",
                        "value": "old",
                        "case_input_signature": "c" * 64,
                        "resume_base_signature": BASE_SIGNATURE,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-1",
                        "value": "new",
                        "case_input_signature": CASE_SIGNATURE,
                        "resume_base_signature": BASE_SIGNATURE,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = confidence._load_latest_rows(path)

    assert len(rows) == 1
    assert rows[0]["value"] == "new"


def test_result_exit_code_fails_on_provider_or_protocol_errors():
    assert (
        confidence._result_exit_code(
            {"overall": {"provider_failures": 0, "protocol_failures": 0}}
        )
        == 0
    )
    assert (
        confidence._result_exit_code(
            {"overall": {"provider_failures": 1, "protocol_failures": 0}}
        )
        == 2
    )
    assert (
        confidence._result_exit_code(
            {"overall": {"provider_failures": 0, "protocol_failures": 1}}
        )
        == 2
    )


def test_judge_samples_include_baseline_and_use_fresh_samples(tmp_path, monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs["sample_index"])
        return (
            SimpleNamespace(
                text='{"answer_grounded":true,"answer_correct":false,"abstained":false,"reason_code":"x"}',
                input_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                model="m",
                attempt_count=1,
                no_event_attempts=0,
            ),
            False,
        )

    monkeypatch.setattr(confidence, "_generate_cached", fake_generate)
    baseline = {
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "reason_code": "ok",
    }
    samples = confidence._judge_samples(
        prompt=_prompt(),
        baseline_decision=baseline,
        repeats=3,
        axis="test",
        transport=object(),
        model="m",
        effort="medium",
        cache_dir=tmp_path,
        provider_attempts=1,
        provider_backoff=0,
    )
    assert calls == [1, 2]
    assert len(samples) == 3
    assert confidence._majority_decision(samples)["answer_correct"] is False
