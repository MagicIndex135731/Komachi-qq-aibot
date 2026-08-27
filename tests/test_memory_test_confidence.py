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
    assert public["overall"]["cross_mode_sampled_cases"] == 0
    assert public["overall"]["cross_mode_outcome_disagreement_cases"] == 0
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
    assert {
        row["baseline_judge_packet_mode"] for row in private_rows
    } == {confidence.LEGACY_JUDGE_PACKET_MODE}
    assert {
        row["resample_judge_packet_mode"] for row in private_rows
    } == {confidence.UNSAMPLED_JUDGE_PACKET_MODE}


@pytest.mark.parametrize("missing_field", ("allowed_citation_ids", "judge_prompt_full"))
def test_answer_resampling_fails_closed_without_new_private_contract(
    tmp_path, missing_field
):
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
        "answer_prompt": ["answer prompt"],
        "judge_prompt": _prompt(),
        "judge_prompt_full": _prompt(),
        "allowed_citation_ids": ["m1"],
    }
    del row[missing_field]
    checkpoint = tmp_path / "existing-private.jsonl"
    checkpoint.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(ValueError, match=missing_field):
        confidence.run_confidence_replay(
            [row],
            answer_transport=object(),
            judge_transport=object(),
            cache_dir=tmp_path / "cache",
            answer_repeats=3,
            checkpoint_path=checkpoint,
        )

    assert checkpoint.read_text(encoding="utf-8") == "preserve me\n"


def test_input_protocol_failure_fails_before_transport_or_checkpoint_mutation(
    tmp_path, monkeypatch
):
    def unexpected_call(**_kwargs):
        raise AssertionError("transport path must not run for failed fullchain input")

    monkeypatch.setattr(confidence, "_judge_samples", unexpected_call)
    monkeypatch.setattr(confidence, "_generate_cached", unexpected_call)
    checkpoint = tmp_path / "existing-private.jsonl"
    checkpoint.write_text("preserve me\n", encoding="utf-8")
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
        "answer_prompt": ["answer prompt"],
        "judge_prompt": _prompt(),
        "protocol_failure_codes": ["citation_repair_invalid"],
    }

    with pytest.raises(ValueError, match="input contains protocol failures"):
        confidence.run_confidence_replay(
            [row],
            answer_transport=object(),
            judge_transport=object(),
            cache_dir=tmp_path / "cache",
            answer_repeats=1,
            checkpoint_path=checkpoint,
        )

    assert checkpoint.read_text(encoding="utf-8") == "preserve me\n"


def test_answer_resampling_uses_full_packet_and_explicit_allowlist(
    tmp_path, monkeypatch
):
    judge_prompts = []
    judge_axes = []
    repair_allowlists = []
    generated = SimpleNamespace(
        text=json.dumps(
            {
                "answer": "new answer",
                "cited_source_message_ids": ["allowed"],
                "abstained": False,
            }
        ),
        input_tokens=1,
        output_tokens=1,
        ttft_ms=1.0,
        model="m",
        attempt_count=1,
        no_event_attempts=0,
    )

    def fake_generate_cached(**kwargs):
        assert kwargs["axis"].startswith("answer|")
        return generated, False

    def fake_repair_answer(**kwargs):
        repair_allowlists.append(tuple(kwargs["allowed_ids"]))
        return kwargs["parsed"], None

    def fake_judge_samples(**kwargs):
        judge_prompts.append("\n".join(kwargs["prompt"]))
        judge_axes.append(kwargs["axis"])
        grounded_correct = not kwargs["axis"].startswith("baseline-full-judge|")
        return [
            {
                "decision": {
                    "answer_grounded": grounded_correct,
                    "answer_correct": grounded_correct,
                    "abstained": False,
                    "reason_code": "ok",
                }
            }
        ]

    monkeypatch.setattr(confidence, "_generate_cached", fake_generate_cached)
    monkeypatch.setattr(confidence, "_repair_answer", fake_repair_answer)
    monkeypatch.setattr(confidence, "_judge_samples", fake_judge_samples)
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
        "category": "event",
        "answer_prompt": ["answer prompt"],
        "judge_prompt": [
            line.replace("packet sentinel", "FOCUSED_PACKET")
            for line in _prompt()
        ],
        "judge_prompt_full": [
            line.replace("packet sentinel", "FULL_PACKET")
            for line in _prompt()
        ],
        "judge_packet_mode": "citation-focused",
        "answer": "old",
        "cited_source_message_ids": ["m1"],
        "allowed_citation_ids": ["allowed"],
        "packed_source_ids": ["allowed", "recent-only"],
        "generated_abstained": False,
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "judge_reason_code": "ok",
        "protocol_failure_codes": [],
    }

    private_rows, public = confidence.run_confidence_replay(
        [row],
        answer_transport=object(),
        judge_transport=object(),
        cache_dir=tmp_path / "cache",
        answer_repeats=3,
        judge_repeats=1,
        provider_attempts=1,
        provider_backoff=0,
    )

    assert len(judge_prompts) == 4
    assert judge_axes == [
        "baseline-judge|case-1",
        "baseline-full-judge|case-1",
        "answer-1-judge|case-1",
        "answer-2-judge|case-1",
    ]
    assert "FOCUSED_PACKET" in judge_prompts[0]
    assert all("FULL_PACKET" in prompt for prompt in judge_prompts[1:])
    assert all("FOCUSED_PACKET" not in prompt for prompt in judge_prompts[1:])
    assert "Generated answer:\nold" in judge_prompts[1]
    assert all("Generated answer:\nnew answer" in prompt for prompt in judge_prompts[2:])
    assert repair_allowlists == [("allowed",), ("allowed",)]
    assert private_rows[0]["baseline_judge_packet_mode"] == "citation-focused"
    assert private_rows[0]["resample_judge_packet_mode"] == "full"
    assert (
        private_rows[0]["answer_samples"][0]["judge_packet_mode"]
        == "citation-focused"
    )
    assert private_rows[0]["answer_samples"][0]["sample_role"] == "baseline"
    assert (
        private_rows[0]["answer_samples"][1]["sample_role"]
        == "baseline-full-rejudge"
    )
    assert private_rows[0]["answer_samples"][1]["answer"] == "old"
    assert all(
        sample["judge_packet_mode"] == "full"
        for sample in private_rows[0]["answer_samples"][1:]
    )
    assert all(
        sample["sample_role"] == "resample"
        for sample in private_rows[0]["answer_samples"][2:]
    )
    assert public["schema_version"] == 5
    assert public["overall"]["baseline_judge_packet_modes"] == ["citation-focused"]
    assert public["overall"]["resample_judge_packet_modes"] == ["full"]
    assert public["overall"]["answer_sampled_cases"] == 1
    assert public["overall"]["answer_outcome_flip_cases"] == 1
    assert public["overall"]["cross_mode_sampled_cases"] == 1
    assert public["overall"]["cross_mode_outcome_disagreement_cases"] == 1

    judge_prompts.clear()
    judge_axes.clear()
    repair_allowlists.clear()
    full_mode_row = {
        **row,
        "judge_prompt": row["judge_prompt_full"],
        "judge_packet_mode": "full",
    }
    full_mode_private, _ = confidence.run_confidence_replay(
        [full_mode_row],
        answer_transport=object(),
        judge_transport=object(),
        cache_dir=tmp_path / "full-mode-cache",
        answer_repeats=3,
        judge_repeats=1,
        provider_attempts=1,
        provider_backoff=0,
    )

    assert judge_axes == [
        "baseline-judge|case-1",
        "answer-1-judge|case-1",
        "answer-2-judge|case-1",
    ]
    assert [
        sample["sample_role"] for sample in full_mode_private[0]["answer_samples"]
    ] == ["baseline", "resample", "resample"]


def test_compare_full_judge_packet_rejudges_frozen_answer_without_resampling(
    tmp_path, monkeypatch
):
    judge_axes = []
    judge_prompts = []

    def unexpected_generate(**_kwargs):
        raise AssertionError("packet A/B must not generate a new answer")

    def fake_judge_samples(**kwargs):
        judge_axes.append(kwargs["axis"])
        judge_prompts.append("\n".join(kwargs["prompt"]))
        return [
            {
                "decision": {
                    "answer_grounded": True,
                    "answer_correct": True,
                    "abstained": False,
                    "reason_code": "ok",
                }
            }
        ] * kwargs["repeats"]

    monkeypatch.setattr(confidence, "_generate_cached", unexpected_generate)
    monkeypatch.setattr(confidence, "_judge_samples", fake_judge_samples)
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
        "category": "event",
        "answer_prompt": ["answer prompt"],
        "judge_prompt": [
            line.replace("packet sentinel", "FOCUSED_PACKET") for line in _prompt()
        ],
        "judge_prompt_full": [
            line.replace("packet sentinel", "FULL_PACKET") for line in _prompt()
        ],
        "judge_packet_mode": "citation-focused",
        "answer": "frozen answer",
        "cited_source_message_ids": ["m1"],
        "generated_abstained": False,
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "judge_reason_code": "ok",
        "protocol_failure_codes": [],
    }

    private, public = confidence.run_confidence_replay(
        [row],
        answer_transport=object(),
        judge_transport=object(),
        cache_dir=tmp_path / "cache",
        answer_repeats=1,
        judge_repeats=3,
        compare_full_judge_packet=True,
    )

    assert judge_axes == ["baseline-judge|case-1", "baseline-full-judge|case-1"]
    assert "FOCUSED_PACKET" in judge_prompts[0]
    assert "FULL_PACKET" in judge_prompts[1]
    assert all("Generated answer:\nfrozen answer" in prompt for prompt in judge_prompts)
    assert [sample["sample_role"] for sample in private[0]["answer_samples"]] == [
        "baseline",
        "baseline-full-rejudge",
    ]
    assert private[0]["compare_full_judge_packet"] is True
    assert private[0]["resample_judge_packet_mode"] == "full"
    assert public["overall"]["cross_mode_sampled_cases"] == 1
    assert public["overall"]["full_judge_packet_comparison_requested_cases"] == 1
    assert public["overall"]["full_judge_packet_comparison_performed_cases"] == 1


def test_compare_full_judge_packet_skips_duplicate_when_baseline_is_full(
    tmp_path, monkeypatch
):
    judge_axes = []

    def fake_judge_samples(**kwargs):
        judge_axes.append(kwargs["axis"])
        return [
            {
                "decision": {
                    "answer_grounded": True,
                    "answer_correct": True,
                    "abstained": False,
                    "reason_code": "ok",
                }
            }
        ]

    monkeypatch.setattr(confidence, "_judge_samples", fake_judge_samples)
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
        "answer_prompt": ["answer prompt"],
        "judge_prompt": _prompt(),
        "judge_prompt_full": _prompt(),
        "judge_packet_mode": "full",
        "answer": "frozen answer",
        "cited_source_message_ids": ["m1"],
        "generated_abstained": False,
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "judge_reason_code": "ok",
        "protocol_failure_codes": [],
    }

    private, public = confidence.run_confidence_replay(
        [row],
        answer_transport=object(),
        judge_transport=object(),
        cache_dir=tmp_path / "cache",
        answer_repeats=1,
        judge_repeats=1,
        compare_full_judge_packet=True,
    )

    assert judge_axes == ["baseline-judge|case-1"]
    assert [sample["sample_role"] for sample in private[0]["answer_samples"]] == [
        "baseline"
    ]
    assert public["overall"]["full_judge_packet_comparison_requested_cases"] == 1
    assert public["overall"]["full_judge_packet_comparison_performed_cases"] == 0


@pytest.mark.parametrize(
    ("field", "invalid_value", "error"),
    [
        ("case_input_signature", "not-a-signature", "signed fullchain"),
        ("answer_prompt", "not-a-private-prompt", "answer_prompt"),
        ("judge_prompt", "not-a-private-prompt", "judge_prompt"),
        ("judge_prompt_full", None, "judge_prompt_full"),
        ("allowed_citation_ids", None, "allowed_citation_ids"),
    ],
)
def test_batch_preflight_rejects_later_invalid_row_before_checkpoint_or_transport(
    tmp_path, monkeypatch, field, invalid_value, error
):
    def private_row(case_id):
        return {
            "case_id": case_id,
            "case_input_signature": CASE_SIGNATURE,
            "resume_base_signature": BASE_SIGNATURE,
            "answer_prompt": ["answer prompt"],
            "judge_prompt": _prompt(),
            "judge_prompt_full": _prompt(),
            "allowed_citation_ids": ["m1"],
            "protocol_failure_codes": [],
        }

    def unexpected_transport(**_kwargs):
        raise AssertionError("batch preflight must run before transport")

    monkeypatch.setattr(confidence, "_judge_samples", unexpected_transport)
    monkeypatch.setattr(confidence, "_generate_cached", unexpected_transport)
    rows = [private_row("case-1"), private_row("case-2")]
    if invalid_value is None:
        del rows[1][field]
    else:
        rows[1][field] = invalid_value
    checkpoint = tmp_path / "existing-private.jsonl"
    checkpoint.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        confidence.run_confidence_replay(
            rows,
            answer_transport=object(),
            judge_transport=object(),
            cache_dir=tmp_path / "cache",
            answer_repeats=3,
            judge_repeats=1,
            checkpoint_path=checkpoint,
        )

    assert checkpoint.read_text(encoding="utf-8") == "preserve me\n"


def test_cli_parser_exposes_full_judge_packet_comparison_flag():
    args = confidence.build_argument_parser().parse_args(
        [
            "--input-detail",
            "private-input.jsonl",
            "--output-private",
            "private-output.jsonl",
            "--output-public",
            "public-output.json",
            "--cache-dir",
            "cache",
            "--compare-full-judge-packet",
        ]
    )

    assert args.compare_full_judge_packet is True


def test_cli_passes_full_judge_packet_comparison_to_runtime(tmp_path, monkeypatch):
    captured = {}
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
    }

    monkeypatch.setattr(confidence, "_load_latest_rows", lambda _path: [row])
    monkeypatch.setattr(confidence, "AppSettings", lambda: object())
    monkeypatch.setattr(
        confidence, "_build_eval_clients", lambda *_args, **_kwargs: (object(), object())
    )
    monkeypatch.setattr(
        confidence, "ObservedResponsesTransport", lambda client, **_kwargs: client
    )

    def fake_run_confidence_replay(_rows, **kwargs):
        captured["rows"] = _rows
        captured.update(kwargs)
        return [], {
            "overall": {"provider_failures": 0, "protocol_failures": 0},
        }

    monkeypatch.setattr(confidence, "run_confidence_replay", fake_run_confidence_replay)
    output_private = tmp_path / "private.jsonl"
    output_public = tmp_path / "public.json"

    exit_code = confidence.main(
        [
            "--input-detail",
            str(tmp_path / "input.jsonl"),
            "--output-private",
            str(output_private),
            "--output-public",
            str(output_public),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--compare-full-judge-packet",
        ]
    )

    assert exit_code == 0
    assert captured["rows"] == [row]
    assert captured["compare_full_judge_packet"] is True
    assert output_public.exists()


def test_repair_answer_sanitizes_mixed_citations_without_model_call(
    tmp_path, monkeypatch
):
    def unexpected_generate(**_kwargs):
        raise AssertionError("deterministic citation sanitation must not call a model")

    monkeypatch.setattr(confidence, "_generate_cached", unexpected_generate)
    parsed = SimpleNamespace(
        answer="supported answer",
        cited_source_message_ids=("allowed", "invented"),
        abstained=False,
    )

    repaired, metadata = confidence._repair_answer(
        answer_prompt=["answer prompt"],
        parsed=parsed,
        allowed_ids=("allowed",),
        transport=object(),
        judge_model="judge",
        judge_effort="medium",
        cache_dir=tmp_path,
        axis="repair-test",
        provider_attempts=1,
        provider_backoff=0,
    )

    assert repaired.cited_source_message_ids == ("allowed",)
    assert metadata == {
        "protocol_failure_codes": [],
        "observations": [],
        "citation_sanitized": True,
        "method": "deterministic_allowlist_filter",
    }


def test_public_summary_fails_closed_on_repair_protocol_failure_codes():
    invalid_repair_sample = {
        "judge_packet_mode": "full",
        "repair": {"protocol_failure_codes": ["citation_repair_invalid"]},
        # A stale/legacy decision must not make a failed repair comparable.
        "judge_majority": {"grounded_correct": False, "valid_samples": 1},
        "judge_samples": [],
    }
    deterministic_sanitation_sample = {
        "judge_packet_mode": "full",
        "repair": {
            "protocol_failure_codes": [],
            "citation_sanitized": True,
        },
        "judge_majority": {"grounded_correct": True, "valid_samples": 1},
        "judge_samples": [],
    }
    private = [
        {
            "category": "event",
            "baseline_judge_packet_mode": "full",
            "resample_judge_packet_mode": "full",
            "answer_samples": [
                {
                    "judge_packet_mode": "full",
                    "judge_majority": {
                        "grounded_correct": True,
                        "valid_samples": 1,
                    },
                    "judge_samples": [],
                },
                invalid_repair_sample,
                deterministic_sanitation_sample,
            ],
        }
    ]

    public = confidence.build_public_summary(private)
    overall = public["overall"]

    assert overall["protocol_failures"] == 1
    assert overall["answer_sampled_cases"] == 1
    assert overall["answer_outcome_flip_cases"] == 0
    assert confidence._result_exit_code(public) == 2


def test_answer_resample_repair_protocol_failure_skips_judge(tmp_path, monkeypatch):
    generated = SimpleNamespace(
        text=json.dumps(
            {
                "answer": "unsupported",
                "cited_source_message_ids": [],
                "abstained": False,
            }
        ),
        input_tokens=1,
        output_tokens=1,
        ttft_ms=1.0,
        model="m",
        attempt_count=1,
        no_event_attempts=0,
    )
    judge_axes = []

    monkeypatch.setattr(
        confidence, "_generate_cached", lambda **_kwargs: (generated, False)
    )
    monkeypatch.setattr(
        confidence,
        "_repair_answer",
        lambda **kwargs: (
            kwargs["parsed"],
            {"protocol_failure_codes": ["citation_repair_invalid"]},
        ),
    )

    def fake_judge_samples(**kwargs):
        judge_axes.append(kwargs["axis"])
        return [
            {
                "decision": {
                    "answer_grounded": True,
                    "answer_correct": True,
                    "abstained": False,
                    "reason_code": "ok",
                }
            }
        ]

    monkeypatch.setattr(confidence, "_judge_samples", fake_judge_samples)
    row = {
        "case_id": "case-1",
        "case_input_signature": CASE_SIGNATURE,
        "resume_base_signature": BASE_SIGNATURE,
        "category": "event",
        "answer_prompt": ["answer prompt"],
        "judge_prompt": _prompt(),
        "judge_prompt_full": _prompt(),
        "judge_packet_mode": "citation-focused",
        "answer": "old",
        "cited_source_message_ids": ["m1"],
        "allowed_citation_ids": ["m1"],
        "generated_abstained": False,
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "judge_reason_code": "ok",
        "protocol_failure_codes": [],
    }

    private, public = confidence.run_confidence_replay(
        [row],
        answer_transport=object(),
        judge_transport=object(),
        cache_dir=tmp_path / "cache",
        answer_repeats=3,
        judge_repeats=1,
        provider_attempts=1,
        provider_backoff=0,
    )

    assert judge_axes == ["baseline-judge|case-1", "baseline-full-judge|case-1"]
    assert all(
        sample["repair_protocol_failure_codes"] == ["citation_repair_invalid"]
        and "judge_samples" not in sample
        for sample in private[0]["answer_samples"][2:]
    )
    assert public["overall"]["protocol_failures"] == 2
    assert public["overall"]["answer_sampled_cases"] == 0
    assert confidence._result_exit_code(public) == 2


def test_public_summary_separates_same_mode_answer_flip_from_cross_mode_disagreement():
    def answer_sample(sample_role, packet_mode, grounded_correct, answer):
        return {
            "sample_role": sample_role,
            "judge_packet_mode": packet_mode,
            "answer": answer,
            "cited_source_message_ids": ["m1"],
            "abstained": False,
            "judge_majority": {
                "grounded_correct": grounded_correct,
                "valid_samples": 1,
            },
            "judge_samples": [],
        }

    private = [
        {
            "category": "event",
            "baseline_judge_packet_mode": "citation-focused",
            "resample_judge_packet_mode": "full",
            "answer_samples": [
                answer_sample("baseline", "citation-focused", True, "frozen"),
                answer_sample("baseline-full-rejudge", "full", True, "frozen"),
                answer_sample("resample", "full", False, "changed"),
            ],
        }
    ]

    overall = confidence.build_public_summary(private)["overall"]

    assert overall["answer_sampled_cases"] == 1
    assert overall["answer_outcome_flip_cases"] == 1
    assert overall["answer_outcome_flip_rate"] == 1.0
    # Packet-mode disagreement compares only the same frozen answer pair.
    assert overall["cross_mode_sampled_cases"] == 1
    assert overall["cross_mode_outcome_disagreement_cases"] == 0
    assert overall["cross_mode_outcome_disagreement_rate"] == 0.0


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
