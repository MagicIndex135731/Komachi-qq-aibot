from scripts.audit_memory_observed_answers import (
    _build_observed_audit_prompt,
    public_summary,
)
from scripts.memory_test_confidence import _extract_judge_answer


def test_observed_answer_public_summary_contains_only_aggregate_results() -> None:
    summary = public_summary(
        [
            {
                "observed_answer": "private answer",
                "valid_samples": 3,
                "grounded_correct": True,
                "judge_samples": [
                    {
                        "decision": {
                            "answer_grounded": True,
                            "answer_correct": True,
                            "reason_code": "supported",
                        }
                    }
                ],
            },
            {
                "observed_answer": "another private answer",
                "valid_samples": 3,
                "grounded_correct": False,
                "judge_samples": [{"protocol_error": "ValueError"}],
            },
        ]
    )

    assert summary["cases"] == 2
    assert summary["grounded_correct"] == 1
    assert summary["incorrect_cases"] == 1
    assert summary["protocol_failures"] == 1
    assert "private answer" not in str(summary)


def test_observed_audit_prompt_does_not_fabricate_historical_citations() -> None:
    frozen = [
        "Generated answer:\nold\nGenerated citation IDs:\n[\"m-1\"]"
        "\nGenerated abstained flag:\nfalse\nRetrieved packet:\nevidence"
    ]

    prompt = _build_observed_audit_prompt(frozen, observed_answer="historical reply")
    answer, citations, abstained = _extract_judge_answer(prompt)

    assert answer == "historical reply"
    assert citations == ()
    assert abstained is False
    assert "Do not penalize empty citation IDs" in prompt[-1]
