from __future__ import annotations

import json
from pathlib import Path

from scripts.run_persona_memory_incident_replay import (
    run_incident_contract,
    validate_fullchain_report,
)


def test_versioned_persona_memory_incident_contract() -> None:
    report = run_incident_contract(
        Path("tests/fixtures/persona_memory_incidents.json")
    )

    assert report["passed"] is True, report["failures"]
    assert report["query_case_count"] >= 12
    assert report["clarification_case_count"] >= 2
    assert report["case_count"] >= 14


def test_fullchain_report_must_prove_selected_sources_and_final_answer(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "fullchain.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id_map": {
                    "source-progress": "real-progress",
                    "source-stage": "real-stage",
                    "source-tendency": "real-tendency",
                    "source-confirmation": "real-confirmation",
                    "source-other-member": "real-other",
                    "source-expired-state": "real-expired",
                    "source-anime-a": "real-anime-a",
                    "source-anime-b": "real-anime-b",
                    "source-discussion-only": "real-discussion",
                },
                "cases": [
                    {
                        "id": "implicit-work-progress-and-company-tendency",
                        "subject_binding": "impersonated",
                        "answer_mode": "current_fact",
                        "topic_terms": ["工作", "公司"],
                        "selected_source_ids": [
                            "real-progress",
                            "real-stage",
                            "real-tendency",
                            "real-confirmation",
                        ],
                        "answer_text": "目前有多家公司在推进，记录更支持星河科技，但没有数值概率，也不能确认最终结果。",
                    },
                    {
                        "id": "implicit-plural-animation",
                        "subject_binding": "impersonated",
                        "answer_mode": "current_fact",
                        "topic_terms": ["动画"],
                        "selected_source_ids": ["real-anime-a", "real-anime-b"],
                        "answer_text": "最近记录里有两部作品。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = validate_fullchain_report(
        Path("tests/fixtures/persona_memory_incidents.json"),
        report_path,
    )

    assert report["passed"] is True, report["failures"]


def test_fullchain_report_fails_when_declared_sources_were_not_selected(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "fullchain-missing.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "implicit-work-progress-and-company-tendency",
                        "subject_binding": "impersonated",
                        "answer_mode": "current_fact",
                        "topic_terms": ["工作", "公司"],
                        "selected_source_ids": ["source-progress"],
                        "answer_text": "还在推进。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = validate_fullchain_report(
        Path("tests/fixtures/persona_memory_incidents.json"),
        report_path,
    )

    assert report["passed"] is False
    assert any("missing_sources" in failure for failure in report["failures"])
