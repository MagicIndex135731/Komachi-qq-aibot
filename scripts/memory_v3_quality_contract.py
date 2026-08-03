from __future__ import annotations

import hashlib
from pathlib import Path


ANSWER_CONTRACT_VERSION = "memory-v3-answer-replay-v10"
JUDGE_CONTRACT_VERSION = "memory-v3-answer-judge-v5"
FIXED_ABSTENTION_ANSWER = "没有足够的记忆素材回答这个问题。"
QUALITY_REPLAY_PROVIDER = "responses-controlled-replay"


def answer_contract_failure_codes(
    *,
    answer: object,
    citations: object,
    abstained: object,
) -> tuple[str, ...]:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("generated answer is empty")
    if (
        not isinstance(citations, list)
        or any(not isinstance(item, str) or not item for item in citations)
        or len(set(citations)) != len(citations)
    ):
        raise ValueError("generated citations are invalid")
    if not isinstance(abstained, bool):
        raise ValueError("generated abstention is invalid")

    failures: list[str] = []
    if len(citations) > 2:
        failures.append("citation_count_over_limit")
    if abstained:
        if answer.strip() != FIXED_ABSTENTION_ANSWER:
            failures.append("abstention_text_mismatch")
        if citations:
            failures.append("abstention_citations_nonempty")
    elif answer.strip() == FIXED_ABSTENTION_ANSWER:
        failures.append("abstention_flag_mismatch")
    elif not citations:
        failures.append("citation_missing")
    return tuple(failures)


def prompt_contract_sha256() -> str:
    replay_source = Path(__file__).with_name("run_memory_v3_quality_replay.py").read_bytes()
    digest = hashlib.sha256()
    digest.update(f"{ANSWER_CONTRACT_VERSION}\n{JUDGE_CONTRACT_VERSION}\n".encode("utf-8"))
    digest.update(hashlib.sha256(replay_source).digest())
    return digest.hexdigest()
