from datetime import UTC, datetime

from app.core.proactive_judge import (
    ProactiveJudgeResult,
    build_proactive_judge_prompt,
    judge_proactive_interjection,
    parse_proactive_judge,
)


class JudgeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[str]] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        return self.response


class ExplodingLlm:
    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        del prompt_lines
        raise RuntimeError("provider down")


def test_build_proactive_judge_prompt_includes_recent_and_target() -> None:
    now = datetime(2026, 5, 8, 12, 0, tzinfo=UTC)
    lines = build_proactive_judge_prompt(
        bot_name="小町",
        target_message="这也太离谱了吧",
        recent_messages=["A: 今天好热", "B: 确实"],
        now=now,
        context_messages=5,
        max_chars_per_message=120,
    )
    joined = "\n".join(lines)
    assert "小町" in joined
    assert "这也太离谱了吧" in joined
    assert "A: 今天好热" in joined
    assert "DECISION: yes|no" in joined
    assert "2026-05-08" in joined


def test_build_proactive_judge_prompt_truncates_and_limits_context() -> None:
    now = datetime(2026, 5, 8, 12, 0, tzinfo=UTC)
    recent = [f"long-{i}-" + "x" * 500 for i in range(10)]
    lines = build_proactive_judge_prompt(
        bot_name="bot",
        target_message="t",
        recent_messages=recent,
        now=now,
        context_messages=3,
        max_chars_per_message=50,
    )
    joined = "\n".join(lines)
    assert "long-6-" not in joined
    assert "long-7-" in joined
    assert "long-8-" in joined
    assert "long-9-" in joined
    assert "x" * 60 not in joined


def test_parse_proactive_judge_yes() -> None:
    result = parse_proactive_judge("DECISION: yes\nREASON: 这是个好梗")
    assert result == ProactiveJudgeResult(True, "这是个好梗")


def test_parse_proactive_judge_no() -> None:
    result = parse_proactive_judge("DECISION: no\nREASON: 无聊的琐事")
    assert result == ProactiveJudgeResult(False, "无聊的琐事")


def test_parse_proactive_judge_single_line_slash_format() -> None:
    result = parse_proactive_judge("DECISION: yes / REASON: 天气吐槽正热闹")
    assert result.should_interject is True
    assert result.reason == "天气吐槽正热闹"


def test_parse_proactive_judge_decision_anywhere_in_text() -> None:
    result = parse_proactive_judge("让我想想…DECISION: no / REASON: 太无聊了")
    assert result.should_interject is False
    assert result.reason == "太无聊了"


def test_parse_proactive_judge_case_insensitive_and_loose_whitespace() -> None:
    result = parse_proactive_judge("  decision : YES \n reason: 有槽点 ")
    assert result.should_interject is True
    assert result.reason == "有槽点"


def test_parse_proactive_judge_malformed_degrades_to_no() -> None:
    assert parse_proactive_judge("") == ProactiveJudgeResult(False, "malformed")
    assert parse_proactive_judge("maybe?") == ProactiveJudgeResult(False, "malformed")
    assert parse_proactive_judge(None) == ProactiveJudgeResult(False, "malformed")


def test_judge_returns_yes_when_model_agrees() -> None:
    client = JudgeLlm("DECISION: yes\nREASON: 接一下")
    result = judge_proactive_interjection(client=client, prompt_lines=["p"])
    assert result.should_interject is True
    assert result.reason == "接一下"


def test_judge_degrades_to_no_on_provider_error() -> None:
    result = judge_proactive_interjection(client=ExplodingLlm(), prompt_lines=["p"])
    assert result == ProactiveJudgeResult(False, "judge_error")
