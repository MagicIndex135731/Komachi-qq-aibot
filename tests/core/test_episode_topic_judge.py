from datetime import UTC, datetime

from app.core.episode_topic_judge import (
    TopicJudgeResult,
    build_topic_judge_prompt,
    judge_topic_switch,
    parse_topic_judge,
)


class TopicJudgeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[str]] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        return self.response


class ExplodingTopicJudgeLlm:
    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        del prompt_lines
        raise RuntimeError("provider down")


def test_build_topic_judge_prompt_includes_recent_and_current() -> None:
    now = datetime(2026, 8, 8, 22, 0, tzinfo=UTC)
    lines = build_topic_judge_prompt(
        recent_messages=["1: 今天好热", "2: 确实"],
        current_message="有人玩原神吗",
        now=now,
        context_messages=8,
        max_chars_per_message=160,
    )
    joined = "\n".join(lines)
    assert "TOPIC: same|different" in joined
    assert "1: 今天好热" in joined
    assert "有人玩原神吗" in joined
    assert "2026-08-09 06:00" in joined


def test_build_topic_judge_prompt_truncates_context() -> None:
    now = datetime(2026, 8, 8, 22, 0, tzinfo=UTC)
    recent = [f"m-{i}-" + "x" * 500 for i in range(10)]
    lines = build_topic_judge_prompt(
        recent_messages=recent,
        current_message="t",
        now=now,
        context_messages=3,
        max_chars_per_message=50,
    )
    joined = "\n".join(lines)
    assert "m-6-" not in joined
    assert "m-7-" in joined
    assert "x" * 60 not in joined


def test_parse_topic_judge_same_and_different() -> None:
    assert parse_topic_judge("TOPIC: same\nREASON: 还在聊天气") == TopicJudgeResult(False, "还在聊天气")
    assert parse_topic_judge("TOPIC: different\nREASON: 跳到游戏了") == TopicJudgeResult(True, "跳到游戏了")


def test_parse_topic_judge_single_line_and_loose_case() -> None:
    assert parse_topic_judge("topic : different / reason: 换话题") == TopicJudgeResult(True, "换话题")
    assert parse_topic_judge("TOPIC: SAME") == TopicJudgeResult(False, "")


def test_parse_topic_judge_malformed_defaults_to_switch() -> None:
    assert parse_topic_judge("") == TopicJudgeResult(True, "malformed")
    assert parse_topic_judge("maybe?") == TopicJudgeResult(True, "malformed")
    assert parse_topic_judge(None) == TopicJudgeResult(True, "malformed")


def test_judge_topic_switch_returns_model_result() -> None:
    client = TopicJudgeLlm("TOPIC: same\nREASON: 接着聊")
    result = judge_topic_switch(client=client, prompt_lines=["p"])
    assert result.switched is False
    assert result.reason == "接着聊"


def test_judge_topic_switch_degrades_to_switch_on_error() -> None:
    result = judge_topic_switch(client=ExplodingTopicJudgeLlm(), prompt_lines=["p"])
    assert result == TopicJudgeResult(True, "judge_error")
