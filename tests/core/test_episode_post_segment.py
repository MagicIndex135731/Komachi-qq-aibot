from types import SimpleNamespace

from app.core.episode_post_segment import (
    build_post_segment_prompt,
    parse_post_segment_boundaries,
    post_segment_episode,
    split_messages,
)


def _message(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        platform_msg_id=f"m-{index}",
        user_id=42,
        plain_text=f"message {index}",
    )


class PostSegmentLlm:
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


def test_build_post_segment_prompt_includes_numbered_messages() -> None:
    lines = build_post_segment_prompt([_message(1), _message(2)])
    joined = "\n".join(lines)
    assert "1. message 1" in joined
    assert "2. message 2" in joined
    assert "segments" in joined


def test_parse_post_segment_boundaries_valid_json() -> None:
    raw = '{"segments": [{"start": 1, "end": 3, "topic": "a"}, {"start": 4, "end": 6, "topic": "b"}]}'
    assert parse_post_segment_boundaries(raw) == [(1, 3), (4, 6)]


def test_parse_post_segment_boundaries_markdown_wrapped() -> None:
    raw = '```json\n{"segments": [{"start": 1, "end": 2}]}\n```'
    assert parse_post_segment_boundaries(raw) == [(1, 2)]


def test_parse_post_segment_boundaries_malformed_returns_empty() -> None:
    assert parse_post_segment_boundaries("") == []
    assert parse_post_segment_boundaries("nope") == []
    assert parse_post_segment_boundaries(None) == []


def test_split_messages_by_boundaries() -> None:
    messages = [_message(i) for i in range(1, 7)]
    pieces = split_messages(messages, [(1, 3), (4, 6)])
    assert [len(piece) for piece in pieces] == [3, 3]
    assert [piece[0].platform_msg_id for piece in pieces] == ["m-1", "m-4"]


def test_split_messages_with_gaps_covers_everything() -> None:
    messages = [_message(i) for i in range(1, 7)]
    pieces = split_messages(messages, [(2, 4)])
    assert [len(piece) for piece in pieces] == [1, 3, 2]


def test_post_segment_episode_splits_when_model_agrees() -> None:
    messages = [_message(i) for i in range(1, 30)]
    client = PostSegmentLlm(
        '{"segments": [{"start": 1, "end": 15}, {"start": 16, "end": 29}]}'
    )
    pieces = post_segment_episode(client=client, messages=messages, min_messages=25)
    assert len(pieces) == 2
    assert [len(piece) for piece in pieces] == [15, 14]
    assert len(client.calls) == 1


def test_post_segment_episode_below_min_messages_keeps_single_piece() -> None:
    messages = [_message(i) for i in range(1, 10)]
    client = PostSegmentLlm('{"segments": [{"start": 1, "end": 5}, {"start": 6, "end": 9}]}')
    pieces = post_segment_episode(client=client, messages=messages, min_messages=25)
    assert len(pieces) == 1
    assert len(client.calls) == 0


def test_post_segment_episode_degrades_on_error() -> None:
    messages = [_message(i) for i in range(1, 30)]
    pieces = post_segment_episode(client=ExplodingLlm(), messages=messages, min_messages=25)
    assert len(pieces) == 1
    assert len(pieces[0]) == 29
