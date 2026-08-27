from __future__ import annotations

from app.core.style_distill import (
    assemble_persona,
    build_style_samples,
    parse_persona_yaml,
    speaker_label,
)


def _records() -> list[dict]:
    return [
        {
            "timestamp": "2026-01-01T00:00:01+00:00",
            "user_id": 111,
            "nickname": "路人甲",
            "group_card": "",
            "plain_text": "在吗",
        },
        {
            "timestamp": "2026-01-01T00:00:02+00:00",
            "user_id": 222,
            "nickname": "测试君",
            "group_card": "测试君",
            "plain_text": "我玩",
            "reply_to_msg_id": "1",
        },
        {
            "timestamp": "2026-01-01T00:00:03+00:00",
            "user_id": 111,
            "nickname": "路人甲",
            "group_card": "",
            "plain_text": "好的",
        },
        {
            "timestamp": "2026-01-01T00:00:04+00:00",
            "user_id": 333,
            "nickname": "路人乙",
            "group_card": "",
            "plain_text": "无关",
        },
    ]


def test_build_style_samples_filters_target_and_keeps_context() -> None:
    samples = build_style_samples(
        records=_records(),
        user_id=222,
        context_before=2,
        context_after=2,
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample["text"] == "我玩"
    assert [item["text"] for item in sample["context_before"]] == ["在吗"]
    assert [item["text"] for item in sample["context_after"]] == ["好的", "无关"]


def test_speaker_label_prefers_group_card() -> None:
    assert speaker_label({"nickname": "昵称", "group_card": "名片", "user_id": 1}) == "名片"
    assert speaker_label({"nickname": "昵称", "group_card": "", "user_id": 1}) == "昵称"


def test_parse_persona_yaml_extracts_fenced_block() -> None:
    text = '输出如下：\n```yaml\nname: 测试君\ncore_traits:\n  - casual\n```'
    profile = parse_persona_yaml(text)
    assert profile == {"name": "测试君", "core_traits": ["casual"]}


def test_assemble_persona_normalizes_runtime_fields() -> None:
    persona = assemble_persona(
        {
            "name": "测试君",
            "identity": "群友",
            "core_traits": ["casual", ""],
            "speech_habits": ["短句"],
            "style_avoid": [],
            "example_lines": ["我玩"],
        },
        target_name="测试君",
        group_card="测试君",
        source_user_id=222,
        aliases=["测试君"],
    )
    assert persona["name"] == "测试君"
    assert persona["core_traits"] == ["casual"]
    assert persona["source_user_id"] == 222
    assert persona["group_card"] == "测试君"
    assert persona["aliases"] == ["测试君"]
