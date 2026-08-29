from app.core.message_mentions import (
    bot_mention_names,
    collect_bot_display_names,
    message_mentions_bot,
)


def test_at_qq_mention_detected() -> None:
    raw = {
        "message": [
            {"type": "at", "data": {"qq": "900001"}},
            {"type": "text", "data": {"text": " 如何评价我"}},
        ]
    }
    assert message_mentions_bot(raw, bot_qq=900001, bot_names={"比企谷小町"})


def test_cq_at_text_detected() -> None:
    raw = {"message": "[CQ:at,qq=900001] 你好"}
    assert message_mentions_bot(raw, bot_qq=900001)


def test_at_name_detected() -> None:
    raw = {"message": "@比企谷小町 你是谁"}
    assert message_mentions_bot(raw, bot_qq=900001, bot_names={"比企谷小町"})


def test_historical_persona_names_detected() -> None:
    for name in ("小町", "阿渣", "逆蝶蝶"):
        raw = {"message": f"@{name} 在吗"}
        assert message_mentions_bot(raw, bot_qq=900001, bot_names={"比企谷小町", "阿渣", "逆蝶蝶", "小町"})


def test_normal_message_not_detected() -> None:
    raw = {"message": [{"type": "text", "data": {"text": "今天天气不错"}}]}
    assert not message_mentions_bot(raw, bot_qq=900001, bot_names={"比企谷小町"})


def test_mention_of_other_member_not_detected() -> None:
    raw = {
        "message": [
            {"type": "at", "data": {"qq": "900002"}},
            {"type": "text", "data": {"text": " 你晚上吃啥"}},
        ]
    }
    assert not message_mentions_bot(raw, bot_qq=900001, bot_names={"比企谷小町"})


def test_string_raw_json_detected() -> None:
    import json

    raw = json.dumps({"message": [{"type": "at", "data": {"qq": "900001"}}]})
    assert message_mentions_bot(raw, bot_qq=900001)


def test_bot_mention_names_includes_short_and_historical() -> None:
    names = bot_mention_names(
        bot_qq=900001,
        default_name="比企谷小町",
        display_names={"比企谷小町", "阿渣", "逆蝶蝶"},
    )
    assert names == {"900001", "比企谷小町", "阿渣", "逆蝶蝶", "小町"}


def test_collect_bot_display_names_parses_senders() -> None:
    values = [
        {"sender": {"card": "阿渣", "nickname": "比企谷小町"}},
        {"sender": {"card": "逆蝶蝶", "nickname": ""}},
        "not json",
        None,
        '{"sender": {"card": "比企谷小町", "nickname": "x"}}',
    ]
    assert collect_bot_display_names(values) == {"阿渣", "逆蝶蝶", "比企谷小町", "x"}
