from app.core.message_mentions import message_mentions_bot


def test_at_qq_mention_detected() -> None:
    raw = {
        "message": [
            {"type": "at", "data": {"qq": "1807533371"}},
            {"type": "text", "data": {"text": " 如何评价我"}},
        ]
    }
    assert message_mentions_bot(raw, bot_qq=1807533371, bot_name="比企谷小町")


def test_cq_at_text_detected() -> None:
    raw = {"message": "[CQ:at,qq=1807533371] 你好"}
    assert message_mentions_bot(raw, bot_qq=1807533371)


def test_at_name_detected() -> None:
    raw = {"message": "@比企谷小町 你是谁"}
    assert message_mentions_bot(raw, bot_qq=1807533371, bot_name="比企谷小町")


def test_normal_message_not_detected() -> None:
    raw = {"message": [{"type": "text", "data": {"text": "今天天气不错"}}]}
    assert not message_mentions_bot(raw, bot_qq=1807533371, bot_name="比企谷小町")


def test_mention_of_other_member_not_detected() -> None:
    raw = {
        "message": [
            {"type": "at", "data": {"qq": "1357318398"}},
            {"type": "text", "data": {"text": " 你晚上吃啥"}},
        ]
    }
    assert not message_mentions_bot(raw, bot_qq=1807533371, bot_name="比企谷小町")


def test_string_raw_json_detected() -> None:
    import json

    raw = json.dumps({"message": [{"type": "at", "data": {"qq": "1807533371"}}]})
    assert message_mentions_bot(raw, bot_qq=1807533371)
