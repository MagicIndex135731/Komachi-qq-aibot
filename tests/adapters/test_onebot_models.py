from app.adapters.onebot_models import parse_group_message_event, parse_private_message_event


def test_parse_group_message_event_extracts_plain_text_and_mentions() -> None:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 42,
        "group_id": 10001,
        "user_id": 20001,
        "raw_message": "@小柚 在吗",
        "message": [{"type": "text", "data": {"text": "@小柚 在吗"}}],
        "sender": {"nickname": "Alice", "card": "阿狸"},
        "time": 1715150000,
    }

    event = parse_group_message_event(payload, bot_qq=123456789, bot_name="小柚")

    assert event.group_id == 10001
    assert event.user_id == 20001
    assert event.plain_text == "@小柚 在吗"
    assert event.mentioned_bot is True


def test_parse_group_message_event_does_not_treat_numeric_substring_as_mention() -> None:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 43,
        "group_id": 10001,
        "user_id": 20002,
        "raw_message": "123456789 在吗",
        "message": [{"type": "text", "data": {"text": "123456789 在吗"}}],
        "sender": {"nickname": "Bob", "card": ""},
        "time": 1715150001,
    }

    event = parse_group_message_event(payload, bot_qq=123456789, bot_name="小柚")

    assert event.mentioned_bot is False


def test_parse_group_message_event_treats_leading_whitespace_prefix_mention_as_mention() -> None:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 44,
        "group_id": 10001,
        "user_id": 20003,
        "raw_message": "  @小柚 在吗",
        "message": "  @小柚 在吗",
        "sender": {"nickname": "Carol", "card": ""},
        "time": 1715150002,
    }

    event = parse_group_message_event(payload, bot_qq=123456789, bot_name="小柚")

    assert event.mentioned_bot is True


def test_parse_group_message_event_extracts_images_and_mixed_message_type() -> None:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 45,
        "group_id": 10001,
        "user_id": 20004,
        "raw_message": "[CQ:at,qq=123456789] look at this [CQ:image,file=cat.png]",
        "message": [
            {"type": "at", "data": {"qq": "123456789"}},
            {"type": "text", "data": {"text": "look at this"}},
            {
                "type": "image",
                "data": {
                    "file": "cat.png",
                    "url": "https://img.example.test/cat.png",
                },
            },
        ],
        "sender": {"nickname": "Dana", "card": ""},
        "time": 1715150003,
    }

    event = parse_group_message_event(payload, bot_qq=123456789, bot_name="Mira")

    assert event.plain_text == "look at this"
    assert event.msg_type == "mixed"
    assert len(event.images) == 1
    assert event.images[0].file_id == "cat.png"
    assert event.images[0].url == "https://img.example.test/cat.png"


def test_parse_group_message_event_extracts_reply_to_msg_id() -> None:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 46,
        "group_id": 10001,
        "user_id": 20005,
        "raw_message": "[CQ:reply,id=765262399][CQ:at,qq=123456789] 这张",
        "message": [
            {"type": "reply", "data": {"id": "765262399"}},
            {"type": "at", "data": {"qq": "123456789"}},
            {"type": "text", "data": {"text": " 这张"}},
        ],
        "sender": {"nickname": "Eve", "card": ""},
        "time": 1715150004,
    }

    event = parse_group_message_event(payload, bot_qq=123456789, bot_name="Mira")

    assert event.reply_to_msg_id == "765262399"


def test_parse_private_message_event_extracts_plain_text() -> None:
    payload = {
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "message_id": 47,
        "user_id": 10001,
        "raw_message": "check logs",
        "message": [{"type": "text", "data": {"text": "check logs"}}],
        "sender": {"nickname": "owner"},
        "time": 1715150005,
    }

    event = parse_private_message_event(payload)

    assert event.platform_msg_id == "47"
    assert event.user_id == 10001
    assert event.nickname == "owner"
    assert event.plain_text == "check logs"


def test_parse_private_message_event_extracts_images_and_reply_to_msg_id() -> None:
    payload = {
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "message_id": 48,
        "user_id": 10001,
        "raw_message": "[CQ:reply,id=765262399][CQ:image,file=cat.png]",
        "message": [
            {"type": "reply", "data": {"id": "765262399"}},
            {
                "type": "image",
                "data": {
                    "file": "cat.png",
                    "url": "https://img.example.test/cat.png",
                },
            },
        ],
        "sender": {"nickname": "owner"},
        "time": 1715150006,
    }

    event = parse_private_message_event(payload)

    assert event.msg_type == "image"
    assert event.reply_to_msg_id == "765262399"
    assert len(event.images) == 1
    assert event.images[0].file_id == "cat.png"
    assert event.images[0].url == "https://img.example.test/cat.png"
