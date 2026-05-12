from app.core.chat_style import (
    build_human_chat_style_lines,
    normalize_chat_reply,
    normalize_proactive_chat_reply,
)


def test_build_human_chat_style_lines_blocks_markdownish_formatting() -> None:
    lines = build_human_chat_style_lines()

    assert any("Do not use Markdown" in line for line in lines)
    assert any("real person" in line for line in lines)


def test_build_human_chat_style_lines_for_proactive_turn_pushes_short_human_interjections() -> None:
    lines = build_human_chat_style_lines(proactive_turn=True)

    assert any("one short sentence" in line for line in lines)
    assert any("few words" in line for line in lines)
    assert any("one QQ line" in line for line in lines)
    assert any("有点贵啊" in line for line in lines)
    assert any("stops making sense" in line for line in lines)
    assert any("empty filler" in line for line in lines)
    assert any("spoken Chinese" in line for line in lines)
    assert any("mini-analysis" in line for line in lines)


def test_normalize_chat_reply_flattens_markdown_list_into_chat_text() -> None:
    raw = "先说结论：\n- 第一，确实有点怪。\n- 第二，你再等等看。"

    assert normalize_chat_reply(raw) == "先说结论：确实有点怪。你再等等看。"


def test_normalize_chat_reply_strips_headings_and_emphasis() -> None:
    raw = "### 其实很简单\n**你现在就改**\n别拖了"

    assert normalize_chat_reply(raw) == "其实很简单 你现在就改 别拖了"


def test_normalize_proactive_chat_reply_keeps_only_the_first_short_sentence() -> None:
    raw = "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真的打两小时的话，钱包先累瘫了。"

    assert normalize_proactive_chat_reply(raw) == "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。"


def test_normalize_proactive_chat_reply_strips_formal_leadin_before_trimming() -> None:
    raw = "总的来说，这价格确实有点离谱。再看看吧。"

    assert normalize_proactive_chat_reply(raw) == "这价格确实有点离谱。"
