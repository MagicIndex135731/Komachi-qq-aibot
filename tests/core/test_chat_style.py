from app.core.chat_style import (
    build_human_chat_style_lines,
    normalize_brief_group_interjection_reply,
    normalize_chat_reply,
    normalize_chat_reply_burst_aware,
    normalize_proactive_chat_reply,
    retrieve_relevant_examples,
    retrieve_relevant_facts,
    scrub_banned_address_terms,
    split_burst_reply,
)


def test_build_human_chat_style_lines_blocks_markdownish_formatting() -> None:
    lines = build_human_chat_style_lines()

    assert any("Do not use Markdown" in line for line in lines)
    assert any("real person" in line for line in lines)
    assert any("not a dislike" in line for line in lines)


def test_build_human_chat_style_lines_for_proactive_turn_pushes_short_human_interjections() -> None:
    lines = build_human_chat_style_lines(proactive_turn=True)

    assert any("8-16 Chinese characters" in line for line in lines)
    assert any("Do not rely on later truncation" in line for line in lines)
    assert not any("do not artificially cut off a useful point" in line for line in lines)
    assert any("one compact QQ message" in line for line in lines)
    assert any("empty filler" in line for line in lines)
    assert any("spoken Chinese" in line for line in lines)
    assert any("mini-analysis" in line for line in lines)
    assert any("one complete short sentence" in line for line in lines)
    assert any("mild opinion" in line for line in lines)
    assert any("not just agree" in line for line in lines)
    assert any("small fresh angle" in line for line in lines)
    assert any("mesugaki" in line for line in lines)
    assert any("teasing" in line for line in lines)


def test_normalize_chat_reply_flattens_markdown_list_into_chat_text() -> None:
    raw = "先说结论：\n- 确实有点怪\n- 你再等等看"

    assert normalize_chat_reply(raw) == "先说结论：确实有点怪。你再等等看。"


def test_normalize_chat_reply_strips_headings_and_emphasis() -> None:
    raw = "### 其实很简单\n**你现在就改**\n别拖了"

    assert normalize_chat_reply(raw) == "其实很简单 你现在就改 别拖了"


def test_normalize_proactive_chat_reply_keeps_full_content_for_normal_proactive_reply() -> None:
    raw = "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。"

    assert normalize_proactive_chat_reply(raw) == "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。"


def test_normalize_proactive_chat_reply_strips_formal_leadin_without_truncating_followup() -> None:
    raw = "总的来说，这价格确实有点离谱。再看看吧。"

    assert normalize_proactive_chat_reply(raw) == "这价格确实有点离谱。再看看吧。"


def test_normalize_chat_reply_keeps_full_addressed_reply_content() -> None:
    raw = "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。"

    assert normalize_chat_reply(raw) == "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。"
def test_normalize_chat_reply_strips_leading_control_json_prefix() -> None:
    raw = '{"queries":["解析图片的笑点"],"sourcefilter":["filelibrary"]}啊，这张图主要靠夸张表情在搞笑。'

    assert normalize_chat_reply(raw) == "啊，这张图主要靠夸张表情在搞笑。"


def test_normalize_brief_group_interjection_reply_does_not_hard_truncate_long_clause() -> None:
    raw = "### 插一句\n今天这个价格已经从离谱升级成离谱plus了吧，钱包看了都想先下线喘口气。"

    assert normalize_brief_group_interjection_reply(raw) == (
        "插一句 今天这个价格已经从离谱升级成离谱plus了吧，钱包看了都想先下线喘口气。"
    )



def test_normalize_brief_group_interjection_keeps_first_complete_sentence_only() -> None:
    raw = "First jab. Second unnecessary sentence."
    assert normalize_brief_group_interjection_reply(raw) == "First jab."


def test_build_human_chat_style_lines_can_drop_komachi_voice() -> None:
    lines = build_human_chat_style_lines(komachi_style=False)
    joined = "\n".join(lines)

    assert "mesugaki" not in joined
    assert "Komachi" not in joined
    assert any("real person" in line for line in lines)

    proactive = build_human_chat_style_lines(
        proactive_turn=True, komachi_style=False
    )
    proactive_joined = "\n".join(proactive)
    assert "mesugaki" not in proactive_joined
    assert "就这？" not in proactive_joined

def test_normalize_chat_reply_strips_model_think_blocks() -> None:
    raw = (
        "<think>Considering concise responses I should keep this short.</think> "
        "当然开车去啊，不然你走过去是让老板洗你吗。"
    )

    assert normalize_chat_reply(raw) == "当然开车去啊，不然你走过去是让老板洗你吗。"


def test_split_burst_reply_disabled_without_burst_config() -> None:
    assert split_burst_reply("来了|人呢", None) == ["来了|人呢"]
    assert split_burst_reply("来了|人呢", {"enabled": False}) == ["来了|人呢"]


def test_split_burst_reply_splits_and_caps_segments() -> None:
    burst = {"enabled": True, "separator": "|", "max_messages": 3}
    assert split_burst_reply("来了|人呢", burst) == ["来了", "人呢"]
    assert split_burst_reply("一|二|三|四", burst) == ["一", "二", "三|四"]
    assert split_burst_reply("一条消息", burst) == ["一条消息"]


def test_scrub_banned_address_terms_replaces_honorifics() -> None:
    assert scrub_banned_address_terms(
        "主人，阿渣啊。大人您稍等", ("主人", "大人", "您")
    ) == "你，阿渣啊。你稍等"


def test_retrieve_relevant_examples_ranks_by_topic_overlap() -> None:
    bank = ["上号", "明天看球吗", "写日报好烦", "吃啥"]
    context = ["明天有比赛吗", "看球不"]

    picked = retrieve_relevant_examples(bank, context, limit=2)

    texts = [entry["text"] for entry in picked]
    assert "明天看球吗" in texts
    assert "写日报好烦" not in texts
    assert "吃啥" not in texts


def test_burst_aware_normalize_keeps_newlines_as_separators() -> None:
    burst = {"enabled": True, "separator": "|", "max_messages": 3}

    normalized = normalize_chat_reply_burst_aware("季挺nb\n前半像日常番", burst)

    assert normalized == "季挺nb|前半像日常番"


def test_retrieve_relevant_facts_ranks_by_topic_overlap() -> None:
    facts = [
        {"category": "游戏", "fact": "主玩英雄联盟手游"},
        {"category": "工作", "fact": "在快手实习"},
    ]

    picked = retrieve_relevant_facts(
        facts, ["你最擅长什么lol英雄"], limit=1
    )

    assert picked == [{"category": "游戏", "fact": "主玩英雄联盟手游"}]


def test_split_burst_reply_auto_splits_long_sentences() -> None:
    burst = {
        "enabled": True,
        "separator": "|",
        "max_messages": 3,
        "max_chars": 8,
    }
    text = "季挺挺nb呀哈。前半像日常番哦。后半直接精神污染了。"

    parts = split_burst_reply(text, burst)

    assert parts == [
        "季挺挺nb呀哈。",
        "前半像日常番哦。",
        "后半直接精神污染了。",
    ]
