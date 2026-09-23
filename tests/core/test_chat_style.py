from types import SimpleNamespace

from app.core.chat_style import (
    burst_delays,
    build_human_chat_style_lines,
    build_reply_split_config,
    format_example_pairs,
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


def test_build_human_chat_style_lines_private_context_only_changes_the_opening_line() -> None:
    group_lines = build_human_chat_style_lines()
    private_lines = build_human_chat_style_lines(chat_context="private")

    assert group_lines[0] == "Talk like a real person chatting in a group."
    assert private_lines[0] == "Talk like a real person chatting on QQ."
    # Everything after the opening line is the shared work-style, unchanged.
    assert private_lines[1:] == group_lines[1:]


def test_build_human_chat_style_lines_never_claims_no_web_access() -> None:
    """A plain turn without tools must not tell the user the bot cannot search."""

    expected = (
        "You do have live web search in this chat when a turn needs it: never tell the user "
        "you cannot browse or search the web; when a turn has no fresh results, say the "
        "information may be out of date instead of claiming you have no web access."
    )

    group_lines = build_human_chat_style_lines()
    private_lines = build_human_chat_style_lines(chat_context="private")

    assert expected in group_lines
    assert expected in private_lines
    # The capability line sits in the shared block, so neither the voice nor the
    # proactive path may drop it.
    assert expected in build_human_chat_style_lines(proactive_turn=True, voice="clingy")


def test_build_reply_split_config_uses_group_settings_and_safe_defaults() -> None:
    defaults = build_reply_split_config()
    assert defaults == {
        "enabled": True,
        "separator": "|",
        "max_messages": 3,
        "max_chars": 64,
        "auto_split_long_segments": True,
        "min_delay_seconds": 0.0,
        "max_delay_seconds": 0.0,
    }

    settings = SimpleNamespace(
        group_reply_split_enabled=False,
        group_reply_split_max_messages=2,
        group_reply_split_max_chars=24,
        group_reply_split_min_delay_seconds=1.5,
        group_reply_split_max_delay_seconds=0.5,
    )
    configured = build_reply_split_config(settings)

    assert configured["enabled"] is False
    assert configured["max_messages"] == 2
    assert configured["max_chars"] == 24
    # The maximum delay is never allowed to fall below the minimum.
    assert configured["min_delay_seconds"] == 1.5
    assert configured["max_delay_seconds"] == 1.5


def test_format_example_pairs_includes_context_after() -> None:
    entries = [
        {
            "text": "来了",
            "reply_target": "加菲猫: 上号",
            "context_before": [{"speaker": "加菲猫", "text": "上号"}],
            "context_after": [{"speaker": "逆蝶蝶", "text": "人呢"}],
        }
    ]
    rendered = format_example_pairs(entries)
    assert "上文「加菲猫: 上号」→ 他回「来了」→ 下文「人呢」" in rendered


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


def test_normalize_chat_reply_preserves_ordering_words_at_start_of_answer() -> None:
    reply = "第二名呀，主人～你超过的是原来的第二名嘛，自己就占第二名的位置啦 😼"

    assert normalize_chat_reply(reply) == reply
    assert normalize_chat_reply("第一名就是你。") == "第一名就是你。"
    assert normalize_chat_reply("第2名是你。") == "第2名是你。"
    assert normalize_chat_reply("第一、先看题。") == "第一、先看题。"
    assert normalize_chat_reply("首先：先看题。") == "首先：先看题。"
    assert normalize_chat_reply("其次再看答案。") == "其次再看答案。"


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
    assert split_burst_reply("一|二|三|四", burst) == ["一", "二", "三，四"]
    assert split_burst_reply("一条消息", burst) == ["一条消息"]


def test_split_burst_reply_merges_overflow_without_leaking_the_separator() -> None:
    """More segments than the cap must never put the separator on the wire."""

    burst = {"enabled": True, "separator": "|", "max_messages": 3, "max_chars": 64}
    text = (
        "诶…主人问今晚的呀，小町刚搜了下～这是 LPL 季后赛，iG 打 AL"
        "|Rookie 那种气势，小町悄悄押 iG 赢"
        "|谁赢都不意外啦。"
        "|主人更看好哪边呀？🍙"
    )

    parts = split_burst_reply(text, burst)

    assert parts == [
        "诶…主人问今晚的呀，小町刚搜了下～这是 LPL 季后赛，iG 打 AL",
        "Rookie 那种气势，小町悄悄押 iG 赢",
        "谁赢都不意外啦。主人更看好哪边呀？🍙",
    ]
    assert all("|" not in part for part in parts)


def test_split_burst_reply_strips_a_dangling_separator() -> None:
    """A separator with nothing on one side is an artifact, not message text."""

    burst = {"enabled": True, "separator": "|", "max_messages": 3}

    assert split_burst_reply("来了|", burst) == ["来了"]
    assert split_burst_reply("|来了", burst) == ["来了"]
    assert split_burst_reply("|", burst) == []
    assert split_burst_reply("来了", burst) == ["来了"]


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


def test_split_burst_reply_can_leave_the_reply_exactly_as_written() -> None:
    """The persona can opt out of the forced long-segment split."""

    burst = {
        "enabled": True,
        "separator": "|",
        "max_messages": 3,
        "max_chars": 8,
        "auto_split_long_segments": False,
    }
    text = "这是一句明显超过上限但必须原样发出的完整回复，不许被切开。"

    assert split_burst_reply(text, burst) == [text]
    assert split_burst_reply("来了|人呢", burst) == ["来了", "人呢"]


def test_split_burst_reply_packs_a_long_run_on_reply_at_clause_boundaries() -> None:
    burst = {"enabled": True, "separator": "|", "max_messages": 3, "max_chars": 24}
    text = (
        "哼，看在你诚心诚意求教的份上，小町就大发慈悲告诉你一次，"
        "不过下次再问这种蠢问题，小町分数可要扣到底了，笨蛋。"
    )

    parts = split_burst_reply(text, burst)

    assert 2 <= len(parts) <= 3
    assert "".join(parts) == text
    assert all(len(part) <= 40 for part in parts)


def test_split_burst_reply_keeps_an_unpunctuated_clause_whole() -> None:
    """No punctuation means no natural cut point, so the line stays intact."""

    burst = {"enabled": True, "separator": "|", "max_messages": 3, "max_chars": 24}
    text = "小町就是要把这句话一口气说完中间一个标点都不带的完整长句所以不会被切开"

    assert split_burst_reply(text, burst) == [text]


def test_burst_delays_share_the_group_window_across_surfaces() -> None:
    """Group and private delivery must space bursts identically."""

    assert burst_delays({"min_delay_seconds": 0.0, "max_delay_seconds": 0.0}, segment_count=3) == (
        0.8,
        2.5,
    )
    assert burst_delays({"min_delay_seconds": 0.4, "max_delay_seconds": 0.9}, segment_count=2) == (
        0.4,
        0.9,
    )
    # A single segment never waits, and a low max is clamped up to the min.
    assert burst_delays({"min_delay_seconds": 0.4, "max_delay_seconds": 0.9}, segment_count=1) == (
        0.0,
        0.0,
    )
    assert burst_delays({"min_delay_seconds": 1.5, "max_delay_seconds": 0.2}, segment_count=2) == (
        1.5,
        1.5,
    )


def test_clingy_voice_swaps_the_mesugaki_edge_for_little_sister_warmth() -> None:
    lines = build_human_chat_style_lines(voice="clingy")
    joined = "\n".join(lines)

    assert "clingy little-sister warmth" in joined
    assert "mesugaki" not in joined
    assert "sharp roast" not in joined


def test_clingy_proactive_lines_never_land_a_jab() -> None:
    joined = "\n".join(build_human_chat_style_lines(proactive_turn=True, voice="clingy"))

    assert "quietly leaning in" in joined
    assert "sharp roast" not in joined
    assert "land the jab" not in joined
    assert "teasing put-downs like" not in joined


def test_clingy_voice_is_ignored_without_the_komachi_style() -> None:
    """Impersonation turns pass ``komachi_style=False`` and keep their own text."""

    assert build_human_chat_style_lines(
        voice="clingy",
        komachi_style=False,
    ) == build_human_chat_style_lines(komachi_style=False)
