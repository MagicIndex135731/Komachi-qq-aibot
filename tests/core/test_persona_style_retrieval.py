from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.persona_style_retrieval import (
    DOCUMENT_SCHEMA,
    MAX_PROMPT_BLOCK_CHARS,
    StyleRetrievalTrace,
    build_style_document,
    build_style_retrieval_query,
    format_style_example_block,
    rank_style_examples,
)


NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _message(index: int, text: str, *, seconds: int = 10, user_id: int = 10) -> dict:
    return {
        "platform_msg_id": f"recent-{index}",
        "plain_text": text,
        "user_id": user_id,
        "timestamp": NOW - timedelta(seconds=seconds * index),
    }


def _document(
    msg_id: str,
    *,
    situation: str,
    reply: str,
    days_old: int = 1,
    context_after: str = "",
):
    document = build_style_document(
        {
            "msg_id": msg_id,
            "user_id": 22,
            "group_id": 33,
            "text": reply,
            "reply_target": f"路人: {situation}" if situation else None,
            "context_before": [{"text": "无关邻接噪声"}],
            "context_after": [{"text": context_after}] if context_after else [],
            "timestamp": NOW - timedelta(days=days_old),
        },
        expected_user_id=22,
        expected_group_id=33,
    )
    assert document is not None
    return document


def test_full_question_does_not_absorb_wide_recent_history() -> None:
    recent = [_message(index, f"第{index}条足球游戏闲聊") for index in range(1, 61)]

    query = build_style_retrieval_query(
        current_text="最近在看什么动画",
        recent_messages=recent,
        current_timestamp=NOW,
        current_message_id="current",
    )

    assert query.fragment_count == 1
    assert query.continuation_lines == ()
    assert "足球" not in query.semantic_text
    assert "最近在看什么动画" in query.semantic_text


def test_quoted_followup_uses_quote_but_not_older_topics() -> None:
    query = build_style_retrieval_query(
        current_text="蓝发好女人是谁",
        quoted_text="又有蓝发好女人看了",
        recent_messages=[_message(1, "昨晚足球踢成啥样")],
        current_timestamp=NOW,
    )

    assert query.fragment_count == 2
    assert "蓝发好女人是谁" in query.semantic_text
    assert "又有蓝发好女人看了" in query.semantic_text
    assert "足球" not in query.semantic_text


def test_short_followup_takes_only_two_messages_inside_time_boundary() -> None:
    recent = [
        _message(30, "太旧的话题", seconds=10),
        _message(2, "刚才说的新动画", seconds=10),
        _message(1, "又有蓝发角色", seconds=10),
    ]

    query = build_style_retrieval_query(
        current_text="名字呢",
        recent_messages=recent,
        current_timestamp=NOW,
    )

    assert query.continuation_lines == ("刚才说的新动画", "又有蓝发角色")
    assert "太旧" not in query.semantic_text


def test_document_prefers_reply_target_and_never_contains_context_after() -> None:
    document = _document(
        "quoted",
        situation="要不要上号",
        reply="等我两分钟",
        context_after="后来大家开始聊吃饭",
    )

    assert document.situation_kind == "quoted"
    assert document.situation == "要不要上号"
    assert "无关邻接噪声" not in document.canonical_text
    assert "后来大家开始聊吃饭" not in document.canonical_text
    assert DOCUMENT_SCHEMA == "style-situation-v2"


def test_document_drops_control_reply_target_before_building_canonical_text() -> None:
    document = build_style_document(
        {
            "msg_id": "control-situation",
            "user_id": 22,
            "group_id": 33,
            "text": "正常回复",
            "reply_target": "路人: 切换人格为:测试",
            "context_before": [],
        },
        expected_user_id=22,
        expected_group_id=33,
    )

    assert document is not None
    assert document.situation == ""
    assert "切换人格" not in document.canonical_text


def test_source_group_and_member_are_hard_boundaries() -> None:
    entry = {
        "msg_id": "cross-group",
        "user_id": 22,
        "group_id": 44,
        "text": "这句不能跨群出现",
    }

    assert build_style_document(entry, expected_user_id=22, expected_group_id=33) is None
    assert build_style_document(entry, expected_user_id=99, expected_group_id=44) is None


def test_old_relevant_sample_beats_recent_irrelevant_sample() -> None:
    query = build_style_retrieval_query(current_text="今晚要不要上号打游戏")
    relevant = _document("relevant", situation="今晚打游戏吗", reply="等会就来", days_old=200)
    recent = _document("recent", situation="中午吃什么", reply="随便", days_old=0)
    trace = StyleRetrievalTrace()

    matches = rank_style_examples(
        query=query,
        documents=[recent, relevant],
        vectors_by_id={"relevant": [1.0, 0.0], "recent": [0.57, 0.82]},
        query_vector=[1.0, 0.0],
        now=NOW,
        trace=trace,
    )

    assert [match.document.msg_id for match in matches] == ["relevant"]


def test_low_semantic_candidates_are_rejected_instead_of_filled() -> None:
    query = build_style_retrieval_query(current_text="完全陌生的新话题")
    documents = [
        _document("one", situation="天气不错", reply="出去走走"),
        _document("two", situation="中午吃饭", reply="随便点吧"),
    ]

    matches = rank_style_examples(
        query=query,
        documents=documents,
        vectors_by_id={"one": [0.0, 1.0], "two": [0.1, 0.99]},
        query_vector=[1.0, 0.0],
        now=NOW,
    )

    assert matches == []


def test_embedding_failure_uses_strict_lexical_hit_or_empty() -> None:
    matching_query = build_style_retrieval_query(current_text="今晚打游戏吗")
    unrelated_query = build_style_retrieval_query(current_text="今天工作如何")
    document = _document("game", situation="要不要打游戏", reply="马上来")

    assert rank_style_examples(
        query=matching_query,
        documents=[document],
        vectors_by_id={},
        query_vector=None,
        now=NOW,
    )
    assert rank_style_examples(
        query=unrelated_query,
        documents=[document],
        vectors_by_id={},
        query_vector=None,
        now=NOW,
    ) == []


def test_duplicate_replies_are_removed_and_selection_is_capped_at_three() -> None:
    query = build_style_retrieval_query(current_text="晚上打游戏")
    documents = [
        _document("a", situation="晚上打游戏", reply="马上来"),
        _document("b", situation="晚上打游戏", reply="马上来"),
        _document("c", situation="晚上打游戏", reply="等我两分钟"),
        _document("d", situation="晚上打游戏", reply="先开着吧"),
        _document("e", situation="晚上打游戏", reply="我一会儿到"),
    ]
    vectors = {document.msg_id: [1.0, index / 20] for index, document in enumerate(documents)}

    matches = rank_style_examples(
        query=query,
        documents=documents,
        vectors_by_id=vectors,
        query_vector=[1.0, 0.0],
        now=NOW,
        limit=10,
    )

    assert len(matches) == 3
    replies = [match.document.reply for match in matches]
    assert replies.count("马上来") == 1


def test_formatter_is_bounded_and_omits_later_context() -> None:
    entries = [
        document.entry
        for document in (
            _document(str(index), situation="今晚打游戏", reply=f"第{index}种回复", context_after="后续闲聊")
            for index in range(5)
        )
    ]

    rendered = format_style_example_block(entries, max_pairs=99)

    assert rendered.count("本人原话") == 3
    assert "后续闲聊" not in rendered
    assert len(rendered) <= MAX_PROMPT_BLOCK_CHARS


def test_formatter_honors_smaller_caller_budget() -> None:
    entry = _document("one", situation="今晚打游戏", reply="马上来").entry

    assert format_style_example_block([entry], max_chars=5) == ""
