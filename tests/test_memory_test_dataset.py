import json
import random
import sqlite3
from datetime import UTC, datetime

import pytest

import scripts.build_memory_test_dataset as memory_test_dataset
from scripts.build_memory_test_dataset import (
    _build_ambiguous_case,
    _build_fact_case,
    _build_first_person_case,
    _build_identity_audit_cases,
    _is_supported_relationship_item,
    _load_group_aliases,
    _load_messages,
    _load_retrievable_raw_message_ids,
    _topic_keywords,
    _validate_answer_contract,
    build_cases,
)
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


def _make_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, group_id INTEGER, platform_msg_id TEXT,
            user_id INTEGER, timestamp TEXT, plain_text TEXT,
            reply_to_msg_id TEXT, mentioned_bot INTEGER
        );
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY, nickname TEXT, group_card TEXT
        );
        CREATE TABLE memory_items (
            id INTEGER PRIMARY KEY, scope_type TEXT, scope_id TEXT,
            subject_id TEXT, memory_kind TEXT, predicate TEXT, object_text TEXT,
            content TEXT, source_msg_id TEXT, source_msg_ids TEXT,
            status TEXT
        );
        CREATE TABLE summaries (
            id INTEGER PRIMARY KEY, scope_type TEXT, scope_id TEXT,
            summary_level TEXT, start_at TEXT, end_at TEXT, content TEXT,
            source_start_msg_id TEXT, source_end_msg_id TEXT, status TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO messages (id, group_id, platform_msg_id, user_id, timestamp, "
        "plain_text, reply_to_msg_id, mentioned_bot) VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, 1001, "p1", 11, "2026-08-01T10:00:00+08:00", "阿渣喜欢看动画", None, 0),
            (2, 1001, "p2", 12, "2026-08-01T10:01:00+08:00", "小町 帮我查一下昨天说的计划", None, 1),
            (3, 1001, "p3", 11, "2026-08-01T10:02:00+08:00", "我讨厌吃香菜", None, 0),
            (4, 1002, "p4", 21, "2026-08-01T10:03:00+08:00", "晚上吃什么", None, 0),
            (5, 1001, "p5", 11, "2026-01-01T10:00:00+08:00", "阿渣打算开发一键上号功能", None, 0),
            (6, 1001, "p6", 11, "2026-07-20T10:00:00+08:00", "阿渣打算周六去海雅唱歌", None, 0),
            (7, 1001, "p7", 11, "2026-08-05T10:00:00+08:00", "阿渣打算下月搬家", None, 0),
        ],
    )
    connection.executemany(
        "INSERT INTO users (user_id, nickname, group_card) VALUES (?,?,?)",
        [(11, "阿渣", "阿渣"), (12, "逆蝶蝶", "逆蝶蝶"), (21, "灰泽满", "灰泽满")],
    )
    connection.executemany(
        "INSERT INTO memory_items (id, scope_type, scope_id, subject_id, memory_kind, "
        "predicate, object_text, content, source_msg_id, source_msg_ids, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "group", "1001", "11", "preference", "likes", "动画", "阿渣喜欢看动画", "p1", '["p1"]', "active"),
            (2, "group", "1001", "11", "taboo", "dislikes", "香菜", "我讨厌吃香菜", "p3", '["p3"]', "active"),
            (3, "group", "1001", "12", "profile", "is", "很会接梗", "逆蝶蝶很会接梗", "p2", '["p2"]', "active"),
            (4, "group", "1001", "11", "plan", "plans", "一键上号", "阿渣打算开发一键上号功能", "p5", '["p5"]', "active"),
            (5, "group", "1001", "11", "plan", "plans", "唱歌", "阿渣打算周六去海雅唱歌", "p6", '["p6"]', "active"),
            (7, "group", "1001", "11", "plan", "plans", "搬家", "阿渣打算下月搬家", "p7", '["p7"]', "active"),
            (6, "group", "1001", "group", "event", "happened", "球赛", "群里聊了球赛", "p6", '["p6"]', "active"),
        ],
    )
    connection.executemany(
        "INSERT INTO summaries (id, scope_type, scope_id, summary_level, start_at, "
        "end_at, content, source_start_msg_id, source_end_msg_id, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "group", "1001", "episode", "2026-08-01T10:00:00+08:00",
             "2026-08-01T10:05:00+08:00", "大家聊了动画和计划", "p1", "p3", "active"),
        ],
    )
    connection.commit()
    connection.close()


def test_build_cases_schema_and_determinism(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    first = build_cases(engine, count=120, seed=7)
    second = build_cases(engine, count=120, seed=7)
    assert 0 < len(first) <= 120
    assert [case["query"] for case in first] == [case["query"] for case in second]
    assert len({case["case_id"] for case in first}) == len(first)
    fact_cases = [case for case in first if case["expected_layer"] == "fact"]
    assert fact_cases
    assert all(case["expected_evidence_message_ids"] for case in fact_cases)
    assert any(case["category"] == "abstention" for case in first)
    assert any(case["category"] == "summary" for case in first)
    assert all("case_id" in case and "gold_text" in case for case in first)
    assert all(
        case["answer_expectation"]
        in {"must_answer", "must_abstain", "either"}
        for case in first
    )


def test_build_cases_never_treats_bot_reply_sender_as_member_subject(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO messages (id, group_id, platform_msg_id, user_id, timestamp, "
        "plain_text, reply_to_msg_id, mentioned_bot) VALUES (?,?,?,?,?,?,?,?)",
        (
            99,
            1001,
            "bot-reply-question-1",
            999999999,
            "2026-08-05T10:01:00+08:00",
            "你是群里的阿渣。",
            "question-1",
            0,
        ),
    )
    connection.execute(
        "INSERT INTO memory_items (id, scope_type, scope_id, subject_id, memory_kind, "
        "predicate, object_text, content, source_msg_id, source_msg_ids, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            99,
            "group",
            "1001",
            "999999999",
            "profile",
            "is",
            "群成员",
            "小町是群成员",
            "bot-reply-question-1",
            '["bot-reply-question-1"]',
            "active",
        ),
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    cases = build_cases(engine, count=120, seed=7)

    assert all(
        "999999999" not in tuple(case.get("allowed_subject_user_ids") or ())
        for case in cases
    )
    assert all("QQ号999999999" not in case["query"] for case in cases)


def test_build_cases_keeps_member_with_deleted_delivery_state(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE messages ADD COLUMN raw_json TEXT")
    connection.execute(
        "INSERT INTO messages (id, group_id, platform_msg_id, user_id, timestamp, "
        "plain_text, reply_to_msg_id, mentioned_bot, raw_json) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            98,
            1001,
            "member-deleted-1",
            77,
            "2026-08-05T10:00:00+08:00",
            "我喜欢做模型",
            None,
            0,
            '{"delivery_state":"deleted","sender":{"nickname":"普通成员"}}',
        ),
    )
    connection.execute(
        "INSERT INTO users (user_id, nickname, group_card) VALUES (?,?,?)",
        (77, "普通成员", "普通成员"),
    )
    connection.execute(
        "INSERT INTO memory_items (id, scope_type, scope_id, subject_id, memory_kind, "
        "predicate, object_text, content, source_msg_id, source_msg_ids, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            98,
            "group",
            "1001",
            "77",
            "preference",
            "likes",
            "做模型",
            "普通成员喜欢做模型",
            "member-deleted-1",
            '["member-deleted-1"]',
            "active",
        ),
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    cases = build_cases(engine, count=120, seed=7)

    assert any(
        "77" in tuple(case.get("allowed_subject_user_ids") or ())
        for case in cases
    )


def test_dataset_datetime_parser_treats_naive_sqlite_values_as_shanghai(
    monkeypatch,
):
    class UtcHostDatetime(datetime):
        """Simulate parsing a naive SQLite clock on a UTC evaluation host."""

        @classmethod
        def fromisoformat(cls, value):
            parsed = datetime.fromisoformat(value)
            return cls(
                parsed.year,
                parsed.month,
                parsed.day,
                parsed.hour,
                parsed.minute,
                parsed.second,
                parsed.microsecond,
                tzinfo=parsed.tzinfo,
            )

        def astimezone(self, tz=None):
            if self.tzinfo is None:
                return self.replace(tzinfo=UTC).astimezone(tz)
            return super().astimezone(tz)

    monkeypatch.setattr(memory_test_dataset, "datetime", UtcHostDatetime)

    assert memory_test_dataset._parse_dt("2026-08-01 10:05:00") == (
        "2026-08-01T02:05:00+00:00"
    )


def test_answer_expectation_contracts_are_explicit(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    cases = build_cases(engine, count=120, seed=11)

    assert all(
        case["answer_expectation"] == "either"
        for case in cases
        if case["category"] == "mention"
    )
    assert all(
        case["answer_expectation"] == "must_abstain"
        for case in cases
        if case["category"]
        in {"abstention", "ambiguous", "cross_group", "distractor"}
    )
    assert all(
        case["answer_expectation"] == "must_answer"
        for case in cases
        if case["gold_text"]
    )


def test_relationship_fact_case_never_conflicts_with_multi_subject_fail_closed():
    item = {
        "group_id": 1001,
        "subject_id": "11",
        "kind": "relationship",
        "object_text": "关系对象",
        "content": "主体与关系对象存在明确关系",
        "source_ids": ("source-1",),
    }
    aliases = {11: ("成员甲",), 12: ("成员乙",)}
    cases = [
        _build_fact_case(item, aliases, random.Random(1), index)
        for index in range(4)
    ]

    assert all("multi_subject" not in case["tags"] for case in cases)
    assert all(case["allowed_subject_user_ids"] == ("11",) for case in cases)
    assert all(case["expected_evidence_message_ids"] for case in cases)
    assert all(case["answer_expectation"] == "must_answer" for case in cases)
    assert all(case["query"] == "成员甲和谁是什么关系" for case in cases)

    ambiguous = _build_ambiguous_case(1001, ("成员甲", "成员乙"), 1)
    assert ambiguous["allowed_subject_user_ids"] == ()
    assert ambiguous["expected_evidence_message_ids"] == ()
    assert ambiguous["answer_expectation"] == "must_abstain"

    contradictory = dict(cases[0])
    contradictory["tags"] = (*contradictory["tags"], "multi_subject")
    with pytest.raises(ValueError, match="multi-subject"):
        _validate_answer_contract(contradictory)


def test_relationship_dataset_requires_member_bound_explicit_relation():
    base = {
        "group_id": 1001,
        "subject_id": "11",
        "kind": "relationship",
        "predicate": "works_with",
        "object_text": "组内一名同事",
        "content": "成员甲的组内一名同事正在做仿真器",
        "source_ids": ("source-1",),
    }

    assert _is_supported_relationship_item(base) is True
    assert _is_supported_relationship_item(
        {**base, "subject_id": "group", "content": "群里讨论过其他成员"}
    ) is False
    assert _is_supported_relationship_item(
        {
            **base,
            "predicate": "mentioned",
            "object_text": "一起游泳",
            "content": "成员甲说等人回来一起游泳",
        }
    ) is False
    assert _is_supported_relationship_item(
        {**base, "kind": "profile", "content": "没有关系词也不影响其他 kind"}
    ) is True


def test_identity_audit_cases_use_exact_requester_and_exact_bot_reply():
    messages = [
        {
            "id": 1,
            "group_id": 1001,
            "platform_msg_id": "source-before",
            "user_id": 11,
            "timestamp": "2026-08-24T10:00:00+00:00",
            "plain_text": "我住在深圳",
            "reply_to_msg_id": None,
            "mentioned_bot": False,
            "raw_json": None,
        },
        {
            "id": 2,
            "group_id": 1001,
            "platform_msg_id": "question-1",
            "user_id": 11,
            "timestamp": "2026-08-24T11:00:00+00:00",
            "plain_text": "我是谁",
            "reply_to_msg_id": None,
            "mentioned_bot": True,
            "raw_json": None,
        },
        {
            "id": 3,
            "group_id": 1001,
            "platform_msg_id": "bot-reply-unrelated",
            "user_id": 999,
            "timestamp": "2026-08-24T11:00:01+00:00",
            "plain_text": "另一个问题的回答",
            "reply_to_msg_id": None,
            "mentioned_bot": False,
            "raw_json": None,
        },
        {
            "id": 4,
            "group_id": 1001,
            "platform_msg_id": "bot-reply-question-1",
            "user_id": 999,
            "timestamp": "2026-08-24T11:00:02+00:00",
            "plain_text": "你是成员甲",
            "reply_to_msg_id": None,
            "mentioned_bot": False,
            "raw_json": None,
        },
        {
            "id": 5,
            "group_id": 1001,
            "platform_msg_id": "question-2",
            "user_id": 12,
            "timestamp": "2026-08-24T11:05:00+00:00",
            "plain_text": "我是谁？",
            "reply_to_msg_id": None,
            "mentioned_bot": True,
            "raw_json": None,
        },
        {
            "id": 6,
            "group_id": 1001,
            "platform_msg_id": "bot-question",
            "user_id": 12,
            "timestamp": "2026-08-24T11:06:00+00:00",
            "plain_text": "我问的是你是谁，不是我是谁",
            "reply_to_msg_id": None,
            "mentioned_bot": True,
            "raw_json": None,
        },
        {
            "id": 7,
            "group_id": 1001,
            "platform_msg_id": "source-after",
            "user_id": 11,
            "timestamp": "2026-08-24T12:00:00+00:00",
            "plain_text": "之后才说的事实",
            "reply_to_msg_id": None,
            "mentioned_bot": False,
            "raw_json": None,
        },
    ]
    items = [
        {
            "group_id": 1001,
            "subject_id": "11",
            "kind": "profile",
            "predicate": "lives_in",
            "object_text": "深圳",
            "content": "成员甲住在深圳",
            "source_ids": ("source-before",),
        },
        {
            "group_id": 1001,
            "subject_id": "11",
            "kind": "profile",
            "predicate": "future_claim",
            "object_text": "之后",
            "content": "问题之后才出现的画像",
            "source_ids": ("source-after",),
        },
    ]

    cases = _build_identity_audit_cases(
        messages,
        items,
        start=datetime(2026, 8, 24, 10, 30, tzinfo=UTC),
        end=datetime(2026, 8, 24, 11, 30, tzinfo=UTC),
    )

    assert len(cases) == 2
    first, second = cases
    assert first["requester_uin"] == "11"
    assert first["allowed_subject_user_ids"] == ("11",)
    assert first["expected_evidence_message_ids"] == ("source-before",)
    assert "之后才出现" not in first["gold_text"]
    assert first["observed_answer"] == "你是成员甲"
    assert first["observed_answer_message_id"] == "bot-reply-question-1"
    assert first["answer_expectation"] == "must_answer"
    assert second["requester_uin"] == "12"
    assert second["observed_answer"] is None
    assert second["answer_expectation"] == "must_abstain"
    assert all("你是谁" not in case["query"] for case in cases)


def test_identity_audit_requires_both_time_bounds(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    with pytest.raises(ValueError, match="provided together"):
        build_cases(
            engine,
            count=20,
            seed=1,
            identity_audit_start="2026-08-24T18:00:00+08:00",
        )


def test_generic_fact_queries_union_same_subject_kind_without_superlatives():
    first = {
        "group_id": 1001,
        "subject_id": "11",
        "kind": "preference",
        "object_text": "动画",
        "content": "成员喜欢动画",
        "source_ids": ("source-1",),
    }
    second = {
        **first,
        "object_text": "音乐",
        "content": "成员喜欢音乐",
        "source_ids": ("source-2",),
    }
    aliases = {11: ("成员甲",)}
    generic = _build_fact_case(
        first,
        aliases,
        random.Random(1),
        1,
        (first, second),
    )
    specific = _build_fact_case(
        first,
        aliases,
        random.Random(1),
        0,
        (first, second),
    )

    assert "最喜欢" not in generic["query"]
    assert generic["expected_evidence_message_ids"] == ("source-1", "source-2")
    assert "成员喜欢动画" in generic["gold_text"]
    assert "成员喜欢音乐" in generic["gold_text"]
    assert specific["expected_evidence_message_ids"] == ("source-1",)


def test_fact_case_skips_unresolvable_single_character_member_alias():
    item = {
        "group_id": 1001,
        "subject_id": "1365923420",
        "kind": "current",
        "object_text": "最近动态",
        "content": "成员最近有新动态",
        "source_ids": ("source-1",),
    }

    alternate = _build_fact_case(
        item,
        {1365923420: ("🐷", "成员甲")},
        random.Random(1),
        0,
    )
    explicit_id = _build_fact_case(
        item,
        {1365923420: ("🐷",)},
        random.Random(1),
        0,
    )

    assert alternate["query"].startswith("成员甲")
    assert explicit_id["query"].startswith("QQ号1365923420")


def test_first_person_cases_are_kind_bound_member_gold_and_keep_the_bucket(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    cases = build_cases(engine, count=120, seed=11)
    first_person_cases = [case for case in cases if case["category"] == "first_person"]

    assert 0 < len(first_person_cases) <= int(120 * 0.08)
    assert len(
        {
            (
                case["kind"],
                case["requester_uin"],
                case["query"],
                case["expected_evidence_message_ids"],
            )
            for case in first_person_cases
        }
    ) == len(first_person_cases)
    for case in first_person_cases:
        kind = case["kind"]
        assert kind in memory_test_dataset.FIRST_PERSON_TEMPLATES_BY_KIND
        assert case["query"] in memory_test_dataset.FIRST_PERSON_TEMPLATES_BY_KIND[kind]
        assert case["requester_uin"].isdigit()
        assert case["allowed_subject_user_ids"] == (case["requester_uin"],)
        assert case["expected_evidence_message_ids"]
        assert case["gold_text"]
        assert case["answer_expectation"] == "must_answer"
        assert "主人" not in case["query"]
        assert "最喜欢" not in case["query"]
        assert "在玩什么" not in case["query"]
        assert "聊过什么" not in case["query"]
        assert "subject=group" not in case["tags"]
        if kind in memory_test_dataset.TEMPORAL_KINDS:
            assert "temporal_recent=1" in case["tags"]
            assert "p5" not in case["expected_evidence_message_ids"]


def test_first_person_contract_rejects_kind_or_requester_mismatch_and_build_fails_closed(
    tmp_path,
    monkeypatch,
):
    item = {
        "group_id": 1001,
        "subject_id": "11",
        "kind": "preference",
        "object_text": "动画",
        "content": "成员喜欢动画",
        "source_ids": ("source-1",),
    }
    valid = _build_first_person_case(item, {}, 0)
    wrong_kind = dict(valid, query="我的计划是什么")
    with pytest.raises(ValueError, match="query-kind"):
        _validate_answer_contract(wrong_kind)
    wrong_requester = dict(valid, requester_uin="group")
    with pytest.raises(ValueError, match="requester"):
        _validate_answer_contract(wrong_requester)

    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    real_builder = memory_test_dataset._build_first_person_case

    def build_bad_first_person_case(*args, **kwargs):
        case = real_builder(*args, **kwargs)
        case["query"] = "我的计划是什么"
        return case

    monkeypatch.setattr(
        memory_test_dataset,
        "_build_first_person_case",
        build_bad_first_person_case,
    )
    with pytest.raises(ValueError, match="query-kind"):
        build_cases(engine, count=120, seed=11)

def test_temporal_kind_cases_skip_stale_items(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    cases = build_cases(engine, count=80, seed=3)
    plan_cases = [case for case in cases if case["category"] == "plan"]
    assert plan_cases
    assert all(
        "p5" not in case["expected_evidence_message_ids"]
        for case in plan_cases
    )


def test_temporal_kind_cases_keep_recent_gold_newest_first(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    cases = build_cases(engine, count=120, seed=3)
    plan_cases = [case for case in cases if case["category"] == "plan"]
    assert plan_cases
    assert plan_cases[0]["expected_evidence_message_ids"] == ("p7", "p6")
    assert "p5" not in plan_cases[0]["expected_evidence_message_ids"]


def test_group_subject_alias_and_mention_expected_abstention(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    cases = build_cases(engine, count=120, seed=11)
    group_cases = [
        case
        for case in cases
        if case.get("tags") and "subject=group" in case["tags"]
    ]
    assert group_cases
    assert all("群里" in case["query"] for case in group_cases)
    assert all(case["allowed_subject_user_ids"] is None for case in group_cases)
    mention_cases = [case for case in cases if case["category"] == "mention"]
    assert mention_cases
    assert all(case["gold_text"] == "" for case in mention_cases)
    assert all(not case["expected_evidence_message_ids"] for case in mention_cases)
    assert all(case["allowed_subject_user_ids"] is None for case in mention_cases)
    summary_cases = [case for case in cases if case["category"] == "summary"]
    assert summary_cases
    assert all("Recent chat summary" not in case["query"] for case in summary_cases)
    assert all("summary:" not in case["query"] for case in summary_cases)
    assert all(case["query"] in {"昨天群里说了什么", "昨天群里聊了什么"} for case in summary_cases)
    assert all(case["allowed_subject_user_ids"] is None for case in summary_cases)
    assert all(case["now_iso"] == "2026-08-01T16:00:00+00:00" for case in summary_cases)
    raw_cases = [case for case in cases if case["category"] == "raw_history"]
    assert raw_cases
    assert all("以前" in case["query"] or "之前" in case["query"] for case in raw_cases)
    distractor_cases = [case for case in cases if case["category"] == "distractor"]
    assert distractor_cases
    assert any(case["allowed_subject_user_ids"] for case in distractor_cases)
    assert all(
        case["allowed_subject_user_ids"] is None
        or len(case["allowed_subject_user_ids"]) == 1
        for case in distractor_cases
    )
    assert all(case["kind"] == "distractor" for case in distractor_cases)
    assert all("喜欢什么游戏" not in case["query"] for case in distractor_cases)


def test_summary_cases_collapse_same_shanghai_day_and_union_runtime_gold(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO summaries (id, scope_type, scope_id, summary_level, start_at, "
        "end_at, content, source_start_msg_id, source_end_msg_id, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            2,
            "group",
            "1001",
            "semantic_window",
            "2026-08-01 11:00:00",
            "2026-08-01 11:30:00",
            "大家还聊了测试",
            "p2",
            "p7",
            "active",
        ),
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    cases = build_cases(engine, count=120, seed=11)
    summary_cases = [case for case in cases if case["category"] == "summary"]

    assert len(summary_cases) == 1
    assert summary_cases[0]["now_iso"] == "2026-08-01T16:00:00+00:00"
    assert summary_cases[0]["time_range"] == (
        "2026-07-31T16:00:00+00:00",
        "2026-08-01T16:00:00+00:00",
    )
    assert summary_cases[0]["expected_evidence_message_ids"] == (
        "p1",
        "p3",
        "p2",
        "p7",
    )


def test_group_aliases_prefer_real_sender_snapshot_and_do_not_leak_cards(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE messages ADD COLUMN raw_json TEXT")
    connection.execute(
        "UPDATE messages SET raw_json = ? WHERE id = 7",
        ('{"sender":{"nickname":"阿渣","card":"群一卡"}}',),
    )
    connection.execute(
        "INSERT INTO messages (id, group_id, platform_msg_id, user_id, timestamp, "
        "plain_text, reply_to_msg_id, mentioned_bot, raw_json) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            8,
            1002,
            "p8",
            11,
            "2026-08-06T10:00:00+08:00",
            "同一个人在另一个群",
            None,
            0,
            '{"sender":{"nickname":"阿渣","card":"群二卡"}}',
        ),
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    aliases = _load_group_aliases(engine, _load_messages(engine))

    assert aliases[1001][11][0] == "群一卡"
    assert aliases[1002][11][0] == "群二卡"
    assert "群二卡" not in aliases[1001][11]


def test_group_aliases_do_not_fall_back_to_global_users_table(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE messages ADD COLUMN raw_json TEXT")
    connection.execute(
        "UPDATE users SET nickname = ?, group_card = ? WHERE user_id = ?",
        ("global-nickname", "other-group-card", 11),
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    aliases = _load_group_aliases(engine, _load_messages(engine))

    assert 11 not in aliases.get(1001, {})


@pytest.mark.parametrize("delivery_state", ("reserved", "blocked", "uncertain", "deleted"))
def test_group_aliases_use_only_latest_eligible_sender_snapshot(
    tmp_path, delivery_state
):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE messages ADD COLUMN raw_json TEXT")
    connection.execute(
        "UPDATE messages SET raw_json = ? WHERE id = 1",
        ('{"sender":{"nickname":"old-name","card":"历史可用名片"}}',),
    )
    connection.execute(
        "INSERT INTO messages (id, group_id, platform_msg_id, user_id, timestamp, "
        "plain_text, reply_to_msg_id, mentioned_bot, raw_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            8,
            1001,
            "latest-empty-text",
            11,
            "2026-08-06T10:00:00+08:00",
            "",
            None,
            0,
            '{"sender":{"nickname":"🐷","card":"🐷"}}',
        ),
    )
    connection.execute(
        "INSERT INTO messages (id, group_id, platform_msg_id, user_id, timestamp, "
        "plain_text, reply_to_msg_id, mentioned_bot, raw_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            9,
            1001,
            "latest-blocked",
            12,
            "2026-08-06T11:00:00+08:00",
            "blocked",
            None,
            0,
            json.dumps(
                {
                    "delivery_state": delivery_state,
                    "sender": {"card": "不可用新名片"},
                },
                ensure_ascii=False,
            ),
        ),
    )
    connection.execute(
        "UPDATE messages SET raw_json = ? WHERE id = 2",
        ('{"sender":{"card":"可用旧名片"}}',),
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    aliases = _load_group_aliases(engine, _load_messages(engine))

    assert aliases[1001][11] == ["🐷"]
    assert "历史可用名片" not in aliases[1001][11]
    assert aliases[1001][12] == ["可用旧名片"]


def test_retrievable_raw_ids_require_active_raw_v3_projection(tmp_path):
    database = tmp_path / "snapshot.db"
    _make_db(database)
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE retrieval_documents (
            id INTEGER PRIMARY KEY, group_id INTEGER, document_kind TEXT,
            source_table TEXT, status TEXT
        );
        CREATE TABLE retrieval_document_messages (
            document_id INTEGER, message_id INTEGER, group_id INTEGER
        );
        INSERT INTO retrieval_documents VALUES
            (1, 1001, 'raw_message_v3', 'messages', 'active'),
            (2, 1001, 'raw_message_v3', 'messages', 'inactive'),
            (3, 1001, 'episode_summary', 'summaries', 'active');
        INSERT INTO retrieval_document_messages VALUES
            (1, 1, 1001), (2, 2, 1001), (3, 3, 1001);
        """
    )
    connection.commit()
    connection.close()
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    assert _load_retrievable_raw_message_ids(engine) == {1}


def test_topic_keywords_reject_generic_fragments_and_keep_specific_phrases():
    assert _topic_keywords("我印象里你和thf是不是投了") == []
    assert _topic_keywords("有没有把safe的tag改成nsfw") == []
    candidates = _topic_keywords("日本球迷球队赛后爱清理看台和更衣室")
    assert candidates
    assert any("日本球" in candidate or "球迷球队" in candidate for candidate in candidates)
