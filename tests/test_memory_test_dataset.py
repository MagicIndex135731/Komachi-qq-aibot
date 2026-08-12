import sqlite3

from scripts.build_memory_test_dataset import build_cases
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