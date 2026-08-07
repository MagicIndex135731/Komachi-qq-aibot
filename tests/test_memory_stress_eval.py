from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    UserRepository,
)


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "memory_stress_eval.py"
_SPEC = importlib.util.spec_from_file_location("memory_stress_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
stress_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stress_eval)


def _seed(sqlite_engine) -> None:
    now = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        for group_id in (10001, 10002):
            GroupRepository(session).upsert_group(
                group_id=group_id,
                group_name=str(group_id),
                enabled=True,
                speak_enabled=True,
            )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="用户A",
            group_card="",
        )
        UserRepository(session).upsert_user(
            user_id=20002,
            nickname="用户B",
            group_card="",
        )
        messages = MessageRepository(session)
        for index in range(2):
            messages.add_group_message(
                platform_msg_id=f"a-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=now,
                plain_text="台风路径好冷",
                raw_json={"sender": {"nickname": "用户A", "card": ""}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        messages.add_group_message(
            platform_msg_id="a-other",
            group_id=10002,
            user_id=20001,
            timestamp=now,
            plain_text="台风路径好冷",
            raw_json={"sender": {"nickname": "用户A", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        for index in range(2):
            messages.add_group_message(
                platform_msg_id=f"b-{index}",
                group_id=10002,
                user_id=20002,
                timestamp=now,
                plain_text="动画很好看",
                raw_json={"sender": {"nickname": "用户B", "card": ""}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )


def test_build_cases_only_generates_target_group_cases(sqlite_engine) -> None:
    _seed(sqlite_engine)

    cases = stress_eval._build_cases(
        sqlite_engine,
        limit_cases=None,
        excluded_user_ids=set(),
        target_group_ids=(10001,),
    )

    assert cases
    assert {int(case["group_id"]) for case in cases} == {10001}
    categories = {case["category"] for case in cases}
    assert "raw_history" in categories
    assert "cross_group" in categories
    assert all(
        int(case["forbidden_groups"][0]) == 10002
        for case in cases
        if case["category"] == "cross_group"
    )


def test_build_cases_without_target_groups_keeps_all_groups(sqlite_engine) -> None:
    _seed(sqlite_engine)

    cases = stress_eval._build_cases(
        sqlite_engine,
        limit_cases=None,
        excluded_user_ids=set(),
    )

    assert {int(case["group_id"]) for case in cases} == {10001, 10002}
