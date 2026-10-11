import pytest

from app.core.memory_source_guard import is_usable_memory_source


@pytest.mark.parametrize("state", ["reserved", "blocked", "uncertain", "deleted"])
def test_source_guard_rejects_every_ineligible_delivery_state(state):
    assert not is_usable_memory_source(
        {
            "group_id": 100,
            "user_id": 11,
            "platform_msg_id": "m",
            "raw_json": {"delivery_state": state},
        },
        group_id=100,
    )
    assert not is_usable_memory_source(
        {
            "group_id": 100,
            "user_id": 11,
            "platform_msg_id": "m",
            "raw_json": '{"delivery_state":"' + state + '"}',
        },
        group_id=100,
    )


def test_source_guard_validates_author_group_and_real_source():
    row = {"group_id": 100, "user_id": 11, "platform_msg_id": "m", "raw_json": {}}
    assert is_usable_memory_source(row, group_id=100, author_id=11)
    assert not is_usable_memory_source(row, group_id=200)
    assert not is_usable_memory_source(row, author_id=22)
    assert not is_usable_memory_source({**row, "platform_msg_id": ""})
