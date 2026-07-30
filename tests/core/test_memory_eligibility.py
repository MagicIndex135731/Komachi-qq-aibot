from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from app.core.memory_eligibility import eligible
from app.core.memory_query_resolver import MemoryQueryPlan, TimeRange


@dataclass(frozen=True)
class Source:
    source_msg_id: str
    group_id: int
    user_id: int
    sent_at: datetime
    blocked: bool = False
    is_reserved: bool = False
    mentioned_uins: tuple[str, ...] = ()
    raw_json: object = None
    delivery_state: str = ""


def plan(**changes) -> MemoryQueryPlan:
    base = MemoryQueryPlan(
        original_query="昨天评价我",
        retrieval_query="昨天评价我",
        group_id=10,
        requester_id="42",
        subject_ids=("42",),
        subject_binding="requester",
        time_range=TimeRange(
            datetime(2026, 7, 21, 16, tzinfo=UTC),
            datetime(2026, 7, 22, 16, tzinfo=UTC),
        ),
        answer_mode="assessment",
        coverage_mode="time_buckets",
    )
    return replace(base, **changes)


def test_eligible_enforces_group_subject_and_half_open_utc_time() -> None:
    matching = Source("m1", 10, 42, datetime(2026, 7, 22, 8, tzinfo=UTC))

    assert eligible(matching, plan()) is True
    assert eligible(replace(matching, group_id=11), plan()) is False
    assert eligible(replace(matching, user_id=43), plan()) is False
    assert eligible(
        replace(matching, sent_at=datetime(2026, 7, 21, 15, 59, 59, tzinfo=UTC)),
        plan(),
    ) is False
    assert eligible(
        replace(matching, sent_at=datetime(2026, 7, 22, 16, tzinfo=UTC)),
        plan(),
    ) is False


def test_eligible_fails_closed_for_missing_scope_provenance_and_ambiguous_subject() -> None:
    matching = Source("m1", 10, 42, datetime(2026, 7, 22, 8, tzinfo=UTC))

    assert eligible(None, plan()) is False
    assert eligible(replace(matching, source_msg_id=""), plan()) is False
    assert eligible(matching, plan(group_id=None)) is False
    assert eligible(matching, plan(subject_ids=(), subject_binding="explicit")) is False


def test_eligible_rejects_blocked_reserved_and_uncertain_sources() -> None:
    matching = Source("m1", 10, 42, datetime(2026, 7, 22, 8, tzinfo=UTC))

    assert eligible(replace(matching, blocked=True), plan()) is False
    assert eligible(replace(matching, is_reserved=True), plan()) is False
    assert eligible(
        replace(matching, raw_json={"delivery_state": "uncertain"}),
        plan(),
    ) is False
    assert eligible(
        replace(matching, delivery_state="deleted"),
        plan(),
    ) is False


def test_unbound_subject_allows_any_author_but_mention_mode_requires_typed_context() -> None:
    other = Source("m1", 10, 99, datetime(2026, 7, 22, 8, tzinfo=UTC))

    assert eligible(other, plan(subject_ids=None, subject_binding="unbound")) is True
    assert eligible(
        replace(other, mentioned_uins=("42",)),
        plan(answer_mode="mention"),
    ) is True
    assert eligible(
        replace(other, mentioned_uins=("77",)),
        plan(answer_mode="mention"),
    ) is False
    assert eligible(
        replace(other, user_id=42, mentioned_uins=()),
        plan(answer_mode="mention"),
    ) is False


def test_naive_sqlite_source_timestamp_is_interpreted_as_utc() -> None:
    source = Source("m1", 10, 42, datetime(2026, 7, 22, 8))

    assert eligible(source, plan()) is True
