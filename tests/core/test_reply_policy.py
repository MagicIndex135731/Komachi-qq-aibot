from datetime import UTC, datetime, timedelta

from app.core.reply_policy import PolicyInput, ReplyPolicy


def make_policy_input(**overrides) -> PolicyInput:
    values = {
        "group_speak_enabled": True,
        "mentioned_bot": False,
        "named_bot": False,
        "direct_question": False,
        "same_thread_followup": False,
        "addressed_without_at": False,
        "has_interjection_opportunity": False,
        "recent_bot_reply_at": None,
        "now": datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
        "quiet_hours": None,
        "proactive_enabled": True,
        "group_traffic_last_minute": 1,
        "proactive_interval_seconds": (180, 480),
        "event_id": "evt-1",
    }
    values.update(overrides)
    return PolicyInput(**values)


def test_reply_policy_blocks_non_allowlisted_groups() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(group_speak_enabled=False, mentioned_bot=True, direct_question=True))
    assert decision.should_reply is False
    assert decision.reason == "group_not_allowlisted"


def test_reply_policy_allows_direct_mention_when_not_in_cooldown() -> None:
    policy = ReplyPolicy()
    now = datetime.now(UTC)
    decision = policy.decide(
        make_policy_input(
            mentioned_bot=True,
            direct_question=True,
            recent_bot_reply_at=now - timedelta(minutes=5),
            now=now,
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_named_bot_bypasses_quiet_hours() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(
        make_policy_input(
            named_bot=True,
            now=datetime(2026, 5, 8, 2, 0, tzinfo=UTC),
            quiet_hours=(datetime(2026, 5, 8, 1, 0, tzinfo=UTC).time(), datetime(2026, 5, 8, 8, 0, tzinfo=UTC).time()),
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_direct_mention_bypasses_quiet_hours() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(
        make_policy_input(
            mentioned_bot=True,
            now=datetime(2026, 5, 8, 2, 0, tzinfo=UTC),
            quiet_hours=(datetime(2026, 5, 8, 1, 0, tzinfo=UTC).time(), datetime(2026, 5, 8, 8, 0, tzinfo=UTC).time()),
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_same_thread_followup_bypasses_cooldown() -> None:
    policy = ReplyPolicy(cooldown_seconds=90)
    now = datetime.now(UTC)
    decision = policy.decide(
        make_policy_input(
            same_thread_followup=True,
            recent_bot_reply_at=now - timedelta(seconds=30),
            now=now,
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_same_thread_followup_bypasses_quiet_hours() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(
        make_policy_input(
            same_thread_followup=True,
            now=datetime(2026, 5, 8, 2, 0, tzinfo=UTC),
            quiet_hours=(datetime(2026, 5, 8, 1, 0, tzinfo=UTC).time(), datetime(2026, 5, 8, 8, 0, tzinfo=UTC).time()),
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_addressed_without_at_counts_as_direct_trigger() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(addressed_without_at=True))
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_guaranteed_trigger_wins_when_proactive_disabled() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(named_bot=True, proactive_enabled=False))
    assert decision.should_reply is True
    assert decision.reason == "direct_trigger"


def test_reply_policy_allows_high_traffic_question_to_join_discussion() -> None:
    policy = ReplyPolicy(proactive_threshold=4)
    decision = policy.decide(make_policy_input(direct_question=True, group_traffic_last_minute=9))
    assert decision.should_reply is True
    assert decision.reason == "proactive_score"
    assert decision.score == 4


def test_reply_policy_low_traffic_alone_is_not_enough_for_proactive_reply() -> None:
    policy = ReplyPolicy(proactive_threshold=4)
    decision = policy.decide(make_policy_input(group_traffic_last_minute=1))
    assert decision.should_reply is False
    assert decision.reason == "below_threshold"
    assert decision.score == 0


def test_reply_policy_time_sensitive_turn_stays_silent_when_group_is_idle() -> None:
    policy = ReplyPolicy(proactive_threshold=4)
    decision = policy.decide(make_policy_input(group_traffic_last_minute=1, has_interjection_opportunity=True))

    assert decision.should_reply is False
    assert decision.reason == "below_threshold"
    assert decision.score == 0


def test_reply_policy_blocks_proactive_turn_inside_active_interval() -> None:
    policy = ReplyPolicy(proactive_threshold=4)
    now = datetime(2026, 5, 8, 12, 4, tzinfo=UTC)
    decision = policy.decide(
        make_policy_input(
            direct_question=True,
            has_interjection_opportunity=True,
            recent_bot_reply_at=datetime(2026, 5, 8, 12, 2, 30, tzinfo=UTC),
            now=now,
            event_id="evt-active-gap",
        )
    )

    assert decision.should_reply is False
    assert decision.reason == "cooldown"


def test_reply_policy_allows_proactive_turn_after_interval_and_opportunity() -> None:
    policy = ReplyPolicy(proactive_threshold=4)
    now = datetime(2026, 5, 8, 12, 9, tzinfo=UTC)
    decision = policy.decide(
        make_policy_input(
            direct_question=True,
            has_interjection_opportunity=True,
            group_traffic_last_minute=3,
            recent_bot_reply_at=datetime(2026, 5, 8, 12, 1, tzinfo=UTC),
            now=now,
            event_id="evt-active-pass",
        )
    )

    assert decision.should_reply is True
    assert decision.reason == "proactive_score"
    assert decision.score == 6


def test_reply_policy_constructor_cooldown_still_blocks_proactive_reply() -> None:
    policy = ReplyPolicy(cooldown_seconds=600, proactive_threshold=4)
    now = datetime(2026, 5, 8, 12, 10, tzinfo=UTC)
    decision = policy.decide(
        make_policy_input(
            direct_question=True,
            has_interjection_opportunity=True,
            recent_bot_reply_at=datetime(2026, 5, 8, 12, 1, 40, tzinfo=UTC),
            now=now,
            proactive_interval_seconds=(180, 180),
            event_id="evt-fixed-interval",
        )
    )

    assert decision.should_reply is False
    assert decision.reason == "cooldown"
