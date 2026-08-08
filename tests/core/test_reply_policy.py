from datetime import UTC, datetime, timedelta

from app.core.reply_policy import PolicyInput, ReplyPolicy


def make_policy_input(**overrides) -> PolicyInput:
    values = {
        "group_speak_enabled": True,
        "mentioned_bot": False,
        "named_bot": False,
        "same_thread_followup": False,
        "addressed_without_at": False,
        "recent_bot_reply_at": None,
        "now": datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
        "quiet_hours": None,
        "proactive_enabled": True,
        "group_traffic_last_minute": 5,
        "proactive_judge_enabled": True,
        "proactive_interval_seconds": (180, 480),
        "event_id": "evt-1",
    }
    values.update(overrides)
    return PolicyInput(**values)


def test_reply_policy_blocks_non_allowlisted_groups() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(group_speak_enabled=False, mentioned_bot=True))
    assert decision.should_reply is False
    assert decision.reason == "group_not_allowlisted"


def test_reply_policy_allows_direct_mention_when_not_in_cooldown() -> None:
    policy = ReplyPolicy()
    now = datetime.now(UTC)
    decision = policy.decide(
        make_policy_input(
            mentioned_bot=True,
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


def test_reply_policy_quiet_hours_blocks_proactive_candidate() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(
        make_policy_input(
            now=datetime(2026, 5, 8, 2, 0, tzinfo=UTC),
            quiet_hours=(datetime(2026, 5, 8, 1, 0, tzinfo=UTC).time(), datetime(2026, 5, 8, 8, 0, tzinfo=UTC).time()),
        )
    )
    assert decision.should_reply is False
    assert decision.reason == "quiet_hours"


def test_reply_policy_cooldown_blocks_proactive_candidate() -> None:
    policy = ReplyPolicy()
    now = datetime(2026, 5, 8, 12, 4, tzinfo=UTC)
    decision = policy.decide(
        make_policy_input(
            recent_bot_reply_at=datetime(2026, 5, 8, 12, 2, 30, tzinfo=UTC),
            now=now,
            event_id="evt-active-gap",
        )
    )
    assert decision.should_reply is False
    assert decision.reason == "cooldown"


def test_reply_policy_proactive_disabled_blocks_candidate() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(proactive_enabled=False))
    assert decision.should_reply is False
    assert decision.reason == "proactive_disabled"


def test_reply_policy_low_traffic_blocks_proactive_candidate() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(group_traffic_last_minute=1))
    assert decision.should_reply is False
    assert decision.reason == "below_threshold"


def test_reply_policy_judge_disabled_blocks_proactive_candidate() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(make_policy_input(proactive_judge_enabled=False))
    assert decision.should_reply is False
    assert decision.reason == "proactive_judge_disabled"


def test_reply_policy_judge_enabled_emits_candidate_without_scoring() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(
        make_policy_input(
            group_traffic_last_minute=2,
            proactive_judge_enabled=True,
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "proactive_candidate"
    assert decision.score == 1


def test_reply_policy_candidate_allows_any_lively_message_not_just_questions() -> None:
    policy = ReplyPolicy()
    decision = policy.decide(
        make_policy_input(
            group_traffic_last_minute=9,
            proactive_judge_enabled=True,
        )
    )
    assert decision.should_reply is True
    assert decision.reason == "proactive_candidate"
