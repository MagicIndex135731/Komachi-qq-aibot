from app.core.summarizer import summarize_recursive, summarize_window


def test_summarize_window_formats_recent_dialogue() -> None:
    summary = summarize_window(
        [
            "Alice: are you online tonight?",
            "Bob: later",
            "Bot: me too",
            "Carol: sounds good",
        ]
    )

    assert summary == "Recent chat summary: Alice: are you online tonight? | Bob: later | Bot: me too | Carol: sounds good"


def test_summarize_recursive_retains_prior_and_new_high_signal_information() -> None:
    summary = summarize_recursive(
        previous_summary="Rolling group memory: Alice plans a trip next week.",
        new_window_summary="Recent chat summary: Bob: the time was changed to Saturday.",
    )

    assert "plans a trip next week" in summary
    assert "changed to Saturday" in summary


def test_summarize_recursive_keeps_latest_window_when_budget_is_exhausted() -> None:
    summary = summarize_recursive(
        previous_summary="old " * 30,
        new_window_summary="latest decision: meet on Saturday",
        max_chars=40,
    )

    assert "latest" in summary


def test_summarize_recursive_does_not_nest_summary_prefixes() -> None:
    summary = "Rolling group memory: Alice plans a trip."
    for index in range(4):
        summary = summarize_recursive(
            previous_summary=summary,
            new_window_summary=f"Recent chat summary: update {index}",
        )

    assert summary.count("Rolling group memory:") == 1
    assert "Recent chat summary:" not in summary
