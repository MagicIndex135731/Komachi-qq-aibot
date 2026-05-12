from app.core.summarizer import summarize_window


def test_summarize_window_formats_recent_dialogue() -> None:
    summary = summarize_window(
        [
            "Alice: are you online tonight?",
            "Bob: later",
            "Bot: me too",
            "Carol: sounds good",
        ]
    )

    assert summary == "Recent chat summary: Alice: are you online tonight? | Bob: later | Bot: me too"
