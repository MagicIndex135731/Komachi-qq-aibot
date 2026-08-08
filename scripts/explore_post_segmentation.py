"""Explore post-hoc episode re-segmentation on real history.

Reads existing episodes, asks the upstream model to find topic boundaries
inside each episode, then compares topic purity of the original episode vs the
re-segmented pieces.

Usage (inside the production container):
    python scripts/explore_post_segmentation.py --group <group_id> --days 7 --samples 8
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, "/workspace")

from app.config import AppSettings
from app.providers.llm_client import LlmClient


def load_episodes(con: sqlite3.Connection, group_id: int, days: int | None) -> list[dict]:
    since = (
        (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        if days
        else "1970-01-01 00:00:00"
    )
    rows = con.execute(
        """
        SELECT e.id, e.started_at, e.ended_at, e.message_count
        FROM conversation_episodes e
        WHERE e.group_id = ? AND (e.ended_at IS NULL OR e.ended_at >= ?)
          AND e.message_count >= 15 AND e.message_count <= 80
        ORDER BY e.id DESC
        """,
        (group_id, since),
    ).fetchall()
    episodes = []
    for episode_id, started_at, ended_at, message_count in rows:
        messages = con.execute(
            """
            SELECT m.plain_text
            FROM episode_messages em
            JOIN messages m ON m.id = em.message_id
            WHERE em.episode_id = ? AND em.group_id = ?
            ORDER BY em.ordinal
            """,
            (episode_id, group_id),
        ).fetchall()
        episodes.append(
            {
                "id": episode_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "message_count": message_count,
                "texts": [row[0] or "" for row in messages],
            }
        )
    return episodes


def build_segment_prompt(texts: list[str]) -> list[str]:
    lines = [
        f"{index}. {text[:120]}"
        for index, text in enumerate(texts, start=1)
        if text.strip()
    ]
    return [
        "System persona: You split a QQ group chat log into topic segments.",
        "Safety rules: Reply with exactly one JSON object, no markdown: "
        '{"segments": [{"start": 1, "end": 5, "topic": "short label"}, ...]}. '
        "Every message must belong to exactly one segment; segments must be contiguous and ordered.",
        "Group policy: Split when the conversation clearly moves to a new topic (different subject, "
        "new question unrelated to the previous flow, or a hard scene change). Brief jokes, "
        "back-and-forth and single-line reactions stay in the current segment.",
        "Chat log:",
        *lines,
    ]


def build_purity_prompt(texts: list[str]) -> list[str]:
    lines = [f"{index}. {text[:120]}" for index, text in enumerate(texts, start=1) if text.strip()]
    return [
        "System persona: You count distinct conversation topics in a QQ group chat segment.",
        "Safety rules: Reply with exactly one line: TOPICS: <1|2|3|4+>.",
        "Segment:",
        *lines,
    ]


def parse_segments(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        return []
    parsed: list[tuple[int, int]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start") or 0)
        end = int(item.get("end") or 0)
        if start >= 1 and end >= start:
            parsed.append((start, end))
    return parsed


def split_texts(texts: list[str], segments: list[tuple[int, int]]) -> list[list[str]]:
    if not segments:
        return [texts]
    pieces: list[list[str]] = []
    cursor = 0
    for start, end in sorted(segments):
        start = max(1, start)
        end = min(len(texts), end)
        if start > cursor + 1:
            pieces.append(texts[cursor : start - 1])
        if start <= end:
            pieces.append(texts[start - 1 : end])
        cursor = max(cursor, end)
    if cursor < len(texts):
        pieces.append(texts[cursor:])
    return [piece for piece in pieces if piece]


def count_topics(client, texts: list[str]) -> int:
    prompt = build_purity_prompt(texts)
    raw = client.generate_text(prompt)
    for token in ("TOPICS:", "TOPICS :"):
        if token in (raw or ""):
            value = (raw or "").split(token, 1)[1].splitlines()[0].strip()
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                return max(1, min(4, int(digits)))
    return 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--days", type=float, default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-messages", type=int, default=20)
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    episodes = load_episodes(con, args.group, args.days)
    con.close()
    episodes = [episode for episode in episodes if len(episode["texts"]) >= args.min_messages]
    episodes = episodes[: args.samples]
    print(f"loaded {len(episodes)} candidate episodes")
    if not episodes:
        return

    settings = AppSettings()
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model or "",
        vision_model="",
        responses_model=settings.llm_model,
        responses_only=True,
        image_responses_model=settings.llm_model,
        builtin_web_search=False,
        web_search_context_size="low",
        reasoning_effort="low",
        max_output_tokens=2048,
        usage_recorder=None,
        tool_event_recorder=None,
    )

    total_original_topics = 0
    total_piece_topics = 0
    total_pieces = 0
    total_segments_found = 0
    successes = 0
    for episode in episodes:
        texts = [text for text in episode["texts"] if text.strip()]
        if len(texts) < args.min_messages:
            continue
        try:
            raw = client.generate_text(build_segment_prompt(texts))
        except Exception as exc:  # noqa: BLE001
            print(f"episode {episode['id']}: segment call failed: {type(exc).__name__}")
            continue
        segments = parse_segments(raw)
        if not segments:
            print(f"episode {episode['id']}: no parseable segments ({len(texts)} msgs)")
            continue
        pieces = split_texts(texts, segments)
        if len(pieces) <= 1:
            print(f"episode {episode['id']}: model kept single segment ({len(texts)} msgs)")
            continue
        original_topics = count_topics(client, texts)
        piece_topics = [count_topics(client, piece) for piece in pieces]
        total_original_topics += original_topics
        total_piece_topics += sum(piece_topics)
        total_pieces += len(pieces)
        total_segments_found += len(segments)
        successes += 1
        print(
            f"episode {episode['id']}: msgs={len(texts)} segments={len(pieces)} "
            f"original_topics={original_topics} piece_topics={piece_topics}"
        )

    if successes:
        print("---")
        print(f"successes={successes}")
        print(f"mean original topics: {total_original_topics / successes:.2f}")
        print(
            f"mean piece topics: {total_piece_topics / total_pieces:.2f} "
            f"(over {total_pieces} pieces)"
        )
        print(f"mean segments found: {total_segments_found / successes:.2f}")


if __name__ == "__main__":
    main()
