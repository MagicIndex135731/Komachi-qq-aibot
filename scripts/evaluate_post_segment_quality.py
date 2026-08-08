"""Compare whole-episode derivation vs post-hoc re-segmented derivation.

Usage (inside the production container):
    python scripts/evaluate_post_segment_quality.py --group <group_id> --samples 3
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, "/workspace")

from app.config import AppSettings
from app.core.episode_post_segment import post_segment_episode
from app.core.memory_background_service import CompactionEpisodeDeriver
from app.providers.llm_client import LlmClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-messages", type=int, default=25)
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    episodes = list(
        con.execute(
            """
            SELECT id, message_count
            FROM conversation_episodes
            WHERE group_id = ? AND status != 'open' AND message_count >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (args.group, args.min_messages, args.samples),
        )
    )
    print(f"loaded {len(episodes)} episodes")
    if not episodes:
        return

    settings = AppSettings()
    llm = LlmClient(
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
        max_output_tokens=4096,
        usage_recorder=None,
        tool_event_recorder=None,
    )
    deriver = CompactionEpisodeDeriver(llm_client=llm, max_facts=24)

    for row in episodes:
        messages = list(
            con.execute(
                """
                SELECT m.id, m.platform_msg_id, m.user_id, m.timestamp, m.plain_text
                FROM episode_messages em
                JOIN messages m ON m.id = em.message_id
                WHERE em.episode_id = ? AND em.group_id = ?
                ORDER BY em.ordinal
                """,
                (row["id"], args.group),
            )
        )
        items = [
            SimpleNamespace(
                id=item[0],
                platform_msg_id=str(item[1]),
                user_id=int(item[2]),
                timestamp=datetime.fromisoformat(str(item[3])),
                plain_text=str(item[4] or ""),
                is_reserved=False,
                is_blocked=False,
            )
            for item in messages
        ]
        whole = deriver.derive(
            episode=SimpleNamespace(id=row["id"], group_id=args.group),
            messages=tuple(items),
            windows=(),
        )
        pieces = post_segment_episode(
            client=llm,
            messages=items,
            min_messages=args.min_messages,
        )
        piece_results = []
        for piece in pieces:
            piece_results.append(
                deriver.derive(
                    episode=SimpleNamespace(id=row["id"], group_id=args.group),
                    messages=tuple(piece),
                    windows=(),
                )
            )
        print("---")
        print(f"episode {row['id']} msgs={len(items)}")
        print(f"whole: summary_len={len(whole.summary)} facts={len(whole.facts)}")
        print(f"  summary: {whole.summary[:200]}")
        print(
            f"post-segment: pieces={len(pieces)} "
            f"summary_len={sum(len(p.summary) for p in piece_results)} "
            f"facts={sum(len(p.facts) for p in piece_results)}"
        )
        for index, result in enumerate(piece_results[:4], start=1):
            print(f"  piece {index}: facts={len(result.facts)} summary: {result.summary[:120]}")

    con.close()


if __name__ == "__main__":
    main()
