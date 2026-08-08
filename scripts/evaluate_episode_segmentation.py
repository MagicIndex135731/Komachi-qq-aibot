"""Compare episode segmentation rules against real chat history.

Usage (inside the production container):
    python scripts/evaluate_episode_segmentation.py --group <group_id> --days 7
    python scripts/evaluate_episode_segmentation.py --group <group_id> --days 7 --with-llm

Simulates the old rules (idle 10min / 50 messages) and the new rules
(idle 5min / 70 messages / bot-reply grace / optional topic judge) over the
same messages and reports episode-count and size distributions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, "/workspace")

from app.config import AppSettings
from app.core.episode_segmenter import decide_episode_boundary
from app.core.episode_topic_judge import build_topic_judge_prompt, judge_topic_switch
from app.providers.llm_client import LlmClient


def load_messages(con: sqlite3.Connection, group_id: int, days: int | None) -> list[dict]:
    if days:
        since = datetime.now(UTC) - timedelta(days=days)
        rows = con.execute(
            "SELECT id, platform_msg_id, user_id, timestamp, plain_text, "
            "reply_to_msg_id, mentioned_bot "
            "FROM messages WHERE group_id=? AND timestamp >= ? ORDER BY timestamp, id",
            (group_id, since.strftime("%Y-%m-%d %H:%M:%S")),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT id, platform_msg_id, user_id, timestamp, plain_text, "
            "reply_to_msg_id, mentioned_bot "
            "FROM messages WHERE group_id=? ORDER BY timestamp, id",
            (group_id,),
        ).fetchall()
    return [
        {
            "id": row[0],
            "platform_msg_id": row[1],
            "user_id": row[2],
            "timestamp": datetime.fromisoformat(str(row[3])),
            "plain_text": row[4] or "",
            "reply_to_msg_id": row[5],
            "mentioned_bot": bool(row[6]),
        }
        for row in rows
    ]


def simulate(
    messages: list[dict],
    *,
    idle_minutes: int,
    max_messages: int,
    bot_grace: bool,
    bot_user_id: int,
    topic_judge=None,
    topic_judge_context: int = 8,
    topic_judge_start: int = 50,
    topic_judge_interval: int = 5,
) -> list[list[dict]]:
    episodes: list[list[dict]] = []
    current: list[dict] = []
    token_count = 0
    open_ids: set[str] = set()
    for message in messages:
        if current:
            previous = _as_message(current[-1])
            current_msg = _as_message(message)
            decision = decide_episode_boundary(
                previous=previous,
                current=current_msg,
                open_message_count=len(current),
                open_token_count=token_count,
                open_platform_msg_ids=open_ids,
                idle_minutes=idle_minutes,
                max_messages=max_messages,
                max_tokens=8000,
                bot_user_id=bot_user_id if bot_grace else None,
            )
            if topic_judge is not None:
                if (
                    not decision.should_close
                    and len(current) >= topic_judge_start
                    and len(current) % topic_judge_interval == 0
                ):
                    if _topic_switched(
                        current,
                        message,
                        topic_judge,
                        topic_judge_context,
                    ):
                        decision = type(decision)(True, "topic_switch")
            if decision.should_close:
                episodes.append(current)
                current = []
                open_ids = set()
                token_count = 0
        current.append(message)
        open_ids.add(str(message["platform_msg_id"]))
        token_count += max(1, len(message["plain_text"]) // 2)
    if current:
        episodes.append(current)
    return episodes


def _as_message(item: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=item["id"],
        platform_msg_id=item["platform_msg_id"],
        timestamp=item["timestamp"],
        plain_text=item["plain_text"],
        reply_to_msg_id=item["reply_to_msg_id"],
        mentioned_bot=item["mentioned_bot"],
        user_id=item["user_id"],
    )


def _topic_switched(
    current: list[dict],
    message: dict,
    topic_judge,
    context: int,
) -> bool:
    recent = [
        f"{item['user_id']}: {item['plain_text']}"
        for item in current[-context:]
        if item["plain_text"].strip()
    ]
    text = message["plain_text"].strip()
    if not text:
        return False
    prompt = build_topic_judge_prompt(
        recent_messages=recent,
        current_message=text,
        now=datetime.now(UTC),
        context_messages=context,
    )
    return judge_topic_switch(client=topic_judge, prompt_lines=prompt).switched


def report(label: str, episodes: list[list[dict]]) -> None:
    sizes = sorted(len(episode) for episode in episodes)
    total = sum(sizes)
    if not sizes:
        print(f"{label}: no episodes")
        return

    def percentile(pct: float) -> int:
        return sizes[int(pct / 100 * (len(sizes) - 1))]

    small = sum(1 for size in sizes if size <= 5)
    large = sum(1 for size in sizes if size >= 50)
    print(
        f"{label}: episodes={len(sizes)} total_msgs={total} "
        f"median={percentile(50)} mean={total / len(sizes):.1f} "
        f"p90={percentile(90)} max={sizes[-1]} "
        f"<=5={small} ({small / len(sizes) * 100:.1f}%) "
        f">=50={large} ({large / len(sizes) * 100:.1f}%)"
    )


def purity_report(
    label: str,
    episodes: list[list[dict]],
    client,
    *,
    samples: int = 8,
) -> None:
    pool = [episode for episode in episodes if len(episode) >= 15]
    if not pool:
        print(f"{label} purity: no episodes >=15 messages")
        return
    chosen = random.sample(pool, min(samples, len(pool)))
    scores: list[int] = []
    for episode in chosen:
        lines = [
            f"{message['user_id']}: {message['plain_text'][:80]}"
            for message in episode[:40]
            if message["plain_text"].strip()
        ]
        if not lines:
            continue
        prompt = [
            "System persona: You count distinct conversation topics inside a QQ group chat segment.",
            "Safety rules: Reply with exactly one line: TOPICS: <1|2|3|4+>.",
            "Segment:",
            "\n".join(lines)[:3000],
        ]
        try:
            raw = client.generate_text(prompt)
        except Exception:  # noqa: BLE001
            continue
        match = re.search(r"TOPICS\s*:\s*(\d+)", raw or "")
        if match:
            scores.append(int(match.group(1)))
    if scores:
        print(
            f"{label} purity: mean_topics={sum(scores) / len(scores):.2f} "
            f"samples={len(scores)} distribution={sorted(scores)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--days", type=float, default=None)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--purity-samples", type=int, default=8)
    parser.add_argument("--bot-qq", type=int, default=0, help="bot user id for reply-grace continuity")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    messages = load_messages(con, args.group, args.days)
    con.close()
    print(f"loaded {len(messages)} messages")
    if not messages:
        return

    old = simulate(
        messages,
        idle_minutes=10,
        max_messages=50,
        bot_grace=False,
        bot_user_id=args.bot_qq,
    )
    report("old (idle10/max50)", old)

    new = simulate(
        messages,
        idle_minutes=5,
        max_messages=70,
        bot_grace=True,
        bot_user_id=args.bot_qq,
    )
    report("new (idle5/max70/grace)", new)

    new2 = simulate(
        messages,
        idle_minutes=10,
        max_messages=70,
        bot_grace=True,
        bot_user_id=args.bot_qq,
    )
    report("new2 (idle10/max70/grace)", new2)

    if args.with_llm:
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
            reasoning_effort=settings.memory_episode_topic_judge_reasoning_effort,
            max_output_tokens=settings.memory_episode_topic_judge_max_output_tokens,
            usage_recorder=None,
            tool_event_recorder=None,
        )
        judged = simulate(
            messages,
            idle_minutes=10,
            max_messages=70,
            bot_grace=True,
            bot_user_id=args.bot_qq,
            topic_judge=client,
            topic_judge_context=settings.memory_episode_topic_judge_context_messages,
        )
        report("new+llm (topic judge)", judged)
        if args.purity_samples:
            purity_report("old", old, client, samples=args.purity_samples)
            purity_report("new+llm", judged, client, samples=args.purity_samples)


if __name__ == "__main__":
    main()
