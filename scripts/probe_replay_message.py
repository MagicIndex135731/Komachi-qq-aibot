"""Replay one stored group message through the production prepare path."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.adapters.onebot_models import parse_group_message_event
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.context_builder import ContextBuilder
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.main import (
    build_llm_client,
    build_memory_runtime,
    build_proactive_judge_client,
    build_web_search_client,
    should_enable_memory_in_group,
)
from app.storage.db import build_engine, session_scope
from app.storage.repositories import MessageRepository


class NullSender:
    gateway = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True)
    parser.add_argument("--tail", type=int, default=30)
    args = parser.parse_args()

    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    llm_client = build_llm_client(settings=settings, engine=engine)
    proactive_judge_client = build_proactive_judge_client(
        settings=settings,
        llm_client=llm_client,
        engine=engine,
    )
    web_search_client = build_web_search_client(settings)
    memory_runtime = build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        bot_display_name=str(runtime.persona.get("name", settings.bot_qq)),
        memory_enabled_group_ids=frozenset(
            int(group_id)
            for group_id in runtime.group_policy.get("groups", {})
            if should_enable_memory_in_group(group_id=int(group_id), group_policy=runtime.group_policy)
        ),
    )
    router = InboundRouter(
        engine=engine,
        runtime=runtime,
        sender=NullSender(),
        llm_client=llm_client,
        proactive_judge_client=proactive_judge_client,
        reply_policy=ReplyPolicy(),
        context_builder=ContextBuilder(),
        admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
        web_search_client=web_search_client,
        dev_control_service=None,
        group_image_service=None,
        memory_compaction_service=None,
        memory_orchestrator=memory_runtime.memory_orchestrator,
    )

    with session_scope(engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id(args.pid)
        if row is None:
            print(json.dumps({"error": "message not found", "pid": args.pid}))
            return 1
        raw = row.raw_json

    event = parse_group_message_event(
        raw,
        bot_qq=settings.bot_qq,
        bot_name=str(runtime.persona.get("name", settings.bot_qq)),
    )
    prepared = asyncio.run(
        asyncio.to_thread(
            router._prepare_group_reply,
            event,
            quoted_raw_payload=None,
        )
    )
    prompt_lines = prepared.prompt_lines or []
    full_prompt = "\n".join(prompt_lines)
    recent_section = ""
    for index, line in enumerate(prompt_lines):
        if line.startswith("Recent messages:"):
            recent_section = "\n".join(prompt_lines[index : index + 25])
            break
    print(
        json.dumps(
            {
                "pid": args.pid,
                "plain_text": event.plain_text,
                "mentioned_bot": event.mentioned_bot,
                "should_reply": prepared.should_reply,
                "image_count": len(prepared.target_images or []),
                "prompt_lines_count": len(prompt_lines),
                "prompt_chars": len(full_prompt),
                "recent_message_count": full_prompt.count("Recent message"),
                "recent_section_lines": len(recent_section.splitlines()) if recent_section else 0,
                "recent_section_chars": len(recent_section) if recent_section else 0,
                "recent_message_estimate": (
                    len(re.findall(r"（QQ昵称：[^）]*）: |比企谷小町: ", recent_section))
                    if recent_section
                    else 0
                ),
                "recent_after_facts": full_prompt.rindex("Recent message") > full_prompt.rindex("Memory fact") if "Recent message" in full_prompt and "Memory fact" in full_prompt else None,
                "recent_before_runtime": full_prompt.rindex("Recent message") < full_prompt.rindex("Runtime facts") if "Recent message" in full_prompt and "Runtime facts" in full_prompt else None,
                "has_context_labels": "Context labels:" in full_prompt,
                "has_prior_instruction": "优先为我跑图" in full_prompt,
                "has_referent_speaker": "900000102" in full_prompt,
                "recent_section": recent_section,
                "prompt_tail": full_prompt[-4000:],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
