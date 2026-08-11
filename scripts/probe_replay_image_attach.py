"""Replay the '看看奶子' incident to see whether any image gets attached."""

from __future__ import annotations

import json
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
        messages = MessageRepository(session)
        row = messages.get_by_platform_msg_id("-1518611770")
        if row is None:
            print(json.dumps({"error": "message not found"}))
            return 1
        raw = row.raw_json
    event = parse_group_message_event(
        raw,
        bot_qq=settings.bot_qq,
        bot_name=str(runtime.persona.get("name", settings.bot_qq)),
    )

    import asyncio

    prepared = asyncio.run(
        asyncio.to_thread(
            router._prepare_group_reply,
            event,
            quoted_raw_payload=None,
        )
    )
    images = prepared.target_images or []
    prompt_text = "\n".join(prepared.prompt_lines or []) if prepared.prompt_lines else ""
    print(
        json.dumps(
            {
                "should_reply": prepared.should_reply,
                "image_count": len(images),
                "image_sources": [
                    {
                        "url": image.url,
                        "local_path": image.local_path,
                        "file_id": image.file_id,
                    }
                    for image in images
                ],
                "proactive_turn": prepared.proactive_turn,
                "use_memory_tools": prepared.use_memory_tools,
                "prompt_mentions_news": any(
                    word in prompt_text
                    for word in ("贪污", "举报", "舅舅", "抗抑郁", "反转", "奶子", "图片")
                ),
                "prompt_chars": len(prompt_text),
                "prompt_head": prompt_text[:600],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
