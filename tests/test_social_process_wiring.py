"""Drive enabled social workers through the real group process lifecycle."""

import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

import app.group_main as group_main
from app.providers.semantic_embeddings import DisabledEmbeddingProvider
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.social_agent import SocialScheduledJob
from tests.test_process_split_smoke import _settings


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_enabled_group_process_connects_both_agents_and_closes_social_resources(
    monkeypatch, tmp_path, sqlite_engine, cancelled,
):
    bot, group, friend = 900000101, 900000001, 900000201
    settings = _settings(tmp_path).model_copy(update={
        "bot_qq": bot, "group_agent_enabled": True, "social_agent_enabled": True,
    })
    common = {"enabled": True, "interval_seconds": 60, "cooldown_seconds": 60,
              "max_daily": 1, "quiet_hours": None}
    runtime = SimpleNamespace(
        settings=settings, persona={"name": "小町"}, personas={}, safety={},
        group_policy={"groups": {str(group): {
            "enabled": True, "speak": True, "proactive_reply": True,
            "memory_enabled": False, "archive": False,
        }}},
        social_policy={"enabled": True, "groups": {str(group): {
            **common, "min_idle_seconds": 0, "topics": ["音乐"],
        }}, "qzone": {
            "publish": {**common, "topics": ["音乐"]},
            "comment": {**common, "friend_ids": [friend]},
        }},
    )
    groups, group_planners, socials, social_planners = [], [], [], []
    real_group, real_planner = group_main.GroupAgentRuntime, group_main.GroupParticipationPlanner
    real_social = group_main.build_social_agent

    def capture(factory, instances):
        def build(**kwargs):
            instance = factory(**kwargs)
            instances.append(instance)
            return instance
        return build

    def build_social(**kwargs):
        agent, planner = real_social(**kwargs)
        socials.append(agent)
        social_planners.append(planner)
        return agent, planner

    class Llm:
        def generate_text(self, *_args, **_kwargs):
            pytest.fail("Starting or reconnecting before next_due must not invoke a model")

    class Worker:
        async def run(self):
            await asyncio.Future()

    class ImageService:
        engine = None

        async def start(self):
            pass

        async def stop(self):
            pass

    class Gateway:
        def __init__(self, **_kwargs):
            self.websocket = None
            self.on_connection_change = None

        async def call_api(self, *_args, **_kwargs):
            pytest.fail("The lifecycle test must not issue any QQ API request")

        async def connect_and_consume(self, handler, on_connect=None):
            assert len(groups) == len(socials) == 1
            assert groups[0].running and socials[0].running
            assert not groups[0].connected and not socials[0].connected
            assert callable(self.on_connection_change)
            self.websocket = object()
            self.on_connection_change(True)
            try:
                await on_connect()
                assert groups[0].connected and socials[0].connected
                await socials[0].tick(datetime.now(UTC))
                with session_scope(sqlite_engine) as session:
                    assert len(list(session.scalars(select(SocialScheduledJob)))) == 3
                    assert list(session.scalars(select(Message))) == []
                generation = socials[0]._connection_generation
                self.websocket = None
                self.on_connection_change(False)
                assert not groups[0].connected and not socials[0].connected
                self.websocket = object()
                self.on_connection_change(True)
                await on_connect()
                assert groups[0].connected and socials[0].connected
                assert socials[0]._connection_generation > generation
                # Private events belong to the other process even when social
                # participation is enabled for this account.
                await handler({"post_type": "message", "message_type": "private",
                               "user_id": friend, "message_id": "synthetic-private"})
                if cancelled:
                    raise asyncio.CancelledError()
            finally:
                self.websocket = None
                self.on_connection_change(False)

    async def noop(**_kwargs):
        pass

    llm = Llm()
    monkeypatch.setattr(group_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(group_main, "load_runtime_config", lambda _settings: runtime)
    monkeypatch.setattr(group_main, "build_engine", lambda _path: sqlite_engine)
    monkeypatch.setattr(group_main, "sync_history_archives", lambda *_args: None)
    monkeypatch.setattr(group_main, "NapCatGateway", Gateway)
    monkeypatch.setattr(group_main, "build_llm_client", lambda **_kwargs: llm)
    monkeypatch.setattr(group_main, "build_proactive_judge_client", lambda **_kwargs: llm)
    monkeypatch.setattr(group_main, "build_group_image_llm_client", lambda **_kwargs: object())
    monkeypatch.setattr(group_main, "build_web_search_client", lambda _settings: None)
    monkeypatch.setattr(group_main, "build_image_reference_search_client", lambda _settings: None)
    monkeypatch.setattr(group_main, "build_group_image_reference_planner_client", lambda **_kwargs: None)
    monkeypatch.setattr(group_main, "build_group_image_service", lambda **_kwargs: ImageService())
    monkeypatch.setattr(group_main, "build_memory_runtime", lambda **_kwargs: SimpleNamespace(
        memory_compaction_service=None, memory_orchestrator=None,
        embedding_provider=DisabledEmbeddingProvider(),
    ))
    monkeypatch.setattr(group_main, "PersonaLiveSyncService", lambda **_kwargs: Worker())
    monkeypatch.setattr(group_main, "MemberFactRefreshService", lambda **_kwargs: Worker())
    monkeypatch.setattr(group_main, "GroupAgentRuntime", capture(real_group, groups))
    monkeypatch.setattr(group_main, "GroupParticipationPlanner", capture(real_planner, group_planners))
    monkeypatch.setattr(group_main, "build_social_agent", build_social)
    monkeypatch.setattr(group_main, "backfill_recent_group_history", noop)
    monkeypatch.setattr(group_main, "_replay_startup_window_mentions", noop)

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await group_main.run()
    else:
        await group_main.run()

    assert not groups[0].running and not groups[0].connected
    assert not socials[0].running and not socials[0].connected
    assert socials[0]._loop_task is None and not socials[0]._active_ticks
    assert social_planners[0].blocking._closed and group_planners[0].blocking._closed
    heartbeat = json.loads((settings.log_dir / "group.heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["state"] == "stopped"
