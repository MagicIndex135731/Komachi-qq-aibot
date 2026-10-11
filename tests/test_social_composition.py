"""Exercise all scheduled surfaces through real composition with fake networks."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
import httpx

import pytest
from sqlalchemy import select

from app.adapters.sender import Sender
from app.core.router import InboundRouter
from app.runtime.social import build_social_agent
from app.providers.llm_client import LlmClient
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.social_agent import SocialActionRecord, SocialScheduledJob


@pytest.mark.asyncio
async def test_composed_agent_wakes_and_sends_group_post_and_friend_comment_once(sqlite_engine):
    bot, group, friend = 900000101, 900000001, 900000201
    stamp = int(datetime.now(UTC).timestamp()) - 60
    calls = []
    class Gateway:
        websocket = object()
        async def call_api(self, action, params):
            calls.append((action, params))
            if action == "send_group_msg":
                data = {"message_id": "synthetic-social-group-receipt"}
            elif action == "send_qzone_msg":
                data = {"tid": "synthetic-own-tid", "time": stamp}
            elif action == "comment_qzone":
                data = {"comment_id": ""}  # Confirmed success need not have a comment ID.
            elif action == "get_qzone_feeds":
                data = {"has_more": False, "feeds": [{"uin": friend, "nickname": "合成好友", "time": stamp,
                        "appid": 311, "key": "feed-handle", "html":
                        f'<li data-uin="{friend}" data-tid="synthetic-friend-tid">春天来了</li>'}]}
            elif action == "get_qzone_msg_list":
                assert params["target_uin"] == friend
                data = {"total": 1, "msglist": [{"tid": "synthetic-friend-tid", "content": "春天来了",
                        "time": stamp, "comment_num": 0, "is_private": False, "images": []}]}
            else:
                raise AssertionError(action)
            return {"status": "ok", "retcode": 0, "data": data}
    class Llm:
        prompts = []
        def generate_text(self, lines, **kwargs):
            self.prompts.append(lines)
            assert kwargs["allow_web_search"] is False
            observation = json.loads(next(line for line in lines if line.startswith("Scheduled observation JSON:")).split("\n", 1)[1])
            surface = observation["surface"]
            if surface == "group":
                return json.dumps({"action": "send_group", "text": "最近听到什么好听的歌？"})
            if surface == "qzone_publish":
                assert "PRIVATE_SYNTHETIC" not in "\n".join(lines)
                return json.dumps({"action": "publish_qzone", "text": "今天也想听一首轻快的歌。"})
            assert observation["observed_posts"][0]["post_id"] == "synthetic-friend-tid"
            return json.dumps({"action": "comment_qzone", "owner_id": friend,
                               "post_id": "synthetic-friend-tid", "text": "终于等到春天了。"})
    gateway = Gateway()
    sender = Sender(gateway)
    llm = Llm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.settings = router.runtime.settings.model_copy(update={"bot_qq": bot, "social_agent_enabled": True})
    router.runtime.persona = {"name": "小町", "facts": ["PRIVATE_SYNTHETIC"], "speaking_style": {"tone": "自然"}}
    router.runtime.group_policy = {"groups": {str(group): {"enabled": True, "speak": True,
                                    "proactive_reply": True, "memory_enabled": False}}}
    common = {"enabled": True, "interval_seconds": 60, "cooldown_seconds": 60,
              "max_daily": 1, "quiet_hours": None}
    router.runtime.social_policy = {"enabled": True, "groups": {str(group): {**common, "min_idle_seconds": 0, "topics": ["音乐"]}},
            "qzone": {"publish": {**common, "topics": ["音乐"]},
                      "comment": {**common, "friend_ids": [friend]}}}
    agent, planner = build_social_agent(router=router, sender=sender, gateway=gateway, llm_client=llm)
    assert agent is not None
    try:
        await agent.start()
        agent.set_connected(True)
        now = datetime.now(UTC)
        await agent.tick(now)
        await agent.tick(now + timedelta(seconds=61))
        writes = [call for call in calls if call[0] in {"send_group_msg", "send_qzone_msg", "comment_qzone"}]
        with session_scope(sqlite_engine) as session:
            outcomes = [(row.kind, row.last_reason) for row in session.scalars(select(SocialScheduledJob))]
        assert len(writes) == 3, outcomes
        assert {call[0] for call in writes} == {"send_group_msg", "send_qzone_msg", "comment_qzone"}
        with session_scope(sqlite_engine) as session:
            actions = list(session.scalars(select(SocialActionRecord)))
            assert len(actions) == 3 and {row.status for row in actions} == {"sent"}
            rows = list(session.scalars(select(Message)))
            assert len(rows) == 1 and rows[0].user_id == bot
            assert rows[0].reply_to_msg_id is None
            assert len(list(session.scalars(select(SocialScheduledJob)))) == 3
        await agent.tick(now + timedelta(seconds=122))
        assert len(llm.prompts) == 3  # Exhausted budget blocks model calls too.
        assert len([call for call in calls if call[0] in {"send_group_msg", "send_qzone_msg", "comment_qzone"}]) == 3
    finally:
        await agent.stop()
        planner.close()
    assert not agent.running


def test_disabled_composition_creates_no_social_worker(sqlite_engine):
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=object(), llm_client=object())
    assert build_social_agent(router=router, sender=None, gateway=None, llm_client=None) == (None, None)


def test_social_factory_preserves_configured_vision_model_and_bounded_no_tools_client(sqlite_engine):
    def reject_network(request):
        raise AssertionError("factory must not issue model requests")

    with httpx.Client(transport=httpx.MockTransport(reject_network)) as transport:
        primary = LlmClient(base_url="https://synthetic.example.test/v1", api_key="test",
                            model="synthetic-primary", http_client=transport)
        router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=object(), llm_client=primary)
        router.runtime.settings = router.runtime.settings.model_copy(update={
            "social_agent_enabled": True, "llm_vision_model": "synthetic-vision",
            "proactive_judge_model": "synthetic-text-judge", "social_agent_max_output_tokens": 768,
            "social_agent_decision_timeout_seconds": 45,
        })
        router.runtime.social_policy = {"enabled": True, "qzone": {"publish": {
            "enabled": True, "topics": ["音乐"],
        }}}
        agent, planner = build_social_agent(router=router, sender=None, gateway=None, llm_client=primary)
        client = planner.decision_client
        try:
            assert client is not primary
            assert client.model == "synthetic-text-judge"
            assert client.vision_model == "synthetic-vision"
            assert client._image_chat_model(default_model=client.responses_model) == "synthetic-vision"
            assert client.max_output_tokens == 768 and client.http_client.timeout.read == 45
            assert client.builtin_web_search is False and client.responses_only is True
            assert agent.running is False
        finally:
            planner.close()
            client.http_client.close()
