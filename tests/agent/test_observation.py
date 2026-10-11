from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.adapters.onebot_models import GroupMessageEvent
from app.adapters.outbound_guard import GroupDispatchScope, group_dispatch_scope
from app.adapters.sender import Sender
from app.agent.contracts import AgentBatch, AgentDecision
from app.agent.observation import GroupObservationService
from app.core.group_image_generation import GroupImageGenerationRequest
from app.core.router import InboundRouter
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository, UserRepository


def make_event():
    return GroupMessageEvent(
        "observer-source",
        900000001,
        900000101,
        "test-member",
        "",
        "普通话题讨论",
        {},
        datetime.now(UTC),
        "text",
        [],
        False,
        None,
    )


class Gateway:
    def __init__(self):
        self.calls = []

    async def call_api(self, action, params):
        self.calls.append((action, params))
        return {"status": "ok", "data": {"message_id": 900000500 + len(self.calls)}}


def make_router(engine, tmp_path):
    gateway = Gateway()
    router = InboundRouter.build_for_test(
        sqlite_engine=engine,
        sender=Sender(gateway),
        llm_client=object(),
    )
    router.runtime.settings.data_dir = tmp_path / "isolated"
    router.runtime.group_policy["groups"]["900000001"] = {
        "enabled": True,
        "speak": True,
        "memory_enabled": False,
        "archive": False,
        "proactive_reply": True,
        "proactive_interval_seconds": "60-60",
    }
    incoming = make_event()
    router.ingest_live_group_message(incoming)
    return router, incoming, gateway


def test_forged_reason_and_event_mention_do_not_grant_authority(sqlite_engine, tmp_path):
    router, incoming, _ = make_router(sqlite_engine, tmp_path)
    router.runtime.group_policy["groups"]["900000001"]["proactive_reply"] = False
    observer = GroupObservationService(router)
    decision = AgentDecision("reply", incoming.platform_msg_id, "direct_trigger")
    forged = replace(incoming, mentioned_bot=True, plain_text="@Mira 请回复")
    assert not observer.dispatch_allowed(forged, decision=decision)
    observation = observer.build_agent_observation(AgentBatch(incoming.group_id, (forged,), "turn"))
    assert observation["direct_target"] is None
    assert observation["suppression"] == "proactive_disabled"

def test_owner_directed_output_does_not_refresh_chat_cooldown(sqlite_engine,tmp_path):
    from app.storage.agent_effects import AgentToolEffectRepository
    router,event,_=make_router(sqlite_engine,tmp_path)
    now=datetime.now(UTC)
    bot=router.runtime.settings.bot_qq
    repo=AgentToolEffectRepository(sqlite_engine,account_id=bot,clock=lambda:now)
    effect=repo.reserve(f'owner:group:{event.group_id}',run_id='owner-directive',
        tool_call_id='send',operation='send_group',payload_hash='a'*64)
    assert repo.mark_dispatching(effect['action_id'],daily_limit=None)
    repo.finish(effect['action_id'],'sent',receipt='synthetic-owner-bot-output')
    with session_scope(sqlite_engine) as db:
        UserRepository(db).upsert_user(user_id=bot,nickname='bot',group_card='')
        MessageRepository(db).add_group_message(platform_msg_id='synthetic-owner-bot-output',group_id=event.group_id,
            user_id=bot,timestamp=now,plain_text='主人指令结果',msg_type='text',reply_to_msg_id=None,mentioned_bot=False,
            raw_json={'social_action_id':effect['action_id'],'delivery_state':'sent'})
    observer=GroupObservationService(router,clock=lambda:now)
    assert observer.proactive_send_allowed(event.group_id,'new-message')
    with session_scope(sqlite_engine) as db:
        MessageRepository(db).add_group_message(platform_msg_id='synthetic-normal-bot-output',group_id=event.group_id,
            user_id=bot,timestamp=now,plain_text='普通自主发言',msg_type='text',reply_to_msg_id=None,mentioned_bot=False,
            raw_json={'delivery_state':'sent'})
    assert not observer.proactive_send_allowed(event.group_id,'new-message')


def test_working_state_newer_than_ledger_preserves_cooldown(sqlite_engine, tmp_path):
    router, incoming, _ = make_router(sqlite_engine, tmp_path)
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(
            user_id=router.runtime.settings.bot_qq, nickname="Mira", group_card=""
        )
        MessageRepository(session).add_group_message(
            platform_msg_id="old-bot-reply",
            group_id=incoming.group_id,
            user_id=router.runtime.settings.bot_qq,
            timestamp=incoming.timestamp - timedelta(hours=1),
            plain_text="先前发言",
            raw_json={"delivery_state": "sent"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
    observation = GroupObservationService(router).build_agent_observation(
        AgentBatch(
            incoming.group_id,
            (incoming,),
            "turn",
            last_sent_at=datetime.now(UTC),
        )
    )
    assert observation["suppression"] == "cooldown"


@pytest.mark.asyncio
async def test_actual_autonomous_burst_collects_multiple_receipts_without_self_cooldown(
    sqlite_engine, tmp_path
):
    router, incoming, gateway = make_router(sqlite_engine, tmp_path)
    router.runtime.settings.group_reply_split_enabled = True
    router.runtime.settings.group_reply_split_max_messages = 3
    router.runtime.settings.group_reply_split_min_delay_seconds = 0
    router.runtime.settings.group_reply_split_max_delay_seconds = 0
    observer = GroupObservationService(router)
    scope = GroupDispatchScope(
        incoming.group_id,
        lambda: observer.dispatch_allowed(
            incoming, decision=AgentDecision("reply", incoming.platform_msg_id)
        ),
        lambda: None,
    )
    with group_dispatch_scope(scope):
        await router._send_chat_reply(incoming, "这段挺有意思|后面还可以继续聊")
    assert len(gateway.calls) == 2
    assert len(scope.receipts) == 2


def test_image_job_authority_ignores_cooldown_but_requires_current_source_and_speak(
    sqlite_engine, tmp_path
):
    router, incoming, _ = make_router(sqlite_engine, tmp_path)
    request = GroupImageGenerationRequest(
        incoming.group_id, incoming.platform_msg_id, "a picture", incoming.user_id
    )
    observer = GroupObservationService(router)
    router._reserve_outbound_reply(incoming, "已经受理出图")
    router._mark_outbound_reply_sent(incoming, "已经受理出图")
    # The accepted image job stays authorized while the acknowledgement causes
    # normal participation to enter cooldown.
    assert observer.image_job_allowed(request)
    assert not observer.image_job_allowed(replace(request, requester_user_id=900000102))
    with session_scope(sqlite_engine) as session:
        from app.storage.repositories import GroupRepository

        GroupRepository(session).set_speak_enabled(incoming.group_id, False)
    assert not observer.image_job_allowed(request)


def test_quiet_hours_are_shanghai_time_and_recalled_sources_revoke_images(sqlite_engine, tmp_path):
    router, incoming, _ = make_router(sqlite_engine, tmp_path)
    router.runtime.group_policy["groups"]["900000001"]["quiet_hours"] = "00:00-05:00"
    night = GroupObservationService(router, clock=lambda: datetime(2026, 10, 9, 17, tzinfo=UTC))
    decision = AgentDecision("reply", incoming.platform_msg_id)
    assert not night.dispatch_allowed(incoming, decision=decision)  # Shanghai 01:00.
    day = GroupObservationService(router, clock=lambda: datetime(2026, 10, 9, 8, tzinfo=UTC))
    assert day.dispatch_allowed(incoming, decision=decision)  # Shanghai 16:00.
    request = GroupImageGenerationRequest(
        incoming.group_id, incoming.platform_msg_id, "a picture", incoming.user_id
    )
    with session_scope(sqlite_engine) as session:
        source = MessageRepository(session).get_by_platform_msg_id(incoming.platform_msg_id)
        source.raw_json = {"delivery_state": "deleted"}
    assert not day.dispatch_allowed(incoming, decision=decision)
    assert not day.image_job_allowed(request)
