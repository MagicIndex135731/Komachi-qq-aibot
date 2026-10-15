from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.adapters.sender import QQMessageDeliveryUncertainError
from app.agent.social_contracts import SocialDecision, SocialJob, SocialObservation, SocialPost
from app.agent.social_executor import SocialExecutor
from app.agent.social_policy import configured_social_jobs
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository
from tests.agent.test_social_policy import runtime


class Guard:
    action_id = "synthetic-action"
    dispatched = False

    def require_valid(self):
        pass

    def valid(self):
        return True

    def on_dispatch(self):
        self.dispatched = True


class Sender:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def send_group_text(self, outbound):
        from app.adapters.outbound_guard import current_group_dispatch
        scope = current_group_dispatch(outbound.group_id)
        assert scope.can_send()
        scope.on_dispatch()
        self.calls.append(outbound)
        if self.error:
            raise self.error
        return ("synthetic-receipt",)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['sent', 'uncertain', 'reserved'])
async def test_reply_recovery_fences_exact_source_and_ignores_unconfirmed_reservations(sqlite_engine, state):
    router, observation = group_setup(sqlite_engine)
    executor = SocialExecutor(router=router, sender=Sender(), qzone_adapter=None)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=observation.job.scope_id, group_name='', enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=900000103, nickname='synthetic', group_card='')
        ids=[]
        for index in (1, 2):
            row=MessageRepository(session).add_group_message(platform_msg_id=f'synthetic-source-{index}',
                group_id=observation.job.scope_id, user_id=900000103, timestamp=datetime.now(UTC), plain_text='synthetic',
                raw_json={}, msg_type='text', reply_to_msg_id=None, mentioned_bot=False)
            session.flush()
            ids.append(row.id)
    observation=replace(observation, source_message_ids=tuple(ids))
    provisional=executor._reserve_group_output(observation, SocialDecision('send_group', 'synthetic reply'),
        'synthetic-answer', reply_source_id=ids[0])
    if state!='reserved':executor._finish_group_output(provisional, state=state, receipt='synthetic-receipt' if state=='sent' else None)
    assert executor.reply_delivery_state(observation.job.scope_id, ids[0]) == (None if state=='reserved' else state)
    assert executor.reply_delivery_state(observation.job.scope_id, ids[1]) is None
    assert executor.reply_delivery_state(900000002, ids[0]) is None


def group_setup(sqlite_engine):
    config = runtime({"enabled": True, "groups": {"900000001": {"enabled": True, "min_idle_seconds": 0,
                                                                               "quiet_hours": None}}},
                     {"900000001": {"enabled": True, "speak": True, "proactive_reply": True}})
    router = SimpleNamespace(runtime=config, engine=sqlite_engine, _resolve_group_policy=lambda **kw:
                             (True, True, True, (60, 60), None, False))
    job = configured_social_jobs(config)[0]
    observation = SocialObservation(job, datetime.now(UTC), "小町")
    return router, observation


@pytest.mark.asyncio
async def test_scheduled_group_send_is_outgoing_with_real_receipt_and_no_fabricated_inbound(sqlite_engine):
    router, observation = group_setup(sqlite_engine)
    sender = Sender()
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None)
    result = await executor.execute(observation, SocialDecision("send_group", "秋天到了"), Guard())
    assert result.status == "sent"
    with session_scope(sqlite_engine) as session:
        rows = list(session.scalars(select(Message)))
        assert len(rows) == 1
        assert rows[0].platform_msg_id == "synthetic-receipt"
        assert rows[0].user_id == router.runtime.settings.bot_qq
        assert rows[0].raw_json["social_action_id"] == "synthetic-action"


@pytest.mark.asyncio
async def test_direct_group_long_answer_is_whole_and_autonomous_caption_keeps_its_bound(sqlite_engine):
    router, observation=group_setup(sqlite_engine)
    sender=Sender();executor=SocialExecutor(router=router,sender=sender,qzone_adapter=None)
    executor.policy_check=lambda *args,**kwargs:True
    reply='核对的文件与触发条件如下。\n' * 100
    decision=SocialDecision('send_group',reply)
    assert (await executor._send_group(observation,decision,Guard())).status=='blocked'
    assert not sender.calls
    result=await executor._send_group(observation,decision,Guard(),max_text_chars=12000)
    assert result.status=='sent' and sender.calls[0].text==reply


@pytest.mark.asyncio
async def test_uncertain_group_send_retains_ineligible_provisional_record(sqlite_engine):
    router, observation = group_setup(sqlite_engine)
    sender = Sender(QQMessageDeliveryUncertainError("unknown"))
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None)
    result = await executor.execute(observation, SocialDecision("send_group", "秋天到了"), Guard())
    assert result.status == "uncertain"
    with session_scope(sqlite_engine) as session:
        row = session.scalar(select(Message))
        assert row.raw_json["delivery_state"] == "uncertain"
        assert row.raw_json["delivery_attempts"] == 1
    assert len(sender.calls) == 1


@pytest.mark.asyncio
async def test_comment_rechecks_post_after_model_and_rejects_changed_source(sqlite_engine):
    config = runtime({"enabled": True, "qzone": {"comment": {"enabled": True, "friend_ids": [900000103],
                                                              "quiet_hours": None}}})
    job = configured_social_jobs(config)[0]
    now = datetime.now(UTC)
    post = SocialPost(900000103, "actual-tid", "原文", now)
    class Space:
        calls = 0
        async def get_posts(self, user_id, count=20):
            return (SocialPost(user_id, "actual-tid", "已修改", now),)
        async def comment(self, *args):
            self.calls += 1
    space = Space()
    router = SimpleNamespace(runtime=config, engine=sqlite_engine)
    executor = SocialExecutor(router=router, sender=None, qzone_adapter=space)
    result = await executor.execute(SocialObservation(job, now, "小町", posts=(post,)),
                                    SocialDecision("comment_qzone", "评价", "actual-tid", 900000103), Guard())
    assert result.status == "blocked"
    assert space.calls == 0


def add_source(engine, *, group_id, platform_id, timestamp):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic", enabled=True,
                                              speak_enabled=True)
        UserRepository(session).upsert_user(user_id=900000103, nickname="synthetic", group_card="")
        row = MessageRepository(session).add_group_message(
            platform_msg_id=platform_id, group_id=group_id, user_id=900000103, timestamp=timestamp,
            plain_text="合成群聊消息", msg_type="text", reply_to_msg_id=None, mentioned_bot=False, raw_json={},
        )
        session.flush()
        return row.id


@pytest.mark.asyncio
@pytest.mark.parametrize("late_offset", [-120, 120])
async def test_new_group_arrival_revokes_topic_even_when_its_timestamp_is_older(sqlite_engine, late_offset):
    from app.core.time_utils import ASIA_SHANGHAI

    router, observation = group_setup(sqlite_engine)
    now = datetime.now(UTC)
    original_time = (now - timedelta(hours=1)).astimezone(ASIA_SHANGHAI).replace(tzinfo=None)
    original_id = add_source(sqlite_engine, group_id=observation.job.scope_id,
                             platform_id="observed-source", timestamp=original_time)
    observation = replace(observation, source_message_ids=(original_id,))
    decision = SocialDecision("send_group", "根据之前话题聊聊", source_message_ids=(original_id,))
    sender = Sender()
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None, clock=lambda: now)
    assert executor.policy_check(observation, decision)

    add_source(sqlite_engine, group_id=observation.job.scope_id, platform_id="new-arrival",
               timestamp=original_time + timedelta(seconds=late_offset))

    result = await executor.execute(observation, decision, Guard())
    assert result.status == "blocked"
    assert sender.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_material", [{"context": ("合成群聊上下文",)}, {"source_message_ids": (1,)}])
async def test_comment_cannot_dispatch_an_observation_carrying_chat_material(sqlite_engine, chat_material):
    config = runtime({"enabled": True, "qzone": {"comment": {"enabled": True, "friend_ids": [900000103],
                                                              "quiet_hours": None}}})
    job = configured_social_jobs(config)[0]
    now = datetime.now(UTC)
    post = SocialPost(900000103, "public-tid", "合成公开动态", now)
    class Space:
        async def get_posts(self, *args, **kwargs):
            return (post,)
        async def comment(self, *args, **kwargs):
            pytest.fail("chat material escaped to a public comment")
    router = SimpleNamespace(runtime=config, engine=sqlite_engine)
    executor = SocialExecutor(router=router, sender=None, qzone_adapter=Space(), clock=lambda: now)
    observation = SocialObservation(job, now, "公开人格", posts=(post,), **chat_material)
    result = await executor.execute(observation, SocialDecision("comment_qzone", "评价", "public-tid", 900000103), Guard())
    assert result.status == "blocked"


@pytest.mark.asyncio
async def test_new_arrival_in_a_different_group_does_not_revoke_this_groups_topic(sqlite_engine):
    from app.core.time_utils import ASIA_SHANGHAI

    router, observation = group_setup(sqlite_engine)
    now = datetime.now(UTC)
    stamp = (now - timedelta(hours=1)).astimezone(ASIA_SHANGHAI).replace(tzinfo=None)
    original_id = add_source(sqlite_engine, group_id=observation.job.scope_id,
                             platform_id="this-groups-source", timestamp=stamp)
    observation = replace(observation, source_message_ids=(original_id,))
    decision = SocialDecision("send_group", "本群的公开话题", source_message_ids=(original_id,))
    add_source(sqlite_engine, group_id=observation.job.scope_id + 1,
               platform_id="another-groups-new-source", timestamp=stamp + timedelta(hours=1))
    sender = Sender()
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None, clock=lambda: now)

    result = await executor.execute(observation, decision, Guard())

    assert result.status == "sent"
    assert [item.group_id for item in sender.calls] == [observation.job.scope_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("old_state", ["reserved", "deleted", "uncertain"])
async def test_old_social_provisional_does_not_permanently_block_independent_schedules(sqlite_engine, old_state):
    from app.core.time_utils import ASIA_SHANGHAI

    router, observation = group_setup(sqlite_engine)
    now = datetime.now(UTC)
    stamp = (now - timedelta(hours=2)).astimezone(ASIA_SHANGHAI).replace(tzinfo=None)
    original_id = add_source(sqlite_engine, group_id=observation.job.scope_id,
                             platform_id="confirmed-observed-source", timestamp=stamp)
    observation = replace(observation, source_message_ids=(original_id,))
    decision = SocialDecision("send_group", "一个独立的新话题", source_message_ids=(original_id,))
    sender = Sender()
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None, clock=lambda: now)
    old_platform_id = executor._reserve_group_output(observation, decision, "interrupted-action")
    with session_scope(sqlite_engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id(old_platform_id)
        row.timestamp = stamp
        row.raw_json = {**row.raw_json, "delivery_state": old_state}
    guard = Guard()
    guard.action_id = "new-independent-action"

    result = await executor.execute(observation, decision, guard)

    assert result.status == "sent"
    assert len(sender.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_identity", ["author", "platform", "missing_action"])
async def test_provisional_filter_cannot_hide_unconfirmed_inbound_messages(sqlite_engine, invalid_identity):
    from app.core.time_utils import ASIA_SHANGHAI

    router, observation = group_setup(sqlite_engine)
    now = datetime.now(UTC)
    stamp = (now - timedelta(hours=2)).astimezone(ASIA_SHANGHAI).replace(tzinfo=None)
    original_id = add_source(sqlite_engine, group_id=observation.job.scope_id,
                             platform_id="confirmed-before-ineligible-source", timestamp=stamp)
    observation = replace(observation, source_message_ids=(original_id,))
    decision = SocialDecision("send_group", "新话题", source_message_ids=(original_id,))
    sender = Sender()
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None, clock=lambda: now)
    old_platform_id = executor._reserve_group_output(observation, decision, "untrusted-action")
    with session_scope(sqlite_engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id(old_platform_id)
        row.timestamp = stamp
        if invalid_identity == "author":
            row.user_id = 900000103
        elif invalid_identity == "platform":
            row.platform_msg_id = "actual-inbound-id"
        else:
            row.raw_json = {"delivery_state": "reserved"}

    result = await executor.execute(observation, decision, Guard())

    assert result.status == "blocked"
    assert sender.calls == []
