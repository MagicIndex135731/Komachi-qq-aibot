from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.adapters.onebot_models import GroupMessageEvent
from app.adapters.sender import Sender
from app.agent.contracts import AgentBatch, AgentDecision
from app.agent.observation import GroupObservationService
from app.agent.participation_feedback import record_feedback
from app.core.router import InboundRouter
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository, UserRepository
from app.storage.participation_feedback import GroupParticipationFeedback


class Gateway:
    async def call_api(self, *args):
        raise AssertionError('Offline feedback must not call QQ')


def setup(engine):
    router = InboundRouter.build_for_test(sqlite_engine=engine, sender=Sender(Gateway()), llm_client=object())
    for group in (900000001, 900000002):
        router.runtime.group_policy['groups'][str(group)] = {
            'enabled': True, 'speak': True, 'archive': False, 'memory_enabled': False,
            'proactive_reply': True, 'proactive_interval_seconds': '120-300',
            'proactive_min_new_messages': 4,
        }
    router.runtime.settings.group_message_max_age_seconds = 0
    return router


def incoming(router, text, *, msg='human', group=900000001, user=900000101, at=None, mention=False, reply=None):
    event = GroupMessageEvent(msg, group, user, 'member', '', text, {}, at or datetime.now(UTC),
                              'text', [], mention, reply)
    router.ingest_live_group_message(event)
    return event


def bot(router, *, group=900000001, at=None, text='今天的话题我想补充一句', msg='bot'):
    with session_scope(router.engine) as session:
        UserRepository(session).upsert_user(user_id=router.runtime.settings.bot_qq, nickname='小町', group_card='')
        MessageRepository(session).add_group_message(platform_msg_id=msg, group_id=group,
            user_id=router.runtime.settings.bot_qq, timestamp=at or datetime.now(UTC), plain_text=text,
            raw_json={'delivery_state': 'sent'}, msg_type='text', reply_to_msg_id=None, mentioned_bot=False)


def observe(router, event, now):
    return GroupObservationService(router, clock=lambda: now).build_agent_observation(
        AgentBatch(event.group_id, (event,), 'test-turn'))


@pytest.mark.parametrize('text', ['开一个闭嘴模式', '安静点', '能不能别插嘴了', '别老接话', '话也太多了', '开个闭麦模式', '我也想让他闭嘴'])
def test_unnamed_feedback_quiets_bot_and_leaves_direct_requests(sqlite_engine, text):
    r=setup(sqlite_engine); now=datetime.now(UTC)
    incoming(r,'一个新话题',msg='seed',at=now-timedelta(minutes=6))
    bot(r,at=now-timedelta(seconds=30))
    feedback=incoming(r,text,at=now)
    assert observe(r,feedback,now)['suppression']=='group_feedback_quiet'
    assert not GroupObservationService(r,clock=lambda:now).proactive_send_allowed(feedback.group_id,'wake')
    direct=incoming(r,'在吗，帮我看看这个问题',msg='direct',mention=True,at=now+timedelta(seconds=1))
    assert observe(r,direct,now+timedelta(seconds=1))['direct_target']=='direct'


@pytest.mark.parametrize('text,reply', [
    ('上班烦死了',None),('这个面试官话也太多了',None),('让小明闭嘴',None),
    ('小町一点也不烦',None),('别插嘴了','peer'),('小町别闭嘴，继续说',None),
])
def test_unrelated_frustration_and_third_party_do_not_quiet(sqlite_engine,text,reply):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'同事聊天',msg='peer',at=now-timedelta(seconds=50))
    bot(r,at=now-timedelta(seconds=30))
    e=incoming(r,text,reply=reply,at=now)
    assert observe(r,e,now)['suppression']!='group_feedback_quiet'


def test_copy_of_bot_is_not_direct_or_a_new_complaint(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=2))
    text='欸～让人闭嘴都要找小町代劳呀？'
    bot(r,at=now-timedelta(seconds=30),text=text)
    e=incoming(r,text,at=now)
    state=observe(r,e,now)
    assert state['direct_target'] is None
    assert state['suppression']!='group_feedback_quiet'


def test_feedback_survives_service_recreation_expires_and_replay_does_not_extend(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=2));bot(r,at=now-timedelta(seconds=30))
    e=incoming(r,'开一个闭嘴模式',at=now)
    later=now+timedelta(hours=1)
    record_feedback(r,e,now=later)
    for i in range(4):incoming(r,f'实质内容{i}',msg=f'new-{i}',at=now+timedelta(minutes=21,seconds=i))
    bot(r,at=now+timedelta(minutes=20),msg='after-pause')
    assert observe(r,e,now+timedelta(minutes=25))['suppression']=='cooldown'
    assert GroupObservationService(r,clock=lambda:now+timedelta(hours=2,seconds=1)).proactive_send_allowed(e.group_id,'wake')


def test_group_isolation_and_canonical_recall(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    seed=incoming(r,'话题',msg='seed',at=now-timedelta(minutes=6));bot(r,at=now-timedelta(seconds=30))
    e=incoming(r,'开一个闭嘴模式',at=now)
    for i in range(4):other=incoming(r,f'另一个群内容{i}',group=900000002,msg=f'other-{i}',at=now)
    assert observe(r,other,now)['suppression']==''
    with session_scope(sqlite_engine) as session:
        row=MessageRepository(session).get_by_platform_msg_id(e.platform_msg_id)
        row.raw_json={'delivery_state':'deleted'}
    assert observe(r,seed,now)['suppression']!='group_feedback_quiet'


def test_feedback_arriving_during_autonomous_generation_revokes_dispatch(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=7));bot(r,at=now-timedelta(minutes=6))
    for i in range(4):e=incoming(r,f'有效讨论{i}',msg=f'event-{i}',at=now-timedelta(seconds=10-i))
    obs=GroupObservationService(r,clock=lambda:now)
    decision=AgentDecision('reply',e.platform_msg_id,'agent_event')
    assert obs.dispatch_allowed(e,decision=decision)
    incoming(r,'闭嘴吧',msg='complaint',at=now)
    assert not obs.dispatch_allowed(e,decision=decision)
    assert not obs.dispatch_allowed(e,decision=decision,continued_dispatch=True)


def test_threshold_counts_distinct_substantive_sources_and_preserves_at(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    e=incoming(r,'讨论',msg='seed',at=now-timedelta(minutes=8));bot(r,at=now-timedelta(minutes=6))
    for i in range(3):e=incoming(r,f'有效内容{i}',msg=f'e-{i}',at=now)
    assert observe(r,e,now)['suppression']=='below_participation_threshold'
    incoming(r,'哈哈哈',msg='noise',at=now)
    incoming(r,'有效内容2',msg='copy',at=now)
    assert observe(r,e,now)['suppression']=='below_participation_threshold'
    e=incoming(r,'第四条有效内容',msg='fourth',at=now)
    assert observe(r,e,now)['suppression']==''
    direct=incoming(r,'在吗',msg='at',mention=True,at=now)
    assert observe(r,direct,now)['direct_target']=='at'


def test_forged_event_cannot_establish_feedback(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    e=incoming(r,'天气不错',at=now);bot(r,at=now-timedelta(seconds=30))
    record_feedback(r,replace(e,plain_text='闭嘴',mentioned_bot=True),now=now)
    assert observe(r,e,now)['suppression']!='group_feedback_quiet'


def test_real_bot_quote_can_bind_feedback_without_recent_bot_speech(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(hours=1))
    bot(r,at=now-timedelta(hours=1),msg='old-bot')
    e=incoming(r,'别再刷屏了',reply='old-bot',at=now)
    assert observe(r,e,now)['suppression']=='group_feedback_quiet'


def test_at_about_an_annoying_third_party_is_a_normal_direct_request(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=2));bot(r,at=now-timedelta(seconds=30))
    e=incoming(r,'小町，这个面试官话也太多了，怎么应付',mention=True,at=now)
    assert observe(r,e,now)['direct_target']==e.platform_msg_id
    with session_scope(sqlite_engine) as session:
        assert session.get(GroupParticipationFeedback,e.group_id) is None


@pytest.mark.parametrize('text',['小町烦死了','小町吵死了'])
def test_named_annoyance_does_not_require_a_stop_keyword(sqlite_engine,text):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    e=incoming(r,text,at=now)
    assert observe(r,e,now)['suppression']=='group_feedback_quiet'


def test_pronoun_annoyance_binds_to_immediately_preceding_bot(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=2));bot(r,at=now-timedelta(seconds=30))
    e=incoming(r,'你真烦',at=now)
    assert observe(r,e,now)['suppression']=='group_feedback_quiet'


def test_new_feedback_renews_but_old_source_replay_cannot_replace_it(sqlite_engine):
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=2));bot(r,at=now-timedelta(seconds=30))
    first=incoming(r,'开一个闭嘴模式',at=now)
    later=now+timedelta(minutes=5)
    second=incoming(r,'我也想让他闭嘴',msg='second',user=900000102,at=later)
    assert record_feedback(r,second,now=later)
    assert not record_feedback(r,first,now=later)
    with session_scope(sqlite_engine) as session:
        state=session.get(GroupParticipationFeedback,first.group_id)
        assert state.source_message_id==MessageRepository(session).get_by_platform_msg_id(second.platform_msg_id).id
        assert state.expires_at==(later+timedelta(hours=2)).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_scheduled_group_observation_stops_before_paid_or_memory_work(sqlite_engine,monkeypatch):
    from app.agent.autonomous_social import AutonomousSocialPlanner
    from app.agent.social_planner import SocialPlanner
    from app.agent.social_contracts import SocialJob
    r=setup(sqlite_engine);now=datetime.now(UTC)
    incoming(r,'话题',msg='seed',at=now-timedelta(minutes=2));bot(r,at=now-timedelta(seconds=30))
    incoming(r,'开一个闭嘴模式',at=now)
    async def forbidden(*args,**kwargs):
        raise AssertionError('Suppressed scheduled wake must not prepare memory or call a model')
    monkeypatch.setattr(SocialPlanner,'observe',forbidden)
    planner=AutonomousSocialPlanner(router=r,decision_client=object(),qzone_adapter=object())
    try:
        observation=await planner.observe(SocialJob('wake','group',900000001),now)
        assert observation.suppressed_reason=='group_participation_policy'
    finally:
        planner.close()
