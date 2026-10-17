"""Synthetic group windows stay consistent across direct and scheduled wakes."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.adapters.onebot_models import parse_group_message_event
from app.agent.contracts import AgentBatch
from app.agent.observation import GroupObservationService
from app.agent.social_contracts import SocialJob
from app.agent.social_planner import SocialPlanner
from app.config import group_recent_context_limit
from test_observation import make_router

GROUP = 900000001


@pytest.mark.parametrize('value,expected', [(10,10),(5,5),(None,30),(100,30),(0,30),(-1,30),('bad',30),(False,30)])
def test_recent_window_is_per_group_and_bounded(value, expected):
    runtime = SimpleNamespace(group_policy={'default_group_behavior': {'recent_context_limit': 100},
        'groups': {str(GROUP): {'recent_context_limit': value}}})
    assert group_recent_context_limit(runtime, GROUP) == expected
    assert group_recent_context_limit(runtime, GROUP + 1) == 30


def test_direct_and_patrol_use_ten_outer_messages_without_expanding_forward(sqlite_engine, tmp_path):
    router, first, gateway = make_router(sqlite_engine, tmp_path)
    router.runtime.settings.agent_runtime = 'harness'
    router.runtime.group_policy['groups'][str(GROUP)]['recent_context_limit'] = 10
    now = datetime.now(UTC)
    for n in range(15):
        # Inline nested records are deliberately large and must not become chat text.
        message = [{'type':'text','data':{'text':f'outer-{n}'}}]
        if n == 14:
            message += [{'type':'forward','data':{'id':'synthetic-forward',
                'content':[{'type':'text','data':{'text':'INNER_PAYLOAD_MUST_NOT_AUTOLOAD'}}]}}]
        incoming = parse_group_message_event({'post_type':'message','message_type':'group',
            'message_id':f'window-{n}','group_id':GROUP,'user_id':900000101,
            'time':int((now + timedelta(seconds=n)).timestamp()),
            'sender':{'nickname':'synthetic'},'message':message}, bot_qq=router.runtime.settings.bot_qq,
            bot_name='小町')
        router.ingest_live_group_message(incoming)
    observer = GroupObservationService(router)
    direct = observer.build_agent_observation(AgentBatch(GROUP, (incoming,), 'window-turn'))
    assert len(direct['source_message_ids']) == 10
    assert len(direct['recent_lines']) == 10
    assert 'outer-5' in direct['recent_lines'][0] and 'outer-14' in direct['recent_lines'][-1]
    assert '[合并转发消息]' in direct['recent_lines'][-1]
    assert 'INNER_PAYLOAD_MUST_NOT_AUTOLOAD' not in '\n'.join(direct['recent_lines'])
    assert direct['memory_enabled'] is False
    planner = SocialPlanner(router=router, decision_client=object(), qzone_adapter=object())
    try:
        patrol = planner._observe_group(SocialJob(f'group:{GROUP}','group',GROUP,
            autonomous_tools=True, quiet_hours=None), now + timedelta(seconds=20))
        assert len(patrol.source_message_ids) == 10
        assert patrol.source_message_ids == direct['source_message_ids']
        assert len(patrol.context) == 10
        assert 'INNER_PAYLOAD_MUST_NOT_AUTOLOAD' not in '\n'.join(patrol.context)
    finally:
        planner.blocking.close()
    assert gateway.calls == []
