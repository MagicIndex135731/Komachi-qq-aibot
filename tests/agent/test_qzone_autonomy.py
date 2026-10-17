"""Autonomous publication can stop while canonical owner commands still work."""
import pytest

from app.agent.owner_actions import OwnerActionService, OwnerActionWorker
from app.agent.public_tools import build_public_tools
from app.agent.social_policy import configured_scheduled_jobs, configured_social_jobs
from tests.agent.test_social_policy import runtime


def test_disabled_autonomy_preserves_capabilities_and_comment_schedule():
    config = runtime({'enabled': True, 'qzone': {
        'publish': {'enabled': True, 'autonomous_enabled': False, 'topics': ['音乐']},
        'comment': {'enabled': True, 'friend_ids': [900000103]}}})
    assert [j.kind for j in configured_social_jobs(config)] == ['qzone_publish', 'qzone_comment']
    assert [j.kind for j in configured_scheduled_jobs(config)] == ['qzone_comment']
    config.social_policy['qzone']['publish'].pop('autonomous_enabled')
    assert [j.kind for j in configured_scheduled_jobs(config)] == ['qzone_publish', 'qzone_comment']


@pytest.mark.parametrize('value', ['false', 0, None])
def test_autonomy_flag_rejects_ambiguous_values(value):
    config = runtime({'enabled': True, 'qzone': {'publish': {
        'enabled': True, 'autonomous_enabled': value, 'topics': ['音乐']}}})
    with pytest.raises(ValueError, match='autonomous_enabled'):
        configured_social_jobs(config)


async def test_existing_public_tool_rechecks_autonomy_before_dispatch(sqlite_engine):
    from tests.agent.test_current_friend_permissions import setup
    from tests.agent.test_social_executor import Guard
    executor, observation = setup(sqlite_engine, object())
    publish = {'enabled': True, 'topics': ['音乐'], 'quiet_hours': None}
    executor.router.runtime.social_policy['qzone']['publish'] = publish
    tool = next(t for t in build_public_tools(executor, observation, Guard(), object())
                if t.name == 'publish_qzone')
    publish['autonomous_enabled'] = False
    with pytest.raises(ValueError, match='public_tool_autonomy_disabled'):
        await tool.handler({'text': '合成公开表达', 'artifact_ids': []})


@pytest.mark.parametrize('private_origin', [False, True])
async def test_owner_can_publish_with_autonomy_disabled(sqlite_engine, tmp_path, private_origin):
    from tests.agent.test_autonomous_social import composed, GROUP, NOW, writes
    from tests.agent.test_mature_composition import model_steps
    from tests.agent.test_owner_actions import source
    client, requests, _ = model_steps(
        ('publish_qzone', {'text': '主人要求公开的合成内容', 'artifact_ids': []}),
        ('report_owner_result', {'text': '发布已获平台回执', 'artifact_ids': []}), None)
    async with composed(sqlite_engine, tmp_path, client, kind='qzone_publish', runtime='harness',
                        options={'autonomous_enabled': False}) as fixture:
        await fixture.agent.stop()
        assert not any(j.kind == 'qzone_publish' for j in fixture.agent.jobs())
        service = OwnerActionService(sqlite_engine, runtime=fixture.router.runtime, clock=lambda: NOW)
        sid = source(sqlite_engine, author=fixture.router.runtime.settings.owner_qq,
                     group=None if private_origin else GROUP, text='发布这条合成空间内容')
        assert service.enqueue(sid, target='public')['status'] == 'queued'
        result = await OwnerActionWorker(service, fixture.executor, connected=lambda: True).run_once()
        assert result['status'] == 'sent', result
        assert [name for name, _ in writes(fixture)] == ['send_qzone_msg']
        assert requests
