"""Own public actions and explicit working thoughts survive independent tasks."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from app.agent.public_tools import build_public_tools
from app.agent.tool_agent import AgentTool, ToolOutcome
from tests.agent.test_current_friend_permissions import setup
from tests.agent.test_social_executor import Guard


def test_public_tool_catalog_offers_bounded_shared_history(sqlite_engine):
    executor, obs = setup(sqlite_engine, object())
    tools = {t.name: t for t in build_public_tools(executor, obs, Guard(), object())}
    assert 'read_social_activity' in tools
    assert tools['read_social_activity'].parameters['properties']['limit']['maximum'] <= 10


async def test_public_thoughts_are_archived_but_owner_private_notes_are_not(sqlite_engine):
    from app.agent.tool_dispatch import ToolDispatchBoundary
    from app.storage.public_social import PublicSocialRepository
    from tests.agent.test_autonomous_social import BOT, NOW
    async def save(args):
        return ToolOutcome({'saved': True})
    tool = AgentTool('remember_intention', '', {}, save)
    for scope, text in [('public', '第一份公开打算'), ('public', '第二份公开打算'),
                        ('owner:public:42', 'PRIVATE_OWNER_INSTRUCTION'), ('group:900000001', 'PRIVATE_GROUP')]:
        boundary = ToolDispatchBoundary(engine=sqlite_engine, account_id=BOT, scope=scope,
            guard=Guard(), clock=lambda: NOW)
        await boundary.invoke(tool, {'goal': text, 'progress': '已看到公开原帖', 'next_step': '等原帖更新再核对'}, text)
    await boundary.invoke(tool, {'goal': '空间例行巡检', 'progress': '', 'next_step': '等待下次统一巡检唤醒'}, 'routine')
    data = PublicSocialRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW).read(kind='thought')
    encoded = json.dumps(data, ensure_ascii=False)
    assert '第一份公开打算' in encoded and '第二份公开打算' in encoded
    assert 'PRIVATE_' not in encoded
    assert '空间例行巡检' not in encoded


async def test_native_failed_comment_is_recovered_by_separate_owner_task_without_feed(sqlite_engine, tmp_path):
    from app.agent.owner_actions import OwnerActionService, OwnerActionWorker
    from tests.agent.test_autonomous_social import composed, wake, writes, FRIEND, GROUP, NOW
    from tests.agent.test_mature_composition import model_steps, latest
    from tests.agent.test_owner_actions import source
    draft = '这条合成公开评论此前尚未派发'
    def failed_comment(body):
        return 'comment_qzone', {'owner_id': FRIEND, 'post_id': 'observed-photo', 'text': draft, 'artifact_ids': []}
    def remember(body):
        assert latest(body, 'comment_qzone')['status'] == 'blocked'
        return 'remember_intention', {'goal': '保留这条公开评论', 'progress': '已准备评论文字', 'next_step': '复验原帖后完成'}
    def detail(body):
        records = latest(body, 'read_social_activity')['data']['records']
        row = next(r for r in records if r['kind'] == 'comment')
        assert row['status'] == 'blocked' and row['write_dispatched'] is False
        return 'read_social_activity', {'record_id': row['record_id']}
    def read_original(body):
        row = latest(body, 'read_social_activity')['data']['record']
        assert row['payload']['text'] == draft
        return 'read_friend_posts', {'owner_id': row['owner_id']}
    client, requests, _ = model_steps(
        ('read_friend_posts', {'owner_id': FRIEND}), failed_comment, remember, None,
        ('read_social_activity', {'kind': 'comment'}), detail, read_original, failed_comment,
        ('report_owner_result', {'text': '已取得合成评论回执', 'artifact_ids': []}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime='harness', kind='qzone_comment') as fixture:
        original = fixture.executor.legacy.policy_check
        fixture.executor.legacy.policy_check = lambda obs, decision, **kw: False
        await wake(fixture)
        assert not writes(fixture)
        fixture.executor.legacy.policy_check = original
        await fixture.agent.stop()
        service = OwnerActionService(sqlite_engine, runtime=fixture.router.runtime, clock=lambda: NOW)
        sid = source(sqlite_engine, author=fixture.router.runtime.settings.owner_qq, group=None,
                     text='把早上尚未派发的空间评论重发')
        assert service.enqueue(sid, target='public')['status'] == 'queued'
        await OwnerActionWorker(service, fixture.executor, connected=lambda: True).run_once()
        assert [name for name, _ in writes(fixture)] == ['comment_qzone']
        assert all(call[0] != 'get_qzone_feeds' for call in fixture.gateway.calls)
        assert 'PRIVATE_PERSONA_FACT' not in json.dumps(requests)


async def test_owner_public_cache_key_is_stable_and_records_are_not_preloaded(sqlite_engine, tmp_path):
    from app.agent.owner_actions import OwnerActionService, OwnerActionWorker
    from app.storage.public_social import PublicSocialRepository
    from tests.agent.test_autonomous_social import composed, NOW, BOT
    from tests.agent.test_mature_composition import model_steps
    from tests.agent.test_owner_actions import source
    client, requests, _ = model_steps(
        ('report_owner_result', {'text': '合成查询完成', 'artifact_ids': []}), None,
        ('report_owner_result', {'text': '合成查询完成', 'artifact_ids': []}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime='harness', kind='qzone_comment') as fixture:
        await fixture.agent.stop()
        service = OwnerActionService(sqlite_engine, runtime=fixture.router.runtime, clock=lambda: NOW)
        for n in range(2):
            PublicSocialRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW).save(
                event_key=str(n), kind='thought', operation='save_work_note', status='ok',
                payload={'note': 'UNLOADED_PUBLIC_NOTE_' + str(n) + '长' * 1000})
            sid = source(sqlite_engine, author=fixture.router.runtime.settings.owner_qq, group=None,
                         text='查看公开评论情况' + str(n))
            service.enqueue(sid, target='public')
            await OwnerActionWorker(service, fixture.executor, connected=lambda: True).run_once()
        assert len(requests) == 4
        left, right = requests[0], requests[2]
        assert left['tools'] == right['tools']
        assert left['input'][0] == right['input'][0]
        assert left['prompt_cache_key'] == right['prompt_cache_key']
        assert 'UNLOADED_PUBLIC_NOTE_' not in json.dumps(requests)
        from app.agent.context_policy import estimate_text_tokens
        proof = {'same_tool_definitions': left['tools'] == right['tools'],
                 'same_system_prefix': left['input'][0] == right['input'][0],
                 'same_cache_key_across_owner_tasks': left['prompt_cache_key'] == right['prompt_cache_key'],
                 'public_records_preloaded': False, 'real_channel_cache_hit_measured': False,
                 'first_request_estimated_tokens': estimate_text_tokens(json.dumps(left, ensure_ascii=False)),
                 'second_request_estimated_tokens': estimate_text_tokens(json.dumps(right, ensure_ascii=False)),
                 'history_tool_definition_estimated_tokens': estimate_text_tokens(json.dumps(
                     next(t for t in left['tools'] if t['name'] == 'read_social_activity'), ensure_ascii=False))}
        (tmp_path/'cache-proof.json').write_text(json.dumps(proof, indent=2))


@pytest.mark.parametrize('fail_before_write', [True, False])
async def test_history_failure_never_duplicates_a_confirmed_write(sqlite_engine, monkeypatch, fail_before_write):
    from app.agent.tool_dispatch import ToolDispatchBoundary
    from app.storage.public_social import PublicSocialRepository
    from tests.agent.test_autonomous_social import BOT, NOW, FRIEND
    boundary = ToolDispatchBoundary(engine=sqlite_engine, account_id=BOT, scope='public',
        guard=Guard(), clock=lambda: NOW)
    sent = []
    async def write(args):
        boundary.on_dispatch()
        sent.append(args)
        return ToolOutcome({}, 'sent', 'synthetic-receipt')
    tool = AgentTool('comment_qzone', '', {}, write, 'write', scope='public:qzone_comment')
    original = PublicSocialRepository.save
    calls = []
    def save(self, **args):
        calls.append(args)
        if fail_before_write or len(calls) > 1:
            raise ValueError('synthetic_projection_failure')
        return original(self, **args)
    monkeypatch.setattr(PublicSocialRepository, 'save', save)
    args = {'owner_id': FRIEND, 'post_id': 'synthetic', 'text': 'public draft', 'artifact_ids': []}
    result = await boundary.invoke(tool, args, 'first')
    assert result.status == ('failed' if fail_before_write else 'sent')
    assert len(sent) == (0 if fail_before_write else 1)
    if not fail_before_write:
        duplicate = await boundary.invoke(tool, args, 'second')
        assert duplicate.status == 'blocked' and len(sent) == 1
        assert PublicSocialRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW).read(
            kind='comment')['records'][0]['status'] == 'sent'
