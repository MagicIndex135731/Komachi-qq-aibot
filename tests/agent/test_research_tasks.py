import pytest

from app.agent.research_tasks import ResearchTaskWorkspace


def workspace(root, scope='group:900000001', actor=900000101, text='分析这个仓库有哪些漏洞', valid=lambda _: True):
    return ResearchTaskWorkspace(root, scope=scope, actor_id=actor, source_id=123,
                                 request=text, valid_source=valid)


def test_task_keeps_progress_and_evidence_out_of_other_people_and_chat(tmp_path):
    task = workspace(tmp_path)
    task.update(progress='已查启动文件', next_step='检查认证', findings=[], status='active')
    ref = task.save_evidence({'content': 'L1: actual source', 'sources': [{'source': 'https://example.invalid'}]})
    assert task.read_evidence(ref)['content'] == 'L1: actual source'
    assert task.summary()['next_step'] == '检查认证'
    with pytest.raises(ValueError):
        workspace(tmp_path, actor=900000102).read_evidence(ref)
    chat = workspace(tmp_path, text='今天吃啥')
    assert not chat.formal
    assert 'actual source' not in str(chat.recent_summary())


def test_main_conversation_owns_routing_until_model_explicitly_begins_task(tmp_path):
    task = workspace(tmp_path, text='请分析这个仓库')
    assert task.formal is False
    assert task.context_scope == 'group:900000001'
    task.begin('主对话决定需要逐项核对')
    assert task.formal is True
    assert task.context_scope.startswith('research:group:900000001:')


def test_revoked_task_cannot_be_resumed_or_read(tmp_path):
    task = workspace(tmp_path)
    ref = task.save_evidence({'content': 'revoked source'})
    task.valid_source = lambda _: False
    with pytest.raises(ValueError):
        task.read_evidence(ref)
    with pytest.raises(ValueError):
        task.summary()


def test_restore_admits_current_original_and_dependencies_in_one_transaction(tmp_path):
    old = workspace(tmp_path)
    old.begin()
    admitted = []
    restored = ResearchTaskWorkspace(tmp_path, scope=old.scope, actor_id=old.actor_id, source_id=124,
        request='continue', valid_source=lambda _: True,
        valid_dependencies=lambda metadata: admitted.append(metadata['sources']) or True)
    assert restored.recent_summary()['goal'] == old.request
    assert admitted == [[124, 123]]


def test_continuation_stays_in_main_until_task_explicitly_selected(tmp_path):
    task = workspace(tmp_path)
    task.update(progress='checked', next_step='continue', findings=[], status='active')
    continuation = ResearchTaskWorkspace(tmp_path, scope='group:900000001', actor_id=900000101,
        source_id=124, request='继续刚才那个仓库', valid_source=lambda _: True)
    assert continuation.task_id != task.task_id
    assert continuation.context_scope == 'group:900000001'
    continuation.resume(task.task_id)
    assert continuation.task_id == task.task_id
    assert continuation.context_scope == task.context_scope
    with pytest.raises(ValueError):
        workspace(tmp_path, actor=900000102).resume(task.task_id)


def test_evidence_ranges_are_bounded_and_path_traversal_is_rejected(tmp_path):
    task = workspace(tmp_path)
    ref = task.save_evidence({'content': 'x' * 50000})
    assert len(task.read_evidence(ref, offset=20000, max_chars=16000)['content']) == 16000
    with pytest.raises(ValueError):
        task.read_evidence('../checkpoint')


def test_expired_and_changed_memory_dependencies_cannot_restore_a_checkpoint(tmp_path):
    current = [100.0]
    revision = ['r1']
    task = ResearchTaskWorkspace(tmp_path, scope='group:900000001', actor_id=900000101, source_id=123,
        request='认真审查', valid_source=lambda _: True, clock=lambda: current[0],
        dependency_snapshot=lambda: {'revision': revision[0]},
        valid_dependencies=lambda data: data['revision'] == revision[0])
    task.update(progress='source bound',next_step='continue',findings=[],status='active')
    revision[0] = 'revoked'
    with pytest.raises(ValueError): task.summary()
    revision[0] = 'r1';current[0] += 8 * 86400
    with pytest.raises(ValueError): task.summary()


def test_delivery_preserves_results_without_claiming_complete_or_importing_raw_web(tmp_path):
    task=workspace(tmp_path)
    task.begin()
    task.record_delivery('已核对入口；认证代码尚未检查', 'sent')
    assert task.summary()['status'] == 'active'
    assert '认证代码' in task.prompt_summary()['progress']


def test_first_archived_page_already_depends_on_quoted_source_before_model_checkpoint(tmp_path):
    valid=[True]
    task=ResearchTaskWorkspace(tmp_path,scope='private:900000101',actor_id=900000101,source_id=1,
        request='审查引用来源',valid_source=lambda _:True,
        dependency_snapshot=lambda:{'sources':[1,2]},valid_dependencies=lambda _:valid[0])
    ref=task.save_evidence({'content':'QUOTED_SOURCE_EVIDENCE'})
    valid[0]=False
    with pytest.raises(ValueError):task.read_evidence(ref)


@pytest.mark.asyncio
async def test_archive_reads_share_the_live_web_content_budget(tmp_path):
    class Guard:
        def require_valid(self):pass
    task=workspace(tmp_path)
    ref=task.save_evidence({'content':'x'*32000})
    task.web_result_chars=95000
    read=next(tool for tool in task.tools(Guard()) if tool.name=='read_research_evidence')
    result=await read.handler({'evidence_id':ref,'max_chars':32000})
    assert result.status=='blocked' and 'budget' in result.data['error']
