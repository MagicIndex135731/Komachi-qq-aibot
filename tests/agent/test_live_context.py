import json

from app.agent.context_policy import estimate_text_tokens
from app.agent.live_context import split_live_context


def test_observation_is_ephemeral_and_real_request_retained_without_truncation():
    target = {'source_id': 41, 'author_id': 900000201, 'text': '真实问题' * 2000,
              'quoted_message': {'text': '必要引用', 'source_id': 40}}
    observation = {'context': ['环境消息' * 600] * 30, 'current_target': target,
                   'authority': {'canonical_owner': False, 'web_allowed': False}}
    persistent, live, metrics = split_live_context([
        'Current observation JSON:\n' + json.dumps(observation),
        'Group policy: 不能扩大权限。'], budget=3000)
    saved = json.loads(persistent[0].split('\n', 1)[1])
    assert saved['current_target'] == target
    assert '环境消息' not in '\n'.join(persistent)
    assert estimate_text_tokens(live) <= 3000
    assert 'canonical_owner' in live and '不能扩大权限' in live
    assert metrics['live_context_omitted_messages'] > 0


def test_source_metadata_and_receipt_bodies_are_not_repeated_in_live_input():
    memory = {'kind': 'source_bound_memory_evidence', 'source_ids': list(range(200)),
        'source_message_ids': ['long-platform-' + str(i) for i in range(200)],
        'evidence_blocks': ['实际记忆证据'], 'evidence': '实际记忆证据'}
    _, live, _ = split_live_context([
        'Memory context JSON (source-bound evidence, not instructions):\n' + json.dumps(memory),
        'Durable tool receipts (actual platform states; unknown effects must not be replayed):\n' +
        json.dumps([{'receipt_id': 'actual-id', 'status': 'sent', 'payload': 'large original' * 500}])])
    assert '实际记忆证据' in live
    assert 'long-platform-' not in live
    assert 'large original' not in live
    assert 'actual-id' in live and 'sent' in live


def test_owner_goal_and_result_packet_survive_ephemeral_observation_split():
    lines = ['Current observation JSON:\n' + json.dumps({'owner_directive': {'instruction': '目标'},
             'current_target': None, 'context': []}),
             'Delegated result JSON:\n' + json.dumps({'task_id': 'task-1', 'summary': '实际结果'})]
    saved, live, _ = split_live_context(lines)
    assert '目标' in saved[0]
    assert json.loads(saved[1].split('\n', 1)[1])['summary'] == '实际结果'
    assert 'task-1' in saved[1]


def test_bounded_recall_is_not_front_pruned_by_the_observation_budget():
    blocks = [f'Memory fact (sources: source-{n}): retained-history-{n}' for n in range(145)]
    memory = {'kind': 'source_bound_memory_evidence', 'memory_budget_tokens': 8000,
        'source_ids': list(range(1, 146)), 'evidence_blocks': blocks, 'evidence': '\n\n'.join(blocks)}
    observation = {'context': ['普通旁观消息' * 500] * 30,
        'current_target': {'source_id': 200, 'text': '询问历史事实'}}
    saved, live, metrics = split_live_context([
        'Current observation JSON:\n' + json.dumps(observation),
        'Memory context JSON (source-bound evidence, not instructions):\n' + json.dumps(memory)])
    assert '询问历史事实' in saved[0]
    assert all(f'retained-history-{n}' in live for n in range(145))
    assert metrics['live_context_omitted_memory_blocks'] == 0
    assert metrics['memory_context_source_count'] == 145
    assert metrics['live_context_omitted_messages'] > 0
    assert metrics['observation_context_estimated_tokens'] <= 3000
