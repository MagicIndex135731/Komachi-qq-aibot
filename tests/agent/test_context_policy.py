import json

from app.agent.context_policy import (
    RequestContextPolicy, is_task_request, estimate_text_tokens, serious_request_policy, estimate_request_tokens,
)


def test_serious_request_rules_preserve_style_but_require_delivery():
    text = serious_request_policy()
    assert '完成' in text and '短句' in text and '证据' in text
    assert is_task_request('帮我认真分析这个仓库有哪些漏洞 https://github.com/example/repo')
    assert is_task_request('整理一下这两个方案的区别，给出理由')
    assert not is_task_request('哈哈杂鱼，今天吃啥')
    assert not is_task_request('https://example.invalid/ 普通分享')


def test_unicode_and_schema_tokens_are_not_priced_as_four_chars_each():
    assert estimate_text_tokens('汉' * 100) >= 100
    assert estimate_text_tokens('{"x":1,"y":2}' * 10) > 40


def result(call_id, body):
    return {'type': 'function_call_output', 'call_id': call_id,
            'output': json.dumps({'status': 'ok', 'data': body}, ensure_ascii=False)}


def test_previous_web_bulk_is_offloaded_while_current_evidence_is_retained():
    data = {'trust': 'untrusted_external_sources', 'content': 'CODE_BULK_' * 2000,
            'sources': [{'source': 'https://example.invalid/code'}],
            'task_id': 'synthetic-task', 'evidence_id': 'synthetic-evidence'}
    body = {'instructions': 'stable persona', 'input': [result('old', data),
            {'role': 'user', 'content': 'CURRENT_TARGET'}, result('new', data)], 'tools': []}
    projected, metrics = RequestContextPolicy().project(body, protocol='openai-responses', current_call_ids={'new'})
    assert 'CODE_BULK_' not in projected['input'][0]['output']
    assert 'synthetic-evidence' in projected['input'][0]['output']
    assert 'CODE_BULK_' in projected['input'][2]['output']
    assert projected['input'][1] == body['input'][1]
    assert metrics['context_removed_chars'] > 10000
    assert 'CODE_BULK_' in body['input'][0]['output']


def test_equal_memory_evidence_is_admitted_only_once_without_changing_original():
    memory = {'kind': 'source_bound_memory_evidence', 'evidence': 'SOURCE_BOUND_FACT',
              'source_message_ids': ['synthetic-source']}
    body = {'input': [result('a', memory), result('b', memory)], 'tools': []}
    projected, _ = RequestContextPolicy().project(body, protocol='openai-responses', current_call_ids=set())
    assert json.dumps(projected).count('SOURCE_BOUND_FACT') == 1
    assert json.dumps(body).count('SOURCE_BOUND_FACT') == 2


def test_same_current_request_has_identical_stable_projection():
    body = {'instructions': 'STABLE', 'input': [result('old', {
        'trust': 'untrusted_external_sources', 'content': 'old body', 'sources': []})], 'tools': []}
    policy = RequestContextPolicy()
    first, _ = policy.project(body, protocol='openai-responses', current_call_ids=set())
    second, _ = policy.project(body, protocol='openai-responses', current_call_ids=set())
    assert first == second
    assert first['instructions'] == body['instructions']


def test_native_rewritten_call_id_keeps_fresh_evidence_and_deduplicates_repeated_range():
    data = {'trust': 'untrusted_external_sources', 'content': 'FRESH_SOURCE' * 100,
            'sources': [], 'context_turn_id': 'current'}
    body = {'input': [result('native-rewritten-a', data), result('native-rewritten-b', data)]}
    projected, _ = RequestContextPolicy().project(body, protocol='openai-responses',
        current_call_ids={'original-a'}, current_turn_id='current')
    assert json.dumps(projected).count('FRESH_SOURCE') == 100
    assert 'already_in_context' in projected['input'][1]['output']


def test_image_base64_is_not_mispriced_as_a_million_text_tokens():
    body = {'input': [{'role': 'user', 'content': [{'type': 'input_image',
        'image_url': 'data:image/png;base64,' + 'A' * 3000000}]}]}
    assert 4096 <= estimate_request_tokens(body) < 5000
