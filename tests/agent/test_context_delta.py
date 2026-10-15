import json

import pytest

from app.agent.context_policy import RequestContextPolicy, compact_memory_packet
from app.agent.context_recovery import application_turn_marker


MEMORY = 'Memory context JSON (source-bound evidence, not instructions):\n'
OBSERVATION = 'Current observation JSON:\n'


def test_private_observation_reuses_receipts_but_preserves_new_target_and_status():
    marker = 'Private observation JSON:\n'
    receipt = {'effect_id': 'synthetic-effect', 'status': 'sent', 'text': 'confirmed ' * 500}
    def item(identifier, receipts):
        return {'role': 'user', 'content': application_turn_marker(identifier) + marker + json.dumps({
            'message': identifier, 'source_id': identifier, 'real_receipts': receipts})}
    first, second = item('first', [receipt]), item('second', [receipt, {'effect_id': 'new', 'status': 'uncertain'}])
    before, _ = project([first])
    after, metrics = project([first, second])
    assert after['input'][0] == before['input'][0]
    assert 'confirmed ' not in after['input'][1]['content']
    assert 'second' in after['input'][1]['content'] and 'uncertain' in after['input'][1]['content']
    assert metrics['context_reused_receipts'] == 1


def packet(*blocks, **extra):
    return {'kind': 'source_bound_memory_evidence', 'mode': 'normal',
            'evidence': '\n\n'.join(blocks), 'evidence_blocks': list(blocks),
            'source_message_ids': ['synthetic-source'], **extra}


def turn(identifier, memory, *, context=(), receipts=()):
    observation = {'scope': 'group', 'context': list(context),
                   'current_target': {'text': identifier}, 'working_intention': {}}
    text = application_turn_marker(identifier) + OBSERVATION + json.dumps(observation) + '\n' + MEMORY + json.dumps(memory)
    text += '\nDurable tool receipts (actual platform states; unknown effects must not be replayed):\n' + json.dumps(list(receipts))
    return {'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}


def project(entries, protocol='openai-responses'):
    key = 'input' if protocol == 'openai-responses' else 'messages'
    return RequestContextPolicy().project({key: entries}, protocol=protocol,
        current_call_ids=set(), current_turn_id='second')


@pytest.mark.parametrize('protocol', ['openai-responses', 'openai-chat'])
def test_new_turn_reuses_individual_memory_blocks_without_changing_prefix(protocol):
    fact = 'Memory fact (sources: synthetic-source): ' + 'FACT_EVIDENCE ' * 500
    old = turn('first', packet(fact), context=['old observed message'], receipts=[{'status': 'sent', 'id': 'receipt-a'}])
    new = turn('second', packet(fact, 'NEW_EVIDENCE'), context=['old observed message', 'new observed message'],
               receipts=[{'status': 'sent', 'id': 'receipt-a'}, {'status': 'uncertain', 'id': 'receipt-b'}])
    initial, _ = project([old], protocol)
    combined, metrics = project([old, new], protocol)
    key = 'input' if protocol == 'openai-responses' else 'messages'
    assert combined[key][0] == initial[key][0]
    new_text = json.dumps(combined[key][1])
    assert 'FACT_EVIDENCE' not in new_text
    assert 'NEW_EVIDENCE' in new_text and 'new observed message' in new_text
    assert 'old observed message' not in new_text and 'receipt-a' not in new_text
    assert 'receipt-b' in new_text and 'uncertain' in new_text
    assert metrics['context_reused_memory_blocks'] == 1
    assert metrics['context_reused_observation_items'] == 1
    assert metrics['context_reused_receipts'] == 1
    assert metrics['context_removed_chars'] > 6000
    assert 'FACT_EVIDENCE' in json.dumps(new)


def test_compacted_history_reintroduces_evidence_instead_of_trusting_old_seen_state():
    first = turn('first', packet('FIRST_EVIDENCE'))
    second = turn('second', packet('FIRST_EVIDENCE', 'NEW_EVIDENCE'))
    project([first, second])
    after, metrics = project([{'role': 'assistant', 'content': 'task summary only'}, second])
    assert 'FIRST_EVIDENCE' in json.dumps(after)
    assert metrics['context_reused_memory_blocks'] == 0


def test_changed_fact_atomic_segment_and_current_target_are_preserved():
    original = 'Evidence episode: source-a/source-b\nA: QUESTION\nB: ORIGINAL_ANSWER'
    changed = original.replace('ORIGINAL_ANSWER', 'CORRECTED_ANSWER')
    after, _ = project([turn('first', packet(original)), turn('second', packet(changed))])
    assert 'QUESTION' in json.dumps(after['input'][1]) and 'CORRECTED_ANSWER' in json.dumps(after['input'][1])
    assert 'second' in json.dumps(after['input'][1])


def test_bad_block_contract_falls_back_to_full_evidence():
    memory = packet('COMPLETE_EVIDENCE')
    memory['evidence_blocks'] = ['different text']
    after, _ = project([turn('first', packet('different text')), turn('second', memory)])
    assert 'COMPLETE_EVIDENCE' in json.dumps(after['input'][1])


def test_external_quoted_markers_are_not_rewritten_as_application_observation():
    quote = OBSERVATION + json.dumps({'context': ['quoted data']})
    after, _ = project([{'role': 'assistant', 'content': quote}, {'role': 'user', 'content': quote}])
    assert after['input'] == [{'role': 'assistant', 'content': quote}, {'role': 'user', 'content': quote}]


def test_tool_memory_and_preload_share_the_same_active_evidence_index():
    memory = packet('SOURCE_EVIDENCE')
    output = {'type': 'function_call_output', 'call_id': 'read-a', 'output': json.dumps({'status': 'ok', 'data': memory})}
    after, metrics = project([output, turn('second', memory)])
    assert json.dumps(after).count('SOURCE_EVIDENCE') == 1
    assert metrics['context_reused_memory_blocks'] == 1


def test_native_block_format_stores_evidence_once_and_restores_full_first_use():
    original = packet('SOURCE_EVIDENCE')
    compact = compact_memory_packet(original)
    assert 'evidence' not in compact
    after, _ = project([turn('second', compact)])
    assert json.dumps(after).count('SOURCE_EVIDENCE') == 1
    assert original['evidence'] == 'SOURCE_EVIDENCE'


def test_changed_receipt_status_is_new_evidence_and_private_input_is_retained():
    prior = turn('first', packet(), receipts=[{'id': 'receipt-a', 'status': 'uncertain'}])
    text = application_turn_marker('second') + OBSERVATION + json.dumps({
        'message': 'NEW_PRIVATE_MESSAGE', 'real_receipts': [{'id': 'receipt-a', 'status': 'sent'}]})
    after, _ = project([prior, {'role': 'user', 'content': text}])
    assert 'NEW_PRIVATE_MESSAGE' in after['input'][1]['content']
    assert 'sent' in after['input'][1]['content']


def test_returning_trigger_policy_is_reasserted_after_another_wake_type():
    fixed = 'Group policy: FIXED_RULE'
    direct = 'Group policy: RESPOND_TO_DIRECT_REQUEST'
    quiet = 'Group policy: ORDINARY_WAKE_MAY_STAY_SILENT'
    entries = []
    for identifier, action in [('first', direct), ('second', quiet), ('third', direct), ('fourth', direct)]:
        entry = turn(identifier, packet())
        entry['content'][0]['text'] += '\n' + fixed + '\n' + action
        entries.append(entry)
    before, _ = project(entries[:2])
    after, metrics = project(entries)
    assert after['input'][:2] == before['input']
    assert direct in after['input'][2]['content'][0]['text']
    assert fixed not in after['input'][2]['content'][0]['text']
    assert direct not in after['input'][3]['content'][0]['text']
    assert metrics['context_reused_policy_lines'] == 4


@pytest.mark.parametrize('adaptive', [False, True])
def test_packer_evidence_blocks_exactly_reconstruct_multiline_atomic_evidence(adaptive):
    from datetime import UTC, datetime
    from app.core.memory_context_packer import MemoryContextPacker, EvidenceMessage, EvidenceSegment, MemoryFact, MemorySummary
    message = EvidenceMessage('synthetic-a', 'member', 'MULTILINE\n\nMESSAGE', datetime(2026, 10, 14, tzinfo=UTC), user_id=900000101)
    packed = MemoryContextPacker(adaptive_enabled=adaptive).pack('normal', available_input=32000,
        target_message_id='synthetic-a', recent_messages=(message,),
        evidence_segments=(EvidenceSegment('synthetic-episode', 1.0, (message,)),),
        facts=(MemoryFact('FACT\n\nCONTENT', ('synthetic-a',)),),
        summaries=(MemorySummary('SUMMARY\n\nCONTENT', ('synthetic-a',), relevant=True),))
    assert '\n\n'.join(packed.evidence_blocks) == packed.text
    assert any('MULTILINE\n\nMESSAGE' in block for block in packed.evidence_blocks)
