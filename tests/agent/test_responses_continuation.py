from copy import deepcopy

from app.agent.responses_continuation import ResponsesContinuation, continuation_supported


def body(items):
    return {'model': 'offline-agent', 'instructions': 'fixed', 'tools': [], 'input': items, 'stream': True}


def completed(identifier='resp-one', text='answer'):
    return {'id': identifier, 'status': 'completed', 'output': [{'type': 'message', 'id': 'msg-one',
        'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}]}


def test_unknown_compatible_providers_require_explicit_capability_opt_in():
    assert not continuation_supported('auto', 'https://synthetic.invalid/v1', 'openai-responses')
    assert continuation_supported('auto', 'https://api.openai.com/v1', 'openai-responses')
    assert continuation_supported('on', 'https://synthetic.invalid/v1', 'openai-responses')
    assert not continuation_supported('on', 'https://api.openai.com/v1', 'openai-completions')


def test_provider_storage_opt_out_always_keeps_full_local_replay():
    state = ResponsesContinuation(enabled=True)
    initial = body([{'role':'user','content':'first'}])
    state.prepare(initial); response = completed(); state.complete(response)
    second = {**initial, 'input':[*initial['input'], *response['output']], 'store':False}
    wire, full, meta = state.prepare(second)
    assert wire == full and 'previous_response_id' not in wire and wire['store'] is False
    assert meta['continuation_reason'] == 'response_storage_disabled'


def test_append_only_history_uses_delta_and_full_replay_matches_provider_context():
    state = ResponsesContinuation(enabled=True)
    first = body([{'role': 'user', 'content': 'first'}])
    wire, _, metrics = state.prepare(first)
    assert metrics['continuation_mode'] == 'full'
    response = completed(); state.complete(response)
    replay = {'role': 'assistant', 'content': response['output'][0]['content']}
    second = body([*first['input'], replay, {'role': 'user', 'content': 'second'}])
    snapshot = deepcopy(second)
    wire, full, metrics = state.prepare(second)
    assert second == snapshot
    assert wire['previous_response_id'] == 'resp-one'
    assert wire['input'] == [{'role': 'user', 'content': 'second'}]
    assert full['input'] == [*first['input'], *response['output'], *wire['input']]


def test_live_clock_and_observations_are_kept_in_bounded_current_turn_fallback():
    state = ResponsesContinuation(enabled=True)
    canonical = [{'role': 'user', 'content': 'request'}]
    first = body([*canonical, {'role': 'developer', 'content': 'clock-one'}])
    state.prepare(first, canonical_input=canonical)
    response = completed(); state.complete(response)
    canonical_two = [*canonical, *response['output'], {'role': 'user', 'content': 'follow-up'}]
    second = body([*canonical_two, {'role': 'developer', 'content': 'clock-two'}])
    wire, full, metrics = state.prepare(second, canonical_input=canonical_two)
    assert metrics['continuation_mode'] == 'incremental'
    assert wire['input'] == [canonical_two[-1], second['input'][-1]]
    assert full['input'] == [*first['input'], *response['output'], *wire['input']]


def test_compaction_model_tool_or_history_edits_force_clean_full_input():
    for change in ('history', 'tools', 'model'):
        state = ResponsesContinuation(enabled=True)
        state.prepare(body([{'role': 'user', 'content': 'old'}])); state.complete(completed())
        next_body = body([{'role': 'user', 'content': 'new'}])
        if change == 'tools': next_body['tools'] = [{'name': 'changed'}]
        if change == 'model': next_body['model'] = 'other'
        wire, full, metrics = state.prepare(next_body)
        assert 'previous_response_id' not in wire
        assert wire == next_body and full == next_body


def test_only_explicit_pre_output_state_rejection_can_trigger_full_retry():
    state = ResponsesContinuation(enabled=True)
    assert state.reject(404, {'error': {'code': 'previous_response_not_found'}})
    assert state.reject(400, {'error': {'code': 'unsupported_parameter', 'param': 'previous_response_id'}})
    assert state.reject(503, {'error': {'code': 'previous_response_not_found'}}) is None
    assert state.reject(400, {'error': {'message': 'arbitrary provider rejection'}}) is None


def test_new_runtime_has_no_upstream_state_and_incomplete_response_cannot_be_chained():
    state = ResponsesContinuation(enabled=True)
    state.prepare(body([]))
    state.complete({**completed(), 'status': 'incomplete'})
    wire, _, _ = state.prepare(body([{'role': 'user', 'content': 'new'}]))
    assert 'previous_response_id' not in wire
    other = ResponsesContinuation(enabled=True)
    assert other.prepare(body([]))[2]['continuation_mode'] == 'full'
