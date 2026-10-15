import pytest
import json
from copy import deepcopy

from app.agent.context_policy import RequestContextPolicy
from app.agent.harness_runtime import HarnessToolAgent
from test_harness_runtime import Guard, model_client, responses_sse, ModelHandler, request_body, tool, bridge_factory


@pytest.mark.asyncio
async def test_native_five_wakes_append_memory_deltas_and_keep_history_prefix(tmp_path, monkeypatch):
    from app.agent.context_policy import estimate_request_tokens
    from test_context_delta import packet, MEMORY, OBSERVATION
    original = RequestContextPolicy.project
    baselines = []
    projected_inputs = []

    def measured(policy, body, **kwargs):
        legacy = deepcopy(body)
        for entry in legacy.get('input', []):
            for part in entry.get('content', []) if isinstance(entry.get('content'), list) else []:
                text = part.get('text', '')
                at = text.find(MEMORY)
                if at < 0:
                    continue
                begin = at + len(MEMORY)
                value, size = json.JSONDecoder().raw_decode(text[begin:])
                if value.get('evidence_format') == 'blocks_v1':
                    value['evidence'] = '\n\n'.join(value.pop('evidence_blocks'))
                    value.pop('evidence_format')
                    part['text'] = text[:begin] + json.dumps(value, ensure_ascii=False) + text[begin+size:]
        baselines.append(estimate_request_tokens(legacy))
        projected, metrics = original(policy, body, **kwargs)
        projected_inputs.append(estimate_request_tokens(projected))
        return projected, metrics

    monkeypatch.setattr(RequestContextPolicy, 'project', measured)
    client, requests, _ = model_client(lambda _, n: responses_sse(number=n))
    traces = []
    stable = 'Memory fact (sources: synthetic-fact): ' + 'STABLE_FACT_EVIDENCE ' * 800
    messages = []
    try:
        for number in range(5):
            messages.append('Recent source synthetic-' + str(number) + ': ' + 'NEW_MESSAGE_' + str(number) + ' ' * 10)
            prompt = ['System persona: stable synthetic persona',
                OBSERVATION + json.dumps({'scope': 'group', 'context': messages,
                    'current_target': {'source_id': number+1, 'text': 'CURRENT_TARGET_' + str(number)}}),
                MEMORY + json.dumps(packet(stable, *messages))]
            agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'delta-native', session_id='stable-delta-session')
            try:
                result = await agent.run(prompt, (), Guard())
                assert result.status == 'silent', (result.reason, result.trace)
                traces.extend(result.trace)
            finally:
                agent.close()
        assert len(requests) == 5
        for prior, current in zip(requests, requests[1:]):
            prefix = prior['input'][:-2]
            assert current['input'][:len(prefix)] == prefix
        for index, body in enumerate(requests):
            text = json.dumps(body, ensure_ascii=False)
            assert text.count('STABLE_FACT_EVIDENCE') == 800
            assert 'CURRENT_TARGET_' + str(index) in text
        actual = [estimate_request_tokens(body) for body in requests]
        baselines = [baseline + optimized - before_transport for baseline, optimized, before_transport
            in zip(baselines, actual, projected_inputs)]
        assert actual[-1] < baselines[-1] * 0.4
        assert sum(row.get('context_reused_memory_blocks', 0) for row in traces) >= 4
        report = {'scope': 'actual native Harness, synthetic upstream, no paid calls',
            'unoptimized_estimated_input_tokens': baselines, 'optimized_estimated_input_tokens': actual,
            'last_request_reduction_ratio': round(1-actual[-1]/baselines[-1], 4),
            'requests': len(requests), 'stable_history_prefix': True,
            'real_upstream_cache_savings': 'not measured'}
        report_path = tmp_path/'delta-growth.json'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('DELTA_GROWTH_REPORT=' + json.dumps(report))
    finally:
        client.http_client.close()


@pytest.mark.asyncio
async def test_native_compaction_readmits_memory_removed_from_active_history(tmp_path):
    from test_context_delta import packet, MEMORY
    summaries = []
    def script(body, number):
        summary = body.get('max_output_tokens') == 512
        summaries.append(summary)
        stream = responses_sse(text='CHECKPOINT: task goal; no verbatim evidence retained' if summary else 'CLOSED_STEP', number=number)
        return stream.replace(b'"input_tokens": 10', b'"input_tokens": 100' if summary else b'"input_tokens": 7000')
    client, requests, _ = model_client(script)
    policy = RequestContextPolicy(context_window=16000, safety_tokens=1000, tool_reserve_tokens=1000,
        threshold_ratio=0.2, retain_tokens=1000, summary_tokens=512)
    evidence = 'REUSABLE_MEMORY_EVIDENCE ' * 250
    try:
        for index in range(3):
            agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'compact-delta',
                session_id='synthetic-memory-compaction', context_policy=policy)
            try:
                result = await agent.run(['System persona: stable persona',
                    'CURRENT_TARGET_' + str(index) + '; ' + 'dated closed context ' * (700 if index == 0 else 300),
                    MEMORY + json.dumps(packet(evidence))], (), Guard())
                assert result.status == 'silent', (result.reason, result.trace)
            finally:
                agent.close()
        assert any(summaries)
        final = requests[-1]
        assert 'CHECKPOINT' in json.dumps(final) and 'CURRENT_TARGET_0' not in json.dumps(final)
        newest = next(item for item in reversed(final['input']) if item.get('role') == 'user')
        assert json.dumps(final).count('REUSABLE_MEMORY_EVIDENCE') == 250
        # A retained older input can supply the block. The newest input need
        # not repeat it, but it cannot disappear with the first turn's removal.
        assert 'CURRENT_TARGET_2' in json.dumps(newest)
    finally:
        client.http_client.close()


@pytest.mark.asyncio
async def test_actual_native_compaction_composition_starts(tmp_path, monkeypatch):
    from deepseek_harness import DeepSeekHarness
    original=DeepSeekHarness.start
    diagnostics=[]
    def start(runtime):
        try:return original(runtime)
        except Exception as exc:
            diagnostics.append(str(exc));raise
    monkeypatch.setattr(DeepSeekHarness,'start',start)
    client, requests, _ = model_client(lambda _, n: responses_sse(number=n))
    agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'home')
    try:
        result = await agent.run(['System persona: synthetic persona', 'synthetic current request'], (), Guard())
    finally:
        agent.close()
    assert result.status == 'silent', (result.reason, result.trace, diagnostics)


def test_internal_summary_cannot_inherit_delivery_requirement(bridge_factory):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse(text='summary'), model='gemini-3.8-flash-medium', require_reply=True)
    bridge.tools={'send_group':tool('send_group',None,effect='write')}
    body=request_body();body['model']=bridge.model
    body['tools']=[{'type':'function','name':'send_group'}]
    bridge.forward(ModelHandler(),body,compaction=True)
    assert requests[0]['tools']==body['tools'] and requests[0]['tool_choice']=='none'
    assert bridge.model_calls==0 and bridge.compaction_calls==1 and bridge.write_calls==0
    assert bridge.trace[-1]['purpose']=='compaction'


def test_tool_definition_order_is_stable_without_changing_allowed_names(bridge_factory):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse())
    bridge.tools = {'beta': tool('beta', None), 'alpha': tool('alpha', None)}
    body = request_body()
    body['tools'] = [{'type': 'function', 'name': 'beta'}, {'type': 'function', 'name': 'alpha'}]
    bridge.forward(ModelHandler(), body)
    body['tools'].reverse()
    bridge.forward(ModelHandler(), body)
    assert requests[0]['tools'] == requests[1]['tools']
    assert [item['name'] for item in requests[0]['tools']] == ['alpha', 'beta']
    assert set(bridge.tools) == {'alpha', 'beta'}


def test_full_envelope_ceiling_includes_schema_and_blocks_before_paid_request(bridge_factory):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse())
    bridge.owner.context_policy = RequestContextPolicy(context_window=6000,safety_tokens=1000)
    body=request_body();body['input']=[{'role':'user','content':'汉' * 6000}]
    sink=ModelHandler();bridge.forward(sink,body)
    assert not requests and sink.replies[0][0]==413
    assert bridge.trace[-1]['status']=='context_budget_blocked'


@pytest.mark.asyncio
async def test_native_threshold_compacts_and_resumes_same_task(tmp_path):
    requests_seen = []
    def script(body, number):
        summary = body.get('max_output_tokens') == 512
        requests_seen.append(summary)
        stream = responses_sse(text='CHECKPOINT: audit goal; already checked entry; next check auth' if summary else
                               'CLOSED_STEP_EVIDENCE', number=number)
        return stream.replace(b'"input_tokens": 10', b'"input_tokens": 7000' if not summary else b'"input_tokens": 100')
    client, requests, _ = model_client(script)
    policy = RequestContextPolicy(context_window=16000, safety_tokens=1000, tool_reserve_tokens=1000,
                                  threshold_ratio=0.2, retain_tokens=1000, summary_tokens=512)
    traces = []
    for prompt in ('Audit goal; ' + 'closed evidence ' * 1200,
                   'Recent step: ' + 'retain fresh evidence ' * 500, 'CURRENT_TARGET: continue auth check'):
        agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'resume',
                                 session_id='synthetic-research-scope', context_policy=policy)
        try:
            result = await agent.run(['System persona: stable persona', prompt], (), Guard())
            assert result.status == 'silent', (result.reason, result.trace)
            traces.extend(result.trace)
        finally:
            agent.close()
    assert any(requests_seen), traces
    assert any(row.get('purpose') == 'compaction' for row in traces)
    final = json.dumps(requests[-1], ensure_ascii=False)
    assert 'CURRENT_TARGET' in final and 'CHECKPOINT' in final
    assert requests[-1]['instructions'] == requests[0]['instructions']
    client.http_client.close()
