import json

import pytest

from app.agent.context_recovery import context_reusable, safe_closed_items
from app.agent.tool_agent import ToolOutcome, ToolTurnResult
from test_harness_runtime import bridge_factory, ModelHandler, request_body, responses_sse, model_client, Guard, tool


def rejected_sse(text=None, *, tool=False):
    payload = responses_sse(name='send_group', arguments={'text':'SYNTHETIC_UNCONFIRMED_BODY'}) if tool else responses_sse(
        text=text or "The prompt could not be submitted. The prompt contains sensitive words that violate Google's Generative AI Prohibited Use policy.")
    lines=[]
    for line in payload.decode().splitlines():
        if line.startswith('event: response.completed'):
            line='event: response.incomplete'
        if line.startswith('data: '):
            value=json.loads(line[6:])
            if value.get('type')=='response.completed':
                value['type']='response.incomplete'
                value['response'].update(status='incomplete', incomplete_details={'reason':'content_filter' if tool else 'interrupted'})
                line='data: '+json.dumps(value)
        lines.append(line)
    return ('\n'.join(lines)+'\n\n').encode()


def test_relay_text_refusal_is_rejected_once_and_never_committed(bridge_factory):
    bridge,requests,_=bridge_factory(lambda *_:rejected_sse(),model='gemini-3.8-flash-medium')
    body=request_body();body['model']=bridge.model
    handler=ModelHandler();bridge.forward(handler,body)
    assert bridge.stop_reason=='model_input_rejected' and len(requests)==1
    assert b'sensitive words' not in handler.wfile.getvalue()
    assert bridge.trace[-1]['output_disposition']=='discarded'


def test_tool_call_followed_by_content_filter_is_not_committed(bridge_factory):
    # Test only the response transaction; no tool is authorized or executed.
    bridge,requests,_=bridge_factory(lambda *_:rejected_sse(tool=True),model='gemini-3.8-flash-medium')
    body=request_body();body['model']=bridge.model
    handler=ModelHandler();bridge.forward(handler,body)
    assert b'SYNTHETIC_UNCONFIRMED_BODY' not in handler.wfile.getvalue()
    assert b'function_call' not in handler.wfile.getvalue()
    assert len(requests)==1 and bridge.stop_reason=='model_input_rejected'


def test_real_receipt_survives_late_rejection_but_history_is_not_reusable(bridge_factory):
    bridge,_,_=bridge_factory(lambda *_:rejected_sse(),model='gemini-3.8-flash-medium')
    bridge.last_write=ToolOutcome({},'sent','synthetic-receipt');bridge.last_write_name='send_group'
    body=request_body();body['model']=bridge.model
    bridge.forward(ModelHandler(),body)
    result=bridge.result(bridge.stop_reason)
    assert result.status=='sent' and result.receipt_id=='synthetic-receipt'
    assert not context_reusable(result)


@pytest.mark.parametrize('outcome',['incomplete','stream_error','model_progress_timeout'])
def test_sent_receipt_does_not_make_incomplete_generation_safe_for_bootstrap(outcome):
    trace=({'status':'model_request','purpose':'agent','outcome':outcome},)
    result=ToolTurnResult('sent','synthetic-receipt','tool:send_group',trace)
    assert not context_reusable(result)
    pair=[{'role':'user','content':'SYNTHETIC_UNCONFIRMED_INPUT'},
          {'role':'assistant','content':json.dumps({'status':'sent','trace':trace})}]
    assert safe_closed_items(pair)==[]


def test_complete_generation_remains_cacheable_and_bootstrappable():
    trace=({'status':'model_request','purpose':'agent','outcome':'completed'},)
    result=ToolTurnResult('sent','synthetic-receipt','tool:send_group',trace)
    assert context_reusable(result)
    pair=[{'role':'user','content':'SYNTHETIC_COMPLETED_INPUT'},
          {'role':'assistant','content':json.dumps({'status':'sent','trace':trace})}]
    assert safe_closed_items(pair)==pair


@pytest.mark.asyncio
@pytest.mark.parametrize('late', [False, True])
async def test_native_rejection_never_executes_unconfirmed_tool_and_preserves_prior_receipt(tmp_path, late):
    from app.agent.harness_runtime import HarnessToolAgent
    calls=[]
    async def send(args):
        calls.append(args)
        return ToolOutcome({}, 'sent', 'synthetic-confirmed-receipt')
    client, requests, _ = model_client(lambda _, n: responses_sse(name='send_group', arguments={'text':'confirmed'}, number=n)
        if late and n==1 else rejected_sse(tool=True))
    client.model=client.responses_model='gemini-3.8-flash-medium'
    agent=HarnessToolAgent(decision_client=client, require_reply=True, session_home=tmp_path/'home')
    try:
        result=await agent.run(['synthetic request'], (tool('send_group', send, effect='write',
            properties={'text':{'type':'string'}}, required=('text',)),), Guard())
        assert len(requests)==(2 if late else 1)
        assert len(calls)==(1 if late else 0)
        assert result.status==('sent' if late else 'failed')
        assert result.receipt_id==('synthetic-confirmed-receipt' if late else None)
        assert not context_reusable(result)
    finally:
        agent.close();client.http_client.close()
