import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.context_policy import estimate_text_tokens
from app.agent.participation_judge import judge_packet, parse_decision


def row(n, text='普通讨论'):
    return SimpleNamespace(id=n, user_id=900000100+n, plain_text=text, mentioned_bot=False,
        reply_to_msg_id=None, platform_msg_id=str(n), raw_json={})


def test_judge_packet_bounds_long_window_and_keeps_addressing():
    rows=[row(n, '很长的群聊内容'*300) for n in range(1,31)]
    rows[-1].reply_to_msg_id='29'
    rows[-1].raw_json={'message':[{'type':'at','data':{'qq':'900000129'}}]}
    packet=judge_packet(rows,bot_id=900000001)
    assert estimate_text_tokens(json.dumps(packet,ensure_ascii=False))<=1500
    last=packet['messages'][-1]
    assert last['id']==30 and last['reply_to']==29 and last['mentions_other']
    assert not last['mentions_bot'] and packet['messages'][0]['id']>=1


def test_judge_decision_cannot_invent_target_or_execute_instructions():
    assert parse_decision('{"decision":"participate","source_id":12,"reason":"natural_hook"}',{12})==(12,'natural_hook')
    for value in ('yes','{"decision":"participate","source_id":99}',
            '{"decision":"participate","source_id":true}',
            '{"decision":"silent","source_id":12}', 'not JSON'):
        assert parse_decision(value,{12})[0] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('participate',[False,True])
async def test_native_judge_skips_persona_memory_and_uses_one_tool_free_request(sqlite_engine,tmp_path,participate,record_property):
    from test_autonomous_social import composed,add_source,NOW,GROUP
    from test_harness_runtime import model_client,responses_sse
    from app.agent.social_contracts import SocialJob,SocialObservation
    from app.agent.tool_agent import ToolTurnResult
    ids=[]
    def script(body,n):
        assert body['tools']==[] and n==1
        assert 'PRIVATE_PERSONA_FACT' not in json.dumps(body)
        assert estimate_text_tokens(json.dumps(body,ensure_ascii=False))<2300
        return responses_sse(text=json.dumps({'decision':'participate' if participate else 'silent',
            'source_id':ids[-1] if participate else None,'reason':'natural_hook' if participate else 'quiet'}))
    client,requests,_=model_client(script)
    async with composed(sqlite_engine,tmp_path,client,runtime='harness') as fixture:
        ids[:]=[add_source(sqlite_engine,platform_id=f'judge-{n}',text=f'合成讨论内容{n}') for n in range(8)]
        full=AsyncMock(return_value=ToolTurnResult('sent','synthetic-receipt'))
        fixture.executor._run_tools=full
        guard=SimpleNamespace(require_valid=lambda:None,valid=lambda:True,source_message_ids=tuple(ids),
            trace_trigger={'trigger_evidence':{'source_ids':ids}})
        observation=SocialObservation(SocialJob(key=f'group:{GROUP}',kind='group',scope_id=GROUP,
            autonomous_tools=True),NOW,'PRIVATE_PERSONA_FACT',source_message_ids=tuple(ids))
        result=await fixture.executor.run_tools(observation,guard,interactive=True)
        assert result.status==('sent' if participate else 'silent'),result
        assert len(requests)==1 and full.await_count==int(participate)
        from app.agent.context_policy import estimate_request_tokens
        record_property('judge_estimated_input_tokens',estimate_request_tokens(requests[0]))
        record_property('upstream_requests',len(requests))
        record_property('tools',len(requests[0]['tools']))
        if participate:assert full.call_args.kwargs['participation_approved']
