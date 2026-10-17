"""Real native HTTP capture proves clean history survives targeted isolation."""
import json
from datetime import UTC, datetime
import pytest

from app.core.memory_source_guard import is_usable_memory_source
from app.storage.db import session_scope
from app.storage.models import Message
from test_autonomous_social import composed
from test_group_context_reset import event, turn
from test_harness_runtime import model_client, responses_sse
from test_mature_composition import feedback


@pytest.mark.asyncio
async def test_group_isolation_keeps_clean_history_but_no_retrieval_leaks(sqlite_engine,tmp_path):
    results=[]
    def model(body,n):
        observed=feedback(body)
        if observed and observed[-1]['name']=='send_group':
            return responses_sse(number=n)
        if not any(x['name']=='search_history' for x in observed):
            return responses_sse(name='search_history',arguments={'query':'HIST'},number=n,call_id=f'history-{n}')
        result=next(x for x in reversed(observed) if x['name']=='search_history')
        results.append(result)
        return responses_sse(name='send_group',arguments={'text':'干净历史仍可查询','artifact_ids':[]},number=n,call_id=f'history-{n}')
    client,requests,_=model_client(model)
    async with composed(sqlite_engine,tmp_path,client,runtime='harness') as fixture:
        await fixture.agent.stop()
        fixture.clock[0]=datetime.now(UTC)
        good=event('clean-history-149','HIST NORMAL_SOURCE_149')
        bad=event('quarantine-history-149','HIST QUARANTINED_SOURCE_149')
        fixture.router.ingest_live_group_message(good)
        fixture.router.ingest_live_group_message(bad)
        command=event('isolate-149','隔离这条并重置上下文',user=fixture.router.runtime.settings.owner_qq)
        command.raw_payload['message'].insert(0,{'type':'reply','data':{'id':bad.platform_msg_id}})
        fixture.router.ingest_live_group_message(command)
        assert await fixture.router.try_handle_context_reset(command)
        assert requests==[]
        result=await turn(fixture,event('new-clean-request-149','查询HIST里的正常记录'))
        assert result.status=='sent'
        inputs=json.dumps(requests,ensure_ascii=False)
        assert 'NORMAL_SOURCE_149' in inputs and 'QUARANTINED_SOURCE_149' not in inputs
        texts=json.dumps(results,ensure_ascii=False)
        assert 'NORMAL_SOURCE_149' in texts and 'QUARANTINED_SOURCE_149' not in texts
        with session_scope(sqlite_engine) as session:
            from app.storage.repositories import MessageRepository
            assert is_usable_memory_source(MessageRepository(session).get_by_platform_msg_id(good.platform_msg_id))


@pytest.mark.asyncio
async def test_private_scoped_history_after_isolation_uses_real_tool_input(sqlite_engine,tmp_path):
    from tests.private_chat.test_agent_service import ACCOUNT,OWNER,event as private_event,setup,sse
    from tests.storage.test_context_resets import source
    seen=[]
    def model(body,n):
        if not any(x.get('type')=='function_call_output' for x in body['input']):
            return sse('search_private_history',{'query':'HIST'},number=n)
        seen.append(body)
        return sse('send_private',{'text':'正常私聊历史保留'},number=n)
    service,agent,dispatcher,sender,requests=setup(sqlite_engine,tmp_path,model)
    source(sqlite_engine,f'private-inbound-{OWNER}-good149',group=None,user=OWNER,text='HIST PRIVATE_NORMAL_149')
    source(sqlite_engine,f'private-inbound-{OWNER}-bad149',group=None,user=OWNER,text='HIST PRIVATE_QUARANTINED_149')
    source(sqlite_engine,'foreign-private-149',group=None,user=900000103,text='HIST FOREIGN_PRIVATE_149')
    await service.start()
    try:
        command=private_event('isolate-private-149',text='隔离这条并重置上下文')
        command.raw_payload['message'].insert(0,{'type':'reply','data':{'id':'bad149'}})
        await dispatcher.handle_private_message(command)
        assert requests==[]
        await dispatcher.handle_private_message(private_event('new-private-149',text='找一下HIST的正常记录'))
        inputs=json.dumps(requests,ensure_ascii=False)
        assert 'PRIVATE_NORMAL_149' in inputs
        assert 'PRIVATE_QUARANTINED_149' not in inputs and 'FOREIGN_PRIVATE_149' not in inputs
        assert seen and sender.texts[-1]==(OWNER,'正常私聊历史保留')
    finally:
        await service.stop()
