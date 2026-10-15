import asyncio
import base64

import pytest

from app.adapters.sender import Sender, QQMessageDeliveryUncertainError
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.providers.web_search import WebSearchClient
from app.storage.agent_effects import AgentToolEffect
from app.storage.db import session_scope
from sqlalchemy import select
from test_autonomous_social import FRIEND, NOW, add_source, composed, image_bytes
from test_delegation_composition import observation
from test_harness_runtime import model_client, responses_sse
from test_twitter_media_tools import URL, provider
from test_web_image_delivery import output
from test_web_video_delivery import mp4
from tests.private_chat.test_agent_service import OTHER, event, setup, sse


@pytest.mark.parametrize('surface',['group','private'])
@pytest.mark.parametrize('uncertain',[False,True])
@pytest.mark.parametrize('delivery_mode',['review','auto'])
async def test_native_queued_mixed_media_preserves_order_scope_and_receipt(sqlite_engine,tmp_path,monkeypatch,mp4,surface,uncertain,delivery_mode,record_property):
    requests=provider(monkeypatch,video=mp4)
    web=WebSearchClient(provider='ddgs',base_url='',api_key='',timeout_seconds=10)
    sent_result=[]
    model_calls=[]
    def command(body,n):
        model_calls.append(n)
        try:
            packet=observation(body,'Delegated result JSON:\n') if surface=='group' else observation(body,'Private observation JSON:\n')
        except AssertionError:
            packet={}
        result=packet if surface=='group' else packet.get('delegated_result',{}) if packet.get('kind')=='research_result' else {}
        if result.get('kind')=='media_result':
            if sent_result:return None
            sent_result.append(True);data=result['data']
            assert [item['kind'] for item in data['media']]==['photo','video','gif']
            return 'send_'+surface,{'text':'原帖共3项媒体','artifact_ids':[],
                'media_artifact_ids':data['media_artifact_ids'],'as_forward':True}
        if n==1:return 'download_twitter_media',{'ref_id':URL,'delivery_mode':delivery_mode,'delivery_text':'原帖共3项媒体'}
        if n==2:
            assert output(body)['status']=='queued'
            return 'send_'+surface,{'text':'已安排下载','artifact_ids':[],'task_status':'progress'}
        return None
    class Gateway:
        def __init__(self):self.calls=[]
        async def call_api(self,action,params):
            self.calls.append((action,params))
            for node in params.get('messages',()):
                content=node['data']['content']
                if any(item['type']=='video' for item in content):
                    assert len(content)==1, 'SnowLuma requires standalone video nodes'
            if uncertain and action.endswith('forward_msg'):
                raise QQMessageDeliveryUncertainError('synthetic lost ack')
            return {'status':'ok','retcode':0,'data':{'message_id':'synthetic-receipt-'+str(len(self.calls))}}
    gateway=Gateway()
    try:
        if surface=='group':
            def script(body,n):
                cmd=command(body,n)
                return responses_sse(name=cmd[0],arguments=cmd[1],call_id=f'x-{n}',number=n) if cmd else responses_sse(number=n)
            client,_,_=model_client(script)
            async with composed(sqlite_engine,tmp_path,client,runtime='harness') as fixture:
                fixture.executor.sender=Sender(gateway);fixture.router.web_search_client=web
                queue=fixture.executor.media_jobs;queue.running=True
                job=configured_social_jobs(fixture.router.runtime)[0]
                source=add_source(sqlite_engine,platform_id='x-download-request',text='小町请下载这个推文全部图片和视频 '+URL)
                class DirectGuard:
                    action_id='x-download-request'
                    def require_valid(self):pass
                    def valid(self):return True
                    def on_dispatch(self):pass
                result=await fixture.executor.run_tools(SocialObservation(job,NOW,'synthetic persona',source_message_ids=(source,)),
                    DirectGuard(),interactive=True,target_user_id=FRIEND,target_message_id='x-download-request',direct_group_reply=True)
                assert result.status=='sent', result
                assert await queue.tick()
                assert await queue.tick()
                await queue.stop()
        else:
            def script(body,n):
                cmd=command(body,n)
                return sse(cmd[0],cmd[1],number=n) if cmd else sse(number=n)
            service,agent,dispatcher,_,_=setup(sqlite_engine,tmp_path,script)
            service.sender=Sender(gateway);service.web_search_client=web
            await service.start()
            try:
                incoming=event('x-download-request',user_id=OTHER,text='请下载这个推文全部图片和视频 '+URL)
                await dispatcher.handle_private_message(incoming)
                async with asyncio.timeout(20):
                    while not any(action.endswith('forward_msg') for action,_ in gateway.calls):await asyncio.sleep(.05)
            finally:await service.stop()
        forwards=[params for action,params in gateway.calls if action.endswith('forward_msg')]
        assert len(forwards)==1 and len(requests)==4
        nodes=forwards[0]['messages']
        assert nodes[0]['data']['content']==[{'type':'text','data':{'text':'原帖共3项媒体'}}]
        media=[item for node in nodes for item in node['data']['content'] if item['type'] in {'image','video'}]
        assert [item['type'] for item in media]==['image','video','video']
        assert [base64.b64decode(item['data']['file'][9:]) for item in media]==[image_bytes(),mp4,mp4]
        if delivery_mode=='auto':
            # A progress acknowledgement with the actual queued job can close
            # locally. Formal tasks with remaining work have separate coverage.
            assert len(model_calls)==2 and sent_result==[]
        record_property('upstream_requests',len(model_calls))
        record_property('delivery_mode',delivery_mode)
        record_property('surface',surface)
        with session_scope(sqlite_engine) as session:
            effects=list(session.scalars(select(AgentToolEffect).where(AgentToolEffect.operation=='send_'+surface)))
            assert effects[-1].status==('uncertain' if uncertain else 'sent')
    finally:web.http_client.close()
