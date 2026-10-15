import base64
import io
import json
import shutil
import subprocess

import httpx
import pytest
from sqlalchemy import select

from app.adapters.sender import Sender,QQMessageDeliveryUncertainError
from app.providers.web_images import WebImageClient
from app.providers.web_search import WebSearchClient
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.agent_effects import AgentToolEffect
from test_autonomous_social import composed,wake,writes
from test_harness_runtime import model_client,responses_sse
from test_web_image_delivery import output,requested_group_media
from tests.private_chat.test_agent_service import OTHER,event,setup,sse

NOTE='0123456789abcdef01234567'


@pytest.fixture
def mp4(tmp_path):
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg: pytest.skip('Video integration runs in the Linux image with ffmpeg')
    path=tmp_path/'fixture.mp4'
    subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i','color=c=red:s=64x64:d=0.3',
        '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(path)],check=True,timeout=15)
    return path.read_bytes()


def video_html(pixels):
    note={'noteId':NOTE,'type':'video','video':{'media':{'stream':{'h264':[{
        'masterUrl':'https://cdn.example.com/video.mp4','size':len(pixels),'duration':320,'width':64,'height':64}]}}}}
    return '<script>window.__INITIAL_STATE__='+json.dumps({'noteData':{'data':{'noteData':note}}})+'</script>'


@pytest.mark.parametrize('surface',['group','private'])
@pytest.mark.parametrize('uncertain',[False,True])
async def test_native_video_download_scope_send_and_unknown_receipt(sqlite_engine,tmp_path,monkeypatch,mp4,surface,uncertain):
    requests=[]
    def respond(request):
        requests.append(request)
        return httpx.Response(200,content=mp4,headers={'Content-Type':'video/mp4'}) if request.url.path.endswith('.mp4') else httpx.Response(
            200,text=video_html(mp4),headers={'Content-Type':'text/html'})
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        provider=WebImageClient(http_client=http)
        monkeypatch.setattr('app.agent.web_images.WebImageClient',lambda _:provider)
        web=WebSearchClient(provider='ddgs',base_url='',api_key='',timeout_seconds=10)
        source=[]
        result_steps=[]
        def command(body,n):
            from test_delegation_composition import observation
            current=observation(body,'Private observation JSON:\n') if surface=='private' else {}
            if current.get('kind')=='research_result':
                if result_steps:return None,None
                result_steps.append(True)
                result=current['delegated_result']['data']
                assert result['codec']=='h264' and result['bytes']==len(mp4)
                return 'send_private',{'text':'原笔记视频','artifact_ids':[],'video_artifact_id':result['video_artifact_id']}
            if surface=='private' and n==1:return 'read_message_links',{'source_id':source[0]}
            step=n-int(surface=='private')
            if step==1:return 'download_web_video',{'ref_id':f'https://www.xiaohongshu.com/discovery/item/{NOTE}'}
            if step==2:
                result=output(body)
                if surface=='private':
                    assert result['status']=='queued'
                    return 'send_private',{'text':'排队中'}
                assert result['codec']=='h264' and result['bytes']==len(mp4)
                return 'send_'+surface,{'text':'原笔记视频','artifact_ids':[],'video_artifact_id':result['video_artifact_id']}
            return None,None
        if surface=='group':
            def model(body,n):
                name,args=command(body,n)
                return responses_sse(name=name,arguments=args,number=n,call_id=f'video-{n}') if name else responses_sse(number=n)
            client,_,_=model_client(model)
            async with composed(sqlite_engine,tmp_path,client,runtime='harness') as fixture:
                fixture.router.web_search_client=web
                original=fixture.gateway.call_api
                async def gateway(action,params):
                    if action=='send_group_forward_msg':
                        fixture.gateway.calls.append((action,params))
                        for node in params['messages']:
                            content=node['data']['content']
                            if any(item['type']=='video' for item in content):assert len(content)==1
                        if uncertain:raise QQMessageDeliveryUncertainError('lost ack')
                        return {'status':'ok','retcode':0,'data':{'message_id':'video-group-receipt'}}
                    return await original(action,params)
                fixture.gateway.call_api=gateway
                await requested_group_media(fixture)
                sent=writes(fixture)
                assert len(sent)==1
                params=sent[0][1]
        else:
            def model(body,n):
                name,args=command(body,n)
                return sse(name,args,number=n) if name else sse(number=n)
            service,agent,dispatcher,_,_=setup(sqlite_engine,tmp_path,model)
            from app.adapters.onebot_models import parse_private_message_event
            card=parse_private_message_event({'message_id':'video-card','user_id':OTHER,'time':100,
                'message':[{'type':'share','data':{'url':f'https://www.xiaohongshu.com/discovery/item/{NOTE}','title':'视频卡片'}}]})
            dispatcher._persist_inbound(card)
            with session_scope(sqlite_engine) as session:
                source.append(session.scalar(select(Message.id).where(Message.platform_msg_id==f'private-inbound-{OTHER}-video-card')))
            class Gateway:
                calls=[]
                async def call_api(self,action,params):
                    self.calls.append((action,params))
                    if uncertain and isinstance(params.get('message',params.get('messages')),list):raise QQMessageDeliveryUncertainError('lost ack')
                    return {'status':'ok','retcode':0,'data':{'message_id':'video-private-receipt'}}
            gateway=Gateway();service.sender=Sender(gateway);service.web_search_client=web
            incoming=event('video-request',user_id=OTHER,text='把这个小红书视频扒下来发我')
            incoming.reply_to_msg_id='video-card';incoming.raw_payload['message'].insert(0,{'type':'reply','data':{'id':'video-card'}})
            await agent.start()
            try:
                await dispatcher.handle_private_message(incoming)
                await dispatcher.handle_private_message(incoming)
                for _ in range(160):
                    if any(isinstance(params.get('message',params.get('messages')),list) for _,params in gateway.calls):break
                    import asyncio
                    await asyncio.sleep(0.1)
                assert len(gateway.calls)==2
                params=next(params for _,params in gateway.calls if isinstance(params.get('message',params.get('messages')),list))
            finally:await agent.stop()
        segments=params.get('message') or [item for node in params['messages'] for item in node['data']['content']]
        video=next(item for item in segments if item['type']=='video')
        assert base64.b64decode(video['data']['file'][9:])==mp4 and len(requests)==2
        with session_scope(sqlite_engine) as session:
            effects=list(session.scalars(select(AgentToolEffect).where(AgentToolEffect.operation=='send_'+surface)))
            assert len(effects)==(2 if surface=='private' else 1) and effects[-1].status==('uncertain' if uncertain else 'sent')
        web.http_client.close()


def test_real_video_validator_rejects_non_video_truncation_and_missing_dependency(tmp_path,mp4,monkeypatch):
    from app.core.video_file import validate_video
    path=tmp_path/'sample.mp4';path.write_bytes(mp4)
    assert validate_video(path)['codec']=='h264'
    for invalid in (b'<html>Login</html>',mp4[:64]):
        path.write_bytes(invalid)
        with pytest.raises(ValueError,match='web_video_'):validate_video(path)
    path.write_bytes(mp4)
    monkeypatch.setattr('app.core.video_file.shutil.which',lambda _:None)
    with pytest.raises(ValueError,match='validator_unavailable'):validate_video(path)
