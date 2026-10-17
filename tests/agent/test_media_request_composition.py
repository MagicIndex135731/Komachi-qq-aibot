"""Native group/private requests: parallel downloads, replacement and one receipt."""
import asyncio
import subprocess
import shutil
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from app.adapters.sender import Sender, QQMessageDeliveryUncertainError
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.providers.twitter_media import TwitterMediaError
from app.providers.web_search import WebSearchClient
from app.storage.db import session_scope
from app.storage.models import Job, Message
from test_autonomous_social import composed, add_source, NOW, FRIEND
from test_delegation_composition import observation
from test_harness_runtime import model_client, responses_sse
from tests.private_chat.test_agent_service import setup, event, OTHER


@pytest.fixture
def ordinary_clips(tmp_path):
    values=[]
    for index,color in enumerate(['red','blue','green','yellow']):
        path=tmp_path/f'ordinary-{index}.mp4'
        subprocess.run([shutil.which('ffmpeg'),'-v','error','-f','lavfi','-i',f'color=c={color}:s=64x64:d=0.2',
            '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(path)],check=True,timeout=10)
        values.append(path.read_bytes())
    return values


@pytest.mark.parametrize('surface',['group','private'])
@pytest.mark.parametrize('failed',[False,True])
@pytest.mark.parametrize('unknown',[False,True])
async def test_three_video_request_returns_once_replenishes_and_packs(sqlite_engine,tmp_path,monkeypatch,
        ordinary_clips,surface,failed,unknown):
    base=900000000000000100
    urls=[f'https://x.com/synthetic/status/{base+i}' for i in range(4)]
    class Crawler:
        def search_posts(self,*args,**kwargs):
            return {'source_url':'https://x.com/search?q=space','posts':[
                {'source_url':url,'post_id':str(base+i),'author':'synthetic','identity_status':'matched',
                 'published_at':datetime.now(UTC).isoformat(),'media_count':1,'media_roles':['post_video'],
                 'text':'普通航天纪录短片'} for i,url in enumerate(urls)]}
        def inspect(self,url,**kwargs):
            index=int(url.rsplit('/',1)[1])-base
            if failed and index==1:raise TwitterMediaError('synthetic_404')
            return {'source_url':url,'post_id':str(base+index),'author':'synthetic','identity_status':'matched',
                'protected':False,'media_count':1,'evidence_ref':'x-posts-'+'a'*24+':post:0','text':'普通航天纪录短片',
                'media':[{'kind':'video','url':f'https://video.twimg.com/ordinary-{index}.mp4',
                    'variants':[{'url':f'https://video.twimg.com/ordinary-{index}.mp4',
                        'bitrate':500000,'duration_seconds':20,'codec':'h264'}]}]}
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.from_settings',staticmethod(lambda _:Crawler()))
    transfers=[]
    def respond(request):
        index=int(request.url.path.rsplit('-',1)[1].split('.')[0]);transfers.append(index)
        return httpx.Response(200,content=ordinary_clips[index],headers={'content-type':'video/mp4'})
    monkeypatch.setattr('app.agent.twitter_media.download_client',lambda _:httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    if surface=='group':monkeypatch.setattr('app.agent.context_dependencies.datetime',SimpleNamespace(now=lambda tz:NOW))
    steps={'main':0,'replacement':0};returned=[]
    def goal(count):
        return {'goal':'搜集三个普通航天视频','delivery':'media','requirements':['media_delivery'],
            'media_selection':{'minimum_count':3,'target_count':3,'selection_scope':'topic','subject':'航天',
            'candidate_reviews':[{'ref_id':url,'decision':'selected','reason':'原帖文字标明普通航天视频'} for url in urls[:count]]}}
    def script(body,n):
        try:
            packet=observation(body,'Delegated result JSON:\n' if surface=='group' else 'Private observation JSON:\n')
        except AssertionError:packet={}
        result=packet if surface=='group' else packet.get('delegated_result') or {}
        if result.get('kind')=='media_result':
            data=result['data'];ids=data['media_artifact_ids']
            if failed and len(ids)==2:
                steps['replacement']+=1
                if steps['replacement']==1:name,args='set_task_goal',goal(4)
                else:
                    assert steps['replacement']==2
                    name,args='download_twitter_media',{'ref_id':urls[3],'delivery_mode':'review'}
            else:
                assert len(ids)==3
                returned.append(len(ids));name,args='send_'+surface,{
                    'text':'三个普通航天视频已合并','artifact_ids':[],
                    'media_artifact_ids':ids,'as_forward':True,'task_status':'completed'}
        else:
            steps['main']+=1
            if steps['main']==1:name,args='search_x_posts',{'mode':'semantic','subject_terms':['航天'],'media_types':['video'],'limit':4}
            elif steps['main']==2:name,args='set_task_goal',goal(3)
            elif steps['main']==3:name,args='queue_media_batch',{'operations':[
                {'operation':'download_twitter_media','arguments':{'ref_id':url,'delivery_mode':'review'}} for url in urls[:3]]}
            else:
                assert steps['main']==4
                name,args='send_'+surface,{'text':'正在下载普通视频','artifact_ids':[],'task_status':'progress'}
        return responses_sse(name=name,arguments=args,number=n,call_id=f'ordinary-{n}')
    class Gateway:
        def __init__(self):self.calls=[]
        async def call_api(self,action,params):
            self.calls.append((action,params))
            if unknown and action.endswith('forward_msg'):raise QQMessageDeliveryUncertainError('synthetic lost receipt')
            return {'status':'ok','retcode':0,'data':{'message_id':'ordinary-receipt-'+str(len(self.calls))}}
    gateway=Gateway();web=WebSearchClient(provider='wyzlab',base_url='https://synthetic.invalid',api_key='synthetic',timeout_seconds=5)
    try:
        if surface=='group':
            client,requests,_=model_client(script)
            async with composed(sqlite_engine,tmp_path,client,runtime='harness') as fixture:
                fixture.executor.sender=Sender(gateway);fixture.router.web_search_client=web
                queue=fixture.executor.media_jobs;queue.running=True
                job=next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind=='group')
                source=add_source(sqlite_engine,platform_id='ordinary-three',text='小町请去X搜集三个普通航天视频发来')
                with session_scope(sqlite_engine) as session:
                    row=session.get(Message,source);row.raw_json={'message':[{'type':'text','data':{'text':row.plain_text}}]}
                guard=SimpleNamespace(action_id='ordinary-three',valid=lambda:True,require_valid=lambda:None,on_dispatch=lambda:None)
                origin=await fixture.executor.run_tools(SocialObservation(job,NOW,'synthetic',source_message_ids=(source,)),guard,
                    interactive=True,target_user_id=FRIEND,target_message_id='ordinary-three',direct_group_reply=True)
                assert origin.status in {'sent','silent'},(origin.status,origin.reason,origin.trace)
                for _ in range(50):
                    if not await queue.tick():break
                await queue.stop()
        else:
            service,agent,dispatcher,_,requests=setup(sqlite_engine,tmp_path,script)
            service.sender=Sender(gateway);service.web_search_client=web
            await service.start()
            try:
                await dispatcher.handle_private_message(event('ordinary-three',user_id=OTHER,text='去X搜集三个普通航天视频发给我'))
                try:
                    async with asyncio.timeout(15):
                        while True:
                            with session_scope(sqlite_engine) as session:
                                jobs=list(session.query(Job).filter(Job.job_type=='agent_media_operation'))
                                settled=len(jobs)==(4 if failed else 3) and all(j.status in {'completed','failed','blocked','uncertain'} for j in jobs)
                            if settled and not agent._active and any(action.endswith('forward_msg') for action,_ in gateway.calls):break
                            await asyncio.sleep(.05)
                except TimeoutError:
                    with session_scope(sqlite_engine) as session:
                        debug=[(j.status,j.last_error_code,j.payload_json.get('result',{}).get('data',{}).get('error'))
                            for j in session.query(Job).filter(Job.job_type=='agent_media_operation')]
                    raise AssertionError((steps,returned,debug,len(requests),[a for a,_ in gateway.calls]))
            finally:await service.stop()
        forwards=[params for action,params in gateway.calls if action.endswith('forward_msg')]
        with session_scope(sqlite_engine) as session:
            debug=[(j.status,j.last_error_code,j.payload_json.get('result',{}).get('data',{}).get('error'))
                for j in session.query(Job).filter(Job.job_type=='agent_media_operation')]
        errors=[]
        for request in requests:
            for item in request.get('input',[]):
                if item.get('type')=='function_call_output':
                    try:value=json.loads(item.get('output','{}'))
                    except ValueError:continue
                    if value.get('data',{}).get('error'):
                        error={k:v for k,v in value['data'].items() if k in {'error','missing','guidance'}}
                        if error not in errors:errors.append(error)
        assert len(forwards)==1 and returned==[3],(steps,returned,debug,len(requests),errors)
        media=[part for node in forwards[0]['messages'] for part in node['data']['content'] if part['type']=='video']
        assert len(media)==3
        assert sorted(transfers)==([0,2,3] if failed else [0,1,2])
        assert steps['replacement']==(2 if failed else 0)
        assert len(requests)==(7 if failed else 5)
    finally:web.http_client.close()
