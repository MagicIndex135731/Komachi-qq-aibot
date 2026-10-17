"""Opt-in ordinary public video probe; default test runs remain offline."""
import os
import json
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get('QQ151_LIVE_VIDEO')!='1',reason='explicit ordinary-video network probe only')
async def test_ordinary_public_video_zero_selection_fallback(sqlite_engine,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from app.adapters.sender import Sender
    from app.agent.social_contracts import SocialObservation
    from app.agent.social_policy import configured_social_jobs
    from app.providers.web_search import WebSearchClient
    from app.providers.x_crawler_client import XCrawlerClient
    from app.storage.db import session_scope
    from app.storage.models import Message,Job
    from test_autonomous_social import composed,add_source,NOW,FRIEND
    from test_harness_runtime import model_client,responses_sse
    url=os.environ['QQ151_LIVE_POST_URL']
    assert url.startswith('https://x.com/NASA/status/')
    crawler=XCrawlerClient('/run/xiaomachi-secrets/x-session.json','http://127.0.0.1:18797')
    monkeypatch.setattr(XCrawlerClient,'from_settings',staticmethod(lambda _:crawler))
    # A deliberately small test-only cap forces the real clip's lower native
    # version. Production retains its 128 MiB bound.
    monkeypatch.setattr('app.agent.twitter_media.MAX_VIDEO_BYTES',2*1024*1024)
    def script(body,n):
        if n==1:name,args='inspect_web_resources',{'ref_ids':[url]}
        elif n==2:name,args='set_task_goal',{'goal':'下载普通航空短视频','delivery':'media','requirements':['media_delivery'],
            'media_selection':{'candidate_reviews':[{'ref_id':url,'decision':'excluded','reason':'测试：模拟零选中'}]}}
        else:
            assert n==3
            name,args='send_group',{'text':'没有选中','artifact_ids':[],'task_status':'partial'}
        return responses_sse(name=name,arguments=args,number=n,call_id=f'ordinary-{n}')
    class Gateway:
        def __init__(self):self.calls=[]
        async def call_api(self,action,params):
            self.calls.append((action,params))
            return {'status':'ok','retcode':0,'data':{'message_id':'synthetic-public-video-receipt'}}
    gateway=Gateway();model,requests,_=model_client(script)
    web=WebSearchClient(provider='wyzlab',base_url='https://synthetic.invalid',api_key='synthetic',timeout_seconds=10)
    try:
        async with composed(sqlite_engine,tmp_path,model,runtime='harness') as fixture:
            fixture.executor.sender=Sender(gateway);fixture.router.web_search_client=web
            fixture.router.runtime.settings.web_page_proxy_url='http://127.0.0.1:18797'
            queue=fixture.executor.media_jobs;queue.running=True
            job=next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind=='group')
            source=add_source(sqlite_engine,platform_id='ordinary-public-video',text='请下载这个普通航空短视频 '+url)
            with session_scope(sqlite_engine) as session:
                row=session.get(Message,source)
                row.raw_json={'message':[{'type':'text','data':{'text':row.plain_text}}]}
            guard=SimpleNamespace(action_id='ordinary-public-video',valid=lambda:True,require_valid=lambda:None,on_dispatch=lambda:None)
            await fixture.executor.run_tools(SocialObservation(job,NOW,'synthetic',source_message_ids=(source,)),guard,
                interactive=True,target_user_id=FRIEND,target_message_id='ordinary-public-video',direct_group_reply=True)
            assert await queue.tick() and await queue.tick()
            await queue.stop()
            with session_scope(sqlite_engine) as session:
                task=session.query(Job).filter(Job.job_type=='agent_media_operation').one()
                data=task.payload_json.get('result',{}).get('data',{})
                report={'status':task.status,'source_url':url,'model_calls':len(requests),'synthetic_gateway_calls':len(gateway.calls),
                    'real_qq_sends':0,'paid_model_calls':0,'test_video_cap_bytes':2*1024*1024,
                    'downloaded_count':data.get('downloaded_count'),'media':[{k:m.get(k) for k in ('kind','bytes','quality','sha256')} for m in data.get('media',[])],
                    'video_quality_fallbacks':data.get('video_quality_fallbacks'),'failure_detail':data.get('failure_detail'),'error':data.get('error')}
            Path(os.environ['QQ151_LIVE_REPORT']).write_text(json.dumps(report,indent=2))
            assert task.status=='completed' and len(requests)==3 and len(gateway.calls)==1,report
            assert data['video_quality_fallbacks'] and data['media'][0]['quality']=='lower_native'
    finally:web.http_client.close()
