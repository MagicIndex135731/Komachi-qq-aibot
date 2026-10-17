"""Deferred fallback candidates share the original queue, guard and budget."""
from app.agent.media_tasks import local_fallback_plan
from app.agent.tool_agent import ToolOutcome
from app.storage.db import session_scope
from app.storage.models import Job,Message
from test_media_tasks import setup_queue
import pytest
from test_web_video_delivery import mp4


def enqueue(queue, payload, index, *, minimum=2, types=('video',), key='synthetic-plan'):
    token=local_fallback_plan.set({'key':key,'wave':index//3,'minimum_count':minimum,
        'media_types':list(types),**{k:payload[k] for k in ('scope','actor_id','source_id')}})
    try:
        return queue.submit({**payload,'arguments':{'ref_id':f'https://example.com/ordinary/{index}',
            'delivery_mode':'auto'}},parent_turn=42,before_commit=lambda:None).data['job_id']
    finally:local_fallback_plan.reset(token)


@pytest.mark.parametrize('first_status', ['sent', 'failed', 'uncertain'])
async def test_parallel_delivery_fence_preserves_three_item_goal_and_replacement(sqlite_engine, first_status):
    import asyncio
    from app.agent.media_delivery import send_media_once
    from app.storage.agent_effects import AgentToolEffectRepository
    entered, release = asyncio.Event(), asyncio.Event()
    downloads, sends = [], []
    async def execute(payload, guard):
        index = int(payload['arguments']['ref_id'].rsplit('/', 1)[1])
        downloads.append(index)
        return {'status':'ok', 'data':{'video_artifact_id':f'art-video-{index}'}}
    async def deliver(payload, guard):
        identifier = payload['result']['data']['video_artifact_id']
        index = int(identifier.rsplit('-', 1)[1])
        async def send():
            guard.on_dispatch()
            sends.append(index)
            if index == 0 and first_status != 'sent':
                if first_status == 'uncertain':entered.set()
                return ToolOutcome({}, first_status)
            if len(sends) == 3:entered.set()
            await release.wait()
            return ToolOutcome({}, 'sent', f'receipt-{index}')
        outcome = await send_media_once(AgentToolEffectRepository(sqlite_engine, account_id=payload['account_id']),
            scope=payload['scope'], source_id=payload['source_id'], operation='send_private',
            arguments={'artifact_ids':[identifier]}, require_valid=guard.require_valid, invoke=send)
        return {'sent':'completed', 'failed':'failed', 'uncertain':'uncertain', 'blocked':'blocked'}[outcome.status]
    queue, payload, _ = setup_queue(sqlite_engine, execute=execute, deliver=deliver)
    jobs = [enqueue(queue, payload, i, minimum=3) for i in range(5)]
    async def worker():
        for _ in range(20):
            if not await queue.tick():return
        raise AssertionError('fallback queue did not settle')
    workers = [asyncio.create_task(worker()) for _ in range(3)]
    try:
        async with asyncio.timeout(3):await entered.wait()
        if first_status != 'uncertain':assert sends == [0, 1, 2] and downloads == [0, 1, 2]
    finally:
        release.set()
        async with asyncio.timeout(5):await asyncio.gather(*workers)
    assert sends == {'sent':[0, 1, 2], 'failed':[0, 1, 2, 3], 'uncertain':[0]}[first_status]
    with session_scope(sqlite_engine) as session:
        rows = [session.get(Job, job) for job in jobs]
        if first_status == 'uncertain':
            assert [row.status for row in rows] == ['uncertain'] + ['blocked']*4
        else:
            assert all(row.status == 'completed' for row in rows[1:])
            assert rows[0].status == ('failed' if first_status == 'failed' else 'completed')
            skipped = rows[4:] if first_status == 'failed' else rows[3:]
            assert all(row.attempt_count == 0 and row.last_error_code == 'media_fallback_goal_satisfied' for row in skipped)


async def test_first_three_fail_then_two_remaining_complete_without_another_model(sqlite_engine):
    downloads=[]; delivered=[]
    async def execute(payload,guard):
        index=int(payload['arguments']['ref_id'].rsplit('/',1)[1]);downloads.append(index)
        if index<3:return {'status':'failed','data':{'error':'synthetic_unavailable'}}
        return {'status':'ok','data':{'video_artifact_id':f'art-video-{index}'}}
    async def deliver(payload,guard):
        if payload['result']['status']=='ok':delivered.append(payload['result']['data']['video_artifact_id']);return 'completed'
        return 'failed'
    queue,payload,_=setup_queue(sqlite_engine,execute=execute,deliver=deliver)
    jobs=[enqueue(queue,payload,i) for i in range(5)]
    # The first wave can run concurrently; overflow waits for its settlements.
    for _ in range(6):
        assert await queue.tick()
        if len(downloads)==3:break
    assert downloads==[0,1,2]
    for _ in range(20):
        if not await queue.tick():break
    assert downloads==[0,1,2,3,4] and len(delivered)==2
    with session_scope(sqlite_engine) as session:
        assert [session.get(Job,j).status for j in jobs]==['failed']*3+['completed']*2


async def test_goal_satisfied_skips_remaining_downloads_and_keeps_legacy_jobs(sqlite_engine):
    downloads=[]
    async def execute(payload,guard):
        index=payload['arguments']['ref_id'];downloads.append(index)
        return {'status':'ok','data':{'video_artifact_id':'art-'+index.rsplit('/',1)[1]}}
    async def deliver(payload,guard):return 'completed'
    queue,payload,_=setup_queue(sqlite_engine,execute=execute,deliver=deliver)
    jobs=[enqueue(queue,payload,i) for i in range(5)]
    for _ in range(20):
        if not await queue.tick():break
    assert len(downloads)==2
    with session_scope(sqlite_engine) as session:
        assert all(session.get(Job,j).last_error_code=='media_fallback_goal_satisfied' for j in jobs[2:])
    legacy=queue.submit(payload,parent_turn=100,before_commit=lambda:None)
    assert await queue.tick() and len(downloads)==3


async def test_unknown_first_wave_fences_every_remaining_candidate(sqlite_engine):
    queue,payload,_=setup_queue(sqlite_engine)
    jobs=[enqueue(queue,payload,i) for i in range(5)]
    with session_scope(sqlite_engine) as session:
        session.get(Job,jobs[0]).status='uncertain'
        session.get(Job,jobs[1]).status='result_ready'
    assert queue.claim() is None
    with session_scope(sqlite_engine) as session:
        assert all(session.get(Job,j).status=='blocked' for j in jobs[1:])


async def test_skipped_fallback_candidates_settle_persisted_request_jobs(sqlite_engine,tmp_path):
    from app.agent.research_tasks import ResearchTaskWorkspace
    queue,payload,_=setup_queue(sqlite_engine)
    queue.task_workspace_root=tmp_path
    task=ResearchTaskWorkspace(tmp_path,scope=payload['scope'],actor_id=payload['actor_id'],
        source_id=payload['source_id'],request='普通视频测试',valid_source=lambda _:True,
        clock=lambda:queue.clock().timestamp())
    payload={**payload,'parent_task_id':task.task_id}
    jobs=[enqueue(queue,payload,i) for i in range(5)]
    for job in jobs:task.record_job_result(job,'queued')
    with session_scope(sqlite_engine) as session:
        for job in jobs[:2]:
            row=session.get(Job,job);row.status='completed'
            row.payload_json={**row.payload_json,'result':{'status':'ok','data':{'video_artifact_id':f'art-{job}'}}}
    for job in jobs[:2]:task.record_job_result(job,'completed')
    for _ in range(3):queue.claim()
    assert task.summary()['background_jobs']=={str(job):'completed' for job in jobs}


async def test_type_gap_and_repeated_content_do_not_count_as_completion(sqlite_engine):
    queue,payload,_=setup_queue(sqlite_engine)
    jobs=[enqueue(queue,payload,i,types=('image','video')) for i in range(5)]
    with session_scope(sqlite_engine) as session:
        for job in jobs[:3]:
            row=session.get(Job,job);row.status='completed'
            row.payload_json={**row.payload_json,'result':{'status':'ok','data':{'media':[
                {'artifact_id':'art-same','kind':'video','sha256':'same-bytes'}]}}}
    claimed=queue.claim()
    assert claimed and claimed[0]==jobs[3] and claimed[2]=='running'


def test_fallback_submission_cannot_borrow_another_source(sqlite_engine):
    queue,payload,_=setup_queue(sqlite_engine)
    token=local_fallback_plan.set({'scope':'private:foreign','actor_id':0,'source_id':0})
    try:
        outcome=queue.submit(payload,parent_turn=42,before_commit=lambda:None)
        assert outcome.status=='blocked' and outcome.data['error']=='media_fallback_out_of_scope'
    finally:local_fallback_plan.reset(token)


@pytest.mark.parametrize('surface',['group','private'])
@pytest.mark.parametrize('uncertain',[False,True])
async def test_zero_selected_video_queue_replenishes_after_three_failures(
        sqlite_engine,tmp_path,monkeypatch,mp4,surface,uncertain):
    import asyncio,base64,subprocess,shutil
    from datetime import UTC,datetime
    from types import SimpleNamespace
    import httpx
    from app.adapters.sender import Sender,QQMessageDeliveryUncertainError
    from app.providers.web_search import WebSearchClient
    from app.agent.social_contracts import SocialObservation
    from app.agent.social_policy import configured_social_jobs
    from test_autonomous_social import composed,add_source,NOW,FRIEND
    from test_harness_runtime import model_client,responses_sse
    from tests.private_chat.test_agent_service import setup,event,OTHER
    blue=tmp_path/'blue.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-f','lavfi','-i','color=c=blue:s=64x64:d=0.3',
        '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(blue)],check=True,timeout=15)
    videos=[mp4,blue.read_bytes()]
    base=900000000000000000
    urls=[f'https://x.com/synthetic/status/{base+i}' for i in range(1,6)]
    class Crawler:
        def search_posts(self,*args,**kwargs):
            return {'source_url':'https://x.com/search?q=space','posts':[
                {'source_url':url,'post_id':str(base+i),'author':'synthetic','identity_status':'matched',
                 'published_at':datetime.now(UTC).isoformat(),'media_count':1,'media_roles':['post_video'],
                 'text':'普通航天短片，用于自动测试'} for i,url in enumerate(urls,1)]}
        def inspect(self,url,**kwargs):
            index=int(url.rsplit('/',1)[1])-base
            return {'source_url':url,'post_id':str(base+index),'author':'synthetic','identity_status':'matched',
                'protected':False,'evidence_ref':'x-posts-'+'a'*24+':post:0','media_count':1,
                'media':[{'kind':'video','url':f'https://video.twimg.com/high-{index}.mp4','variants':[
                    {'url':f'https://video.twimg.com/high-{index}.mp4','bitrate':80_000_000,'duration_seconds':20,'codec':'h264'},
                    {'url':f'https://video.twimg.com/low-{index}.mp4','bitrate':500_000,'duration_seconds':1201 if index<=3 else 20,'codec':'h264'}]}]}
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.from_settings',staticmethod(lambda _:Crawler()))
    transfers=[]
    def respond(request):
        transfers.append(request.url.path)
        index=int(request.url.path.rsplit('-',1)[1].split('.')[0])
        assert index in {4,5} and '/low-' in request.url.path
        return httpx.Response(200,content=videos[index-4],headers={'content-type':'video/mp4'})
    monkeypatch.setattr('app.agent.twitter_media.download_client',lambda _:httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    if surface=='group':monkeypatch.setattr('app.agent.context_dependencies.datetime',SimpleNamespace(now=lambda tz:NOW))
    def script(body,n):
        if n==1:name,args='search_x_posts',{'mode':'semantic','subject_terms':['航天'],'media_types':['video'],'limit':5}
        elif n==2:
            from test_web_image_delivery import output
            assert len(output(body)['resources'])==5
            name,args='set_task_goal',{'goal':'找两个普通航天视频','delivery':'media','requirements':['media_delivery'],
            'media_selection':{'candidate_reviews':[{'ref_id':url,'decision':'excluded','reason':'测试中模拟模型选中零条'} for url in urls]}}
        else:
            assert n==3
            name,args='send_'+surface,{'text':'没有选中视频','artifact_ids':[],'task_status':'partial'}
        return responses_sse(name=name,arguments=args,number=n,call_id=f'zero-{n}')
    class Gateway:
        def __init__(self):self.calls=[]
        async def call_api(self,action,params):
            self.calls.append((action,params))
            if uncertain:raise QQMessageDeliveryUncertainError('synthetic lost receipt')
            return {'status':'ok','retcode':0,'data':{'message_id':'video-wave-'+str(len(self.calls))}}
    gateway=Gateway();web=WebSearchClient(provider='wyzlab',base_url='https://synthetic.invalid',api_key='synthetic',timeout_seconds=10)
    try:
        if surface=='group':
            client,requests,_=model_client(script)
            async with composed(sqlite_engine,tmp_path,client,runtime='harness') as fixture:
                fixture.executor.sender=Sender(gateway);fixture.router.web_search_client=web
                queue=fixture.executor.media_jobs;queue.running=True
                job=next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind=='group')
                source=add_source(sqlite_engine,platform_id='wave-video',text='小町请去X找两个航天视频发来')
                with session_scope(sqlite_engine) as session:
                    row=session.get(Message,source)
                    row.raw_json={'message':[{'type':'text','data':{'text':row.plain_text}}]}
                guard=SimpleNamespace(action_id='wave-video',valid=lambda:True,require_valid=lambda:None,on_dispatch=lambda:None)
                await fixture.executor.run_tools(SocialObservation(job,NOW,'synthetic',source_message_ids=(source,)),guard,
                    interactive=True,target_user_id=FRIEND,target_message_id='wave-video',direct_group_reply=True)
                for _ in range(24):
                    if not await queue.tick():break
                await queue.stop()
        else:
            service,agent,dispatcher,_,requests=setup(sqlite_engine,tmp_path,script)
            service.sender=Sender(gateway);service.web_search_client=web
            await service.start()
            try:
                await dispatcher.handle_private_message(event('wave-video',user_id=OTHER,text='去X找两个航天视频发给我'))
                async with asyncio.timeout(30):
                    while True:
                        with session_scope(sqlite_engine) as session:
                            rows=list(session.query(Job).filter(Job.job_type=='agent_media_operation'))
                            if len(rows)==5 and all(r.status in {'completed','failed','blocked','uncertain'} for r in rows):break
                        await asyncio.sleep(.05)
            finally:await service.stop()
        assert len(requests)==3
        assert len(gateway.calls)==(1 if uncertain else 2)
        sent=[]
        for action,params in gateway.calls:
            segments=list(params.get('message',[])) if isinstance(params.get('message'),list) else []
            for node in params.get('messages',[]):segments+=node['data']['content']
            sent.extend(base64.b64decode(s['data']['file'][9:]) for s in segments if s['type']=='video')
            assert any('较低清晰度' in s['data'].get('text','') for s in segments)
        assert sent==videos[:len(sent)]
        assert not any('high-' in path for path in transfers)
        with session_scope(sqlite_engine) as session:
            jobs=list(session.query(Job).filter(Job.job_type=='agent_media_operation').order_by(Job.id))
            assert len(jobs)==5 and all(j.status=='failed' for j in jobs[:3])
            for job in jobs[:3]:
                details=job.payload_json['result']['data']['failure_detail']
                assert details['rejected_by_reason']=={'duration_limit':1,'estimated_size_limit':1}
            assert jobs[3].status==('uncertain' if uncertain else 'completed')
            assert jobs[4].status==('blocked' if uncertain else 'completed')
    finally:web.http_client.close()
