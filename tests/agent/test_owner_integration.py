"""Native Harness -> durable owner task -> real adapter -> original receipt."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy import select

from app.agent.owner_actions import OwnerActionService,OwnerActionWorker,OwnerNoticeWorker,build_owner_tools
from app.agent.owner_inputs import owner_image_inputs
from app.agent.artifacts import AutonomousArtifactStore
from app.storage.db import session_scope
from app.storage.models import Message,Job
from app.storage.agent_effects import AgentToolEffectRepository
from test_owner_actions import setup,source,OWNER,BOT,GROUP,NOW


class Guard:
    action_id='synthetic-owner-input'
    def valid(self): return True
    def require_valid(self): pass
    def on_dispatch(self): pass


def test_private_owner_inputs_use_only_canonical_own_cache_and_remain_revocable(sqlite_engine,tmp_path):
    service,_=setup(sqlite_engine,tmp_path)
    sid=source(sqlite_engine,group=None,text='用这张图片发空间')
    cache=tmp_path/'image_cache'/'private'/str(BOT)/str(OWNER)
    cache.mkdir(parents=True)
    own=cache/'own.png'; Image.new('RGB',(8,8),'red').save(own)
    foreign=tmp_path/'foreign.png'; Image.new('RGB',(8,8),'blue').save(foreign)
    with session_scope(sqlite_engine) as db:
        row=db.get(Message,sid)
        row.raw_json={**row.raw_json,'message':[
            {'type':'image','data':{'local_path':str(own)}},
            {'type':'image','data':{'local_path':str(foreign),'url':'https://synthetic.invalid/foreign.png'}}]}
    original=service.source(sid)
    store=AutonomousArtifactStore(tmp_path/'autonomous_artifacts',account_id=BOT,scope='owner:public:1',
        engine=sqlite_engine,source_message_ids=lambda:())
    with owner_image_inputs(service,original,Guard(),store) as images:
        assert len(images)==1 and images[0].source_user_id==OWNER
        assert images[0].url=='' and 'art-' in images[0].reference_subject
    with session_scope(sqlite_engine) as db:
        row=db.get(Message,sid); row.raw_json={**row.raw_json,'delivery_state':'deleted'}
    with pytest.raises(RuntimeError):
        with owner_image_inputs(service,original,Guard(),store): pass


@pytest.mark.parametrize('private_origin',[False,True])
@pytest.mark.parametrize('patrol_enabled',[False,True])
async def test_native_owner_public_comment_bypasses_quiet_and_returns_to_origin(sqlite_engine,tmp_path,private_origin,patrol_enabled):
    from test_autonomous_social import composed,FRIEND,BOT as ACCOUNT,GROUP as TARGET,NOW as INSTANT,writes
    from test_harness_runtime import model_client,responses_sse
    commands=[('read_friend_posts',{'owner_id':FRIEND}),
        ('comment_qzone',{'owner_id':FRIEND,'post_id':'observed-photo','text':'合成主人评论','artifact_ids':[]}),
        ('report_owner_result',{'text':'已收到评论平台回执','artifact_ids':[]})]
    def script(body,n):
        if n<=len(commands):
            name,args=commands[n-1]
            return responses_sse(name=name,arguments=args,number=n,call_id=f'owner-{n}')
        return responses_sse(number=n)
    client,requests,_=model_client(script)
    controller=900000999
    async with composed(sqlite_engine,tmp_path,client,kind='qzone_comment',runtime='harness',
            options={'max_daily':1,'max_daily_per_friend':1,'quiet_hours':['00:00','23:59']}) as f:
        await f.agent.stop()
        f.router.runtime.settings=f.router.runtime.settings.model_copy(update={'owner_qq':controller})
        repo=AgentToolEffectRepository(sqlite_engine,account_id=ACCOUNT,clock=lambda:INSTANT)
        if patrol_enabled:
            f.router.runtime.social_policy['qzone']['patrol']={'enabled':True,
                'interval_seconds':[5400,10800],'morning_delay_seconds':[1800,5400],
                'shared_cooldown_seconds':3600}
            prior=repo.reserve(f'public:qzone_comment:friend:{FRIEND}',run_id='prior-patrol',
                tool_call_id='prior',operation='comment_qzone',payload_hash='d'*64)
            assert repo.mark_dispatching(prior['effect_id'],daily_limit=1)
            repo.finish(prior['effect_id'],'sent')
        # No group/private context may enter the public agent, even when the
        # owner directs a write during the autonomous quiet period.
        service=OwnerActionService(sqlite_engine,runtime=f.router.runtime,clock=lambda:INSTANT)
        sid=source(sqlite_engine,author=controller,group=None if private_origin else TARGET,text='评论合成好友的最新动态')
        task=service.enqueue(sid,target='public')
        assert task['status']=='queued'
        worker=OwnerActionWorker(service,f.executor,connected=lambda:True)
        result=await worker.run_once()
        assert result['status']=='sent' and result['report']['text']=='已收到评论平台回执'
        assert [name for name,_ in writes(f)]==['comment_qzone']
        assert repo.count_dispatched(scope=f'public:qzone_comment:friend:{FRIEND}')==int(patrol_enabled)
        assert repo.count_dispatched(scope=f'owner:public:qzone_comment:friend:{FRIEND}')==1
        assert 'PRIVATE_PERSONA_FACT' not in str(requests[0])
        class PrivateSender:
            sent=[]
            async def send_private_text(self,message):
                self.sent.append(message); return ('synthetic-owner-private-receipt',)
        sender=PrivateSender() if private_origin else f.executor.sender
        notice=OwnerNoticeWorker(service,sender,surface='private' if private_origin else 'group',connected=lambda:True)
        assert (await notice.run_once()).status=='sent'
        assert await notice.run_once() is None
        if private_origin: assert len(sender.sent)==1 and sender.sent[0].user_id==controller


async def test_native_private_owner_can_enqueue_enabled_public_tools(sqlite_engine,tmp_path):
    from test_agent_service import setup as private_setup,event,sse
    service,runtime_router=setup(sqlite_engine,tmp_path)
    private,agent,dispatcher,sender,requests=private_setup(sqlite_engine,tmp_path,
        lambda body,n: sse('request_owner_action',{'target':'public','group_id':0,'include_current_images':False},number=n)
            if n==1 else sse('send_private',{'text':'已排队，等实际回执'},number=n) if n==2 else sse(number=n))
    agent.owner_service=OwnerActionService(sqlite_engine,runtime=runtime_router.runtime)
    await private.start()
    try:
        await dispatcher.handle_private_message(event(text='读取好友空间'))
        assert 'request_owner_action' in {tool['name'] for tool in requests[0]['tools']}
        with session_scope(sqlite_engine) as db:
            jobs=list(db.scalars(select(Job).where(Job.job_type=='owner_agent_action')))
            assert len(jobs)==1 and jobs[0].payload_json['source_group_id'] is None
    finally: await private.stop()

async def test_owner_can_switch_authorized_group_persona_without_qq_profile_write(sqlite_engine,tmp_path):
    from app.agent.service_tools import ServiceToolContext,build_service_tools
    from app.agent.social_contracts import SocialJob,SocialObservation
    from app.runtime.blocking import BoundedBlockingRunner
    service,router=setup(sqlite_engine,tmp_path)
    router.persona_manager.personas['synthetic_style']={'name':'合成人格','aliases':['合成别名']}
    guard=Guard(); guard.owner_authority=lambda:True
    store=AutonomousArtifactStore(tmp_path/'artifacts',account_id=BOT,scope=f'owner:group:{GROUP}:1',
        engine=sqlite_engine,source_message_ids=lambda:())
    blocking=BoundedBlockingRunner(1)
    try:
        context=ServiceToolContext(router,SocialObservation(SocialJob('group:'+str(GROUP),'group',GROUP),NOW,''),
            guard,set(),blocking,store)
        tool=next(tool for tool in build_service_tools(context) if tool.name=='set_group_persona')
        outcome=await tool.handler({'persona':'合成别名'})
        assert outcome.data['applied'] and router.persona_manager.active_key(GROUP)=='synthetic_style'
    finally: blocking.close()

async def test_native_owner_group_turn_does_not_recover_a_live_completion_notice(sqlite_engine,tmp_path):
    from test_autonomous_social import composed,BOT as ACCOUNT,GROUP as TARGET,NOW as INSTANT,InteractiveIngressGuard
    from test_harness_runtime import model_client,responses_sse
    from app.adapters.onebot_models import parse_group_message_event
    from app.agent.contracts import AgentBatch,AgentDecision
    client,requests,_=model_client(lambda body,n:responses_sse(name='send_group',arguments={'text':'合成主人回应','artifact_ids':[]},
        call_id='owner-chat',number=n) if n==1 else responses_sse(number=n))
    async with composed(sqlite_engine,tmp_path,client,runtime='harness') as f:
        await f.agent.stop()
        repo=AgentToolEffectRepository(sqlite_engine,account_id=ACCOUNT,clock=lambda:INSTANT)
        live=repo.reserve(f'owner:group:{TARGET}',run_id='owner-notice:synthetic-live',tool_call_id='notice',
            operation='owner_action_notice',payload_hash='a'*64)
        assert repo.mark_dispatching(live['action_id'],daily_limit=None)
        actor=f.router.runtime.settings.owner_qq
        event=parse_group_message_event({'post_type':'message','message_type':'group','group_id':TARGET,
            'message_id':'synthetic-owner-direct','user_id':actor,'time':INSTANT.timestamp(),
            'sender':{'nickname':'synthetic owner'},'message':[
                {'type':'at','data':{'qq':str(ACCOUNT)}},{'type':'text','data':{'text':'在吗'}}]},bot_qq=ACCOUNT,bot_name='小町')
        f.router.ingest_live_group_message(event)
        batch=AgentBatch(TARGET,(event,),'synthetic-owner-native')
        result=await f.executor.execute_group_turn(event,batch,AgentDecision('reply',event.platform_msg_id,'direct_trigger'),
            InteractiveIngressGuard(batch.turn_id))
        assert result.status=='sent' and not result.charge_proactive_cadence
        assert 'request_owner_action' in {tool['name'] for tool in requests[0]['tools']}
        assert repo.get(live['action_id'])['status']=='dispatching'
