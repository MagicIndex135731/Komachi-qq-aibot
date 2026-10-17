from types import SimpleNamespace

import pytest

from app.agent.media_tasks import deliver_download, media_result, media_store
from app.agent.tool_agent import ToolOutcome
from app.core.downloaded_media import DownloadedMediaCache
from app.storage.db import session_scope
from app.storage.models import Job, Message
from test_media_tasks import setup_queue


@pytest.mark.parametrize('state',['sent','rejected','uncertain','revoked','cancelled','fallback'])
async def test_automatic_delivery_fences_sources_unknown_results_and_one_failure_notice(sqlite_engine,tmp_path,state):
    from test_autonomous_social import image_bytes
    settings=SimpleNamespace(data_dir=tmp_path)
    class Runner:
        async def run(self,fn,*args,**kwargs):return fn(*args,**kwargs)
    calls=[]
    async def execute(payload,guard):
        store=media_store(payload,guard,engine=sqlite_engine,settings=settings)
        store.root.mkdir(parents=True,exist_ok=True)
        path=store.root/'synthetic.png';path.write_bytes(image_bytes())
        identifier=DownloadedMediaCache.for_store(store,settings).register(store,path,'web_page_image')
        return media_result(ToolOutcome({'artifact_ids':[identifier],
            **({'quality_fallbacks':[{'index':1,'image_size':'large','reason':'http_503'}]} if state=='fallback' else {})}),payload)
    async def send(boundary,files,text):
        boundary.on_dispatch()
        calls.append((len(files),text))
        if files and state=='uncertain':return ToolOutcome({},'uncertain')
        if files and state=='rejected':return ToolOutcome({'error':'qq_send_rejected'},'failed')
        return ToolOutcome({},'sent','synthetic-receipt')
    async def deliver(payload,guard):
        return await deliver_download(payload,guard,engine=sqlite_engine,settings=settings,
            runner=Runner(),send=send,clock=queue.clock,conversation_key=tmp_path/'logical-private')
    queue,payload,_=setup_queue(sqlite_engine,execute=execute,deliver=deliver)
    payload['arguments']['delivery_mode']='auto'
    outcome=queue.submit(payload,parent_turn=42,before_commit=lambda:None)
    assert await queue.tick()
    if state=='revoked':
        with session_scope(sqlite_engine) as session:
            session.get(Message,payload['source_id']).raw_json={'delivery_state':'deleted'}
    if state=='cancelled':queue.cancel_scope(payload['scope_id'])
    await queue.tick()
    assert not await queue.tick()
    assert [n for n,_ in calls]==([1,0] if state=='rejected' else [] if state in {'revoked','cancelled'} else [1])
    if state=='fallback':assert 'large' in calls[0][1] and '非原图' in calls[0][1]
    with session_scope(sqlite_engine) as session:
        row=session.get(Job,outcome.data['job_id'])
        assert row.status=={'sent':'completed','fallback':'completed','rejected':'failed','uncertain':'uncertain',
            'revoked':'blocked','cancelled':'cancelled'}[state]


async def test_delivery_mode_does_not_create_duplicate_download_and_cannot_authorize_background_group(sqlite_engine):
    queue,payload,_=setup_queue(sqlite_engine)
    first=queue.submit(payload,parent_turn=42,before_commit=lambda:None)
    payload['arguments'].update(delivery_mode='auto',delivery_text='说明')
    second=queue.submit(payload,parent_turn=42,before_commit=lambda:None)
    assert first.data['job_id']==second.data['job_id']
    with session_scope(sqlite_engine) as session:
        assert session.get(Job,first.data['job_id']).payload_json['delivery_mode']=='review'
    payload['surface']='group'
    assert queue.submit(payload,parent_turn=43,before_commit=lambda:None).status=='blocked'
