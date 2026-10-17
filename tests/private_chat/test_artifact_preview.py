"""Private media selection can view owned pixels without project control."""
from datetime import UTC,datetime

from PIL import Image
import pytest

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.tool_agent import ToolTurnResult
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository,UserRepository
from tests.private_chat.test_agent_service import setup,event,ACCOUNT,OTHER


@pytest.mark.parametrize('case',['owned','other_user','revoked'])
@pytest.mark.parametrize('media_kind',['image','video'])
async def test_private_read_artifact_is_offered_and_fences_pixels(sqlite_engine,tmp_path,case,media_kind):
    def forbidden(*args):raise AssertionError('fixture must not call a model')
    service,agent,dispatcher,_,_=setup(sqlite_engine,tmp_path,forbidden)
    actor=OTHER if case!='other_user' else OTHER+1
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=actor,nickname='synthetic',group_card='')
        source=MessageRepository(session).add_private_message(platform_msg_id='synthetic-old-image',
            user_id=actor,timestamp=datetime.now(UTC),plain_text='synthetic saved photo',
            raw_json={'message':[{'type':'text','data':{'text':'synthetic saved photo'}}]})
        session.flush()
        identifier=source.id
    photo=tmp_path/'fixture.png';Image.new('RGB',(20,30),'navy').save(photo)
    store=AutonomousArtifactStore(tmp_path/'autonomous_artifacts',account_id=ACCOUNT,
        scope=f'private:{actor}',engine=sqlite_engine,source_message_ids=lambda:(identifier,),
        allowed_input_roots=(tmp_path,))
    if media_kind=='video':
        from app.core.downloaded_media import DownloadedMediaCache
        cache=DownloadedMediaCache.for_store(store);cache.root.mkdir(parents=True,exist_ok=True)
        video=cache.root/'video.part';video.write_bytes(b'synthetic-video')
        artifact=cache.save_video(store,video,{'bytes':15,'duration_seconds':2})
    else:artifact=store.add(photo,'web_page_image')
    if case=='revoked':
        from app.storage.models import Message
        with session_scope(sqlite_engine) as session:
            session.get(Message,identifier).raw_json={'delivery_state':'deleted','message':[]}
    outcomes=[]
    class Harness:
        def __init__(self,**kwargs):pass
        async def run(self,prompt,tools,guard,**kwargs):
            registry={tool.name:tool for tool in tools}
            assert 'read_artifact' in registry
            outcome=await registry['read_artifact'].handler({'artifact_id':artifact})
            outcomes.append(outcome)
            return ToolTurnResult('silent',reason='fixture_finished')
        def close(self):pass
    agent.runtime_factory=Harness
    await service.start()
    try:await dispatcher.handle_private_message(event('preview-request',user_id=OTHER,text='查看已保存图片'))
    finally:await service.stop();service.llm_client.http_client.close()
    assert len(outcomes)==1
    assert bool(outcomes[0].images)==(case=='owned' and media_kind=='image')
    assert outcomes[0].status==('ok' if case=='owned' and media_kind=='image' else 'blocked')


async def test_private_native_read_artifact_attaches_real_pixels_to_main_model(sqlite_engine,tmp_path):
    import json
    from tests.private_chat.test_agent_service import sse
    artifact=None
    observations=[]
    def script(body,n):
        observations.append({'call':n,'pixels':'data:image/' in json.dumps(body),
            'frames':'frames_attached' in json.dumps(body)})
        if n==1:
            assert 'read_artifact' in {t['name'] for t in body['tools']}
            return sse('read_artifact',{'artifact_id':artifact},number=n)
        if n==2:
            assert 'data:image/' in json.dumps(body)
            assert 'pixels_attached' in json.dumps(body)
            return sse('send_private',{'text':'已核对实际图片'},number=n)
        return sse(number=n)
    service,agent,dispatcher,sender,_=setup(sqlite_engine,tmp_path,script)
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=OTHER,nickname='synthetic',group_card='')
        source=MessageRepository(session).add_private_message(platform_msg_id='previous-own-photo',
            user_id=OTHER,timestamp=datetime.now(UTC),plain_text='synthetic photo',
            raw_json={'message':[{'type':'text','data':{'text':'synthetic photo'}}]})
        session.flush();identifier=source.id
    path=tmp_path/'native-preview.png';Image.new('RGB',(24,36),'crimson').save(path)
    store=AutonomousArtifactStore(tmp_path/'autonomous_artifacts',account_id=ACCOUNT,scope=f'private:{OTHER}',
        engine=sqlite_engine,source_message_ids=lambda:(identifier,),allowed_input_roots=(tmp_path,))
    artifact=store.add(path,'web_page_image')
    await service.start()
    try:await dispatcher.handle_private_message(event('native-preview-request',user_id=OTHER,text='分析这张已存图片的颜色'))
    finally:await service.stop();service.llm_client.http_client.close()
    assert sender.texts==[(OTHER,'已核对实际图片')],observations
