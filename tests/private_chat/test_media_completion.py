"""A completed original media goal settles its late callback without a model."""
from dataclasses import replace

from PIL import Image
import pytest

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.tool_agent import ToolTurnResult
from app.private_chat.agent_service import _completed_media_request, _mark_completed_media_request
from app.storage.db import session_scope
from app.storage.models import Job
from app.storage.repositories import MessageRepository, UserRepository
from tests.private_chat.test_agent_service import setup, event, ACCOUNT, OWNER


@pytest.mark.parametrize('case', ['completed', 'partial', 'progress', 'uncertain', 'missing_receipt',
    'text', 'other_source', 'gaps', 'no_media', 'wrong_operation'])
def test_completion_requires_full_goal_and_confirmed_media_receipt(case):
    state = {'source_id': 41, 'delivery': 'media', 'status': 'completed', 'completion_gaps': []}
    result = ToolTurnResult('sent', 'synthetic-receipt', task_status=state)
    receipt = {'reply_source_id': 41, 'operation': 'send_private', 'status': 'sent',
        'receipt_id': result.receipt_id, 'media': True}
    if case in {'partial', 'progress'}: state['status'] = case
    if case == 'uncertain': result = replace(result, status='uncertain')
    if case == 'missing_receipt': result = replace(result, receipt_id=None)
    if case == 'text': state['delivery'] = 'text'
    if case == 'other_source': state['source_id'] = 42
    if case == 'gaps': state['completion_gaps'] = ['media_quantity_incomplete']
    if case == 'no_media': receipt['media'] = False
    if case == 'wrong_operation': receipt['operation'] = 'generate_private_image'
    _mark_completed_media_request([receipt], 41, result)
    assert _completed_media_request({'receipts': [receipt]}, 41) == (case == 'completed')
    assert not _completed_media_request({'receipts': [receipt]}, 42)


@pytest.mark.parametrize('stage', ['completed', 'partial'])
async def test_late_private_media_callback_settles_only_completed_original(sqlite_engine, tmp_path, stage):
    def forbidden(*args): raise AssertionError('fixture must not call a model')
    service, agent, dispatcher, sender, _ = setup(sqlite_engine, tmp_path, forbidden)
    async def send_media(*, user_id, text, image_files):
        sender.texts.append((user_id, text))
        sender.images.extend((user_id, path) for path in image_files)
        return ('synthetic-media-receipt',)
    sender.send_private_media = send_media
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=OWNER, nickname='synthetic', group_card='')
        source = MessageRepository(session).add_private_message(platform_msg_id='synthetic-picture-source',
            user_id=OWNER, timestamp=event().timestamp, plain_text='synthetic saved picture',
            raw_json=event().raw_payload)
        session.flush(); picture_source = source.id
    path = tmp_path/'fixture.png'; Image.new('RGB', (20, 30), 'navy').save(path)
    store = AutonomousArtifactStore(tmp_path/'autonomous_artifacts', account_id=ACCOUNT,
        scope=f'private:{OWNER}', engine=sqlite_engine, source_message_ids=lambda: (picture_source,),
        allowed_input_roots=(tmp_path,))
    artifact = store.add(path, 'web_page_image')
    runs = []
    class Harness:
        def __init__(self, **kwargs):
            self.invoke = kwargs['invoke_tool']
            self.source_id = kwargs['reply_source_id']
        async def run(self, prompt, tools, guard, **kwargs):
            runs.append(self.source_id)
            if len(runs) > 1: return ToolTurnResult('silent', reason='callback_processed')
            tool = next(t for t in tools if t.name == 'send_private')
            outcome = await self.invoke(tool, {'text': '已核对图片', 'artifact_ids': [artifact],
                'task_status': stage}, 'synthetic-send')
            assert outcome.status == 'sent'
            return ToolTurnResult('sent', outcome.receipt_id, task_status={'source_id': self.source_id,
                'delivery': 'media', 'status': stage, 'completion_gaps': [] if stage == 'completed'
                    else ['media_quantity_incomplete']})
        def close(self): pass
    agent.runtime_factory = Harness
    await service.start()
    try:
        await dispatcher.handle_private_message(event('synthetic-media-request', text='查看并发送保存的图片'))
        snapshot = agent.sessions.read(f'private:{OWNER}')
        source_id = snapshot['checkpoint']['reply_source_id']
        assert _completed_media_request(snapshot['checkpoint'], source_id) == (stage == 'completed')
        queued = agent._enqueue(user_id=OWNER, source_id=source_id, private_scope='owner',
            kind='research_result', delegated_result={'kind': 'media_result'},
            dependencies={'sources': [source_id]}, key='synthetic-late-result')
        await agent._drain_user(OWNER)
        with session_scope(sqlite_engine) as session:
            job = session.get(Job, queued['task_id'])
            assert job.status == 'completed'
            assert job.last_error_code == ('original_media_request_already_completed'
                if stage == 'completed' else 'callback_processed')
        assert len(runs) == (1 if stage == 'completed' else 2)
        assert len(sender.images) == 1
        if stage == 'completed':
            assert agent.sessions.read(f'private:{OWNER}')['revision'] == snapshot['revision']
            # The same words from a new canonical source remain a new request.
            await dispatcher.handle_private_message(event('synthetic-new-request', text='再发保存的图片'))
            assert len(runs) == 2 and runs[-1] != source_id
    finally:
        await service.stop(); service.llm_client.http_client.close()
