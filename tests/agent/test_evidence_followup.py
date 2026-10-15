"""Shared Agent wiring recovers private follow-ups before media delivery."""
import asyncio
import json

import httpx
import pytest

from app.adapters.sender import Sender
from app.providers.gemini_search import GeminiSearchClient
from app.providers.web_search import WebSearchClient
from test_twitter_media_tools import URL, provider
from test_web_video_delivery import mp4
from tests.private_chat.test_agent_service import OWNER, OTHER, event, setup, sse


def output(body, name):
    names = {item['call_id']: item['name'] for item in body.get('input', [])
             if item.get('type') == 'function_call'}
    value = next(item['output'] for item in reversed(body['input'])
                 if item.get('type') == 'function_call_output' and names.get(item['call_id']) == name)
    if isinstance(value, list):
        value = next(part['text'] for part in value if part.get('type') in {'text', 'input_text'})
    return json.loads(value)


@pytest.mark.parametrize('owner', [False, True])
async def test_followup_recovers_old_evidence_and_finds_post_inside_article(
        sqlite_engine, tmp_path, monkeypatch, mp4, owner):
    downloads = provider(monkeypatch, video=mp4)
    holder, delivered, page_reads = {}, [], []
    article = 'https://example.com/article'
    def search(request):
        return httpx.Response(200, json={'output': '10月13日的报道介绍某作者的公开照片。',
            'results': [{'type': 'text_result', 'title': '公开报道', 'url': article,
                         'ref_id': 'previous-article', 'snippet': '媒体报道，并非原帖日期'}]})
    def page(request):
        page_reads.append(request)
        return httpx.Response(200, headers={'content-type': 'text/html'}, text=
            '<article><p>报道日期是10月13日，原帖日期是10月11日。</p>'
            '<blockquote><a href="' + URL + '">原始帖子</a></blockquote></article>')
    web = WebSearchClient(provider='wyzlab_alpha', base_url='https://synthetic.invalid/v1',
        api_key='synthetic', timeout_seconds=10,
        http_client=httpx.Client(transport=httpx.MockTransport(search)))
    reader = GeminiSearchClient(provider='local_page_reader', base_url='', api_key='', model='',
        timeout_seconds=10, http_client=httpx.Client(transport=httpx.MockTransport(page)),
        address_resolver=lambda *_: ['93.184.216.34'])
    web.page_reader = reader
    def script(body, n):
        if n == 1: return sse('search_web', {'query': '作者最新帖子'}, number=n)
        if n == 2:
            data = output(body, 'search_web')['data']
            holder.update(task_id=data['task_id'], evidence_id=data['evidence_id'])
            return sse('send_private', {'text': '找到10月13日的报道'}, number=n)
        if n == 3:
            return sse('read_research_evidence', {'evidence_id': holder['evidence_id']}, number=n)
        if n == 4:
            assert output(body, 'read_research_evidence')['data']['error'] == 'task_evidence_unavailable'
            return sse('send_private', {'text': '没有单帖链接所以不能下载', 'task_status': 'blocked'}, number=n)
        if n == 5:
            feedback = output(body, 'send_private')
            assert feedback['status'] == 'blocked'
            assert 'recoverable_lookup_not_task_blocker' in feedback['data']['missing']
            return sse('read_research_evidence', holder, number=n)
        if n == 6:
            assert article in json.dumps(output(body, 'read_research_evidence')['data'])
            return sse('open_web_page', {'ref_id': article}, number=n)
        if n == 7:
            assert URL in output(body, 'open_web_page')['data']['content']
            return sse('download_twitter_media', {'ref_id': URL, 'delivery_mode': 'auto',
                'delivery_text': '报道提到的原帖媒体；报道日期与发帖日期不同'}, number=n)
        if n == 8:
            return sse('send_private', {'text': '已从报道中找到原帖，正在下载', 'task_status': 'progress'}, number=n)
        raise AssertionError('Unexpected extra model request')
    class Gateway:
        async def call_api(self, action, params):
            delivered.append((action, params))
            return {'status': 'ok', 'retcode': 0,
                    'data': {'message_id': 'synthetic-followup-' + str(len(delivered))}}
    service, agent, dispatcher, _, requests = setup(sqlite_engine, tmp_path, script)
    service.sender, service.web_search_client = Sender(Gateway()), web
    actor = OWNER if owner else OTHER
    await service.start()
    try:
        await dispatcher.handle_private_message(event('older-question', user_id=actor, text='有更新的帖子吗'))
        current_text = '那把这个十月十三号的图片发给我'
        await dispatcher.handle_private_message(event('media-followup', user_id=actor, text=current_text))
        assert len(requests) == 8
        async with asyncio.timeout(20):
            while not any(action.endswith('forward_msg') for action, _ in delivered):
                await asyncio.sleep(.05)
        forwards = [params for action, params in delivered if action.endswith('forward_msg')]
        assert len(forwards) == 1 and len(page_reads) == 1 and len(downloads) == 4
        assert '没有单帖链接所以不能下载' not in str(delivered)
        state_file = next((tmp_path / 'research_tasks').glob('*/request-goal.json'))
        assert json.loads(state_file.read_text())['control']['goal'] == current_text
    finally:
        await service.stop()
        web.http_client.close()
        reader.http_client.close()
