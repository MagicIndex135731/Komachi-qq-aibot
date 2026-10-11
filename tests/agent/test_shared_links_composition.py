"""APP card to failed open, supplementary search, body read and QQ receipt."""
import json

import httpx

from app.providers.web_search import WebSearchClient
from app.storage.db import session_scope
from app.storage.models import Message
from test_autonomous_social import add_source, composed, wake, writes
from test_mature_composition import latest, model_steps
from test_shared_links import card


async def test_native_group_share_read_search_fallback_and_reply(sqlite_engine, tmp_path):
    source = add_source(sqlite_engine, platform_id='native-app-card', text='[分享卡片：合成标题] 卡片摘要')
    web_calls = []

    def web(request):
        commands = json.loads(request.content)['commands']
        web_calls.append(commands)
        if len(web_calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={'output': '来源 (https://example.org/full)\nciteturn0view0\nL0: 完整的合成正文资料',
            'results': [{'type': 'text_result', 'title': '完整正文', 'url': 'https://example.org/full',
                         'ref_id': 'turn0view0', 'snippet': '摘要'}]})

    def discover(body):
        found = latest(body, 'read_context')['data']['share_sources']
        assert found[0]['source_id'] == source
        assert found[0]['title'] == '合成标题'
        return 'read_message_links', {'source_id': source}

    def open_target(body):
        data = latest(body, 'read_message_links')['data']
        assert data['metadata_only'] and data['trust'] == 'untrusted_shared_content'
        return 'open_web_page', {'ref_id': data['links'][0]['url']}

    def fallback(body):
        assert latest(body, 'open_web_page')['status'] == 'failed'
        return 'search_web', {'query': '合成标题 example.com'}

    def open_supplement(body):
        data = latest(body, 'search_web')['data']
        assert '完整的合成正文资料' in data['content']
        return 'open_web_page', {'ref_id': data['sources'][0]['ref_id']}

    def reply(body):
        assert '完整的合成正文资料' in latest(body, 'open_web_page')['data']['content']
        return 'send_group', {'text': '目标页无法打开，已根据补充来源核对概要', 'artifact_ids': []}

    client, requests, _ = model_steps(('read_context', {}), discover, open_target, fallback, open_supplement, reply, None)
    async with composed(sqlite_engine, tmp_path, client, runtime='harness', options={'max_steps': 10}) as fixture:
        with session_scope(sqlite_engine) as session:
            session.get(Message, source).raw_json = {'message': [card()]}
        fixture.router.web_search_client = WebSearchClient(provider='wyzlab_alpha', base_url='https://synthetic.invalid/v1',
            api_key='synthetic', timeout_seconds=45, http_client=httpx.Client(transport=httpx.MockTransport(web)))
        await wake(fixture)
        assert len(requests) == 7 and len(writes(fixture)) == 1
        assert web_calls[0]['open'][0]['ref_id'] == 'https://example.com/article'
        assert web_calls[1]['search_query'][0]['q'] == '合成标题 example.com'
        assert web_calls[2]['open'][0]['ref_id'] == 'https://example.org/full'
