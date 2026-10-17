import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.agent.context_policy import RequestContextPolicy, estimate_text_tokens
from app.agent.media_tasks import media_result
from app.agent.resource_tools import build_resource_tool
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.shared_links import wrap_share_web_tools
from app.agent.tool_agent import ToolOutcome
from app.agent.web_tools import build_web_tools
from app.providers.web_images import WebImageError
from tests.agent.test_web_tools import Client, Guard, Runner


async def test_batch_inspection_compact_once_recoverable_after_new_context(tmp_path, monkeypatch):
    calls = []
    def inspect(self, url, **kwargs):
        kwargs['check']();calls.append(url)
        return {'source_url': url, 'post_id': '123', 'title': '作品', 'author': '作者',
            'published_at': '2026-01-01', 'identity_status': 'matched', 'media_count': 1,
            'author_status': 'source_reported', 'date_status': 'source_reported',
            'text': '完整来源正文' * 300, 'media': [{'kind': 'image', 'role': 'post_image',
                'url': 'https://images.example.com/' + 'a' * 1500 + '.jpg'}]}
    monkeypatch.setattr('app.agent.resource_tools.PublicResourceClient.inspect', inspect)
    w = ResearchTaskWorkspace(tmp_path, scope='group:900000001', actor_id=900000101,
        source_id=1, request='找作品图片', valid_source=lambda _: True)
    t = build_resource_tool(None, runner=Runner(), guard=Guard(), workspace=w)
    result = await t.handler({'ref_ids': ['https://example.com/posts/123'] * 2})
    assert calls == ['https://example.com/posts/123']
    assert len(json.dumps(result.data, ensure_ascii=False)) < 2000
    assert 'a' * 1500 not in json.dumps(result.data)
    r = result.data['resources'][0]
    recovered = w.read_evidence(r['evidence_id'])
    assert 'a' * 1500 in recovered['content'] and '完整来源正文' in recovered['content']
    echo = await t.handler({'ref_ids': calls})
    assert len(calls) == 1 and echo.data['resources'][0]['cached_in_turn']
    assert 'author' not in echo.data['resources'][0]
    # A fresh workspace can recover metadata without putting raw URLs/HTML
    # back into the main model history or changing its currently selected goal.
    fresh = ResearchTaskWorkspace(tmp_path, scope=w.scope, actor_id=w.actor_id,
        source_id=2, request='继续筛选', valid_source=lambda _: True)
    page = fresh.resource_page(task_id=w.task_id)
    assert page['resources'][0]['author'] == '作者'
    assert page['resources'][0]['media_roles'] == ['post_image']
    assert fresh.source_id == 2


async def test_failure_is_not_empty_success_and_is_not_repeated(monkeypatch):
    calls = []
    def fail(self, url, **kwargs):
        calls.append(url);raise WebImageError('web_resource_session_required')
    monkeypatch.setattr('app.agent.resource_tools.PublicResourceClient.inspect', fail)
    t = build_resource_tool(None, runner=Runner(), guard=Guard())
    first = await t.handler({'ref_ids': ['https://example.com/a']})
    second = await t.handler({'ref_ids': ['https://example.com/a']})
    assert first.status == second.status == 'failed' and len(calls) == 1
    assert second.data['resources'][0]['error'] == 'web_resource_session_required'


async def test_long_share_urls_are_archived_without_repeating_them_in_context(tmp_path, monkeypatch):
    url = 'https://example.com/post/12?token=' + 'x' * 1500
    def inspect(self, url, **kwargs):
        return {'source_url': url, 'post_id': '12', 'title': '作品', 'media': [], 'media_count': 0}
    monkeypatch.setattr('app.agent.resource_tools.PublicResourceClient.inspect', inspect)
    w = ResearchTaskWorkspace(tmp_path, scope='private:900000101', actor_id=900000101,
        source_id=1, request='读公开分享', valid_source=lambda _: True)
    t = build_resource_tool(None, runner=Runner(), guard=Guard(), workspace=w)
    first = await t.handler({'ref_ids': [url]})
    second = await t.handler({'ref_ids': [url]})
    assert url not in json.dumps(first.data) and url not in json.dumps(second.data)
    assert first.data['resources'][0]['input_index'] == 1
    assert url in w.read_evidence(first.data['resources'][0]['evidence_id'])['content']
    mixed = await t.handler({'ref_ids': ['https://example.com/short', 'https://example.com/short', url]})
    assert mixed.data['resources'][1]['input_index'] == 3


async def test_bad_external_metadata_does_not_discard_other_candidates(monkeypatch):
    def inspect(self, url, **kwargs):
        if url.endswith('bad'):raise TypeError('untrusted payload should not leak')
        return {'source_url': url, 'title': '作品', 'media': []}
    monkeypatch.setattr('app.agent.resource_tools.PublicResourceClient.inspect', inspect)
    t = build_resource_tool(None, runner=Runner(), guard=Guard())
    result = await t.handler({'ref_ids': ['https://example.com/bad', 'https://example.com/good']})
    assert result.status == 'ok'
    assert result.data['resources'][0]['error'] == 'web_resource_invalid_metadata'
    assert result.data['resources'][1]['title'] == '作品'
    assert 'untrusted payload' not in json.dumps(result.data)


async def test_all_targets_are_authorized_before_batch_dispatch(monkeypatch):
    calls = []
    async def handler(args):calls.append(args);return ToolOutcome({})
    from app.agent.tool_agent import AgentTool
    t = AgentTool('inspect_web_resources', '', {}, handler)
    access = SimpleNamespace(web_allowed=lambda: True,
        known_card_target_allowed=lambda target, sources: target != 'https://example.com/unquoted')
    t, = wrap_share_web_tools((t,), access)
    r = await t.handler({'ref_ids': ['https://example.com/allowed', 'https://example.com/unquoted']})
    assert r.status == 'blocked' and not calls
    assert (await t.handler({'ref_ids': [123]})).status == 'blocked'


def test_compact_tool_output_preserves_prefix_across_turns_and_native_state():
    from tests.agent.test_context_policy import result
    data = {'trust': 'untrusted_external_sources', 'context_retention': 'compact_v1',
        'content': 'verified compact evidence', 'task_id': 'task', 'evidence_id': 'evidence', 'sources': []}
    body = {'instructions': 'stable', 'input': [result('a', data)], 'tools': [{'type': 'function', 'name': 'stable'}]}
    body['input'].insert(0, {'type': 'reasoning', 'encrypted_content': 'OPAQUE_NATIVE_STATE'})
    old, _ = RequestContextPolicy().project(body, protocol='openai-responses', current_call_ids={'a'})
    new, _ = RequestContextPolicy().project(body, protocol='openai-responses', current_call_ids=set())
    assert old == new and body['input'][0] == new['input'][0]
    huge = deepcopy(body);huge['input'][-1] = result('a', {**data, 'content': 'bulk' * 20000})
    projected, _ = RequestContextPolicy().project(huge, protocol='openai-responses', current_call_ids=set())
    assert 'bulk' * 100 not in json.dumps(projected)


async def test_web_default_is_smaller_and_full_source_remains_external(tmp_path):
    w = ResearchTaskWorkspace(tmp_path, scope='group:900000001', actor_id=900000101,
        source_id=1, request='读取代码', valid_source=lambda _: True)
    tools = {t.name: t for t in build_web_tools(Client(), settings=SimpleNamespace(), runner=Runner(), guard=Guard(), workspace=w)}
    page = await tools['open_web_page'].handler({'ref_id': 'https://example.com/page'})
    assert len(json.dumps(page.data, ensure_ascii=False)) <= 6000
    raw = w.read_evidence(page.data['evidence_id'], max_chars=32000)
    assert len(raw['content']) > len(page.data['content']) * 3
    assert estimate_text_tokens(json.dumps(page.data)) < estimate_text_tokens(json.dumps(raw)) / 2


async def test_long_source_lists_are_paged_without_corrupting_share_parameters(tmp_path):
    from app.providers.web_search import WebSearchReport, WebSearchResult
    class LongSources(Client):
        def search_report(self, query, max_results):
            return WebSearchReport([WebSearchResult('来源', '摘要',
                f'https://example.com/{i}?xsec_token=' + 'a' * 1500, '') for i in range(8)], 'grounded answer')
    w = ResearchTaskWorkspace(tmp_path, scope='group:900000001', actor_id=900000101,
        source_id=1, request='检索笔记', valid_source=lambda _: True)
    t = {t.name:t for t in build_web_tools(LongSources(),settings=SimpleNamespace(),runner=Runner(),guard=Guard(),workspace=w)}
    result = await t['search_web'].handler({'query':'作品'})
    assert result.status == 'ok' and len(json.dumps(result.data,ensure_ascii=False)) <= 6000
    assert result.data['sources_truncated'] and result.data['total_sources'] == 8
    assert result.data['sources'][0]['source'].endswith('a' * 1500)
    index = w.resource_page(limit=20)
    assert index['total'] == 8 and all(r['url'].endswith('a' * 1500) for r in index['resources'])


async def test_goal_correction_does_not_reuse_a_different_media_search(tmp_path):
    from app.providers.web_search import WebSearchReport, WebSearchResult
    class Discovery(Client):
        def search_resources(self, query, max_results, goal):
            self.calls.append(goal)
            return WebSearchReport([WebSearchResult('来源', '', 'https://example.com/post', '')], goal)
    w = ResearchTaskWorkspace(tmp_path, scope='private:900000101', actor_id=900000101,
        source_id=1, request='找图', valid_source=lambda _: True)
    w.request_control = {'delivery': 'media', 'goal': '找第一季图片'}
    c = Discovery()
    t = {t.name:t for t in build_web_tools(c,settings=SimpleNamespace(),runner=Runner(),guard=Guard(),workspace=w)}
    await t['search_web'].handler({'query': '作品'})
    w.request_control['goal'] = '改成第二季图片'
    await t['search_web'].handler({'query': '作品'})
    repeated = await t['search_web'].handler({'query': '作品'})
    assert c.calls == ['找第一季图片', '改成第二季图片']
    assert repeated.data['cached_in_turn']


async def test_source_only_search_is_archived_even_without_inline_text(tmp_path):
    from app.providers.web_search import WebSearchReport, WebSearchResult
    class SourcesOnly(Client):
        def search_report(self, query, max_results):
            return WebSearchReport([WebSearchResult('来源', '',
                f'https://example.com/{i}?share=' + 'x' * 1500, '') for i in range(8)], '')
    w = ResearchTaskWorkspace(tmp_path, scope='private:900000101', actor_id=900000101,
        source_id=1, request='找公开分享', valid_source=lambda _: True)
    tools = {t.name: t for t in build_web_tools(SourcesOnly(), settings=SimpleNamespace(),
        runner=Runner(), guard=Guard(), workspace=w)}
    result = await tools['search_web'].handler({'query': '作品'})
    assert result.status == 'ok' and result.data['sources_truncated']
    assert w.read_evidence(result.data['evidence_id'])['total_sources'] == 8
    assert w.resource_page(limit=20)['total'] == 8


async def test_evidence_tool_default_matches_compact_reader_default(tmp_path):
    w = ResearchTaskWorkspace(tmp_path, scope='private:900000101', actor_id=900000101,
        source_id=1, request='读原文', valid_source=lambda _: True)
    ref = w.save_evidence({'content': '文' * 25000,
        'sources': [{'source': 'https://example.com/post'}]})
    tools = {t.name: t for t in w.tools(Guard())}
    result = await tools['read_research_evidence'].handler({'evidence_id': ref})
    assert len(result.data['content']) == 6000
    assert result.data['next_offset'] == 6000
    assert 'context_retention' not in result.data
    short_ref = w.save_evidence({'content': '简短来源结论', 'sources': []})
    short = await tools['read_research_evidence'].handler({'evidence_id': short_ref})
    assert short.data['context_retention'] == 'compact_v1'


async def test_cached_receipt_itself_keeps_the_prefix_stable_next_turn():
    from tests.agent.test_context_policy import result as entry
    tools = {t.name: t for t in build_web_tools(Client(), settings=SimpleNamespace(),
        runner=Runner(), guard=Guard())}
    await tools['search_web'].handler({'query': '来源'})
    repeated = await tools['search_web'].handler({'query': '来源'})
    body = {'input': [entry('repeat', repeated.data)]}
    current, _ = RequestContextPolicy().project(body, protocol='openai-responses', current_call_ids={'repeat'})
    later, _ = RequestContextPolicy().project(body, protocol='openai-responses', current_call_ids=set())
    assert repeated.data['cached_in_turn'] and current == later


async def test_raw_source_does_not_stay_in_later_chat_as_a_compact_summary(tmp_path):
    from app.providers.web_search import WebSearchReport, WebSearchResult
    from tests.agent.test_context_policy import result as entry
    class Source(Client):
        def open_page(self, url, lineno=0):
            return WebSearchReport([WebSearchResult('源文件', 'RAW_CODE_BULK', url, '')],
                'RAW_CODE_BULK\n' * 1000, 'raw_source')
    w = ResearchTaskWorkspace(tmp_path, scope='private:900000101', actor_id=900000101,
        source_id=1, request='读源文件', valid_source=lambda _: True)
    tools = {t.name: t for t in build_web_tools(Source(), settings=SimpleNamespace(),
        runner=Runner(), guard=Guard(), workspace=w)}
    page = await tools['open_web_page'].handler({'ref_id': 'https://example.com/source.py'})
    reread = w.read_evidence(page.data['evidence_id'])
    assert 'context_retention' not in page.data and 'context_retention' not in reread
    assert 'RAW_CODE_BULK' in reread['content']
    for data in (page.data, reread):
        projected, _ = RequestContextPolicy().project({'input': [entry('code', data)]},
            protocol='openai-responses', current_call_ids=set())
        assert 'RAW_CODE_BULK' not in json.dumps(projected)


async def test_one_long_source_line_keeps_visible_text_and_can_recover_the_rest(tmp_path):
    from app.providers.web_search import WebSearchReport, WebSearchResult
    class LongLine(Client):
        def open_page(self, url, lineno=0):
            return WebSearchReport([WebSearchResult('源文件', '', url, '')],
                'Source\nL0: ' + 'LONG_LINE_BODY_' * 1200)
    w = ResearchTaskWorkspace(tmp_path, scope='private:900000101', actor_id=900000101,
        source_id=1, request='读源文件', valid_source=lambda _: True)
    tools = {t.name: t for t in build_web_tools(LongLine(), settings=SimpleNamespace(),
        runner=Runner(), guard=Guard(), workspace=w)}
    result = await tools['open_web_page'].handler({'ref_id': 'https://example.com/source'})
    assert result.status == 'ok' and len(json.dumps(result.data, ensure_ascii=False)) <= 6000
    assert 'LONG_LINE_BODY_' in result.data['content'] and result.data['partial_line'] == 0
    assert result.data['next_line'] is None and 'context_retention' not in result.data
    rest = w.read_evidence(result.data['evidence_id'], offset=result.data['evidence_next_offset'], max_chars=32000)
    assert result.data['content'] + rest['content'] == 'Source\nL0: ' + 'LONG_LINE_BODY_' * 1200


def test_downloads_do_not_auto_load_pixels_but_generated_images_keep_preview():
    out = ToolOutcome({'artifact_ids': ['art-a', 'art-b']})
    assert media_result(out, {'operation': 'download_web_images', 'task_id': 't'})['preview_artifact_ids'] == []
    assert media_result(out, {'operation': 'generate_image', 'task_id': 't'})['preview_artifact_ids'] == ['art-a', 'art-b']
