import pytest

from app.agent.xhs_posts import build_xhs_search_tool
from app.providers.xhs_crawler_client import XHSCrawlerClient, XHSCrawlerError
from tests.agent.test_web_tools import Guard, Runner
from tests.agent.test_request_task_control import control
from app.agent.tool_agent import ToolOutcome

URL = 'https://www.xiaohongshu.com/explore/0123456789abcdef01234567'


async def test_xhs_search_is_source_guarded_cached_and_stops_on_refusal(monkeypatch):
    calls = []
    def search(*a, **k):
        k['check']();calls.append(a)
        if len(calls) > 1:raise XHSCrawlerError('xhs_rate_limited')
        return {'posts': [], 'visible_posts': 0, 'coverage': 'candidates_only'}
    monkeypatch.setattr(XHSCrawlerClient, 'search_posts', search)
    tool = build_xhs_search_tool(runner=Runner(), guard=Guard())
    assert (await tool.handler({'query': '主体 主题'})).status == 'ok'
    assert (await tool.handler({'query': '主体 主题'})).data['cached_in_turn']
    assert (await tool.handler({'query': '另一个准确别名'})).data['error'] == 'xhs_rate_limited'
    assert (await tool.handler({'query': '再换词'})).data['stopped_in_turn']
    assert len(calls) == 2


@pytest.mark.parametrize('operation,arguments', [
    ('download_web_video', {'ref_id': URL}),
    ('download_web_images', {'image_ids': ['webimg-synthetic']}),
    ('download_web_image', {'image_id': 'webimg-synthetic'})])
@pytest.mark.parametrize('date', ['', '2020-01-01T00:00:00+00:00', '2099-01-01T00:00:00+00:00'])
def test_recent_xhs_downloads_require_source_date_before_dispatch(tmp_path, operation, arguments, date):
    goal = control(tmp_path, '去小红书找近7天的上海图片和视频发给我')
    goal.observe('list_web_images', {}, ToolOutcome({'page_url': URL, 'post_id': URL.rsplit('/', 1)[1],
        'identity_status': 'matched', 'author': '合成作者', 'published_at': date,
        'images': [{'image_id': 'webimg-synthetic'}]}), 'list')
    result = goal.preflight(operation, arguments)
    assert result.data['error'] == 'resource_freshness_unverified'
    assert result.data['handler_executed'] is False and result.data['background_queued'] is False


def test_search_candidate_is_not_verified_original_evidence(tmp_path):
    goal = control(tmp_path, '去小红书找近7天的图片下载发给我')
    goal.observe('search_xhs_posts', {'query': '主体'}, ToolOutcome({'resources': [{
        'source_url': URL, 'post_id': URL.rsplit('/', 1)[1], 'identity_status': 'candidate',
        'published_at': '', 'date_status': 'unknown'}]}), 'search')
    assert goal.discovery_attempts == 1
    assert 'primary_metadata' not in goal.state()['evidence_kinds']
    assert not goal.post_metadata
    result = goal.preflight('download_web_video', {'ref_id': URL})
    assert result.data['error'] == 'resource_freshness_unverified'


def test_recent_verified_album_is_allowed_and_future_search_cannot_replace_date(tmp_path):
    from datetime import UTC, datetime
    goal = control(tmp_path, '去小红书找近7天图片发给我')
    goal.observe('list_web_images', {}, ToolOutcome({'page_url': URL, 'post_id': URL.rsplit('/', 1)[1],
        'identity_status': 'matched', 'author': '合成作者',
        'published_at': datetime.now(UTC).isoformat(),
        'images': [{'image_id': 'webimg-synthetic'}]}), 'list')
    assert goal.preflight('download_web_images', {'image_ids': ['webimg-synthetic']}) is None


async def test_candidate_alias_resolves_to_its_observed_source_before_inspection(monkeypatch):
    from types import SimpleNamespace
    from app.agent.shared_links import wrap_share_web_tools
    from app.agent.tool_agent import AgentTool
    seen=[]
    def search(*args, **kwargs):
        return {'posts':[{'source_url':URL,'post_id':URL.rsplit('/',1)[1],'title':'合成笔记'}],
            'visible_posts':1,'coverage':'candidate_only'}
    monkeypatch.setattr(XHSCrawlerClient,'search_posts',search)
    async def inspect(args):seen.extend(args['ref_ids']);return ToolOutcome({})
    access=SimpleNamespace(web_allowed=lambda:True,known_card_target_allowed=lambda *a:True)
    tools=wrap_share_web_tools((build_xhs_search_tool(runner=Runner(),guard=Guard()),
        AgentTool('inspect_web_resources','',{},inspect)),access)
    await tools[0].handler({'query':'合成主体'})
    await tools[1].handler({'ref_ids':[URL.rsplit('/',1)[1]]})
    assert seen==[URL]
