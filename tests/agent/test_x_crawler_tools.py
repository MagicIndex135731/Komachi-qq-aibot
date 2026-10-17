from types import SimpleNamespace

from app.agent.web_tools import build_web_tools
from app.agent.shared_links import wrap_share_web_tools
from test_web_tools import Client, Guard, Runner
from test_twitter_account_tools import result, URL, workspace


def fake_crawler(monkeypatch):
    calls = []
    def list_posts(account, **kw):
        calls.append(('list', account)); return result() | {'provider': 'x_crawler'}
    def inspect(url, **kw):
        calls.append(('inspect', url)); return result()['posts'][0] | {
            'media': [{'kind': 'photo', 'url': 'https://pbs.twimg.com/media/synthetic.jpg'}]}
    crawler = SimpleNamespace(list_posts=list_posts, inspect=inspect)
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.from_settings', lambda s: crawler)
    return calls


async def test_existing_tools_route_crawler_and_archive_under_request_scope(tmp_path, monkeypatch):
    calls = fake_crawler(monkeypatch)
    w = workspace(tmp_path)
    tools = {t.name:t for t in build_web_tools(Client(), settings=SimpleNamespace(), runner=Runner(), guard=Guard(), workspace=w)}
    listed = await tools['list_x_posts'].handler({'account':'synthetic'})
    inspected = await tools['inspect_web_resources'].handler({'ref_ids':[URL]})
    assert calls == [('list','synthetic'),('inspect',URL)]
    assert listed.data['resources'][0]['source_url'] == URL
    assert inspected.data['resources'][0]['media_roles'] == ['post_image']
    evidence = w.read_evidence(inspected.data['resources'][0]['evidence_id'])
    assert 'pbs.twimg.com' in evidence['content']


async def test_crawler_never_bypasses_missing_current_web_request(monkeypatch):
    calls = fake_crawler(monkeypatch)
    tools = build_web_tools(Client(), settings=SimpleNamespace(), runner=Runner(), guard=Guard())
    tools = wrap_share_web_tools(tools, SimpleNamespace(web_allowed=lambda:False, known_card_target_allowed=lambda *a:True))
    for tool in tools:
        if tool.name in {'list_x_posts','inspect_web_resources'}:
            result = await tool.handler({'account':'synthetic'} if tool.name=='list_x_posts' else {'ref_ids':[URL]})
            assert result.data['error'] == 'web_requires_explicit_request'
    assert not calls
