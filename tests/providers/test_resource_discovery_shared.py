from types import SimpleNamespace

from app.providers.web_search import WebSearchClient, WebSearchReport, WebSearchResult


def test_gpt_search_reuses_http_only_discovery_without_a_second_model_call():
    client = WebSearchClient(provider='wyzlab', base_url='https://example.com', api_key='synthetic', timeout_seconds=45)
    calls = []
    report = WebSearchReport([WebSearchResult('来源', '摘要', 'https://example.com/post', '')], '摘要')
    def search(query, max_results):calls.append(('model_search', max_results));return report
    def discover(value, **kwargs):
        calls.append(('http_discovery', kwargs['goal']));assert value is report;return value
    client.search_report = search
    client.page_reader = SimpleNamespace(discover_resources=discover)
    try:
        assert client.search_resources('作品', goal='最新原图') is report
        assert calls == [('model_search', 8), ('http_discovery', '最新原图')]
    finally:client.http_client.close()
