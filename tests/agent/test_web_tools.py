"""Public web data stays bounded, turn scoped, and revocable."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.agent.web_tools import build_web_tools
from app.providers.web_search import WebSearchReport, WebSearchResult


class Guard:
    valid = True

    def require_valid(self):
        if not self.valid:
            raise RuntimeError("source revoked")


class Runner:
    async def run(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class Client:
    supports_page_tools = True

    def __init__(self):
        self.calls = []

    def search_report(self, query, max_results):
        self.calls.append(("search", query, max_results))
        return WebSearchReport([WebSearchResult("source", "short", "https://example.com/page", "", "turn0search0")],
            "source (https://example.com/page)\nciteturn0search0 [wordlim: 200]\nDetailed evidence beyond the snippet.")

    def open_page(self, url, *, lineno=0):
        self.calls.append(("open", url, lineno))
        return WebSearchReport([WebSearchResult("source", "", url, "", "turn0view0")],
            "source (" + url + ")\nciteturn0view0 Total lines: 600\n" +
            "\n".join(f"L{i}: " + "x" * 120 for i in range(max(0, lineno - 20), 600)))

    def find_page(self, url, *, pattern):
        self.calls.append(("find", url, pattern))
        return self.open_page(url)


def tools(client=None, guard=None, **settings):
    return {tool.name: tool for tool in build_web_tools(client or Client(), settings=SimpleNamespace(**settings),
        runner=Runner(), guard=guard or Guard())}


async def test_default_eight_sources_body_and_local_reference_open_find():
    client = Client()
    group = tools(client)
    found = await group["search_web"].handler({"query": "public evidence"})
    assert client.calls == [("search", "public evidence", 8)]
    assert "Detailed evidence" in found.data["content"]
    ref = found.data["sources"][0]["ref_id"]
    assert ref.startswith("web_") and "turn0search0" not in found.data["content"]
    page = await group["open_web_page"].handler({"ref_id": ref, "lineno": 250})
    assert client.calls[-1] == ("open", "https://example.com/page", 250)
    assert "L250:" in page.data["content"]
    assert page.data["truncated"] and 250 < page.data["next_line"] < 599
    await group["find_web_page"].handler({"ref_id": ref, "pattern": "detail"})
    assert client.calls[-2][0] == "find"
    other = tools()
    blocked = await other["open_web_page"].handler({"ref_id": ref})
    assert blocked.status == "blocked"


@pytest.mark.parametrize("ref", ["http://localhost/a", "http://127.0.0.1", "http://169.254.169.254/latest",
    "http://[::1]", "https://user:password@example.com", "file:///etc/passwd", "http://2130706433", "http://host.docker.internal", "turn0search0"])
async def test_invalid_targets_never_dispatch(ref):
    client = Client()
    result = await tools(client)["open_web_page"].handler({"ref_id": ref})
    assert result.status == "blocked" and client.calls == []


async def test_ref_collision_across_requests_and_budgets_are_explicit():
    client = Client()
    current = tools(client)
    a = await current["search_web"].handler({"query": "a", "max_results": 20})
    b = await current["search_web"].handler({"query": "b"})
    assert a.data["sources"][0]["ref_id"] != b.data["sources"][0]["ref_id"]
    assert (await current["search_web"].handler({"query": "a", "max_results": 21})).status == "blocked"
    for _ in range(8):
        await current["search_web"].handler({"query": str(_)})
    assert len(client.calls) == 8
    assert (await current["search_web"].handler({"query": "a"})).data["error"] == "web_turn_budget_exhausted"


async def test_page_context_budget_and_next_line_use_visible_content():
    current = tools()
    total = 0
    for _ in range(8):
        result = await current["open_web_page"].handler({"ref_id": "https://example.com/page", 'lineno': _ * 10})
        if result.status == "blocked":
            break
        total += len(__import__("json").dumps(result.data, ensure_ascii=False))
        visible = __import__("re").findall(r"L(\d+):", result.data["content"])
        assert result.data["next_line"] == int(visible[-1]) + 1
    assert total <= 48000


async def test_source_revocation_failure_timeout_and_cancellation():
    guard = Guard()
    class Revoked(Client):
        def search_report(self, *args, **kwargs):
            guard.valid = False
            return super().search_report(*args, **kwargs)
    with pytest.raises(RuntimeError, match="source revoked"):
        await tools(Revoked(), guard)["search_web"].handler({"query": "a"})
    class Failed(Client):
        def search_report(self, *args, **kwargs):
            raise httpx.ReadTimeout("token-containing exception")
    result = await tools(Failed())["search_web"].handler({"query": "a"})
    assert result.status == "failed" and "token" not in str(result.data)
    class Cancelled(Client):
        def search_report(self, *args, **kwargs):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await tools(Cancelled())["search_web"].handler({"query": "a"})


async def test_other_provider_keeps_search_contract_without_new_page_access():
    class Legacy:
        def search(self, query, max_results):
            assert max_results == 8
            return [WebSearchResult("source", "legacy snippet", "https://example.com", "")]
    current = tools(Legacy())
    assert set(current) == {"search_web"}
    assert (await current["search_web"].handler({"query": "a"})).data["sources"][0]["snippet"] == "legacy snippet"


async def test_find_keeps_late_match_visible_and_legacy_feedback_fits():
    class LateMatch(Client):
        def find_page(self, url, *, pattern):
            report = self.open_page(url)
            report.content = report.content.replace("L500: ", "L500: NEEDLE ")
            return report
    result = await tools(LateMatch())["find_web_page"].handler({"ref_id": "https://example.com/page", "pattern": "NEEDLE"})
    assert "L500: NEEDLE" in result.data["content"]
    legacy = tools(agent_runtime="legacy")
    result = await legacy["open_web_page"].handler({"ref_id": "https://example.com/page"})
    assert len(__import__("json").dumps(result.data, ensure_ascii=False)) <= 8000


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
async def test_search_retries_transient_http_once_and_counts_both_attempts(status):
    class Temporary(Client):
        attempts = 0
        def search_report(self,*args,**kwargs):
            self.attempts += 1
            if self.attempts == 1:
                response=httpx.Response(status,request=httpx.Request('POST','https://bridge.example'))
                raise httpx.HTTPStatusError('secret-containing error',request=response.request,response=response)
            return super().search_report(*args,**kwargs)
    c=Temporary();current=tools(c)
    result=await current['search_web'].handler({'query':'public evidence'})
    assert result.status=='ok' and c.attempts==2
    for _ in range(6):await current['search_web'].handler({'query':str(_)})
    assert (await current['search_web'].handler({'query':'new uncached request'})).status=='blocked'
    assert c.attempts==8


@pytest.mark.parametrize('status', [400,401,403,404,422,502])
async def test_http_errors_are_safe_and_retries_are_bounded(status):
    class Failed(Client):
        attempts=0
        def search_report(self,*args,**kwargs):
            self.attempts+=1
            response=httpx.Response(status,request=httpx.Request('POST','https://bridge.example?token=secret'))
            raise httpx.HTTPStatusError('secret',request=response.request,response=response)
    c=Failed();result=await tools(c)['search_web'].handler({'query':'a'})
    assert result.status=='failed' and result.data['error']==f'web_http_{status}'
    assert c.attempts==(2 if status==502 else 1)
    assert 'secret' not in str(result.data)


async def test_metadata_kind_is_visible_and_revocation_prevents_retry():
    class Metadata(Client):
        def open_page(self,*args,**kwargs):
            r=super().open_page(*args,**kwargs);r.content_kind='bilibili_public_metadata';return r
    assert (await tools(Metadata())['open_web_page'].handler({'ref_id':'https://example.com/page'})).data['content_kind']=='bilibili_public_metadata'
    guard=Guard()
    class Revoked(Client):
        attempts=0
        def search_report(self,*args,**kwargs):
            self.attempts+=1;guard.valid=False
            response=httpx.Response(502,request=httpx.Request('POST','https://bridge.example'))
            raise httpx.HTTPStatusError('secret',request=response.request,response=response)
    c=Revoked()
    with pytest.raises(RuntimeError,match='source revoked'):await tools(c,guard)['search_web'].handler({'query':'a'})
    assert c.attempts==1


async def test_retry_backoff_stays_inside_total_timeout_and_call_budget():
    class Temporary(Client):
        attempts=0
        def search_report(self,*args,**kwargs):
            self.attempts+=1
            response=httpx.Response(502,request=httpx.Request('POST','https://bridge.example'))
            raise httpx.HTTPStatusError('secret',request=response.request,response=response)
    c=Temporary()
    result=await tools(c,agent_web_search_timeout_seconds=0.01)['search_web'].handler({'query':'a'})
    assert result.data['error']=='web_timeout' and c.attempts==1
    class AtLimit(Client):
        attempts=0
        def search_report(self,*args,**kwargs):
            self.attempts+=1
            if self.attempts==8:
                response=httpx.Response(502,request=httpx.Request('POST','https://bridge.example'))
                raise httpx.HTTPStatusError('secret',request=response.request,response=response)
            return super().search_report(*args,**kwargs)
    c=AtLimit();current=tools(c)
    for _ in range(7):await current['search_web'].handler({'query':str(_)})
    assert (await current['search_web'].handler({'query':'a'})).data['error']=='web_http_502'
    assert c.attempts==8


async def test_repeated_page_search_uses_compact_same_turn_receipt_and_rechecks_guard():
    c=Client();guard=Guard();current=tools(c,guard)
    first=await current['search_web'].handler({'query':'public evidence'})
    again=await current['search_web'].handler({'query':'public evidence'})
    assert again.status=='ok' and again.data['cached_in_turn'] is True
    assert again.data['source_refs']==[x['ref_id'] for x in first.data['sources']] and not again.data['content']
    assert len(c.calls)==1
    ref=first.data['sources'][0]['ref_id']
    page=await current['open_web_page'].handler({'ref_id':ref})
    repeated=await current['open_web_page'].handler({'ref_id':'https://example.com/page'})
    assert repeated.data['source_refs']==[x['ref_id'] for x in page.data['sources']] and repeated.data['cached_in_turn']
    assert len(c.calls)==2 and repeated.data['next_line']==page.data['next_line']
    guard.valid=False
    with pytest.raises(RuntimeError):await current['open_web_page'].handler({'ref_id':ref})
    other=tools(c)
    await other['search_web'].handler({'query':'public evidence'})
    assert len(c.calls)==3


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
async def test_exhausted_search_service_failure_stops_changed_queries_only_in_this_turn(status):
    class Unavailable(Client):
        attempts = 0
        unavailable = True

        def search_report(self, *args, **kwargs):
            self.attempts += 1
            if self.unavailable:
                response = httpx.Response(status, request=httpx.Request('POST', 'https://bridge.example'))
                raise httpx.HTTPStatusError('private upstream body', request=response.request, response=response)
            return super().search_report(*args, **kwargs)

    client, guard = Unavailable(), Guard()
    current = tools(client, guard)
    first = await current['search_web'].handler({'query': 'first research query'})
    assert first.status == 'failed' and client.attempts == 2
    another = await current['search_web'].handler({'query': 'changed research query'})
    assert another.status == 'failed'
    assert another.data['cached_failure_in_turn'] is True and client.attempts == 2
    assert another.data['error'] == first.data['error'] == f'web_http_{status}'
    assert '同一回合' in another.data['guidance'] and 'private' not in str(another.data)
    # A supplied public URL remains usable, and a new turn can retry a service
    # that recovered. A turn-local failure must not become a global outage.
    assert (await current['open_web_page'].handler({'ref_id': 'https://example.com/page'})).status == 'ok'
    client.unavailable = False
    assert (await tools(client)['search_web'].handler({'query': 'changed research query'})).status == 'ok'
    guard.valid = False
    with pytest.raises(RuntimeError, match='source revoked'):
        await current['search_web'].handler({'query': 'changed again'})


async def test_search_timeout_is_not_retried_by_changing_the_query_in_one_turn():
    class Timeout(Client):
        attempts = 0

        def search_report(self, *args, **kwargs):
            self.attempts += 1
            raise httpx.ReadTimeout('private timeout')

    client = Timeout()
    current = tools(client)
    first = await current['search_web'].handler({'query': 'a'})
    second = await current['search_web'].handler({'query': 'b'})
    assert first.data['error'] == second.data['error'] == 'web_timeout'
    assert second.data['cached_failure_in_turn'] is True and client.attempts == 1
