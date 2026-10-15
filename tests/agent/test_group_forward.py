"""Synthetic forward resources; no QQ identities, URLs or chat data from production."""
import asyncio
import io
import json
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from app.adapters.onebot_models import parse_group_message_event
from app.storage.db import session_scope
from app.storage.models import Message
from test_service_tools import context, seed, GROUP, AUTHOR


def segment(resource="synthetic-root"):
    return {"type": "forward", "data": {"id": resource}}


def node(text="合成引用", **extra):
    return {"sender": {"user_id": 900000333, "nickname": "引用署名"},
            "time": 100, "message": [{"type": "text", "data": {"text": text}}], **extra}


class Gateway:
    websocket = object()

    def __init__(self, resources):
        self.resources, self.calls = resources, []

    async def call_api(self, action, params):
        self.calls.append((action, params))
        return {"status": "ok", "retcode": 0, "data": {"messages": self.resources[params["id"]]}}


def setup_tools(context, resources):
    from app.agent.group_forward import build_forward_tools
    source = next(iter(context.observed_sources))
    with session_scope(context.router.engine) as db:
        db.get(Message, source).raw_json = {"message": [segment()]}
    gateway = Gateway(resources)
    tools = {tool.name: tool for tool in build_forward_tools(context, gateway)}
    return source, gateway, tools


def args(source, **overrides):
    return {"source_id": source, "forward_index": 0, "path": [], "offset": 0, "count": 2, **overrides}


def embedded_card(name="MultiMsg_synthetic-child"):
    return {"type": "xml", "data": {"data":
        f'<msg serviceID="35" action="viewMultiMsg" m_fileName="{name}"/>', "resid": 1}}


@pytest.mark.asyncio
async def test_embedded_xml_has_executable_navigation_and_bound_container(context, monkeypatch):
    from app.adapters.forward_messages import ForwardMessageAdapter, ForwardResource, _node
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node(message=[embedded_card()])]})
    calls = []
    async def embedded(self, container_id, file_name):
        calls.append((container_id, file_name))
        return ForwardResource((_node(node("包内第二层内容")),), False)
    monkeypatch.setattr(ForwardMessageAdapter, "read_embedded", embedded, raising=False)
    read = tools["read_forward_messages"].handler
    root = await read(args(source))
    assert root.data["nodes"][0]["forward_count"] == 1
    locator = root.data["nodes"][0]["nested_reads"][0]
    assert locator == args(source, count=20, path=[{"node_index": 0, "forward_index": 0}])
    result = await read(locator)
    assert result.status == "ok" and result.data["nodes"][0]["text"] == "包内第二层内容"
    assert calls == [("synthetic-root", "MultiMsg_synthetic-child")]
    assert len(gateway.calls) == 1
    assert (await read(locator)).status == "ok" and len(calls) == 1


@pytest.mark.asyncio
async def test_forward_images_page_beyond_first_two(context, monkeypatch):
    from app.agent import group_forward
    picture = io.BytesIO(); Image.new("RGB", (8, 8), "blue").save(picture, "PNG")
    fetched = []
    def http(request):
        fetched.append(str(request.url)); return httpx.Response(200, content=picture.getvalue())
    monkeypatch.setattr(group_forward, "media_client", lambda: httpx.Client(transport=httpx.MockTransport(http)))
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node(message=[
        {"type": "image", "data": {"url": f"https://multimedia.nt.qq.com.cn/{i}.png"}} for i in range(9)])]})
    root = await tools["read_forward_messages"].handler(args(source))
    assert root.data["nodes"][0]["viewable_image_count"] == 9
    observe = tools["observe_forward_images"].handler
    locator = dict(source_id=source, forward_index=0, path=[], node_index=0, image_offset=2)
    result = await observe(locator)
    assert result.status == "ok" and len(result.images) == 2
    assert result.data["next_image_offset"] == 4
    assert fetched == [f"https://multimedia.nt.qq.com.cn/{i}.png" for i in (2, 3)]
    assert [image["image_index"] for image in result.data["images"]] == [2, 3]
    assert (await observe({**locator, "image_offset": 9})).status == "blocked"


@pytest.mark.asyncio
async def test_local_cycle_and_revoked_source_stay_blocked(context, monkeypatch):
    from app.adapters.forward_messages import ForwardMessageAdapter, ForwardResource, _node
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node(message=[embedded_card()])]})
    async def embedded(self, container_id, file_name):
        return ForwardResource((_node(node(message=[embedded_card()])),), False)
    monkeypatch.setattr(ForwardMessageAdapter, "read_embedded", embedded, raising=False)
    read = tools["read_forward_messages"].handler
    root = await read(args(source)); child = root.data["nodes"][0]["nested_reads"][0]
    second = await read(child)
    cycle = await read(second.data["nodes"][0]["nested_reads"][0])
    assert cycle.status == "blocked" and cycle.data["error"] == "forward_cycle_detected"
    with session_scope(context.router.engine) as db:
        db.get(Message, source).raw_json = {"delivery_state": "deleted"}
    assert (await read(child)).status == "blocked"


@pytest.mark.asyncio
async def test_nested_navigation_keeps_page_content_bounded(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node("x" * 320,
        message=[{"type": "text", "data": {"text": "x" * 320}},
                 *[segment(f"synthetic-child-{i}") for i in range(8)]]) for _ in range(20)]})
    root = await tools["read_forward_messages"].handler(args(source, count=20))
    assert sum(len(n["nested_reads"]) for n in root.data["nodes"]) == 20
    assert any(n["nested_navigation_truncated"] for n in root.data["nodes"])
    assert len(json.dumps(root.data, ensure_ascii=False)) < 18000


@pytest.mark.parametrize("message", [
    [segment("opaque+/=")],
    [{"type": "forward", "data": {"res_id": "opaque+/="}}],
    [{"type": "json", "data": {"data": json.dumps({"app": "com.tencent.multimsg", "meta": {"detail": {"resid": "opaque+/="}}})}}],
    [{"type": "xml", "data": {"data": '<msg serviceID="35" m_resid="opaque+/="/>'}}],
    "[CQ:forward,id=opaque+/=]",
])
def test_forward_discovery(message):
    from app.core.forward_content import extract_forward_ids
    assert extract_forward_ids(message) == ("opaque+/=",)


@pytest.mark.parametrize("message", [
    [{"type": "text", "data": {"text": "[CQ:forward,id=fake]"}}],
    [{"type": "json", "data": {"data": '{"app":"other","meta":{"detail":{"resid":"fake"}}}'}}],
    [{"type": "json", "data": {"data": "malformed"}}],
    [{"type": "xml", "data": {"data": '<!DOCTYPE msg [<!ENTITY x "secret">]><msg serviceID="35" m_resid="fake"/>'}}],
    [{"type": "xml", "data": {"data": '<msg serviceID="1" m_resid="fake"/>'}}],
    [segment("file:///etc/passwd")], [segment("\ninvalid")],
    [{"type": "node", "data": {"content": [segment()]}}],
])
def test_non_forward_or_malformed_content_not_discovered(message):
    from app.core.forward_content import extract_forward_ids
    assert extract_forward_ids(message) == ()


def test_root_marker_does_not_promote_quoted_at_or_images():
    payload = {"message_id": "synthetic-message", "group_id": GROUP, "user_id": AUTHOR, "time": 100,
               "message": [segment(), {"type": "node", "data": {"content": [
                   {"type": "at", "data": {"qq": 900000099}},
                   {"type": "image", "data": {"url": "https://example.qq.com/synthetic.png"}}]}}]}
    event = parse_group_message_event(payload, bot_qq=900000099, bot_name="小町")
    assert event.plain_text == "[合并转发消息]"
    assert not event.mentioned_bot and event.images == []


@pytest.mark.asyncio
async def test_paging_nested_and_claimed_authors_stay_quotations(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node(), node(message=[segment("synthetic-child")]), node("尾页")],
        "synthetic-child": [{"type": "node", "data": {"name": "节点署名", "uin": 900000444,
            "content": [{"type": "at", "data": {"qq": 900000099}}, {"type": "text", "data": {"text": "嵌套内容"}}]}}]})
    read = tools["read_forward_messages"].handler
    first = await read(args(source))
    assert first.status == "ok" and first.data["next_offset"] == 2
    assert first.data["nodes"][1]["forward_count"] == 1
    assert first.data["trust"] == "unverified_forwarded_quotation"
    assert first.data["source_id"] == source
    assert first.data["nodes"][0]["reported_author"] == "引用署名"
    end = await read(args(source, offset=2))
    assert end.data["next_offset"] is None and end.data["nodes"][0]["text"] == "尾页"
    nested = await read(args(source, path=[{"node_index": 1, "forward_index": 0}]))
    assert "嵌套内容" in nested.data["nodes"][0]["text"]
    assert nested.data["nodes"][0]["reported_author"] == "节点署名"
    assert [call[1]["id"] for call in gateway.calls] == ["synthetic-root", "synthetic-child"]
    assert context.observed_sources == {source}


@pytest.mark.asyncio
async def test_sources_rechecked_even_for_cached_forward(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node()]})
    read = tools["read_forward_messages"].handler
    assert (await read(args(source))).status == "ok"
    with session_scope(context.router.engine) as db:
        db.get(Message, source).raw_json = {"delivery_state": "deleted", "message": [segment()]}
    assert (await read(args(source))).status == "blocked"
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_private_forward_cannot_read_group_or_other_private_author(sqlite_engine, tmp_path):
    from tests.agent.test_owner_actions import source, OWNER, OTHER
    from app.agent.artifacts import AutonomousArtifactStore
    from app.agent.group_forward import build_forward_tools, observed_forward_sources
    from app.runtime.blocking import BoundedBlockingRunner
    sid = source(sqlite_engine, group=None, text='展开私聊转发')
    other = source(sqlite_engine, author=OTHER, group=None, text='别人转发')
    group = source(sqlite_engine, text='群里转发')
    with session_scope(sqlite_engine) as db:
        for identity in (sid, other, group):
            row = db.get(Message, identity)
            row.raw_json = {**row.raw_json, 'message': [segment()]}
    class Guard:
        def require_valid(self): pass
    runner = BoundedBlockingRunner(1)
    router = SimpleNamespace(engine=sqlite_engine, runtime=SimpleNamespace(settings=SimpleNamespace(llm_supports_vision_input=True)))
    store = AutonomousArtifactStore(tmp_path/'private', account_id=900000099, scope=f'private:{OWNER}',
        engine=sqlite_engine, source_message_ids=lambda:(sid,))
    ctx = SimpleNamespace(router=router, observation=SimpleNamespace(job=SimpleNamespace(kind='private',scope_id=OWNER)),
        private_user_id=OWNER, guard=Guard(), observed_sources={sid,other,group}, store=store, blocking=runner)
    try:
        tools={tool.name:tool for tool in build_forward_tools(ctx,Gateway({'synthetic-root':[node('私聊引用')]}))}
        assert set(tools)=={'read_forward_messages','observe_forward_images'}
        assert (await tools['read_forward_messages'].handler(args(sid))).status=='ok'
        assert (await tools['read_forward_messages'].handler(args(other))).status=='blocked'
        assert (await tools['read_forward_messages'].handler(args(group))).status=='blocked'
        assert observed_forward_sources(router,None,{sid,other,group},private_user_id=OWNER)==[{'source_id':sid,'forward_count':1}]
    finally:
        runner.close()


@pytest.mark.asyncio
async def test_foreign_unknown_and_changed_root_rejected(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node()]})
    read = tools["read_forward_messages"].handler
    foreign = seed(context.router.engine, group_id=900000002, message_id="foreign-source", raw_json={"message": [segment()]})
    context.observed_sources.add(foreign)
    assert (await read(args(foreign))).status == "blocked"
    context.observed_sources.remove(foreign)
    assert (await read(args(source + 100))).status == "blocked"
    assert not gateway.calls
    assert (await read(args(source))).status == "ok"
    with session_scope(context.router.engine) as db:
        db.get(Message, source).raw_json = {"message": [segment("changed-root")]}
    assert (await read(args(source))).status == "blocked"
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_cycle_depth_pages_and_response_budgets(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node(message=[segment()])]})
    read = tools["read_forward_messages"].handler
    assert (await read(args(source, path=[{"node_index": 0, "forward_index": 0}]))).status == "blocked"
    assert (await read(args(source, path=[{"node_index": 0, "forward_index": 0}] * 4))).status == "blocked"
    assert (await read(args(source, count=21))).status == "blocked"
    assert (await read(args(source, offset=-1))).status == "blocked"
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {"status": "failed", "retcode": 1},
    {"status": "ok", "retcode": 0, "data": {"messages": "bad"}},
    {"status": "ok", "retcode": False, "data": {"messages": []}},
    {"status": "ok", "retcode": 0, "data": {"messages": [None]}},
])
async def test_gateway_failures_do_not_create_observations(context, response):
    source, gateway, tools = setup_tools(context, {})
    async def failure(*unused):
        return response
    gateway.call_api = failure
    result = await tools["read_forward_messages"].handler(args(source))
    assert result.status == "failed" and "nodes" not in result.data


@pytest.mark.asyncio
async def test_forward_limits_and_no_hidden_resource_arguments(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node("x" * 1000)] * 1001})
    result = await tools["read_forward_messages"].handler(args(source, count=20))
    assert result.data["resource_truncated"] and result.data["available_nodes"] == 1000
    assert all(n["text_truncated"] for n in result.data["nodes"])
    assert len(json.dumps(result.data, ensure_ascii=False)) < 12000
    assert (await tools["read_forward_messages"].handler(args(source, id="forged"))).status == "blocked"
    assert "id" not in tools["read_forward_messages"].parameters["properties"]


@pytest.mark.asyncio
async def test_images_require_observed_node_and_real_pixels(context, tmp_path, monkeypatch):
    from app.agent import group_forward
    picture = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(picture, "PNG")
    calls = []
    def http(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=picture.getvalue())
    client = httpx.Client(transport=httpx.MockTransport(http))
    monkeypatch.setattr(group_forward, "media_client", lambda: client)
    raw = node(message=[{"type": "image", "data": {"url": "https://multimedia.nt.qq.com.cn/synthetic.png", "local_path": "/etc/passwd"}},
        {"type": "image", "data": {"url": "http://127.0.0.1/secret", "file": str(tmp_path / "secret")}}])
    source, gateway, tools = setup_tools(context, {"synthetic-root": [raw]})
    image_args = {"source_id": source, "forward_index": 0, "path": [], "node_index": 0}
    observe = tools["observe_forward_images"].handler
    assert (await observe(image_args)).status == "blocked" and calls == []
    await tools["read_forward_messages"].handler(args(source))
    result = await observe(image_args)
    assert result.status == "ok" and result.data["actual_pixels"]
    assert len(result.images) == 1 and calls == ["https://multimedia.nt.qq.com.cn/synthetic.png"]
    image = result.images[0]
    assert image.source_user_id == AUTHOR and image.source_message_id == "synthetic-source"
    assert "unverified" in image.reference_subject
    assert Image.open(image.local_path).getpixel((0, 0)) == (0, 0, 255)
    with session_scope(context.router.engine) as db:
        db.get(Message, source).raw_json = {"delivery_state": "deleted", "message": [segment()]}
    assert (await observe(image_args)).status == "blocked" and len(calls) == 1


@pytest.mark.asyncio
async def test_source_revoked_during_gateway_read(context):
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node()]})
    original = gateway.call_api
    async def revoke(*arguments):
        result = await original(*arguments)
        with session_scope(context.router.engine) as db:
            db.get(Message, source).raw_json = {"delivery_state": "deleted"}
        return result
    gateway.call_api = revoke
    assert (await tools["read_forward_messages"].handler(args(source))).status == "blocked"


@pytest.mark.parametrize("url,allowed", [
    ("https://multimedia.nt.qq.com.cn/download", True),
    ("https://multimedia.nt.qq.com.cn.evil.example/download", False),
    ("https://evil.multimedia.nt.qq.com.cn/download", False),
    ("http://multimedia.nt.qq.com.cn/download", False),
    ("https://multimedia.nt.qq.com.cn:444/download", False),
    ("https://other.qq.com.cn/download", False),
])
def test_nt_forward_media_exact_host(url, allowed):
    from app.agent.social_images import allowed_media_url
    assert allowed_media_url(url) is allowed


@pytest.mark.asyncio
async def test_read_deadline_is_shared_across_nested_requests(context):
    from app.adapters.forward_messages import ForwardMessageAdapter, ForwardReadError
    gateway = Gateway({})
    async def hang(*unused):
        await asyncio.Event().wait()
    gateway.call_api = hang
    with pytest.raises(ForwardReadError, match="forward_gateway_unavailable"):
        await ForwardMessageAdapter(gateway, timeout_seconds=0.01).read("synthetic-slow")


@pytest.mark.asyncio
async def test_non_pixel_response_does_not_attach_images(context, monkeypatch):
    from app.agent import group_forward
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not-pixels")))
    monkeypatch.setattr(group_forward, "media_client", lambda: client)
    source, gateway, tools = setup_tools(context, {"synthetic-root": [node(message=[
        {"type": "image", "data": {"url": "https://example.qq.com/synthetic.png"}}])]})
    await tools["read_forward_messages"].handler(args(source))
    result = await tools["observe_forward_images"].handler({"source_id": source, "forward_index": 0, "path": [], "node_index": 0})
    assert result.status == "failed" and not result.images and not result.data["actual_pixels"]
