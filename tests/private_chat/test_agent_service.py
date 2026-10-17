"""Real Harness/private service integration with synthetic HTTP model replies."""
import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest
from sqlalchemy import select

from app.adapters.onebot_models import PrivateMessageEvent
from app.adapters.sender import QQMessageDeliveryUncertainError
from app.private_chat.agent_service import PrivateHarnessService
from app.private_chat.access import PrivateChatAccess
from app.private_chat.dispatcher import PrivateMessageDispatcher
from app.private_chat.service import PrivateChatService
from app.providers.llm_client import LlmClient
from app.storage.agent_effects import AgentToolEffect
from app.storage.db import session_scope
from app.storage.models import Job, Message


ACCOUNT, OWNER, OTHER = 900000099, 900000101, 900000102


def sse(name=None, args=None, *, number=1):
    call_id = f"private-call-{number}"
    item = ({"type": "function_call", "id": "fc_" + call_id, "call_id": call_id,
        "name": name, "arguments": json.dumps(args or {}), "status": "completed"} if name else
        {"type": "message", "id": "msg_" + call_id, "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "internal complete", "annotations": []}]})
    response = {"id": f"private-response-{number}", "object": "response", "status": "in_progress", "model": "offline-agent", "output": []}
    events = [{"type": "response.created", "response": response}, {"type": "response.output_item.added", "output_index": 0,
        "item": {**item, "arguments": ""} if name else {**item, "content": []}}]
    if name:
        events.append({"type": "response.function_call_arguments.delta", "item_id": item["id"], "output_index": 0, "delta": item["arguments"]})
    else:
        events.extend([{"type": "response.content_part.added", "item_id": item["id"], "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}},
            {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": "internal complete"}])
    events.extend([{"type": "response.output_item.done", "output_index": 0, "item": item}, {"type": "response.completed", "response": {
        **response, "status": "completed", "output": [item], "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}}}])
    return "".join("event: " + value["type"] + "\ndata: " + json.dumps(value) + "\n\n" for value in events).encode()


class Sender:
    def __init__(self, uncertain=False):
        self.texts, self.images = [], []
        self.uncertain = uncertain

    async def send_private_text(self, message):
        self.texts.append((message.user_id, message.text))
        if self.uncertain:
            raise QQMessageDeliveryUncertainError("synthetic unknown")
        return (f"private-receipt-{len(self.texts)}",)

    async def send_private_image(self, *, user_id, image_file):
        self.images.append((user_id, image_file))
        return f"private-image-receipt-{len(self.images)}"


def setup(sqlite_engine, tmp_path, script, *, uncertain=False):
    requests = []
    def model(request):
        body = json.loads(request.content)
        requests.append(body)
        value = script(body, len(requests))
        return value if isinstance(value, httpx.Response) else httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=value)
    client = LlmClient(base_url="https://synthetic.invalid/v1", api_key="synthetic", model="offline-agent",
        responses_model="offline-agent", max_output_tokens=1024, http_client=httpx.Client(transport=httpx.MockTransport(model)))
    sender = Sender(uncertain)
    settings = SimpleNamespace(bot_qq=ACCOUNT, data_dir=tmp_path, agent_max_external_writes=2,
        agent_model_request_timeout_seconds=120, agent_model_progress_timeout_seconds=45)
    service = PrivateChatService(engine=sqlite_engine, sender=sender, llm_client=client,
        owner_qq=OWNER, bot_qq=ACCOUNT, private_chat_qqs={OTHER}, data_dir=tmp_path,
        reply_split_config={"enabled": False}, persona={"name": "小町"})
    agent = PrivateHarnessService(service, settings=settings)
    service.agent_handler = agent
    dispatcher = PrivateMessageDispatcher(sqlite_engine, tmp_path, service)
    return service, agent, dispatcher, sender, requests


def event(message_id="native-private-one", *, user_id=OWNER, text="聊聊夜空"):
    return PrivateMessageEvent(message_id, user_id, "synthetic", text,
        {"post_type": "message", "message_type": "private", "message_id": message_id,
            "user_id": user_id, "sender": {"nickname": "synthetic"}, "message": [{"type": "text", "data": {"text": text}}]}, datetime.now(UTC))


async def test_private_can_read_own_qzone_fact_without_public_context_or_owner_control(sqlite_engine, tmp_path):
    from tests.storage.test_self_activity import effect
    effect(sqlite_engine, 'independent-private-query', account=ACCOUNT, stamp=datetime.now(UTC))
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse('read_self_activity', {}, number=n) if n == 1 else
        sse('send_private', {'text': '今天已经发过空间了'}, number=n) if n == 2 else sse(number=n))
    await service.start()
    try:
        await dispatcher.handle_private_message(event('qzone-own-fact', user_id=OTHER, text='今天发空间了吗'))
    finally:
        await service.stop()
    assert sender.texts == [(OTHER, '今天已经发过空间了')]
    output = next(item for item in requests[1]['input'] if item.get('type') == 'function_call_output')
    actual = json.loads(output['output'])
    assert actual['data']['published'] is True and actual['data']['confirmed_count'] == 1
    assert actual['data']['coverage'] == 'system_publication_receipts_only'
    assert 'media_task_status' in {tool['name'] for tool in requests[0]['tools']}


@pytest.mark.parametrize("rotate_reason", ["context_version", "grok_model", "gemini_model"])
async def test_native_private_history_bootstraps_once_and_restores_only_after_rotation(sqlite_engine, tmp_path, rotate_reason):
    service,agent,dispatcher,sender,requests=setup(sqlite_engine,tmp_path,
        lambda body,n:sse('send_private',{'text':'正常回复'},number=n))
    def observation(body):
        for item in reversed(body['input']):
            content=item.get('content','')
            if isinstance(content,list):content='\n'.join(block.get('text','') for block in content)
            if 'Private observation JSON:\n' in content:
                return json.loads(content.split('Private observation JSON:\n',1)[1])
        raise AssertionError('private observation missing')
    await service.start()
    try:
        await dispatcher.handle_private_message(event('native-bootstrap-first',text='第一次普通聊天'))
        first=observation(requests[-1]);assert first['previous_closed_context']==[]
        await dispatcher.handle_private_message(event('native-bootstrap-second',text='第二次普通聊天'))
        second=observation(requests[-1])
        assert second['history_in_native_session'] and second['previous_closed_context']==[]
        assert second['daily_session_summary']==''
        scope=f'private:{OWNER}';state=agent.sessions.read(scope)
        if rotate_reason == "grok_model":
            service.llm_client.model = service.llm_client.responses_model = "grok-4.7"
        elif rotate_reason == "gemini_model":
            service.llm_client.model = service.llm_client.responses_model = "gemini-3.8-flash-low"
        else:
            agent.sessions.save(scope,state['items'],{**state['checkpoint'],'context_format_version':0},
                state['source_message_ids'],expected_revision=state['revision'])
        await dispatcher.handle_private_message(event('native-bootstrap-third',text='第三次恢复聊天'))
        third=observation(requests[-1])
        assert not third['history_in_native_session']
        assert [x['content'] for x in third['previous_closed_context'] if x['role']=='user']==['第一次普通聊天','第二次普通聊天']
    finally:
        await service.stop()


async def test_private_rejected_turn_rotates_without_failure_input_or_summary(sqlite_engine,tmp_path):
    from tests.agent.test_harness_runtime import image_policy_sse
    service,agent,dispatcher,sender,requests=setup(sqlite_engine,tmp_path,
        lambda body,n:image_policy_sse() if n==1 else sse('send_private',{'text':'正常回复'},number=n) if n==2 else sse(number=n))
    await service.start()
    try:
        await dispatcher.handle_private_message(event('rejected-first',text='REJECTED_PRIVATE_INPUT'))
        first=agent.sessions.read(f'private:{OWNER}')
        assert first['checkpoint']['last_reason']=='model_image_input_rejected'
        assert first['items']==[]
        assert '图片被模型接口拒绝' in sender.texts[0][1]
        await dispatcher.handle_private_message(event('normal-second',text='今天好吗'))
        second=agent.sessions.read(f'private:{OWNER}')
        assert second['checkpoint']['runtime_id']!=first['checkpoint']['runtime_id']
        assert 'REJECTED_PRIVATE_INPUT' not in json.dumps(requests[1],ensure_ascii=False)
        assert sender.texts[-1][1]=='正常回复'
        assert len(sender.texts)==2
    finally:
        await service.stop()


async def test_current_friend_outside_fixed_list_can_chat_without_owner_tools(sqlite_engine, tmp_path):
    friend = 900000103
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text": "你好"}) if n == 1 else sse(number=n))

    async def friends():
        return (friend,)

    service.access_policy = PrivateChatAccess(owner_qq=OWNER, allowlist=service.private_chat_qqs,
        all_friends=True, friend_reader=friends)
    await agent.start()
    try:
        await dispatcher.handle_private_message(event(user_id=friend))
    finally:
        await agent.stop()
    assert sender.texts == [(friend, "你好")]
    assert agent.allowed(friend)
    tools = {tool["name"] for tool in requests[0]["tools"]}
    assert "send_private" in tools
    assert not tools.intersection({"request_owner_action", "draw_private", "schedule_wakeup", "run_compute"})
    assert not await service.handle_private_message(event("stranger", user_id=900000104))
    assert len(sender.texts) == 1


async def test_friend_revoked_between_admission_and_execution_never_calls_model(sqlite_engine, tmp_path):
    friend = 900000103
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path, lambda body, n: sse())
    reads = 0

    async def friends():
        nonlocal reads
        reads += 1
        return (friend,) if reads == 1 else ()

    service.access_policy = PrivateChatAccess(owner_qq=OWNER, all_friends=True, friend_reader=friends)
    await agent.start()
    try:
        await dispatcher.handle_private_message(event(user_id=friend))
    finally:
        await agent.stop()
    assert reads == 2
    assert not requests and not sender.texts
    assert not agent.allowed(friend)
    with session_scope(sqlite_engine) as session:
        job = session.scalars(select(Job).where(Job.job_type == agent.job_type)).one()
        assert job.status == "failed"
        assert job.last_error_code == "private_agent_authority_revoked"


async def test_private_native_first_output_has_its_own_budget(sqlite_engine, tmp_path):
    import time
    class Thinking(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"type":"response.in_progress"}\n\n'
            time.sleep(1.2)
            yield sse("send_private", {"text": "已核对公开内容"})
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: httpx.Response(200, stream=Thinking()) if n == 1 else sse(number=n))
    agent.settings.agent_model_first_output_timeout_seconds = 1.7
    agent.settings.agent_model_progress_timeout_seconds = 0.04
    await service.start()
    try:
        await dispatcher.handle_private_message(event())
        assert sender.texts == [(OWNER, "已核对公开内容")] and len(requests) == 1
        with session_scope(sqlite_engine) as session:
            assert session.query(Job).filter(Job.job_type == agent.job_type).one().status == "completed"
    finally:
        await service.stop()


@pytest.mark.parametrize("mode", ["fresh", "sent", "uncertain", "revoked", "notice_unknown"])
async def test_private_failure_notice_uses_durable_authority_without_duplicate_send(
    sqlite_engine, tmp_path, monkeypatch, mode,
):
    from app.agent.tool_agent import ToolTurnResult
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse(number=n), uncertain=mode in {"uncertain", "notice_unknown"})
    class FailedRuntime:
        def __init__(self, **kwargs):
            self.invoke = kwargs["invoke_tool"]
        async def run(self, prompt, tools, guard, **kwargs):
            if mode in {"sent", "uncertain"}:
                send = next(tool for tool in tools if tool.name == "send_private")
                await self.invoke(send, {"text": "已开始的回复"}, "original-send")
            if mode == "revoked":
                monkeypatch.setattr(agent, "_valid_sources", lambda *_: False)
            return ToolTurnResult("failed", reason="model_first_output_timeout")
    agent.runtime_factory = FailedRuntime
    await service.start()
    try:
        normal = event()
        await dispatcher.handle_private_message(normal)
        await dispatcher.handle_private_message(normal)
        assert requests == []
        assert len(sender.texts) == (0 if mode == "revoked" else 1)
        with session_scope(sqlite_engine) as session:
            effects = session.query(AgentToolEffect).all()
            assert len(effects) == (0 if mode == "revoked" else 1)
            if mode != "revoked":
                effect = effects[0]
                assert effect.status == ("uncertain" if mode in {"uncertain", "notice_unknown"} else "sent")
                if mode in {"fresh", "notice_unknown"}:
                    assert effect.tool_call_id == "model_failure_notice"
                    assert "没能完成" in sender.texts[0][1]
                else:
                    assert sender.texts == [(OWNER, "已开始的回复")]
            job = session.query(Job).filter(Job.job_type == agent.job_type).one()
            assert job.status == ("uncertain" if mode == "notice_unknown" else "failed")
            if mode == "fresh":
                assert job.last_error_code == "model_first_output_timeout"
    finally:
        await service.stop()


async def test_private_native_web_body_open_find_and_send_are_wired(sqlite_engine, tmp_path):
    from app.providers.web_search import WebSearchClient
    web_calls = []
    def web(request):
        commands = json.loads(request.content)["commands"]
        web_calls.append(commands)
        return httpx.Response(200, json={"output": "Source (https://example.com/page)\nciteturn0view0\nL0: DETAILED_PUBLIC_EVIDENCE",
            "results": [{"type": "text_result", "url": "https://example.com/page", "ref_id": "turn0view0", "snippet": "short"}]})
    commands = [("search_web", {"query": "public question"}), ("open_web_page", {"ref_id": "https://example.com/page"}),
        ("find_web_page", {"ref_id": "https://example.com/page", "pattern": "EVIDENCE"}), ("send_private", {"text": "已核对公开正文"})]
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse(*commands[n-1], number=n) if n <= len(commands) else sse(number=n))
    service.web_search_client = WebSearchClient(provider="wyzlab_alpha", base_url="https://synthetic.invalid/v1",
        api_key="synthetic", timeout_seconds=45, http_client=httpx.Client(transport=httpx.MockTransport(web)))
    await service.start()
    try:
        await dispatcher.handle_private_message(event(user_id=OTHER, text='联网搜索公开夜空资料并打开链接'))
        assert sender.texts == [(OTHER, "已核对公开正文")]
        assert len(web_calls) == 3 and len(requests) == 4
        assert "DETAILED_PUBLIC_EVIDENCE" in json.dumps(requests[1], ensure_ascii=False)
        assert {"search_web", "open_web_page", "find_web_page"} <= {v["name"] for v in requests[0]["tools"]}
        assert web_calls[1]["open"][0]["ref_id"] == "https://example.com/page"
        assert web_calls[2]["find"][0]["pattern"] == "EVIDENCE"
    finally:
        await service.stop()


async def test_real_private_harness_sends_with_receipt_and_owns_only_restricted_tools(sqlite_engine, tmp_path):
    from app.agent.runtime_trace import TraceReader
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text": "夜空很适合慢慢看"}, number=n) if n == 1 else sse(number=n))
    agent.settings.agent_monitor_enabled = True
    await service.start()
    try:
        await asyncio.sleep(0.02)
        assert requests == []
        await dispatcher.handle_private_message(event())
        await dispatcher.handle_private_message(event())
        assert sender.texts == [(OWNER, "夜空很适合慢慢看")]
        assert len(requests) == 1
        names = {value["name"] for value in requests[0]["tools"]}
        assert names == {"read_private_context", "read_artifact", "read_message_links", "read_self_activity", "send_private", "generate_private_image", "private_image_status", "schedule_private_wakeup", "list_private_wakeups", "cancel_private_wakeup", 'begin_research_task', 'update_research_task', 'read_research_task', 'read_research_evidence', 'read_research_resources', 'resume_research_task', 'delegate_research_task', 'media_task_status', 'set_task_goal', 'review_task_progress'}
        snapshot = agent.sessions.read(f"private:{OWNER}")
        assert snapshot["checkpoint"]["receipts"][0]["receipt_id"] == "private-receipt-1"
        source_id = snapshot["checkpoint"]["reply_source_id"]
        assert snapshot["checkpoint"]["receipts"][0]["reply_source_id"] == source_id
        assert snapshot["items"][-2]["source_id"] == snapshot["items"][-1]["reply_source_id"] == source_id
        reader=TraceReader(tmp_path/'agent_monitor');traces=reader.list_traces()
        assert len(traces)==1 and traces[0]['surface']=='private' and traces[0]['source_id']==source_id
        assert traces[0]['scope_id']==OWNER and traces[0]['model_requests']==1
        assert traces[0]['context_kind']=='private' and traces[0]['context_state']=='new'
        assert traces[0]['trigger_kind']=='private_message' and traces[0]['trigger_text']==event().plain_text
        assert reader.list_conversations()[0]['identity_quality']=='logical'
        assert traces[0]['trigger_reason_code']=='private_message_queued'
        assert traces[0]['trigger_evidence']['job_id']>0
        assert any(e['kind']=='tool.completed' and e['data']['receipt_id']=='private-receipt-1'
            for e in reader.read_trace(traces[0]['trace_id'])['events'])
        with session_scope(sqlite_engine) as session:
            effects = session.query(AgentToolEffect).all()
            assert len(effects) == 1 and effects[0].status == "sent"
    finally:
        await service.stop()


async def test_formal_private_delivers_long_answer_and_returns_to_chat_without_code_bulk(sqlite_engine, tmp_path):
    from app.providers.gemini_search import GeminiSearchClient
    answer='\n'.join(f'文件{i}：已核对入口，认证模块仍待检查。' for i in range(150))
    def script(body,n):
        if n==1:return sse('open_web_page',{'ref_id':'https://example.com/source'},number=n)
        if n==2:
            assert 'RAW_REPOSITORY_CODE' in json.dumps(body)
            return sse('update_research_task',{'progress':'入口已核对','next_step':'检查认证',
                'findings':['入口已核对，未检查认证'],'status':'partial'},number=n)
        if n==3:return sse('send_private',{'text':answer},number=n)
        if n==4:
            return sse('send_private',{'text':'继续聊夜空吧'},number=n)
        return sse(number=n)
    service,agent,dispatcher,sender,requests=setup(sqlite_engine,tmp_path,script)
    service.reply_split_config={'enabled':True,'max_chars':30,'max_messages':3,'separator':'|'}
    service.web_search_client=GeminiSearchClient(provider='gemini',base_url='',api_key='synthetic',model='synthetic',
        timeout_seconds=45,address_resolver=lambda *_:['93.184.216.34'],
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(200,
            headers={'content-type':'text/plain'},text=('RAW_REPOSITORY_CODE\n'*900)))))
    await service.start()
    try:
        await dispatcher.handle_private_message(event('research-private',text='打开 https://example.com/source 认真审查这个仓库的代码'))
        await dispatcher.handle_private_message(event('back-to-chat',text='还是聊夜空吧'))
        assert sender.texts==[(OWNER,answer),(OWNER,'继续聊夜空吧')]
        assert len(requests)==4
        assert 'RAW_REPOSITORY_CODE' not in json.dumps(requests[3])
        assert '入口已核对' in json.dumps(requests[3],ensure_ascii=False)
    finally:await service.stop()


@pytest.mark.parametrize("requested_links", [False, True])
async def test_private_native_link_policy_preserves_complete_answer_without_empty_source_tail(
    sqlite_engine, tmp_path, requested_links,
):
    reply = "先核对当天套餐。\n再核对加价项。\n最后确认门店营业时间。\n查到的门店页：https://example.com/shop"
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text": reply}, number=n) if n == 1 else sse(number=n))
    service.reply_split_config = {"enabled": True, "separator": "|", "max_messages": 3,
        "max_chars": 8, "auto_split_long_segments": True, "min_delay_seconds": 0.001, "max_delay_seconds": 0.001}
    await service.start()
    try:
        # This test only verifies formatting of supplied information; actual
        # lookup/evidence requirements are covered by the web/goal contracts.
        await dispatcher.handle_private_message(event(text="按已提供的当天套餐回复，把链接发我" if requested_links else "按已提供的当天套餐回复"))
        sent = "".join(text for _, text in sender.texts)
        assert "先核对当天套餐。" in sent
        assert "再核对加价项。" in sent
        assert "最后确认门店营业时间。" in sent
        assert len(sender.texts) <= 3
        if requested_links:
            assert "https://example.com/shop" in sent
            assert "Include only the URLs needed" in json.dumps(requests[0], ensure_ascii=False)
        else:
            assert "http" not in sent and "查到的门店页" not in sent
            assert "Do not include URLs" in json.dumps(requests[0], ensure_ascii=False)
        assert requests[0]["input"][0]["role"] in {"system", "developer"}
        assert "Talk like a real person" in requests[0]["input"][0]["content"]
        assert "Stay grounded" in requests[0]["input"][0]["content"]
        snapshot = agent.sessions.read(f"private:{OWNER}")
        assert snapshot["checkpoint"]["last_status"] == "sent"
        with session_scope(sqlite_engine) as session:
            effect = session.query(AgentToolEffect).one()
            assert len(effect.receipt["data"]["message_ids"]) == len(sender.texts)
    finally:
        await service.stop()


async def test_private_source_date_uses_stored_shanghai_and_current_anchor(sqlite_engine, tmp_path):
    from app.core.time_utils import stored_as_utc
    current = datetime(2026, 10, 10, 19, tzinfo=UTC)
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("read_private_context", {}, number=n) if n == 1 else
            sse("send_private", {"text":"已核对本轮日期"}, number=n) if n == 2 else sse(number=n))
    agent.clock = lambda: current
    incoming = event(message_id="private-old-date", text="今天看合成动画")
    incoming.timestamp = datetime(2026, 10, 10, 14, tzinfo=UTC)
    await service.start()
    try:
        await dispatcher.handle_private_message(incoming)
        assert len(requests) == 2 and sender.texts == [(OWNER, "已核对本轮日期")]
        body = json.dumps(requests, ensure_ascii=False)
        assert '2026-10-11T03:00:00+08:00' in body
        assert '2026-10-10T22:00:00+08:00' in body
        assert '2026-10-11T06:00:00' not in body
        with session_scope(sqlite_engine) as session:
            row = session.query(Message).filter_by(platform_msg_id=f"private-inbound-{incoming.user_id}-{incoming.platform_msg_id}").one()
            assert stored_as_utc(row.timestamp) == incoming.timestamp
    finally:
        await service.stop()


async def test_real_private_harness_async_image_queue_and_background_source_receipt(sqlite_engine, tmp_path, monkeypatch):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("generate_private_image", {"prompt": "合成蓝色夜空", "use_current_images": False}, number=n) if n == 1 else sse(number=n))
    path = tmp_path / "synthetic-generated.png"
    Image.new("RGB", (24, 24), "blue").save(path)
    monkeypatch.setattr(service.private_image_service, "_generate_and_store_image", lambda request: path)
    await service.start()
    try:
        await dispatcher.handle_private_message(event(text="画一张夜空"))
        await service.private_image_service.wait_for_idle()
        assert sender.images == [(OWNER, str(path))]
        assert sender.texts == [(OWNER, "图好了")]
        output = next(value for value in reversed(requests[1]["input"]) if value.get("type") == "function_call_output")
        result = json.loads(output["output"])
        assert result["data"]["status"] == "queued"
        with session_scope(sqlite_engine) as session:
            job = session.get(Job, result["data"]["task_id"])
            assert job.job_type == "private_image_generation" and job.status == "completed"
            assert job.payload_json["agent_source_message_id"] == f"private-inbound-{OWNER}-native-private-one"
            assert job.payload_json["user_id"] == OWNER
            assert job.payload_json["image_receipt_ids"] == ["private-image-receipt-1"]
            assert job.payload_json["notice_receipt_ids"] == ["private-receipt-1"]
    finally:
        await service.stop()


async def test_unknown_private_send_stops_without_duplicate_and_survives_restart(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text": "一次合成发送"}, number=n), uncertain=True)
    await service.start()
    try:
        await dispatcher.handle_private_message(event())
        assert len(requests) == 1 and sender.texts == [(OWNER, "一次合成发送")]
        with session_scope(sqlite_engine) as session:
            effect = session.query(AgentToolEffect).one()
            assert effect.status == "uncertain"
    finally:
        await service.stop()
    service2, _, dispatcher2, sender2, requests2 = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text": "不得重放"}, number=n))
    await service2.start()
    try:
        await dispatcher2.handle_private_message(event())
        assert sender2.texts == [] and requests2 == []
    finally:
        await service2.stop()


async def test_private_goal_wakeup_uses_persistent_queue_and_next_native_decision(sqlite_engine, tmp_path):
    requests_by_wake = []
    def script(body, n):
        requests_by_wake.append(body)
        if n == 1:
            return sse("schedule_private_wakeup", {"goal": "以后聊观星", "delay_seconds": 120}, number=n)
        if n == 2:
            return sse("send_private", {"text": "已记下提醒"}, number=n)
        if n == 3:
            return sse("send_private", {"text": "记得我们想聊的观星吗"}, number=n)
        return sse(number=n)
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path, script)
    agent.settings.agent_monitor_enabled = True
    await service.start()
    try:
        await dispatcher.handle_private_message(event())
        assert len(requests) == 2 and sender.texts == [(OWNER, "已记下提醒")]
        with session_scope(sqlite_engine) as session:
            wake = session.query(Job).filter(Job.job_type == agent.job_type, Job.status == "queued").one()
            assert wake.payload_json["kind"] == "wakeup" and wake.payload_json["source_id"] > 0
            wake.run_at = datetime.now(UTC).replace(tzinfo=None)
        await agent._drain_user(OWNER)
        assert sender.texts == [(OWNER, "已记下提醒"), (OWNER, "记得我们想聊的观星吗")]
        assert "以后聊观星" in json.dumps(requests[2], ensure_ascii=False)
        from app.agent.runtime_trace import TraceReader
        traces = TraceReader(tmp_path/'agent_monitor').list_traces()
        wake_trace = next(t for t in traces if t['trigger_kind']=='private_wakeup')
        assert wake_trace['trigger_reason_code']=='private_wakeup_due'
        assert wake_trace['trigger_evidence']['goal']=='以后聊观星'
        assert wake_trace['trigger_evidence']['due_at']
    finally:
        await service.stop()


async def test_private_agent_rejects_other_user_and_recalled_image_request(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path, lambda body, n: sse(number=n))
    await service.start()
    try:
        await dispatcher.handle_private_message(event(user_id=OTHER + 1))
        assert requests == [] and sender.texts == []
        normal = event()
        dispatcher._persist_inbound(normal)
        with session_scope(sqlite_engine) as session:
            source = session.query(Message).filter(Message.platform_msg_id == f"private-inbound-{OWNER}-{normal.platform_msg_id}").one()
            source.raw_json = {**source.raw_json, "delivery_state": "deleted"}
        result = await agent.handle(normal, private_scope="owner_daily")
        assert result.status == "blocked" and requests == []
        from app.core.group_image_generation import PrivateImageGenerationRequest
        queued = await service.private_image_service.enqueue(PrivateImageGenerationRequest(OWNER, normal.platform_msg_id,
            "invalid", agent_source_message_id=f"private-inbound-{OWNER}-{normal.platform_msg_id}"))
        assert not queued.accepted and queued.reason == "source_unusable"
    finally:
        await service.stop()


async def test_private_native_inflight_checkpoint_rotates_before_recovery(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text":"已接续新消息"}, number=n) if n == 1 else sse(number=n))
    original = event("interrupted-private", text="previous canonical source")
    dispatcher._persist_inbound(original)
    with session_scope(sqlite_engine) as session:
        source = session.query(Message).filter(Message.platform_msg_id == f"private-inbound-{OWNER}-{original.platform_msg_id}").one()
        source_id = source.id
    agent.sessions.begin_turn(f"private:{OWNER}", {"runtime_id": "interrupted-native-session", "receipts": []},
        (source_id,), expected_revision=0)
    await service.start()
    try:
        await dispatcher.handle_private_message(event("after-interruption"))
        snapshot = agent.sessions.read(f"private:{OWNER}")
        assert snapshot["checkpoint"]["runtime_id"] != "interrupted-native-session"
        assert snapshot["checkpoint"]["in_flight"] is False
        assert len(requests) == 1 and sender.texts == [(OWNER, "已接续新消息")]
    finally:
        await service.stop()


async def test_private_native_partial_burst_keeps_receipts_and_stops(sqlite_engine, tmp_path, monkeypatch):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text": "合成分段回复"}, number=n))
    async def partial_reply(*, receipt_callback, before_dispatch, **kwargs):
        before_dispatch()
        receipt_callback(("earlier-partial-receipt",))
        return False, "later_part_failed"
    monkeypatch.setattr(service, "_send_private_chat_reply", partial_reply)
    await service.start()
    try:
        await dispatcher.handle_private_message(event())
        assert len(requests) == 1
        with session_scope(sqlite_engine) as session:
            effect = session.query(AgentToolEffect).one()
            assert effect.status == "uncertain"
            assert effect.receipt["data"]["message_ids"] == ["earlier-partial-receipt"]
            assert effect.receipt["receipt_id"] == "earlier-partial-receipt"
    finally:
        await service.stop()


async def test_private_native_receives_canonical_actual_pixels(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text":"合成图片是蓝色"}, number=n) if n == 1 else sse(number=n))
    picture = agent.image_input_root(OWNER) / "unknown" / "private-source.png"
    picture.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "blue").save(picture)
    image_event = event("private-picture", text="这张图是什么颜色")
    from app.core.message_content import extract_images_from_raw_payload
    image_event.raw_payload["message"].append({"type": "image", "data": {"url": "", "local_path": str(picture)}})
    image_event.images = extract_images_from_raw_payload(image_event.raw_payload)
    await service.start()
    try:
        await dispatcher.handle_private_message(image_event)
        assert len(requests) == 1 and sender.texts == [(OWNER, "合成图片是蓝色")]
        blocks = [part for item in requests[0]["input"] for part in item.get("content", []) if isinstance(part, dict)]
        images = [part for part in blocks if part.get("type") == "input_image"]
        assert images and images[0]["image_url"].startswith("data:image/png;base64,")
    finally:
        await service.stop()


@pytest.mark.parametrize("foreign_scope", ["group", "private_user", "host"])
async def test_private_native_rejects_foreign_or_host_image_paths(sqlite_engine, tmp_path, foreign_scope):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse("send_private", {"text":"本轮没有可用图片"}, number=n) if n == 1 else sse(number=n))
    picture = ({"group": tmp_path / "image_cache" / "900000001" / "foreign.png",
        "private_user": agent.image_input_root(OTHER) / "unknown" / "foreign.png",
        "host": tmp_path / "host-image.png"})[foreign_scope]
    picture.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "red").save(picture)
    from app.core.message_content import extract_images_from_raw_payload
    image_event = event("foreign-private-picture", text="这是我的图吗")
    image_event.raw_payload["message"].append({"type": "image", "data": {"url": "", "local_path": str(picture)}})
    image_event.images = extract_images_from_raw_payload(image_event.raw_payload)
    await service.start()
    try:
        await dispatcher.handle_private_message(image_event)
        assert len(requests) == 1 and sender.texts == [(OWNER, "本轮没有可用图片")]
        assert "data:image/" not in json.dumps(requests)
        with session_scope(sqlite_engine) as session:
            source = session.query(Message).filter(Message.platform_msg_id == f"private-inbound-{OWNER}-{image_event.platform_msg_id}").one()
            assert "local_path" not in source.raw_json["message"][-1]["data"]
    finally:
        await service.stop()


def test_private_pre_dispatch_recovery_has_finite_attempts(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse(number=n))
    normal = event()
    dispatcher._persist_inbound(normal)
    with session_scope(sqlite_engine) as session:
        source = session.query(Message).filter(Message.platform_msg_id == f"private-inbound-{OWNER}-{normal.platform_msg_id}").one()
        source_id = source.id
    queued = agent._enqueue(user_id=OWNER, source_id=source_id, private_scope="owner_daily")
    for number in range(3):
        assert agent._claim(OWNER)[0] == queued["task_id"]
        with session_scope(sqlite_engine) as session:
            row = session.get(Job, queued["task_id"])
            row.lease_until = datetime(2000, 1, 1)
    assert agent._claim(OWNER) is None
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, queued["task_id"]).status == "failed"
    agent.runtime_runner.close()
    agent.tool_runner.close()


def test_private_goal_enqueue_idempotence_and_late_guard_rollback(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse(number=n))
    normal = event()
    dispatcher._persist_inbound(normal)
    with session_scope(sqlite_engine) as session:
        source = session.query(Message).filter(Message.platform_msg_id == f"private-inbound-{OWNER}-{normal.platform_msg_id}").one()
        source_id = source.id
    args = {"user_id": OWNER, "source_id": source_id, "private_scope": "owner_daily", "kind": "wakeup",
        "goal": "later sky discussion", "delay_seconds": 3, "key": "same-private-wake-goal"}
    first, repeated = agent._enqueue(**args), agent._enqueue(**args)
    assert first["task_id"] == repeated["task_id"] and first["run_at"] == repeated["run_at"]
    def revoked():
        raise RuntimeError("source authority revoked")
    with pytest.raises(RuntimeError, match="revoked"):
        agent._enqueue(**{**args, "key": "late-revocation"}, before_commit=revoked)
    with session_scope(sqlite_engine) as session:
        assert session.query(Job).filter(Job.job_type == agent.job_type).count() == 1
    agent.runtime_runner.close()
    agent.tool_runner.close()

def test_owner_effect_scope_is_recovered_without_requeue_after_crash(sqlite_engine,tmp_path):
    from datetime import timedelta
    service,agent,dispatcher,_,_=setup(sqlite_engine,tmp_path,lambda body,n:sse(number=n))
    original=event(); dispatcher._persist_inbound(original)
    with session_scope(sqlite_engine) as db:
        sid=db.query(Message).filter(Message.platform_msg_id==f'private-inbound-{OWNER}-{original.platform_msg_id}').one().id
    queued=agent._enqueue(user_id=OWNER,source_id=sid,private_scope='owner_daily',key='crash-after-send')
    assert agent._claim(OWNER)
    effect=agent.effects.reserve(f'owner:private:{OWNER}',run_id=f"private-job:{queued['task_id']}",
        tool_call_id='synthetic-send',operation='send_private',payload_hash='a'*64,sources=(sid,))
    assert agent.effects.mark_dispatching(effect['effect_id'],daily_limit=None)
    with session_scope(sqlite_engine) as db:
        db.get(Job,queued['task_id']).lease_until=datetime.now(UTC).replace(tzinfo=None)-timedelta(seconds=1)
    assert agent._claim(OWNER) is None
    with session_scope(sqlite_engine) as db:
        assert db.get(Job,queued['task_id']).status=='uncertain'
    assert agent.effects.get(effect['effect_id'])['status']=='uncertain'
    agent.runtime_runner.close();agent.tool_runner.close()

def test_nonowner_cannot_schedule_or_keep_legacy_drawing_authority(sqlite_engine,tmp_path):
    service,agent,dispatcher,_,_=setup(sqlite_engine,tmp_path,lambda body,n:sse(number=n))
    original=event(user_id=OTHER); dispatcher._persist_inbound(original)
    with session_scope(sqlite_engine) as db:
        sid=db.query(Message).filter(Message.platform_msg_id==f'private-inbound-{OTHER}-{original.platform_msg_id}').one().id
    assert agent._enqueue(user_id=OTHER,source_id=sid,private_scope='daily',kind='wakeup',goal='提醒')['error']=='owner_required'
    assert agent._enqueue(user_id=OTHER,source_id=sid,private_scope='daily')['status']=='queued'
    assert not service.private_image_service._agent_private_source_usable(SimpleNamespace(user_id=OTHER,agent_source_message_id=None))
    agent.runtime_runner.close();agent.tool_runner.close()


@pytest.mark.parametrize('foreign_quote', [False, True])
async def test_private_quoted_share_discovery_and_search_fallback(sqlite_engine, tmp_path, foreign_quote):
    from app.adapters.onebot_models import parse_private_message_event
    from app.providers.web_search import WebSearchClient
    web_calls = []
    def outputs(body, name):
        calls = {item['call_id']: item['name'] for item in body.get('input', []) if item.get('type') == 'function_call'}
        values = [item for item in body.get('input', []) if item.get('type') == 'function_call_output' and calls[item['call_id']] == name]
        return json.loads(values[-1]['output'])
    def model(body, n):
        if n == 1:
            return sse('read_private_context', {}, number=n)
        if n == 2:
            found = outputs(body, 'read_private_context')['data']['share_sources']
            if foreign_quote:
                assert found == []
                return sse('send_private', {'text':'无法读取其他用户的引用卡片'}, number=n)
            assert found[0]['title'] == '合成分享新闻'
            return sse('read_message_links', {'source_id': found[0]['source_id']}, number=n)
        if n == 3:
            if foreign_quote:
                return sse(number=n)
            data = outputs(body, 'read_message_links')['data']
            assert data['metadata_only'] and data['trust'] == 'untrusted_shared_content'
            return sse('open_web_page', {'ref_id': data['links'][0]['url']}, number=n)
        if n == 4:
            assert outputs(body, 'open_web_page')['status'] == 'failed'
            return sse('search_web', {'query': '合成分享新闻 example.com'}, number=n)
        if n == 5:
            return sse('open_web_page', {'ref_id': outputs(body, 'search_web')['data']['sources'][0]['ref_id']}, number=n)
        if n == 6:
            assert '完整合成新闻正文' in outputs(body, 'open_web_page')['data']['content']
            return sse('send_private', {'text': '已查阅补充来源，概要如下'}, number=n)
        return sse(number=n)
    def web(request):
        web_calls.append(json.loads(request.content)['commands'])
        if len(web_calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={'output': '来源 (https://example.org/news)\nciteturn0view0\nL0: 完整合成新闻正文',
            'results': [{'type': 'text_result', 'url': 'https://example.org/news', 'ref_id': 'turn0view0', 'snippet': '摘要'}]})
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path, model)
    service.web_search_client = WebSearchClient(provider='wyzlab_alpha', base_url='https://synthetic.invalid/v1',
        api_key='synthetic', timeout_seconds=45, http_client=httpx.Client(transport=httpx.MockTransport(web)))
    shared = parse_private_message_event({'message_id': 'quoted-card', 'user_id': OWNER if foreign_quote else OTHER,
        'time': 100, 'message': [{'type': 'share', 'data': {'url': 'https://example.com/news',
                                                      'title': '合成分享新闻', 'content': '卡片摘要'}}]})
    dispatcher._persist_inbound(shared)
    current = event('read-quoted-card', user_id=OTHER, text='看看引用卡片说了什么')
    current.reply_to_msg_id = 'quoted-card'
    current.raw_payload['message'].insert(0, {'type': 'reply', 'data': {'id': 'quoted-card'}})
    await service.start()
    try:
        await dispatcher.handle_private_message(current)
        if foreign_quote:
            assert len(requests) == 2 and sender.texts == [(OTHER, '无法读取其他用户的引用卡片')] and not web_calls
        else:
            assert len(requests) == 6 and len(web_calls) == 3
            assert sender.texts == [(OTHER, '已查阅补充来源，概要如下')]
            assert web_calls[0]['open'][0]['ref_id'] == 'https://example.com/news'
            assert web_calls[2]['open'][0]['ref_id'] == 'https://example.org/news'
            names = {tool['name'] for tool in requests[0]['tools']}
            assert {'read_message_links', 'search_web', 'open_web_page'} <= names
            assert 'request_owner_action' not in names
    finally:
        await service.stop()
