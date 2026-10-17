"""Replay floors and exact source masks have deliberately different authority."""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text

from app.core.context_reset import authored_reset, parse_context_reset, reset_confirmation
from app.core.memory_source_guard import is_usable_memory_source
from app.storage.context_resets import ContextResetRepository, automatic_context_floor
from app.storage.db import create_all, session_scope
from app.storage.models import ContextExcludedMessage, ContextReset, Message
from app.storage.repositories import MessageRepository
from test_context_resets import ACCOUNT, GROUP, OTHER, OWNER, source


def control(engine, name, action, reply=None, *, group=GROUP, user=OWNER):
    identifier = source(engine,name,text=action,group=group,user=user)
    if reply is not None:
        with session_scope(engine) as session:
            row=session.get(Message,identifier)
            row.raw_json={**row.raw_json,'message':[{'type':'reply','data':{'id':reply}},*row.raw_json['message']]}
            row.reply_to_msg_id=reply
    return identifier


def test_plain_reset_does_not_mask_normal_history_or_backfill(sqlite_engine):
    normal=[source(sqlite_engine,f'normal-{i}') for i in range(35)]
    command=control(sqlite_engine,'plain-reset','重置上下文')
    repo=ContextResetRepository(sqlite_engine,account_id=ACCOUNT)
    result=repo.reset(scope=f'group:{GROUP}',actor_id=OWNER,source_id=command)
    assert result['mode']=='session' and result['excluded_count']==0
    assert '没有批量屏蔽' in reset_confirmation(result,private=False)
    create_all(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        assert all(is_usable_memory_source(session.get(Message,i)) for i in normal)
        assert MessageRepository(session).list_recent_group_messages(group_id=GROUP,limit=10)==[]
        assert automatic_context_floor(session,f'group:{OTHER}')==0
        assert automatic_context_floor(session,f'group:{GROUP}')==command
    new=source(sqlite_engine,'after-reset')
    with session_scope(sqlite_engine) as session:
        assert [r.id for r in MessageRepository(session).list_recent_group_messages(group_id=GROUP,limit=10)]==[new]
        # Explicit history queries remain authorized and contain canonical originals.
        assert len(list(session.scalars(select(Message).where(Message.id.in_(normal)))))==35


@pytest.mark.parametrize('private',[False,True])
def test_quote_isolation_restore_and_scope_persistence(sqlite_engine,private):
    group=None if private else GROUP;scope=f'private:{OWNER}' if private else f'group:{GROUP}'
    target=source(sqlite_engine,'bad',group=group,text='QUARANTINED_MARKER')
    good=source(sqlite_engine,'good',group=group,text='NORMAL_MARKER')
    foreign=source(sqlite_engine,'foreign',group=OTHER)
    command=control(sqlite_engine,'isolate','隔离这条并重置上下文','bad',group=group)
    repo=ContextResetRepository(sqlite_engine,account_id=ACCOUNT)
    result=repo.reset(scope=scope,actor_id=OWNER,source_id=command,action='isolate')
    assert result['excluded_count']==1 and result['target_message_id']==target
    create_all(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        assert not is_usable_memory_source(session.get(Message,target))
        assert is_usable_memory_source(session.get(Message,good))
        assert is_usable_memory_source(session.get(Message,foreign))
        assert session.get(Message,target).plain_text=='QUARANTINED_MARKER'
    restore=control(sqlite_engine,'restore','恢复这条上下文','bad',group=group)
    assert repo.reset(scope=scope,actor_id=OWNER,source_id=restore,action='restore')['mode']=='restore'
    with session_scope(sqlite_engine) as session:
        assert is_usable_memory_source(session.get(Message,target))
        assert session.get(ContextExcludedMessage,command) is not None


def test_quarantine_does_not_fill_recent_window_with_older_rows(sqlite_engine):
    old=source(sqlite_engine,'outside-window')
    recent=[source(sqlite_engine,f'row-{i}') for i in range(10)]
    command=control(sqlite_engine,'isolate-last','隔离这条并重置上下文','row-5')
    ContextResetRepository(sqlite_engine,account_id=ACCOUNT).reset(
        scope=f'group:{GROUP}',actor_id=OWNER,source_id=command,action='isolate')
    with session_scope(sqlite_engine) as session:
        ids=[r.id for r in MessageRepository(session).list_recent_group_messages(group_id=GROUP,limit=10)]
        assert ids==[i for i in recent[1:] if i!=recent[5]]
        assert old not in ids and recent[0] not in ids


@pytest.mark.parametrize('action',['isolate','restore'])
def test_cannot_control_foreign_source_or_restore_a_command(sqlite_engine,action):
    source(sqlite_engine,'foreign-target',group=OTHER)
    control_id=control(sqlite_engine,'scoped-control','隔离这条并重置上下文','foreign-target')
    repo=ContextResetRepository(sqlite_engine,account_id=ACCOUNT)
    with pytest.raises(ValueError,match='target_unavailable'):
        repo.reset(scope=f'group:{GROUP}',actor_id=OWNER,source_id=control_id,action=action)
    assert repo.latest_source_id(f'group:{GROUP}')==0


def test_control_requires_authored_text_and_one_real_reply():
    message=[{'type':'reply','data':{'id':'origin'}},{'type':'text','data':{'text':'重置上下文'}}]
    event=SimpleNamespace(raw_payload={'message':message},plain_text='重置上下文')
    assert authored_reset(event).action=='isolate'
    assert parse_context_reset('重置上下文',default_count=30).count is None
    message.append({'type':'image','data':{'url':'https://synthetic.invalid/image'}})
    assert authored_reset(event) is None
    assert authored_reset(SimpleNamespace(raw_payload={},plain_text='恢复这条上下文')).error


def test_old_schema_migration_keeps_legacy_meaning(sqlite_engine):
    with sqlite_engine.begin() as con:
        con.exec_driver_sql('DROP TABLE context_resets')
        con.exec_driver_sql('CREATE TABLE context_resets (source_id INTEGER PRIMARY KEY,account_id INTEGER,scope VARCHAR(160),requested_count INTEGER,excluded_count INTEGER,created_at DATETIME)')
        con.exec_driver_sql("INSERT INTO context_resets VALUES (123,1,'group:900000001',30,30,'2026-01-01')")
    create_all(sqlite_engine);create_all(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        row=session.get(ContextReset,123)
        assert row.mode=='legacy_count' and row.automatic_after_id==0


@pytest.mark.asyncio
async def test_plain_reset_stops_automatic_memory_but_keeps_explicit_tools(sqlite_engine,monkeypatch):
    from app.agent.memory_integration import build_agent_memory,call_agent_memory_tool
    from app.agent.tool_agent import ToolOutcome
    command=control(sqlite_engine,'memory-reset','重置上下文')
    ContextResetRepository(sqlite_engine,account_id=ACCOUNT).reset(scope=f'group:{GROUP}',actor_id=OWNER,source_id=command)
    context=SimpleNamespace(router=SimpleNamespace(engine=sqlite_engine,runtime=SimpleNamespace(settings=SimpleNamespace())),
        observation=SimpleNamespace(job=SimpleNamespace(kind='group',scope_id=GROUP)),
        memory_search_allowed=True,guard=SimpleNamespace(require_valid=lambda:None))
    calls=[]
    async def read(*args,**kwargs):
        calls.append(kwargs);return ToolOutcome({'evidence':'normal historical evidence'})
    monkeypatch.setattr('app.agent.memory_integration._run_read',read)
    assert (await build_agent_memory(context)).data['mode']=='manual_recall_after_reset'
    assert calls==[]
    assert (await call_agent_memory_tool(context,'memory_search',{'query':'normal'})).data['evidence']=='normal historical evidence'
    assert len(calls)==1
    context.target_message_id='synthetic-current'
    monkeypatch.setattr('app.agent.memory_integration._request',lambda *a,**k:
        SimpleNamespace(query='继续聊吧',quoted_message=object()))
    assert (await build_agent_memory(context)).data['memory_search_allowed'] is False
    context.router._group_memory_enabled=lambda **_:False
    await build_agent_memory(context)
    assert len(calls)==2  # Disabled groups keep the original guarded reader path.
