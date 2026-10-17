"""A policy-blocked dispatch closes the real group patrol checkpoint."""
from dataclasses import replace
from types import SimpleNamespace

from app.agent.observation import GroupObservationService
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.agent.tool_agent import ToolTurnResult
from app.agent.tool_dispatch import ToolDispatchBoundary
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository
from test_autonomous_social import composed,ScriptedModel,add_source,_mock_native_session,NOW,GROUP,FRIEND


async def test_blocked_group_patrol_saves_terminal_checkpoint_without_false_reset(
        sqlite_engine,tmp_path,monkeypatch):
    class Harness:
        def __init__(self,**kwargs):
            self.session_id=_mock_native_session(kwargs);self.invoke=kwargs['invoke_tool']
        async def run(self,prompt,tools,guard,**kwargs):
            outcome=await self.invoke(next(t for t in tools if t.name=='send_group'),
                {'text':'synthetic patrol','artifact_ids':[]},'synthetic-patrol-send')
            assert outcome.status=='blocked' and outcome.data['error']=='daily_scope_budget_exhausted'
            guard.parent.require_valid()
            return ToolTurnResult('blocked',reason=outcome.data['error'])
    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent',Harness)
    async with composed(sqlite_engine,tmp_path,ScriptedModel(),runtime='harness') as fixture:
        source=add_source(sqlite_engine,platform_id='patrol-checkpoint',text='合成普通群消息')
        job=replace(next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind=='group'),
            chat_cadence_only=False,max_daily=1,cooldown_seconds=1)
        observer=GroupObservationService(fixture.router,clock=lambda:NOW)
        def valid():
            with session_scope(sqlite_engine) as session:
                return not observer._suppression(group_id=GROUP,event_id='patrol-checkpoint',
                    messages=MessageRepository(session),policy=(True,True,True,(60,60),None,False))
        guard=SimpleNamespace(action_id='synthetic-patrol-turn',valid=valid,
            trace_trigger={'trigger_reason_code':'intention_due'},
            require_valid=lambda:None if valid() else (_ for _ in ()).throw(RuntimeError('policy_revoked')),
            on_dispatch=lambda:None)
        scope=fixture.executor.data_scope(job)
        ledger=ToolDispatchBoundary(engine=sqlite_engine,account_id=fixture.router.runtime.settings.bot_qq,
            scope=scope,guard=guard,clock=lambda:NOW).repository
        row=ledger.reserve(scope,run_id='synthetic-prior',tool_call_id='send',operation='send_group',payload_hash='a'*64)
        assert ledger.mark_dispatching(row['action_id'],daily_limit=None)
        ledger.finish(row['action_id'],'sent',receipt='synthetic-prior-receipt')
        result=await fixture.executor.run_tools(SocialObservation(job,NOW,'synthetic',source_message_ids=(source,)),guard,
            interactive=True,target_user_id=FRIEND,target_message_id='patrol-checkpoint')
        checkpoint=fixture.executor.sessions.read(scope)['checkpoint']
        assert result.status=='blocked' and checkpoint['last_status']=='blocked',(result,checkpoint)
        assert checkpoint['last_reason']=='daily_scope_budget_exhausted' and valid()
