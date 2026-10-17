from types import SimpleNamespace
import pytest
from app.agent.request_budget import RequestBudget
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.request_task import RequestTaskControl
from app.agent.tool_agent import ToolOutcome
from tests.agent.test_harness_runtime import bridge_factory,tool,responses_sse


def workspace(tmp_path,**kwargs):
    return ResearchTaskWorkspace(tmp_path,scope=kwargs.get('scope','private:900000101'),actor_id=900000101,
        source_id=10,request='帮我下载一些旅行图片',valid_source=kwargs.get('valid',lambda _:True),clock=kwargs.get('clock',lambda:1000))


def test_same_request_budget_survives_wakes_and_reserves_last_send(tmp_path):
    w=workspace(tmp_path);budget=RequestBudget(w,limits={'model_calls':1,'writes':2,'media_received':10})
    budget.spend('model_calls');budget.charge(7)
    resumed=RequestBudget(workspace(tmp_path))
    assert resumed.received==7 and resumed.remaining==3
    with pytest.raises(ValueError):resumed.spend('model_calls')
    resumed.spend('writes')
    with pytest.raises(ValueError):resumed.spend('writes')
    resumed.spend('writes',finalization=True)
    with pytest.raises(ValueError):resumed.spend('writes',finalization=True)


def test_expired_or_revoked_budget_never_becomes_a_new_budget(tmp_path):
    clock=[1000];valid=[True]
    w=workspace(tmp_path,clock=lambda:clock[0],valid=lambda _:valid[0]);budget=RequestBudget(w,limits={})
    clock[0]+=601
    with pytest.raises(ValueError):RequestBudget(w).spend('model_calls')
    valid[0]=False
    with pytest.raises(ValueError):budget.spend('writes',finalization=True)


@pytest.mark.parametrize('state',['ready','unreviewed','revoked','uncertain','cancelled','expired'])
async def test_model_zero_finalizer_keeps_ready_partials_and_authority_fences(bridge_factory,tmp_path,state):
    import time
    bridge,requests,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    w=workspace(tmp_path);bridge.owner.task_workspace=w
    bridge.control=RequestTaskControl(w,capability_version='synthetic')
    url='https://x.com/synthetic/status/900000000000000001'
    bridge.control.observe('search_web',{},ToolOutcome({'sources':[{'source':url}]}),'search')
    if state!='unreviewed':bridge.control.describe({'media_selection':{'candidate_reviews':[
        {'ref_id':url,'decision':'selected','reason':'原帖正文标明旅行主题'}]}})
    bridge.control.observe('media_result',{},ToolOutcome({'artifact_ids':['art-a'],
        'media':[{'artifact_id':'art-a','kind':'photo','source_ref':url,'logical_media_id':'source-asset-1'}],
        'requested_count':3,'all_requested_downloaded':False,'failures':[{'index':2,'error':'synthetic'},{'index':3,'error':'synthetic'}]}),'ready')
    sends=[]
    async def send(args):sends.append(args);return ToolOutcome({},'sent','receipt')
    bridge.tools={'send_private':tool('send_private',send,effect='write',properties={
        'text':{'type':'string'},'artifact_ids':{'type':'array','items':{'type':'string'}},
        'as_forward':{'type':'boolean'},'task_status':{'type':'string'}})}
    bridge.owner.max_steps=1;bridge.model_calls=1;bridge.stop_reason='model_budget_exhausted'
    if state=='revoked':bridge.guard.require_valid=lambda:(_ for _ in ()).throw(RuntimeError('revoked'))
    if state=='uncertain':bridge.control.unknown=True
    if state=='cancelled':bridge.cancelled.set()
    if state=='expired':bridge.finalization_deadline=time.monotonic()-1
    await bridge.finalize_media()
    assert len(sends)==(1 if state in {'ready','unreviewed'} else 0) and not requests
    if sends:
        assert sends[0]['artifact_ids']==['art-a'] and sends[0]['task_status']=='partial'
        assert bridge.control.stage=='partial' and bridge.control.minimum_media_count==2
        if state=='unreviewed':assert '尚未逐项核实' in sends[0]['text']


def test_reviews_and_compaction_are_each_one_admission_without_losing_prior_calls(bridge_factory,tmp_path):
    bridge,_,_=bridge_factory(lambda *_:responses_sse())
    bridge.owner.task_workspace=workspace(tmp_path)
    bridge.admit_model_budget(100)  # Before the model identifies a media goal.
    bridge.owner.task_workspace.media_budget=RequestBudget(bridge.owner.task_workspace)
    bridge.admit_model_budget(200)  # Current main call plus the earlier one.
    bridge.model_calls=2
    bridge.admit_model_budget(300)  # Compaction does not change local main count.
    bridge.admit_model_budget(400)  # One review or subsequent main admission.
    state=bridge.owner.task_workspace.media_budget.snapshot()
    assert state['used']['model_calls']==4 and state['used']['estimated_tokens']==1000
