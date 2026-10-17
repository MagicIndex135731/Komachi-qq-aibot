"""One finite media-request budget shared by foreground and queued workers."""
from __future__ import annotations

import time

from app.agent.research_tasks import serialized_checkpoint


class RequestBudget:
    LIMITS={'model_calls':24,'estimated_tokens':256000,'tool_calls':64,'discovery_calls':8,
        'media_received':384*1024*1024,'network_reserved_bytes':768*1024*1024,'transfer_requests':96,'writes':8}

    def __init__(self,workspace,*,create=True,limits=None):
        self.workspace=workspace
        self.path=workspace._file('media-budget.json')
        self._initialize(create,limits)

    @property
    def root(self):return self.workspace.root

    @property
    def formal(self):return True

    @serialized_checkpoint
    def _initialize(self,create,limits):
        old=self.workspace._load(self.path)
        if old:
            self.workspace._check(old)
            if old.get('task_id')!=self.workspace.task_id:raise ValueError('request_budget_identity_mismatch')
            return
        if not create:raise ValueError('request_budget_unavailable')
        now=self.workspace.clock()
        state={'scope':self.workspace.scope,'actor_id':self.workspace.actor_id,'source_id':self.workspace.source_id,
            'task_id':self.workspace.task_id,'created_at':now,'updated_at':now,'expires_at':now+600,
            'limits':{**self.LIMITS,**(limits or {})},'used':{}}
        if self.workspace.dependency_snapshot:state['dependencies']=self.workspace.dependency_snapshot()
        self.workspace._write(self.path,state)

    @serialized_checkpoint
    def transact(self,key=None,amount=0,*,finalization=False,charge=False):
        state=self.workspace._load(self.path)
        if not state:raise ValueError('request_budget_unavailable')
        self.workspace._check(state)
        if state.get('source_id')!=self.workspace.source_id or state.get('task_id')!=self.workspace.task_id:
            raise ValueError('request_budget_identity_mismatch')
        deadline=state['expires_at']-(0 if finalization else 30)
        if self.workspace.clock()>=deadline:raise ValueError('request_budget_deadline')
        if key:
            if type(amount) is not int or amount<0 or key not in state['limits']:raise ValueError('invalid_request_budget_charge')
            used=state['used'].get(key,0)+amount
            cap=state['limits'][key]-(1 if key=='writes' and not finalization else 0)
            if used>cap and not charge:raise ValueError('request_budget_'+key)
            state['used'][key]=used;state['updated_at']=self.workspace.clock()
            self.workspace._write(self.path,state)
            if used>cap:raise ValueError('request_budget_'+key)
        return state

    def spend(self,key,amount=1,*,finalization=False):return self.transact(key,amount,finalization=finalization)

    @serialized_checkpoint
    def record_usage(self,usage):
        state=self.workspace._load(self.path)
        if not state:return
        self.workspace._check(state)
        for name in ('input_tokens','cached_input_tokens','output_tokens'):
            amount=getattr(usage,name,None)
            if type(amount) is int:state['used']['actual_'+name]=state['used'].get('actual_'+name,0)+amount
        state['updated_at']=self.workspace.clock()
        self.workspace._write(self.path,state)

    def snapshot(self):return self.transact(finalization=True)

    @property
    def received(self):return self.snapshot()['used'].get('media_received',0)

    @property
    def remaining(self):
        state=self.snapshot()
        return max(0,min(state['limits']['media_received']-state['used'].get('media_received',0),
            state['limits']['network_reserved_bytes']-state['used'].get('network_reserved_bytes',0)))

    @property
    def deadline(self):
        state=self.snapshot()
        return time.monotonic()+max(0,state['expires_at']-30-self.workspace.clock())

    def check(self):
        self.transact()
        if not self.remaining:raise ValueError('request_budget_media_received')

    def request(self):self.spend('transfer_requests')

    def reserve_network(self,size):self.spend('network_reserved_bytes',size)

    @serialized_checkpoint
    def charge(self,size):
        if type(size) is not int or size<0:raise ValueError('invalid_request_budget_charge')
        state=self.workspace._load(self.path)
        if not state:raise ValueError('request_budget_unavailable')
        self.workspace._check(state)
        if state.get('source_id')!=self.workspace.source_id or state.get('task_id')!=self.workspace.task_id:
            raise ValueError('request_budget_identity_mismatch')
        for key in ('network_reserved_bytes','media_received'):
            state['used'][key]=state['used'].get(key,0)+size
        state['updated_at']=self.workspace.clock()
        self.workspace._write(self.path,state)
        # Account for the chunk already received, including a boundary overrun.
        if self.workspace.clock()>=state['expires_at']-30:raise ValueError('request_budget_deadline')
        for key in ('network_reserved_bytes','media_received'):
            if state['used'][key]>state['limits'][key]:raise ValueError('request_budget_'+key)

    def reference(self):
        return {'task_id':self.workspace.task_id,'scope':self.workspace.scope,
            'actor_id':self.workspace.actor_id,'source_id':self.workspace.source_id}


def worker_budget(payload,guard,settings):
    reference=payload.get('request_budget')
    if not reference:return None  # Legacy jobs keep their existing finite limits.
    if any(reference.get(k)!=payload.get(k) for k in ('scope','actor_id','source_id')):
        raise ValueError('request_budget_out_of_scope')
    from app.agent.research_tasks import ResearchTaskWorkspace
    workspace=ResearchTaskWorkspace(settings.data_dir/'research_tasks',scope=payload['scope'],
        actor_id=payload['actor_id'],source_id=payload['source_id'],request=payload.get('goal',''),
        valid_source=lambda i:guard.valid_source(i,author=i==payload['source_id']),
        valid_dependencies=guard.valid_dependencies)
    if reference.get('task_id')!=workspace.task_id:raise ValueError('request_budget_out_of_scope')
    return RequestBudget(workspace,create=False)
