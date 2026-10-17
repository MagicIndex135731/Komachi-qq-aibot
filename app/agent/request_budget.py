"""Persistent media usage shared by foreground and queued workers."""
from __future__ import annotations

import time
import re
from contextlib import contextmanager
from contextvars import ContextVar

from app.agent.research_tasks import serialized_checkpoint

_partial_sources = ContextVar('partial_request_sources', default=None)


@contextmanager
def source_receipt_scope(workspace, urls):
    """Only the local finalizer may attach these observed source receipts."""
    token=_partial_sources.set((workspace.scope,workspace.source_id,frozenset(urls)))
    try:yield
    finally:_partial_sources.reset(token)


def source_receipt_urls_allowed(scope, source_id, text):
    value=_partial_sources.get()
    urls=set(re.findall(r'https?://\S+',text))
    return bool(value and value[:2]==(scope,source_id) and urls and urls<=value[2])


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

    @property
    def unmetered(self):return self.snapshot().get('policy') == 'unmetered'

    @serialized_checkpoint
    def _initialize(self,create,limits):
        old=self.workspace._load(self.path)
        if old:
            self.workspace._check(old)
            if old.get('task_id')!=self.workspace.task_id:raise ValueError('request_budget_identity_mismatch')
            # Older production checkpoints carried an implicit ten-minute
            # quota. Preserve usage and authority, adopt the current policy.
            if 'policy' not in old:
                old.update(policy='unmetered',expires_at=None,
                    limits={key:None for key in self.LIMITS})
                self.workspace._write(self.path,old)
            self._unmetered=old.get('policy')=='unmetered'
            return
        if not create:raise ValueError('request_budget_unavailable')
        now=self.workspace.clock()
        state={'scope':self.workspace.scope,'actor_id':self.workspace.actor_id,'source_id':self.workspace.source_id,
            'task_id':self.workspace.task_id,'created_at':now,'updated_at':now,
            'policy':'unmetered' if limits is None else 'finite',
            'expires_at':None if limits is None else now+600,
            'limits':{key:None for key in self.LIMITS} if limits is None else {**self.LIMITS,**limits},'used':{}}
        if self.workspace.dependency_snapshot:state['dependencies']=self.workspace.dependency_snapshot()
        self._unmetered=state['policy']=='unmetered'
        self.workspace._write(self.path,state)

    @serialized_checkpoint
    def transact(self,key=None,amount=0,*,finalization=False,charge=False):
        state=self.workspace._load(self.path)
        if not state:raise ValueError('request_budget_unavailable')
        self.workspace._check(state)
        if state.get('source_id')!=self.workspace.source_id or state.get('task_id')!=self.workspace.task_id:
            raise ValueError('request_budget_identity_mismatch')
        deadline=state['expires_at']
        if deadline is not None and self.workspace.clock()>=deadline-(0 if finalization else 30):
            raise ValueError('request_budget_deadline')
        if key:
            if type(amount) is not int or amount<0 or key not in state['limits']:raise ValueError('invalid_request_budget_charge')
            used=state['used'].get(key,0)+amount
            cap=state['limits'][key]
            if cap is not None:cap-=1 if key=='writes' and not finalization else 0
            if cap is not None and used>cap and not charge:raise ValueError('request_budget_'+key)
            state['used'][key]=used;state['updated_at']=self.workspace.clock()
            self.workspace._write(self.path,state)
            if cap is not None and used>cap:raise ValueError('request_budget_'+key)
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
        return min(self.remaining_for(key,state) for key in ('media_received','network_reserved_bytes'))

    def remaining_for(self,key,state=None):
        state=state or self.snapshot()
        cap=state['limits'][key]
        return float('inf') if cap is None else max(0,cap-state['used'].get(key,0))

    @property
    def deadline(self):
        state=self.snapshot()
        if state['expires_at'] is None:return float('inf')
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
        if state['expires_at'] is not None and self.workspace.clock()>=state['expires_at']-30:
            raise ValueError('request_budget_deadline')
        for key in ('network_reserved_bytes','media_received'):
            if state['limits'][key] is not None and state['used'][key]>state['limits'][key]:
                raise ValueError('request_budget_'+key)

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
    budget=RequestBudget(workspace,create=False)
    workspace.media_budget=budget
    from app.agent.media_request_policy import bind_request_workspace
    bind_request_workspace(guard,workspace)
    return budget
