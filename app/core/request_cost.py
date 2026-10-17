"""Request-local cost context, copied into bounded worker threads."""
from contextvars import ContextVar
from contextlib import contextmanager

current_request_budget=ContextVar('request_cost_budget',default=None)

@contextmanager
def request_cost_scope(budget):
    token=current_request_budget.set(budget)
    try:yield
    finally:current_request_budget.reset(token)
