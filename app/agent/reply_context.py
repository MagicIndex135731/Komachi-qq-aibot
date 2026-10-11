"""Reply context scoped to one approved Agent turn, including worker threads."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from app.agent.contracts import AgentBatch, AgentDecision


@dataclass(frozen=True, slots=True)
class AgentReplyContext:
    batch: AgentBatch
    decision: AgentDecision


current_agent_reply: ContextVar[AgentReplyContext | None] = ContextVar("agent_reply", default=None)


@contextmanager
def agent_reply_context(batch: AgentBatch, decision: AgentDecision) -> Iterator[None]:
    token = current_agent_reply.set(AgentReplyContext(batch, decision))
    try:
        yield
    finally:
        current_agent_reply.reset(token)
