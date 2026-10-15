"""Fence each external tool effect without making any agent decisions."""
from __future__ import annotations

from contextvars import ContextVar
from hashlib import sha256
import json

from app.agent.tool_agent import ToolOutcome
from app.storage.agent_effects import AgentToolEffectRepository
from app.adapters.qzone import QzoneRejectedError, QzoneUnavailableError
from app.adapters.account_profile import ProfileRejectedError, ProfileUnavailableError


class ToolDispatchBoundary:
    def __init__(self, *, engine, account_id, scope, guard, clock, max_daily=2, cooldown_seconds=60,
                 interactive_group_send: bool = False, group_send_allowed=None, owner_authority=None,
                 group_interjection_reserve=None):
        if type(interactive_group_send) is not bool:
            raise ValueError("interactive group send authority must be a boolean")
        self.repository = AgentToolEffectRepository(engine, account_id=account_id, clock=clock)
        self.scope, self.parent, self.clock = scope, guard, clock
        self.max_daily, self.cooldown_seconds = max_daily, cooldown_seconds
        self.interactive_group_send = interactive_group_send
        self.group_send_allowed = group_send_allowed
        self.group_interjection_reserve = group_interjection_reserve
        self.owner_authority = owner_authority
        self.current = ContextVar("agent_tool_effect", default=None)
        self.receipts = []
        self.dispatched = False
        self.uncertain = False

    async def invoke(self, tool, arguments, call_id):
        self.require_valid()
        if tool.authorization is not None and tool.authorization() is not True:
            return ToolOutcome({'error': 'owner_required'}, 'blocked')
        if tool.effect != "write":
            return await tool.handler(arguments)
        digest = sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        # Repeated payloads within one durable wake are the same requested
        # effect even if a model invents a different call id after a timeout.
        run_id, identity = self.parent.action_id, tool.name + ":" + digest
        effect_scope = tool.scope or self.scope
        if tool.per_target_daily_limit:
            target = arguments.get("owner_id")
            if type(target) is not int or not 0 < target <= 2**63 - 1:
                return ToolOutcome({"error": "invalid_budget_target"}, "blocked")
            effect_scope += ":friend:" + str(target)
        if self.owner_authority is not None:
            effect_scope = "owner:" + effect_scope.removeprefix("owner:")
        if (self.owner_authority is None and tool.name == "comment_qzone"
                and "owner_id" in arguments and "post_id" in arguments):
            run_id = "post:" + sha256((str(arguments["owner_id"]) + ":" + arguments["post_id"]).encode()).hexdigest()
            identity = "comment_qzone"
        try:
            record = self.repository.reserve(scope=effect_scope, run_id=run_id,
                tool_call_id=identity, operation=tool.name, payload_hash=digest,
                sources=getattr(self.parent, "source_message_ids", ()))
        except ValueError:
            return ToolOutcome({"error": "effect_identity_conflict", "replay_allowed": False}, "blocked")
        if not record["fresh"]:
            return ToolOutcome({"error": "effect_already_recorded", "operation_id": record["action_id"],
                "previous_status": record["status"], "replay_allowed": False},
                "uncertain" if record["status"] in {"dispatching", "uncertain"} else "blocked")
        token = self.current.set((record["action_id"], tool))
        try:
            try:
                outcome = await tool.handler(arguments)
            except BaseException as exc:
                known = isinstance(exc, (QzoneRejectedError, QzoneUnavailableError, ProfileRejectedError, ProfileUnavailableError))
                current_status = self.repository.get(record["action_id"])["status"]
                state = "blocked" if current_status == "blocked" else "uncertain" if current_status == "dispatching" and not known else "failed"
                self.repository.finish(record["action_id"], status=state, reason="tool_interrupted")
                self.uncertain = state == "uncertain"
                self.receipts.append({"call_id": call_id, "action_id": record["action_id"], "tool": tool.name,
                    "status": state, "receipt_id": None, "replay_allowed": False})
                if isinstance(exc, Exception):
                    return ToolOutcome({"error": type(exc).__name__}, state)
                raise
            recorded = self.repository.get(record["action_id"])
            status = outcome.status
            if status == "sent" and recorded["status"] != "dispatching":
                status = "failed"
                outcome = ToolOutcome({"error": "effect_bypassed_dispatch_boundary"}, status)
            self.repository.finish(record["action_id"], status=status if status in {"sent", "uncertain", "failed", "blocked"} else "failed",
                receipt=outcome.receipt_id, reason="tool:" + tool.name)
            self.receipts.append({"call_id": call_id, "action_id": record["action_id"], "tool": tool.name,
                "status": status, "receipt_id": outcome.receipt_id, "replay_allowed": False})
            self.uncertain = status == "uncertain"
            return outcome
        finally:
            self.current.reset(token)

    @property
    def action_id(self):
        current = self.current.get()
        return current[0] if current else self.parent.action_id

    def valid(self):
        return (not self.uncertain and self.parent.valid()
                and (self.owner_authority is None or self.owner_authority() is True))

    def require_valid(self):
        if not self.valid():
            raise RuntimeError("tool dispatch authority revoked")
        self.parent.require_valid()

    def on_dispatch(self):
        self.require_valid()
        current = self.current.get()
        # Only canonical direct ingress grants this exception. It is not a
        # model argument, and all other writes retain their social limits.
        direct_reply = self.interactive_group_send and current is not None and current[1].name == "send_group"
        chat_send = current is not None and current[1].name == "send_group" and self.group_send_allowed is not None
        manual = self.owner_authority is not None
        if chat_send and not direct_reply and not manual and not self.group_send_allowed():
            raise RuntimeError("group chat cadence or policy revoked")
        if (chat_send and not direct_reply and not manual and self.group_interjection_reserve is not None
                and not self.group_interjection_reserve()):
            raise RuntimeError('proactive_speech_budget')
        daily_limit = None if direct_reply or manual else (
            current[1].daily_limit if current is not None and current[1].daily_limit is not None else self.max_daily)
        cooldown = 0 if direct_reply or chat_send or manual or (current and current[1].per_target_daily_limit) else (
            current[1].cooldown_seconds if current is not None and current[1].cooldown_seconds is not None else self.cooldown_seconds)
        if current is None or not self.repository.mark_dispatching(current[0],
                daily_limit=daily_limit, cooldown_seconds=cooldown,
                operation_daily_limit=None if manual else current[1].operation_daily_limit,
                shared_cooldown_seconds=0 if manual else current[1].shared_cooldown_seconds,
                wake_id=self.parent.action_id,
                max_writes_per_wake=None if manual else current[1].max_writes_per_wake,
                before_commit=self.require_valid):
            raise RuntimeError("tool effect has no durable authority")
        self.dispatched = True
        # Scheduled wake is an envelope, with each tool effect in its own
        # ledger. Interactive ingress also keeps its existing source fence.
        parent = getattr(self.parent, "dispatch_parent", self.parent)
        if hasattr(parent, "record_tool_dispatch"):
            parent.record_tool_dispatch(current[0])
        elif not getattr(parent, "dispatched", False):
            self.parent.on_dispatch()
