import asyncio
import json
import pytest

from app.agent.tool_agent import AgentTool, AutonomousToolAgent, ToolOutcome


class Guard:
    revoked = False
    def require_valid(self):
        if self.revoked:
            raise RuntimeError("revoked")


def test_argument_feedback_explains_array_limit_without_echoing_values():
    from app.agent.tool_agent import argument_validation_feedback
    schema={'type':'object','properties':{'aliases':{'type':'array','minItems':1,'maxItems':3,
        'items':{'type':'string'}}},'required':['aliases']}
    feedback=argument_validation_feedback({'aliases':['PRIVATE_VALUE']*4,'PRIVATE_FIELD':'SECRET'},schema)
    assert feedback['invalid_arguments']==[{'parameter':'aliases','expected':{'type':'array','minItems':1,'maxItems':3}}]
    assert feedback['unknown_parameter_count']==1 and 'PRIVATE' not in json.dumps(feedback)


class Model:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.prompts = []
    def generate_text(self, lines, **kwargs):
        self.prompts.append(lines)
        return json.dumps(next(self.calls))


SCHEMA = {"type": "object", "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 50}}, "required": ["text"]}


@pytest.mark.asyncio
async def test_tool_result_changes_next_decision_and_one_write_ends_turn():
    effects = []
    async def image(args):
        effects.append("image")
        return ToolOutcome({"artifact_id": "generated-1"})
    async def send(args):
        effects.append("write")
        return ToolOutcome({}, "sent", "real-receipt")
    model = Model([{"tool": "draw", "arguments": {"text": "cats"}}, {"tool": "send", "arguments": {"text": "look"}}, {"tool": "send", "arguments": {"text": "again"}}])
    agent = AutonomousToolAgent(decision_client=model)
    try:
        result = await agent.run([], (AgentTool("draw", "draw", SCHEMA, image, "image"), AgentTool("send", "send", SCHEMA, send, "write")), Guard())
        assert result.status == "sent" and result.receipt_id == "real-receipt"
        assert effects == ["image", "write"] and len(model.prompts) == 2
        assert "generated-1" in "\n".join(model.prompts[1])
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_unknown_tool_and_cross_target_argument_never_execute():
    effects = []
    async def send(args):
        effects.append(args)
        return ToolOutcome({}, "sent")
    model = Model([{"tool": "shell", "arguments": {}}, {"tool": "send", "arguments": {"text": "x", "group_id": 900000002}}, {"finish": True}])
    agent = AutonomousToolAgent(decision_client=model)
    try:
        result = await agent.run([], (AgentTool("send", "send", SCHEMA, send, "write"),), Guard())
        assert result.status == "silent" and effects == []
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_uncertain_and_failed_write_are_terminal():
    for status in ("uncertain", "failed"):
        async def send(args):
            return ToolOutcome({"error": "gateway"}, status)
        model = Model([{"tool": "send", "arguments": {"text": "x"}}, {"tool": "send", "arguments": {"text": "retry"}}])
        agent = AutonomousToolAgent(decision_client=model)
        try:
            result = await agent.run([], (AgentTool("send", "send", SCHEMA, send, "write"),), Guard())
            assert result.status == status and len(model.prompts) == 1
        finally:
            agent.close()


@pytest.mark.asyncio
async def test_image_failure_feedback_and_budget_do_not_fake_success():
    effects = []
    async def image(args):
        effects.append(args)
        raise ValueError("synthetic")
    model = Model([{"tool": "draw", "arguments": {"text": "x"}}, {"tool": "draw", "arguments": {"text": "x"}}, {"finish": True}])
    agent = AutonomousToolAgent(decision_client=model)
    try:
        result = await agent.run([], (AgentTool("draw", "draw", SCHEMA, image, "image"),), Guard())
        assert result.status == "silent" and len(effects) == 1
        assert "failed" in "\n".join(model.prompts[1]) and "image_budget_exhausted" in "\n".join(model.prompts[2])
    finally:
        agent.close()
