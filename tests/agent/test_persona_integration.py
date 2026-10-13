from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.agent.persona_integration import (
    MAX_PERSONA_PROFILE_CHARS,
    build_group_persona_context,
    build_group_persona_fallback,
)
from app.core.persona_engine import render_persona
from app.adapters.onebot_models import parse_group_message_event
from app.adapters.sender import Sender
from app.agent.contracts import AgentActionResult, AgentBatch
from app.agent.participation import GroupParticipationPlanner
from app.agent.service_tools import ServiceToolContext, build_service_tools
from app.agent.social_contracts import SocialJob, SocialObservation
from app.core.persona_switch import PersonaManager, PersonaSwitchService
from app.core.router import InboundRouter
from app.runtime.blocking import BoundedBlockingRunner
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


GROUP, OTHER_GROUP, ACCOUNT, AUTHOR, MEMBER = 900000001, 900000002, 900000099, 900000101, 900000102
NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)


def seed(engine, message_id, text, *, group_id=GROUP, user_id=AUTHOR, when=NOW,
         reply_to=None, mentioned=False, raw=None):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=user_id, nickname="synthetic member", group_card="")
        row = MessageRepository(session).add_group_message(
            platform_msg_id=message_id, group_id=group_id, user_id=user_id, timestamp=when,
            plain_text=text, msg_type="text", raw_json=raw or {}, reply_to_msg_id=reply_to,
            mentioned_bot=mentioned,
        )
        session.flush()
        return row.id


class Manager:
    def __init__(self, profile, *, key="member"):
        self.profile, self.key = profile, key
        self.queries, self.entries = [], []

    def active_key(self, group_id):
        return self.key

    def active_persona(self, group_id):
        return self.profile

    def is_impersonating(self, group_id):
        return self.key != "default"

    def live_persona(self, group_id):
        return {**self.profile, "relationships": [{"member": "当前群名片", "relation": "群友", "address_terms": ["当前群名片"]}]}

    def retrieve_examples(self, group_id, query, **kwargs):
        self.queries.append((group_id, query, kwargs))
        return self.entries


def router_for(engine, *, key="member", **fields):
    profile = {"name": "合成模仿者", "source_user_id": MEMBER, "source_group_id": GROUP,
               "address_rules": ["直接叫名字"], "speech_habits": ["短句"], **fields}
    manager = Manager(profile, key=key)
    router = SimpleNamespace(engine=engine, persona_manager=manager,
                             runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=ACCOUNT), persona={"name": "小町"}))
    router._active_persona = manager.active_persona
    router._impersonating = manager.is_impersonating
    router._persona_text_for = lambda persona, group_id: InboundRouter._persona_text_for(router, persona, group_id)
    router._bot_addressing_instruction = lambda **kwargs: InboundRouter._bot_addressing_instruction(router, **kwargs)
    return router


def test_persona_reuses_live_profile_focused_query_and_canonical_style_sources(sqlite_engine):
    target = seed(sqlite_engine, "target", "这部动画怎么样", mentioned=True, reply_to="quote")
    quote = seed(sqlite_engine, "quote", "动画真好看", when=NOW - timedelta(seconds=30))
    broad = seed(sqlite_engine, "broad", "无关彩票股票", when=NOW - timedelta(seconds=20))
    sample = seed(sqlite_engine, "sample", "还行吧", user_id=MEMBER, when=NOW - timedelta(days=1))
    router = router_for(sqlite_engine)
    router.persona_manager.entries = [{"msg_id": "sample", "user_id": MEMBER, "group_id": GROUP,
                                      "text": "还行吧", "reply_target": "动画怎么样"}]
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=(target, quote, broad),
                                        target_message_id="target", addressed=True, now=NOW)
    query = router.persona_manager.queries[0][1]
    assert query.current_text == "这部动画怎么样"
    assert query.quoted_text == "动画真好看"
    assert "彩票" not in query.semantic_text
    assert "当前群名片" in result.text
    assert "不是事实来源" in result.text and "还行吧" in result.text
    assert "call name" in result.text
    assert result.source_message_ids == (sample,)
    assert result.impersonated_subject_id == MEMBER
    assert result.impersonated_subject_addressed is True
    assert result.persona_key == "member"


def test_default_persona_does_not_use_member_style_bank(sqlite_engine):
    target = seed(sqlite_engine, "target", "你好")
    router = router_for(sqlite_engine, key="default")
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=(target,),
                                        target_message_id="target", addressed=True)
    assert router.persona_manager.queries == []
    assert result.impersonated_subject_id is None
    assert result.source_message_ids == ()


def test_default_voice_and_imitation_use_existing_shared_expression_rules(sqlite_engine):
    router = router_for(sqlite_engine, key="default", chat_voice="clingy")
    clingy = build_group_persona_context(router, group_id=GROUP, source_message_ids=())
    assert "clingy little-sister warmth" in clingy.text
    router.persona_manager.profile["chat_voice"] = "komachi"
    sharp = build_group_persona_context(router, group_id=GROUP, source_message_ids=())
    assert "mesugaki edge" in sharp.text
    assert sharp.profile_fingerprint != clingy.profile_fingerprint
    router.persona_manager.key = "member"
    imitation = build_group_persona_context(router, group_id=GROUP, source_message_ids=())
    assert "mesugaki edge" not in imitation.text
    assert "Talk like a real person chatting in a group" in imitation.text


@pytest.mark.parametrize("source_group", [OTHER_GROUP, None])
def test_foreign_or_unscoped_member_persona_transfers_only_generic_identity_and_style(sqlite_engine, source_group):
    target = seed(sqlite_engine, "target", "聊聊动画")
    router = router_for(sqlite_engine, source_group_id=source_group, identity="通用扮演身份",
                        speech_habits=["通用短句"], background="来源群个人经历",
                        self_concept="来源群私人自述", external_relations=[{"name": "来源群关系人", "relation": "私人关系"}],
                        facts=[{"fact": "来源群独有事实"}], address_rules=["来源群独有称呼"],
                        example_lines=["来源群真人固定原话"])
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=(target,), target_message_id="target")
    assert "通用扮演身份" in result.text and "通用短句" in result.text
    for foreign in ("来源群个人经历", "来源群私人自述", "来源群关系人", "来源群独有事实", "来源群独有称呼", "来源群真人固定原话", "当前群名片"):
        assert foreign not in result.text
    assert result.impersonated_subject_id is None
    assert result.source_message_ids == () and result.style_text == ""
    assert router.persona_manager.queries == []


def test_foreign_persona_fallback_cannot_restore_relationship_data_or_touch_failed_live_service(sqlite_engine):
    router = router_for(sqlite_engine, source_group_id=OTHER_GROUP, identity="通用扮演身份",
                        relationships=[{"member": "外群成员", "relation": "私人关系"}],
                        address_rules=["外群称呼"], background="外群经历")
    router._persona_text_for = lambda *args: pytest.fail("fallback must not rerun live rendering")
    result = build_group_persona_fallback(router, group_id=GROUP)
    assert "通用扮演身份" in result.text
    assert "外群成员" not in result.text and "外群称呼" not in result.text and "外群经历" not in result.text
    assert result.impersonated_subject_id is None and result.source_message_ids == ()


@pytest.mark.parametrize("invalid", ["other_group", "recalled", "outbound"])
def test_unusable_target_is_rejected_before_querying_persona(sqlite_engine, invalid):
    raw = {"delivery_state": "deleted"} if invalid == "recalled" else {"direction": "outbound"} if invalid == "outbound" else {}
    target = seed(sqlite_engine, "target", "这部动画怎么样", group_id=OTHER_GROUP if invalid == "other_group" else GROUP,
                  user_id=ACCOUNT if invalid == "outbound" else AUTHOR, raw=raw)
    router = router_for(sqlite_engine)
    with pytest.raises(ValueError, match="persona_target_unavailable"):
        build_group_persona_context(router, group_id=GROUP, source_message_ids=(target,), target_message_id="target")
    assert router.persona_manager.queries == []


def test_recalled_or_cross_group_quote_is_not_added(sqlite_engine):
    target = seed(sqlite_engine, "target", "你看了吗", reply_to="quote")
    seed(sqlite_engine, "quote", "机密内容", group_id=OTHER_GROUP)
    router = router_for(sqlite_engine)
    build_group_persona_context(router, group_id=GROUP, source_message_ids=(target,), target_message_id="target")
    assert router.persona_manager.queries[0][1].quoted_text is None


def test_short_followup_uses_preceding_context_without_later_batch_messages(sqlite_engine):
    previous = seed(sqlite_engine, "before", "聊聊动画", when=NOW - timedelta(seconds=30))
    target = seed(sqlite_engine, "target", "然后呢")
    later = seed(sqlite_engine, "after", "后来的无关彩票", when=NOW + timedelta(seconds=1))
    router = router_for(sqlite_engine)
    build_group_persona_context(router, group_id=GROUP, source_message_ids=(previous, target, later), target_message_id="target")
    query = router.persona_manager.queries[0][1]
    assert "聊聊动画" in query.semantic_text
    assert "彩票" not in query.semantic_text


def test_style_timeout_falls_back_to_live_identity_without_fabricating_examples(sqlite_engine):
    target = seed(sqlite_engine, "target", "聊聊动画")
    router = router_for(sqlite_engine)
    def timeout(*args, **kwargs):
        raise TimeoutError()
    router.persona_manager.retrieve_examples = timeout
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=(target,), target_message_id="target")
    assert "当前群名片" in result.text
    assert result.style_text == "" and result.source_message_ids == ()


def test_profile_fingerprint_changes_for_refresh_but_not_dynamic_query(sqlite_engine):
    first = seed(sqlite_engine, "first", "聊聊动画")
    second = seed(sqlite_engine, "second", "聊聊音乐")
    router = router_for(sqlite_engine)
    one = build_group_persona_context(router, group_id=GROUP, source_message_ids=(first,), target_message_id="first")
    two = build_group_persona_context(router, group_id=GROUP, source_message_ids=(second,), target_message_id="second")
    assert one.profile_fingerprint == two.profile_fingerprint
    router.persona_manager.profile["speech_habits"] = ["更新的用词"]
    refreshed = build_group_persona_context(router, group_id=GROUP, source_message_ids=(second,), target_message_id="second")
    assert refreshed.profile_fingerprint != one.profile_fingerprint


def test_large_profile_keeps_addressing_and_imitation_contract_bounded(sqlite_engine):
    router = router_for(sqlite_engine, background="无关背景" * 8000)
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=())
    assert len(result.profile_text) <= MAX_PERSONA_PROFILE_CHARS
    assert "直接叫名字" in result.profile_text
    assert "当前群名片" in result.profile_text
    assert "对话中对方说的'我'指提问者本人" in result.profile_text
    assert "不要照抄整句" in result.profile_text


def test_wrong_author_recalled_and_cross_group_samples_are_not_used(sqlite_engine):
    target = seed(sqlite_engine, "target", "聊聊动画")
    seed(sqlite_engine, "wrong-author", "错误作者语气", user_id=AUTHOR)
    seed(sqlite_engine, "recalled", "撤回语气", user_id=MEMBER, raw={"delivery_state": "deleted"})
    seed(sqlite_engine, "cross", "其他群语气", user_id=MEMBER, group_id=OTHER_GROUP)
    router = router_for(sqlite_engine)
    router.persona_manager.entries = [{"msg_id": message_id, "text": text, "user_id": MEMBER, "group_id": GROUP}
                                      for message_id, text in [("wrong-author", "错误作者语气"), ("recalled", "撤回语气"), ("cross", "其他群语气")]]
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=(target,), target_message_id="target")
    assert result.source_message_ids == ()
    assert "错误作者语气" not in result.text and "撤回语气" not in result.text and "其他群语气" not in result.text


def test_missing_manager_uses_existing_default_persona_without_new_requirements(sqlite_engine):
    router = SimpleNamespace(engine=sqlite_engine, persona_manager=None,
                             runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=ACCOUNT)),
                             _active_persona=lambda group_id: {"name": "小町", "identity": "聊天"})
    result = build_group_persona_context(router, group_id=GROUP, source_message_ids=())
    assert render_persona({"name": "小町", "identity": "聊天"}) in result.text
    assert result.persona_key == "default"


class Guard:
    turn_id = "synthetic-persona-turn"
    action_id = "synthetic-persona-action"
    allowed = True
    dispatched = False

    def valid(self):
        return self.allowed

    def require_valid(self):
        if not self.allowed:
            raise RuntimeError("revoked")

    def on_dispatch(self):
        self.require_valid()
        self.dispatched = True


async def test_read_persona_tool_reuses_live_context_and_retains_style_sources(sqlite_engine):
    target = seed(sqlite_engine, "target", "聊聊动画", mentioned=True)
    sample = seed(sqlite_engine, "sample", "还行吧", user_id=MEMBER)
    router = router_for(sqlite_engine)
    router.runtime.settings.memory_memory_tools_enabled = False
    router.runtime.social_policy = {"services": {}}
    router.persona_manager.entries = [{"msg_id": "sample", "text": "还行吧", "user_id": MEMBER, "group_id": GROUP}]
    job = SocialJob(f"group:{GROUP}", "group", GROUP)
    observation = SocialObservation(job, NOW, "stale persona", source_message_ids=(target,))
    runner = BoundedBlockingRunner(1)
    context = ServiceToolContext(router, observation, Guard(), {target}, runner,
                                 SimpleNamespace(scope=job.key), target_message_id="target")
    try:
        tools = {tool.name: tool for tool in build_service_tools(context)}
        result = await tools["read_persona"].handler({})
    finally:
        runner.close()
    assert "当前群名片" in result.data["expression_style"]
    assert "还行吧" in result.data["expression_style"]
    assert sample in context.observed_sources
    assert result.data["kind"] == "persona_not_fact_evidence"


class NoModel:
    def generate_text(self, *args, **kwargs):
        raise AssertionError("persona command must not use a model")


class Gateway:
    def __init__(self, *, uncertain=False):
        self.calls, self.uncertain = [], uncertain

    async def call_api(self, action, params):
        self.calls.append((action, params))
        return {"status": "ok", "data": {} if self.uncertain else {"message_id": "synthetic-confirmation"}}


class Autonomous:
    runtime_mode = "harness"

    def __init__(self):
        self.calls = []

    def handles_group(self, group_id):
        return True

    async def execute_group_turn(self, event, batch, decision, guard):
        self.calls.append(event.platform_msg_id)
        return AgentActionResult("silent")


def switch_router(engine, tmp_path, *, gateway=None):
    gateway = gateway or Gateway()
    router = InboundRouter.build_for_test(sqlite_engine=engine, sender=Sender(gateway), llm_client=NoModel())
    router.runtime.settings.owner_qq = AUTHOR
    router.runtime.settings.bot_qq = ACCOUNT
    router.runtime.settings.data_dir = tmp_path / "isolated"
    router.runtime.group_policy["groups"][str(GROUP)] = {"enabled": True, "speak": True,
        "memory_enabled": False, "archive": False, "proactive_reply": False}
    default = {"name": "合成小町", "identity": "群聊助手"}
    personas = {"default": default, "member": {"name": "合成成员", "source_user_id": MEMBER,
                                                "source_group_id": GROUP, "identity": "群友"}}
    router.runtime.persona, router.runtime.personas = default, personas
    router.persona_manager = PersonaManager(engine=engine, personas=personas, default_persona=default)
    router.persona_switch_service = PersonaSwitchService(manager=router.persona_manager, sender=router.sender, bot_qq=ACCOUNT)
    return router, gateway


def onebot_command(message_id="owner-command", *, user_id=AUTHOR, mentioned=True, target="合成成员", text=None):
    message = [{"type": "at", "data": {"qq": str(ACCOUNT)}}] if mentioned else []
    message.append({"type": "text", "data": {"text": text or f"切换人格为：{target}"}})
    return parse_group_message_event({"message_id": message_id, "group_id": GROUP, "user_id": user_id,
                                     "sender": {"nickname": "合成主人"}, "time": int(datetime.now(UTC).timestamp()),
                                     "message": message}, bot_qq=ACCOUNT, bot_name="合成小町")


async def test_owner_switch_runs_before_harness_and_uses_guarded_confirmation(sqlite_engine, tmp_path):
    router, gateway = switch_router(sqlite_engine, tmp_path)
    command = onebot_command()
    router.ingest_live_group_message(command)
    autonomous = Autonomous()
    planner = GroupParticipationPlanner(router=router, decision_client=NoModel(), autonomous_executor=autonomous)
    batch = AgentBatch(GROUP, (command,), "switch-turn")
    guard = Guard()
    try:
        decision = await planner.decide(batch)
        result = await planner.execute(command, batch, decision, guard)
    finally:
        planner.close()
    assert decision.reason == "persona_switch"
    assert router.persona_manager.active_key(GROUP) == "member"
    assert router.persona_manager.active_key(OTHER_GROUP) == "default"
    assert autonomous.calls == []
    assert result.status == "sent" and guard.dispatched
    assert [action for action, _ in gateway.calls] == ["send_group_msg"]


async def test_owner_switches_follow_message_order_without_consuming_other_at(sqlite_engine, tmp_path):
    router, gateway = switch_router(sqlite_engine, tmp_path)
    events = (onebot_command("earlier-at", user_id=MEMBER, text="你好"),
              onebot_command("switch-one"), onebot_command("switch-two", target="合成小町"))
    for event in events:
        router.ingest_live_group_message(event)
    switched = []
    actual_switch = router.persona_switch_service.switch
    async def capture(**kwargs):
        switched.append(kwargs["target_key"])
        return await actual_switch(**kwargs)
    router.persona_switch_service.switch = capture
    planner = GroupParticipationPlanner(router=router, decision_client=NoModel(), autonomous_executor=Autonomous())
    batch = AgentBatch(GROUP, events, "switch-turn")
    try:
        decision = await planner.decide(batch)
        assert decision.target_message_id == 'earlier-at' and decision.reason == 'direct_trigger'
        first_switch = AgentBatch(GROUP, events[1:], 'first-switch-turn')
        decision = await planner.decide(first_switch)
        assert decision.target_message_id == 'switch-one' and decision.reason == 'persona_switch'
        await planner.execute(events[1], first_switch, decision, Guard())
        second_switch = AgentBatch(GROUP, events[2:], 'second-switch-turn')
        decision = await planner.decide(second_switch)
        assert decision.target_message_id == 'switch-two' and decision.reason == 'persona_switch'
        await planner.execute(events[2], second_switch, decision, Guard())
    finally:
        planner.close()
    assert switched == ['member', 'default']
    assert len(gateway.calls) == 2


@pytest.mark.parametrize("case", ["non_owner", "not_mentioned", "not_persisted", "recalled", "spoofed_author", "spoofed_text", "wrong_group"])
def test_persona_switch_requires_canonical_author_address_scope_and_text(sqlite_engine, tmp_path, case):
    router, _ = switch_router(sqlite_engine, tmp_path)
    command = onebot_command(user_id=MEMBER if case == "non_owner" else AUTHOR, mentioned=case != "not_mentioned")
    if case != "not_persisted":
        router.ingest_live_group_message(command)
    if case == "recalled":
        with session_scope(sqlite_engine) as session:
            row = MessageRepository(session).get_by_platform_msg_id(command.platform_msg_id)
            row.raw_json = {**row.raw_json, "delivery_state": "deleted"}
    elif case == "spoofed_author":
        command.user_id = MEMBER
    elif case == "spoofed_text":
        command.plain_text = "切换人格为：合成小町"
    elif case == "wrong_group":
        command.group_id = OTHER_GROUP
    assert router.persona_switch_target(command) is None
    assert router.persona_manager.active_key(GROUP) == "default"


async def test_non_owner_switch_text_still_reaches_normal_harness(sqlite_engine, tmp_path):
    router, gateway = switch_router(sqlite_engine, tmp_path)
    command = onebot_command(user_id=MEMBER)
    router.ingest_live_group_message(command)
    autonomous = Autonomous()
    planner = GroupParticipationPlanner(router=router, decision_client=NoModel(), autonomous_executor=autonomous)
    batch = AgentBatch(GROUP, (command,), "switch-turn")
    try:
        decision = await planner.decide(batch)
        await planner.execute(command, batch, decision, Guard())
    finally:
        planner.close()
    assert autonomous.calls == [command.platform_msg_id]
    assert router.persona_manager.active_key(GROUP) == "default"
    assert gateway.calls == []


async def test_revoked_guard_prevents_persona_mutation(sqlite_engine, tmp_path):
    router, gateway = switch_router(sqlite_engine, tmp_path)
    command = onebot_command()
    router.ingest_live_group_message(command)
    planner = GroupParticipationPlanner(router=router, decision_client=NoModel(), autonomous_executor=Autonomous())
    batch = AgentBatch(GROUP, (command,), "switch-turn")
    guard = Guard()
    guard.allowed = False
    try:
        decision = await planner.decide(batch)
        with pytest.raises(RuntimeError, match="revoked"):
            await planner.execute(command, batch, decision, guard)
    finally:
        planner.close()
    assert router.persona_manager.active_key(GROUP) == "default"
    assert gateway.calls == []


async def test_uncertain_persona_confirmation_does_not_replay_platform_write(sqlite_engine, tmp_path):
    router, gateway = switch_router(sqlite_engine, tmp_path, gateway=Gateway(uncertain=True))
    command = onebot_command()
    router.ingest_live_group_message(command)
    planner = GroupParticipationPlanner(router=router, decision_client=NoModel(), autonomous_executor=Autonomous())
    batch = AgentBatch(GROUP, (command,), "switch-turn")
    try:
        decision = await planner.decide(batch)
        result = await planner.execute(command, batch, decision, Guard())
        assert result.status == "uncertain"
        await planner.execute(command, batch, decision, Guard())
    finally:
        planner.close()
    assert len(gateway.calls) == 1
