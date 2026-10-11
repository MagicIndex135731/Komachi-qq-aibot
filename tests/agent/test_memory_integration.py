from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.agent import memory_integration
from app.agent.memory_integration import (
    build_agent_memory, call_agent_memory_tool, create_memory_dependency_receipt,
    memory_dependencies_valid, restore_memory_dependency_receipt,
)
from app.agent.service_tools import ServiceToolContext, build_service_tools
from app.agent.social_contracts import SocialJob, SocialObservation
from app.config import AppSettings
from app.core.time_utils import ASIA_SHANGHAI
from app.providers.semantic_embeddings import EmbeddingIdentity
from app.runtime.blocking import AgentModelBusyError, BoundedBlockingRunner
from app.runtime.memory_composition import build_memory_runtime
from app.storage.db import (
    activate_retrieval_vector_generation,
    create_retrieval_vector_generation,
    refresh_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.models import Job, MemoryItem, Message, RetrievalDocument, Summary
from app.storage.repositories import (
    GroupRepository, MemoryRepository, MessageRepository,
    RetrievalDocumentRepository, SummaryRepository, UserRepository,
)


GROUP, OTHER_GROUP, ACCOUNT, AUTHOR, OTHER_AUTHOR = 900000001, 900000002, 900000099, 900000101, 900000102
NOW = datetime(2026, 10, 10, 10, tzinfo=UTC)


class Guard:
    allowed = True
    def valid(self):
        return self.allowed
    def require_valid(self):
        if not self.allowed:
            raise RuntimeError("revoked")


class Embedding:
    identity = EmbeddingIdentity(provider="synthetic", model="offline", version="v1", dimensions=3)
    available = True
    def __init__(self):
        self.queries = []
    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0, 0.0]
    def embed_documents(self, documents):
        return [[1.0, 0.0, 0.0] for _ in documents]


def seed(engine, text, platform_id, *, group=GROUP, author=AUTHOR, timestamp=NOW,
         nickname="synthetic author", reply_to=None, mentioned_bot=False):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=author, nickname=nickname, group_card="")
        row = MessageRepository(session).add_group_message(platform_msg_id=platform_id,
            group_id=group, user_id=author, timestamp=timestamp, plain_text=text,
            msg_type="text", raw_json={"sender": {"nickname": nickname}},
            reply_to_msg_id=reply_to, mentioned_bot=mentioned_bot)
        session.flush()
        RetrievalDocumentRepository(session).project_raw_message_v3(group_id=group, message_id=row.id)
        return row.id


def attach_runtime(context, embedding, *, generation=None, shadow=False):
    context.router.memory_orchestrator = build_memory_runtime(
        settings=context.router.runtime.settings.model_copy(update={"memory_orchestration_shadow_mode": shadow}),
        engine=context.router.engine, llm_client=SimpleNamespace(), bot_display_name="小町",
        raw_message_embedding_generation_override=generation,
        embedding_provider_factory=lambda **_: embedding,
    ).memory_orchestrator


@pytest.fixture
def context(sqlite_engine, tmp_path):
    source_id = seed(sqlite_engine, "我以前喜欢星星", "historical-star", timestamp=NOW-timedelta(days=1))
    seed(sqlite_engine, "今天的天气", "recent-chat")
    target_id = seed(sqlite_engine, "群里以前关于星星聊过什么", "target-query", mentioned_bot=True)
    settings = AppSettings.model_construct(bot_qq=ACCOUNT, data_dir=tmp_path,
        napcat_ws_url="ws://127.0.0.1:1", llm_base_url="http://127.0.0.1:1", llm_api_key="synthetic",
        memory_orchestration_v2_enabled=True, memory_orchestration_shadow_mode=False,
        memory_raw_v3_enabled=True, memory_adaptive_context_enabled=False,
        memory_layered_memory_enabled=True, memory_memory_tools_enabled=True,
        memory_query_rewrite_enabled=False, memory_compaction_enabled=False,
        memory_memory_tool_timeout_seconds=2, memory_memory_tool_max_results=5,
        context_recent_limit=2, context_summary_limit=3)
    runtime = SimpleNamespace(settings=settings, persona={"name": "小町"}, personas={},
        social_policy={"services": {"memory_write": True}})
    router = SimpleNamespace(engine=sqlite_engine, runtime=runtime,
        _group_memory_enabled=lambda **_: True, _active_persona=lambda _: runtime.persona,
        _impersonating=lambda _: False)
    observation = SocialObservation(SocialJob(f"group:{GROUP}", "group", GROUP), NOW, "synthetic",
                                   source_message_ids=(target_id,))
    runner = BoundedBlockingRunner(1)
    context = ServiceToolContext(router, observation, Guard(), {target_id}, runner, None,
        target_user_id=AUTHOR, target_message_id="target-query")
    context.router.synthetic_source_id = source_id
    context.router.embedding = Embedding()
    attach_runtime(context, context.router.embedding)
    try:
        yield context
    finally:
        runner.close()


async def test_initial_recall_uses_existing_v3_pipeline_and_registers_historical_sources(context):
    before = counts(context)
    outcome = await build_agent_memory(context, addressed=True)
    assert outcome.status == "ok"
    assert "我以前喜欢星星" in outcome.data["evidence"]
    assert "historical-star" in outcome.data["source_message_ids"]
    from app.core.time_utils import shanghai_aware
    assert outcome.data["source_timestamps"]["historical-star"] == shanghai_aware(NOW-timedelta(days=1)).isoformat(timespec="seconds")
    assert context.router.synthetic_source_id in context.observed_sources
    assert outcome.data["mode"] == "v2"
    assert counts(context) == before


def counts(context):
    with session_scope(context.router.engine) as session:
        return tuple(session.scalar(select(func.count()).select_from(model))
                     for model in (Message, MemoryItem, Job, RetrievalDocument))


async def test_vector_only_paraphrase_recall_uses_real_shared_vector_channel(context):
    embedding = context.router.embedding
    generation = create_retrieval_vector_generation(context.router.engine,
        provider=embedding.identity.provider, model=embedding.identity.model,
        dimensions=3, version=embedding.identity.version, document_family="raw_message_v3")
    assert generation is not None, "offline sqlite-vec is required for this regression"
    with session_scope(context.router.engine) as session:
        document = RetrievalDocumentRepository(session).project_raw_message_v3(group_id=GROUP,
            message_id=context.router.synthetic_source_id, embedding_generation=generation)
        session.flush()
        document_id = document.id
    assert write_retrieval_vector_embeddings(context.router.engine, generation=generation,
        rows=[(document_id, GROUP, [1.0, 0.0, 0.0])]) == 1
    # Only this canonical projection is eligible in the isolated generation.
    assert refresh_retrieval_vector_generation(context.router.engine, generation=generation, mark_ready=True).status == "ready"
    assert activate_retrieval_vector_generation(context.router.engine, generation=generation,
        expected_active_generation=None)
    attach_runtime(context, embedding, generation=generation)
    outcome = await call_agent_memory_tool(context, "memory_search",
        {"query": "astronomy preference", "layer": "raw"})
    assert outcome.status == "ok"
    assert "historical-star" in outcome.data["source_message_ids"]
    assert embedding.queries and "astronomy preference" in embedding.queries[-1]


async def test_tools_resolve_historical_member_and_enforce_group_author_time_and_layer(context):
    mine = seed(context.router.engine, "昨天星空很好看", "member-yesterday", timestamp=NOW-timedelta(days=1))
    seed(context.router.engine, "昨天星空很好看", "other-author", author=OTHER_AUTHOR, nickname="another author", timestamp=NOW-timedelta(days=1))
    seed(context.router.engine, "昨天星空很好看", "other-group", group=OTHER_GROUP, timestamp=NOW-timedelta(days=1))
    seed(context.router.engine, "前天星空很好看", "older-member", timestamp=NOW-timedelta(days=2))
    # Explicit tool queries can read newer canonical messages than the trigger.
    context.target_message_id = None
    outcome = await call_agent_memory_tool(context, "memory_search",
        {"query": "昨天星空", "layer": "raw", "member": "synthetic author"})
    assert outcome.status == "ok"
    assert "member-yesterday" in outcome.data["source_message_ids"]
    assert "other-author" not in outcome.data["source_message_ids"]
    assert "other-group" not in outcome.data["source_message_ids"]
    assert "older-member" not in outcome.data["source_message_ids"]
    assert outcome.data["layer_counts"]["facts"] == outcome.data["layer_counts"]["summaries"] == 0
    assert mine in context.observed_sources


async def test_ambiguous_and_out_of_group_members_fail_closed(context):
    seed(context.router.engine, "星星", "ambiguous-alias", author=OTHER_AUTHOR)
    seed(context.router.engine, "星星", "out-of-scope", group=OTHER_GROUP, author=OTHER_AUTHOR+1, nickname="outsider")
    for member in ("synthetic author", "outsider", str(OTHER_AUTHOR+1), str(ACCOUNT)):
        outcome = await call_agent_memory_tool(context, "memory_search", {"query": "星星", "member": member})
        assert outcome.status == "blocked" and outcome.data["error"] == "member_unresolved"


async def test_full_summary_provenance_is_retained_and_middle_recall_removes_summary(context):
    with session_scope(context.router.engine) as session:
        sources = list(session.scalars(select(Message).where(Message.group_id == GROUP).order_by(Message.id)))
        summary = SummaryRepository(session).upsert_summary(scope_type="group", scope_id=str(GROUP),
            summary_level="episode", summary_key="synthetic-window", start_at=sources[0].timestamp,
            end_at=sources[-1].timestamp, content="星星讨论的来源完整摘要", source_count=len(sources),
            source_start_msg_id=sources[0].platform_msg_id, source_end_msg_id=sources[-1].platform_msg_id)
        middle_id = sources[1].id
        session.flush()
    outcome = await call_agent_memory_tool(context, "memory_search", {"query": "星星", "layer": "summaries"})
    assert outcome.status == "ok"
    assert "来源完整摘要" in outcome.data["evidence"]
    assert middle_id in outcome.data["source_ids"]
    assert middle_id in context.observed_sources
    with session_scope(context.router.engine) as session:
        session.get(Message, middle_id).raw_json = {"delivery_state": "deleted"}
    outcome = await call_agent_memory_tool(context, "memory_search", {"query": "星星", "layer": "summaries"})
    assert outcome.status == "ok"
    assert "来源完整摘要" not in outcome.data["evidence"]
    assert outcome.data["layer_counts"]["summaries"] == 0


async def test_readonly_agent_recall_does_not_enqueue_shadow_or_mutate_storage(context):
    attach_runtime(context, context.router.embedding, shadow=True)
    enqueued = []
    context.router.memory_orchestrator.shadow_enqueue = enqueued.append
    before = counts(context)
    await build_agent_memory(context)
    assert enqueued == []
    assert counts(context) == before


async def test_cancelled_and_timed_out_retrieval_never_admits_late_evidence(context, monkeypatch):
    original = context.router.memory_orchestrator.build_context
    started, finish, completed = Event(), Event(), Event()
    def blocked(request):
        started.set()
        assert finish.wait(10)
        try:
            return original(request)
        finally:
            completed.set()
    context.router.memory_orchestrator.build_context = blocked
    context.router.runtime.settings.memory_memory_tool_timeout_seconds = 0.1
    async def wait_worker_release():
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            try:
                await context.blocking.run(lambda: None)
                return
            except AgentModelBusyError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(0.01)
    async def time_after_worker_started(awaitable, *, timeout):
        worker = asyncio.create_task(awaitable)
        try:
            assert await asyncio.to_thread(started.wait, 5)
            return await asyncio.wait_for(worker, timeout=timeout)
        finally:
            if not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
    # Time the deliberately blocked retrieval, independent of cold SQLite setup.
    monkeypatch.setattr(memory_integration, "asyncio", SimpleNamespace(
        wait_for=time_after_worker_started, CancelledError=asyncio.CancelledError))
    before = set(context.observed_sources)
    outcome = await build_agent_memory(context)
    assert started.is_set() and outcome.data["error"] == "tool_timeout"
    assert context.observed_sources == before
    finish.set()
    assert await asyncio.to_thread(completed.wait, 5)
    # Wait until the blocking runner has released its real worker slot.
    await wait_worker_release()
    assert context.observed_sources == before

    monkeypatch.setattr(memory_integration, "asyncio", asyncio)
    started.clear(); finish.clear(); completed.clear()
    context.router.runtime.settings.memory_memory_tool_timeout_seconds = 2
    task = asyncio.create_task(build_agent_memory(context))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finish.set()
    assert await asyncio.to_thread(completed.wait, 5)
    await wait_worker_release()
    assert context.observed_sources == before


async def test_conversation_quote_keeps_tools_out_of_registry(context):
    with session_scope(context.router.engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id("target-query")
        row.plain_text, row.reply_to_msg_id = "继续聊吧", "historical-star"
    outcome = await build_agent_memory(context)
    assert outcome.status == "ok" and outcome.data["memory_search_allowed"] is False
    context.memory_search_allowed = outcome.data["memory_search_allowed"]
    assert not {"memory_search", "memory_read", "memory_write"}.intersection(tool.name for tool in build_service_tools(context))


async def test_initial_impersonated_binding_preserves_requester_and_bot_address_authority(context):
    context.router.runtime.persona["source_user_id"] = OTHER_AUTHOR
    context.router.runtime.persona["source_group_id"] = GROUP
    context.router._impersonating = lambda _: True
    seed(context.router.engine, "我喜欢动画", "persona-source", author=OTHER_AUTHOR, nickname="synthetic donor", timestamp=NOW-timedelta(days=2))
    with session_scope(context.router.engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id("target-query")
        row.plain_text = "你喜欢什么"
    outcome = await build_agent_memory(context, addressed=True)
    assert outcome.data["resolved_subject_binding"] == "impersonated"
    assert outcome.data["resolved_subject_ids"] == (str(OTHER_AUTHOR),)
    with session_scope(context.router.engine) as session:
        MessageRepository(session).get_by_platform_msg_id("target-query").plain_text = "我喜欢什么"
    outcome = await build_agent_memory(context, addressed=True)
    assert outcome.data["resolved_subject_binding"] == "requester"
    assert outcome.data["resolved_subject_ids"] == (str(AUTHOR),)


async def test_explicit_fact_tool_applies_temporal_provenance_and_layer_boundary(context):
    yesterday = seed(context.router.engine, "我喜欢星空摄影", "dated-fact", timestamp=NOW-timedelta(days=1))
    seed(context.router.engine, "我喜欢星空绘画", "old-fact", timestamp=NOW-timedelta(days=2))
    with session_scope(context.router.engine) as session:
        for source, content, observed in (("dated-fact", "喜欢星空摄影", NOW-timedelta(days=1)),
                                          ("old-fact", "喜欢星空绘画", NOW-timedelta(days=2))):
            MemoryRepository(session).add_memory(scope_type="group", scope_id=str(GROUP),
                subject_type="user", subject_id=str(AUTHOR), memory_kind="preference",
                content=content, importance=5, confidence=1, source_msg_id=source, valid_from=observed)
    context.target_message_id = None
    outcome = await call_agent_memory_tool(context, "memory_search",
        {"query": "昨天星空", "layer": "facts", "member": str(AUTHOR)})
    assert "喜欢星空摄影" in outcome.data["evidence"]
    assert "喜欢星空绘画" not in outcome.data["evidence"]
    assert outcome.data["layer_counts"] == {"raw": 0, "facts": 1, "summaries": 0}
    assert yesterday in context.observed_sources


async def test_source_budget_keeps_packet_and_guard_provenance_in_sync(context):
    ids = [seed(context.router.engine, f"scope seed {index}", f"scope-seed-{index}") for index in range(199)]
    context.observed_sources.update(ids)
    assert len(context.observed_sources) == 200
    outcome = await call_agent_memory_tool(context, "memory_search", {"query": "星星", "layer": "raw"})
    assert outcome.status == "ok"
    assert context.router.synthetic_source_id not in outcome.data["source_ids"]
    assert "我以前喜欢星星" not in outcome.data["evidence"]
    assert len(context.observed_sources) == 200
    assert set(outcome.data["source_ids"]).issubset(context.observed_sources)


async def test_disabled_group_and_bad_tool_arguments_do_not_call_provider(context):
    def unexpected(_):
        raise AssertionError("disabled or invalid reads must not run retrieval")
    context.router.memory_orchestrator.build_context = unexpected
    context.router._group_memory_enabled = lambda **_: False
    assert (await build_agent_memory(context)).data["error"] == "memory_disabled"
    assert (await call_agent_memory_tool(context, "memory_search", {"query": "星星"})).data["error"] == "memory_disabled"
    context.router._group_memory_enabled = lambda **_: True
    assert (await call_agent_memory_tool(context, "memory_search", {"query": "", "limit": True})).data["error"] == "invalid_memory_search_arguments"


@pytest.mark.parametrize("rejection", ["mixed_member", "revoked", "source_budget"])
async def test_summary_limit_counts_accepted_evidence_and_skips_unusable_top_rank(context, rejection):
    good = seed(context.router.engine, "星星的有效来源", "valid-summary-source", timestamp=NOW-timedelta(hours=3))
    first = seed(context.router.engine, "星星旧讨论开始", "summary-new-start", timestamp=NOW-timedelta(days=2))
    middle = seed(context.router.engine, "星星旧讨论中间", "summary-new-middle", author=OTHER_AUTHOR, nickname="another author", timestamp=NOW-timedelta(days=2))
    last = seed(context.router.engine, "星星旧讨论结束", "summary-new-end", timestamp=NOW-timedelta(days=2))
    with session_scope(context.router.engine) as session:
        summaries = SummaryRepository(session)
        for key, text, source_count, start_id, end_id, start, end in (
            ("newer-valid", "星星同成员有效摘要", 1, "valid-summary-source", "valid-summary-source", NOW-timedelta(hours=3), NOW-timedelta(hours=3)),
            ("older-unusable", "星星第一候选不可用摘要", 3, "summary-new-start", "summary-new-end", NOW-timedelta(days=2), NOW-timedelta(days=2)),
        ):
            summaries.upsert_summary(scope_type="group", scope_id=str(GROUP), summary_level="episode",
                summary_key=key, start_at=start, end_at=end, content=text, source_count=source_count,
                source_start_msg_id=start_id, source_end_msg_id=end_id)
        if rejection == "revoked":
            session.get(Message, middle).raw_json = {"delivery_state": "deleted"}
    if rejection == "source_budget":
        fillers = [seed(context.router.engine, f"quota filler {index}", f"quota-filler-{index}") for index in range(198)]
        context.observed_sources.update([good, *fillers])
        assert len(context.observed_sources) == 200
    arguments = {"query": "星星", "layer": "summaries", "limit": 1}
    if rejection == "mixed_member":
        arguments["member"] = str(AUTHOR)
    outcome = await call_agent_memory_tool(context, "memory_search", arguments)
    assert outcome.status == "ok"
    assert outcome.data["layer_counts"]["summaries"] == 1
    assert "星星同成员有效摘要" in outcome.data["evidence"]
    assert "第一候选不可用摘要" not in outcome.data["evidence"]
    assert good in outcome.data["source_ids"]
    assert first not in outcome.data["source_ids"]
    assert last not in outcome.data["source_ids"]


async def test_cross_group_persona_donor_does_not_bind_current_group_self_fact(context):
    context.router.runtime.persona.update(source_user_id=OTHER_AUTHOR, source_group_id=OTHER_GROUP)
    context.router._impersonating = lambda _: True
    seed(context.router.engine, "我喜欢动画", "local-donor-source", author=OTHER_AUTHOR,
         nickname="synthetic donor", timestamp=NOW-timedelta(days=2))
    with session_scope(context.router.engine) as session:
        MessageRepository(session).get_by_platform_msg_id("target-query").plain_text = "你喜欢什么"
    outcome = await build_agent_memory(context, addressed=True)
    assert outcome.data["resolved_subject_binding"] != "impersonated"
    with session_scope(context.router.engine) as session:
        MessageRepository(session).get_by_platform_msg_id("target-query").plain_text = "synthetic donor 喜欢什么"
    outcome = await build_agent_memory(context, addressed=True)
    assert outcome.data["resolved_subject_binding"] == "explicit"
    assert outcome.data["resolved_subject_ids"] == (str(OTHER_AUTHOR),)


def add_fact(context, *, source="historical-star", content="喜欢星星", group=GROUP):
    with session_scope(context.router.engine) as session:
        row = MemoryRepository(session).add_memory(scope_type="group", scope_id=str(group),
            subject_type="user", subject_id=str(AUTHOR), memory_kind="preference",
            content=content, importance=5, confidence=1, source_msg_id=source,
            valid_from=NOW-timedelta(days=1))
        session.flush()
        return row.id


@pytest.mark.parametrize("change", ["superseded", "expired", "expires_at", "content", "source", "deleted"])
async def test_admitted_fact_dependencies_invalidate_after_semantic_change(context, change):
    memory_id = add_fact(context)
    outcome = await call_agent_memory_tool(context, "memory_read", {"member": str(AUTHOR)})
    dependencies = dict(context.memory_dependencies)
    assert dependencies and f"fact:{memory_id}" in dependencies
    assert "memory_dependencies" not in outcome.data
    assert not any(revision in outcome.data["evidence"] for revision in dependencies.values())
    assert context.memory_dependencies == dependencies
    assert memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=NOW)
    with session_scope(context.router.engine) as session:
        row = session.get(MemoryItem, memory_id)
        if change == "superseded":
            row.status = "superseded"
        elif change == "expired":
            row.valid_until = NOW-timedelta(seconds=1)
        elif change == "expires_at":
            row.expires_at = NOW-timedelta(seconds=1)
        elif change == "content":
            row.content = "喜欢月亮"
        elif change == "source":
            row.source_msg_ids = ["historical-star", "recent-chat"]
        else:
            session.delete(row)
    assert not memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=NOW)


async def test_dependency_validity_changes_at_expiry_without_any_database_update(context):
    memory_id = add_fact(context)
    expiry = datetime.now(UTC)+timedelta(seconds=60)
    with session_scope(context.router.engine) as session:
        session.get(MemoryItem, memory_id).valid_until = expiry.astimezone(ASIA_SHANGHAI).replace(tzinfo=None)
    outcome = await call_agent_memory_tool(context, "memory_read", {"member": str(AUTHOR)})
    dependencies = dict(context.memory_dependencies)
    assert dependencies
    assert memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=expiry-timedelta(seconds=1))
    assert not memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=expiry)
    receipt = create_memory_dependency_receipt(dependencies)
    assert restore_memory_dependency_receipt(context.router.engine, group_id=GROUP,
        receipt=receipt, now=expiry-timedelta(seconds=1)) == dependencies
    assert restore_memory_dependency_receipt(context.router.engine, group_id=GROUP,
        receipt=receipt, now=expiry) is None


async def test_unrelated_new_messages_facts_and_usage_counters_preserve_dependency_revision(context):
    memory_id = add_fact(context)
    outcome = await call_agent_memory_tool(context, "memory_read", {"member": str(AUTHOR)})
    dependencies = dict(context.memory_dependencies)
    seed(context.router.engine, "无关新消息", "unrelated-new-source")
    add_fact(context, source="unrelated-new-source", content="喜欢新的运动")
    with session_scope(context.router.engine) as session:
        row = session.get(MemoryItem, memory_id)
        row.mention_count += 1
        row.last_seen_at = NOW+timedelta(seconds=1)
        row.confidence = 0.99
    assert memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=NOW)
    assert not memory_dependencies_valid(context.router.engine, group_id=OTHER_GROUP, dependencies=dependencies, now=NOW)
    assert not memory_dependencies_valid(context.router.engine, group_id=GROUP,
        dependencies={"fact:-1": "a"*64}, now=NOW)
    assert not memory_dependencies_valid(context.router.engine, group_id=GROUP,
        dependencies={f"fact:{index+1}": "a"*64 for index in range(201)}, now=NOW)


async def test_summary_dependency_tracks_children_and_ignores_unrelated_summaries(context):
    context.router.runtime.settings.context_summary_limit = 5
    attach_runtime(context, context.router.embedding)
    with session_scope(context.router.engine) as session:
        sources = list(session.scalars(select(Message).where(Message.group_id == GROUP).order_by(Message.id)))
        summaries = SummaryRepository(session)
        children = []
        for row in sources:
            child = summaries.upsert_summary(scope_type="group", scope_id=str(GROUP), summary_level="episode",
                summary_key=f"child-{row.id}", start_at=row.timestamp, end_at=row.timestamp,
                content=f"星星子摘要{row.id}", source_count=1,
                source_start_msg_id=row.platform_msg_id, source_end_msg_id=row.platform_msg_id)
            session.flush()
            children.append(child.id)
        parent = summaries.upsert_summary(scope_type="group", scope_id=str(GROUP), summary_level="semantic_daily",
            summary_key="parent", start_at=sources[0].timestamp, end_at=sources[-1].timestamp,
            content="星星父摘要", source_count=len(sources), source_summary_ids=children,
            source_start_msg_id=sources[0].platform_msg_id, source_end_msg_id=sources[-1].platform_msg_id)
        session.flush()
        parent_id = parent.id
    outcome = await call_agent_memory_tool(context, "memory_search", {"query": "星星", "layer": "summaries", "limit": 5})
    dependencies = dict(context.memory_dependencies)
    assert "父摘要" in outcome.data["evidence"]
    assert set(dependencies) == {f"summary:{identifier}" for identifier in [parent_id, *children]}
    assert memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=NOW)
    with session_scope(context.router.engine) as session:
        SummaryRepository(session).upsert_summary(scope_type="group", scope_id=str(GROUP), summary_level="episode",
            summary_key="unrelated", start_at=NOW, end_at=NOW, content="无关新摘要", source_count=1,
            source_start_msg_id="recent-chat", source_end_msg_id="recent-chat")
    assert memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=NOW)
    with session_scope(context.router.engine) as session:
        session.get(Summary, children[0]).status = "inactive"
    assert not memory_dependencies_valid(context.router.engine, group_id=GROUP, dependencies=dependencies, now=NOW)


@pytest.mark.parametrize("change", ["unchanged", "unrelated", "superseded", "expired"])
async def test_atomic_work_note_dependency_receipt_restores_only_current_facts(context, change):
    context.clock = lambda: NOW
    memory_id = add_fact(context)
    recalled = await call_agent_memory_tool(context, "memory_read", {"member": str(AUTHOR)})
    dependencies = dict(context.memory_dependencies)
    tools = {tool.name: tool for tool in build_service_tools(context)}
    saved = await tools["save_work_note"].handler({"note": "接着讨论历史星星偏好"})
    assert saved.data["saved"]
    with session_scope(context.router.engine) as session:
        row = session.get(MemoryItem, memory_id)
        if change == "superseded":
            row.status = "superseded"
        elif change == "expired":
            row.valid_until = NOW.astimezone(ASIA_SHANGHAI).replace(tzinfo=None)-timedelta(seconds=1)
    if change == "unrelated":
        seed(context.router.engine, "无关新消息", "unrelated-note-source")
        add_fact(context, source="unrelated-note-source", content="喜欢运动")
    # A fresh Agent session must obtain dependencies from the durable note.
    context.memory_dependencies.clear()
    read = await tools["read_work_note"].handler({})
    if change in {"superseded", "expired"}:
        assert read.status == "blocked" and read.data["error"] == "note_memory_dependency_revoked"
        assert context.memory_dependencies == {}
        assert "note" not in read.data
    else:
        assert read.data["note"] == {"note": "接着讨论历史星星偏好"}
        assert context.memory_dependencies == dependencies
        assert "memory_dependencies" not in read.data["note"]


def test_dependency_receipt_is_bounded_content_free_and_rejects_malformed_data(context):
    dependencies = {f"fact:{index+1}": "a"*64 for index in range(200)}
    receipt = create_memory_dependency_receipt(dependencies)
    assert len(str(receipt)) < 2000
    assert "a"*64 not in str(receipt)
    assert restore_memory_dependency_receipt(context.router.engine, group_id=GROUP,
        receipt=receipt, now=NOW) is None
    empty = create_memory_dependency_receipt({})
    assert restore_memory_dependency_receipt(context.router.engine, group_id=GROUP, receipt=empty, now=NOW) == {}
    for invalid in ({**empty, "v": True}, {**empty, "keys": "*invalid*"},
                    {**empty, "revision": "非"*64}, {**empty, "extra": True}):
        assert restore_memory_dependency_receipt(context.router.engine, group_id=GROUP, receipt=invalid, now=NOW) is None
    for invalid in ({"fact:01": "a"*64}, {"fact:9223372036854775808": "a"*64},
                    {"fact:1": "z"*64}, {f"fact:{index+1}": "a"*64 for index in range(201)}):
        with pytest.raises(ValueError, match="invalid_memory_dependencies"):
            create_memory_dependency_receipt(invalid)


async def test_work_note_receipt_budget_preserves_preexisting_note(context):
    tools = {tool.name: tool for tool in build_service_tools(context)}
    await tools["save_work_note"].handler({"note": "原有目标"})
    # Widely spaced canonical IDs defeat compression; metadata shares the total note budget.
    context.memory_dependencies.update({f"fact:{2**62 + index * 9876543212345}": "a"*64 for index in range(200)})
    result = await tools["save_work_note"].handler({"note": "目"*1200})
    assert result.status == "blocked" and result.data["error"] == "note_budget_exhausted"
    context.memory_dependencies.clear()
    assert (await tools["read_work_note"].handler({})).data["note"] == {"note": "原有目标"}
