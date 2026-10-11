import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import threading

from PIL import Image
import pytest
import yaml

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.service_tools import PersonaTaskService, PersonaTaskWorker, ServiceToolContext, build_service_tools
from app.agent.social_contracts import SocialJob, SocialObservation
from app.config import AppSettings
from app.core.persona_live_sync import PersonaLiveSyncService
from app.runtime.blocking import BoundedBlockingRunner
from app.runtime.memory_composition import build_memory_runtime
from app.storage.autonomous_agent import AutonomousIntention, AutonomousRepository
from app.storage.db import session_scope
from app.storage.models import Job, Message
from app.storage.repositories import GroupRepository, MemoryRepository, MessageRepository, UserRepository


GROUP, ACCOUNT, AUTHOR = 900000001, 900000099, 900000101


class Guard:
    action_id = "synthetic-turn"
    allowed = True

    def valid(self):
        return self.allowed

    def require_valid(self):
        if not self.valid():
            raise RuntimeError("revoked")


def seed(engine, *, group_id=GROUP, author=AUTHOR, message_id="synthetic-source", raw_json=None):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=author, nickname="synthetic author", group_card="")
        row = MessageRepository(session).add_group_message(platform_msg_id=message_id, group_id=group_id,
            user_id=author, timestamp=datetime.now(UTC), plain_text="我喜欢星星", msg_type="text",
            raw_json=raw_json or {"sender": {"nickname": "synthetic author"}}, reply_to_msg_id=None, mentioned_bot=False)
        session.flush()
        return row.id


@pytest.fixture
def context(sqlite_engine, tmp_path):
    source = seed(sqlite_engine)
    settings = AppSettings.model_construct(bot_qq=ACCOUNT, data_dir=tmp_path,
        napcat_ws_url="ws://127.0.0.1:1", llm_base_url="http://127.0.0.1:1",
        llm_api_key="synthetic", memory_memory_tools_enabled=True,
        memory_memory_tool_timeout_seconds=2, memory_memory_tool_max_results=5, llm_supports_vision_input=True,
        memory_orchestration_v2_enabled=True, memory_raw_v3_enabled=True,
        memory_layered_memory_enabled=True, memory_embedding_provider="disabled",
        memory_compaction_enabled=False, memory_query_rewrite_enabled=False)
    runtime = SimpleNamespace(settings=settings, persona={"name": "小町", "identity": "自主兴趣"},
        personas={}, social_policy={"services": {"memory_write": True, "persona_refresh": True}})
    router = SimpleNamespace(engine=sqlite_engine, runtime=runtime,
        _group_memory_enabled=lambda **kwargs: True, _active_persona=lambda group_id: runtime.persona)
    router.memory_orchestrator = build_memory_runtime(settings=settings, engine=sqlite_engine,
        llm_client=SimpleNamespace(), bot_display_name="小町").memory_orchestrator
    job = SocialJob(f"group:{GROUP}", "group", GROUP, autonomous_tools=True)
    observation = SocialObservation(job, datetime.now(UTC), "synthetic persona", source_message_ids=(source,))
    observed_sources = {source}
    store = AutonomousArtifactStore(tmp_path / "artifacts", account_id=ACCOUNT, scope=job.key,
        engine=sqlite_engine, source_message_ids=lambda: tuple(sorted(observed_sources)))
    runner = BoundedBlockingRunner(1)
    value = ServiceToolContext(router, observation, Guard(), observed_sources, runner, store,
        target_user_id=AUTHOR, target_message_id="synthetic-source",
        task_service=PersonaTaskService(sqlite_engine, account_id=ACCOUNT))
    try:
        yield value
    finally:
        runner.close()


def registry(context):
    return {tool.name: tool for tool in build_service_tools(context)}


def write_args(subject=str(AUTHOR), source="synthetic-source"):
    return {"kind": "preference", "subject": subject, "predicate": "likes", "object_text": "星星",
        "content": "喜欢星星", "source_msg_ids": [source]}


async def test_memory_tools_reuse_real_service_and_source_author(context):
    tools = registry(context)
    outcome = await tools["memory_write"].handler(write_args())
    assert outcome.status == "ok" and outcome.data["memory_id"] > 0
    read = await tools["memory_read"].handler({"member": "synthetic author"})
    assert "喜欢星星" in read.data["evidence"]
    assert "synthetic-source" in read.data["evidence"]
    wrong_author = await tools["memory_write"].handler(write_args(str(AUTHOR + 1)))
    assert wrong_author.status == "failed" and wrong_author.data["error"] == "subject_out_of_scope"
    cross = seed(context.router.engine, group_id=GROUP + 1, message_id="other-group")
    assert cross not in context.observed_sources
    rejected = await tools["memory_write"].handler(write_args(source="other-group"))
    assert rejected.data["error"] == "source_not_in_conversation"


async def test_memory_write_revocation_rolls_back_service_transaction(context, monkeypatch):
    tools = registry(context)
    original = MemoryRepository.upsert_canonical_memory
    def revoke(repository, **kwargs):
        row = original(repository, **kwargs)
        context.guard.allowed = False
        return row
    monkeypatch.setattr(MemoryRepository, "upsert_canonical_memory", revoke)
    with pytest.raises(RuntimeError, match="revoked"):
        await tools["memory_write"].handler(write_args())
    with session_scope(context.router.engine) as session:
        assert MemoryRepository(session).list_group_memories_for_subject(scope_id=str(GROUP), subject_id=str(AUTHOR), limit=10) == []


async def test_work_note_persists_with_revocable_sources(context):
    tools = registry(context)
    assert (await tools["save_work_note"].handler({"note": "以后研究夜空"})).data["saved"]
    assert (await registry(context)["read_work_note"].handler({})).data["note"] == {"note": "以后研究夜空"}
    with session_scope(context.router.engine) as session:
        row = session.get(Message, context.observation.source_message_ids[0])
        row.raw_json = {"delivery_state": "deleted"}
    context.observed_sources.clear()
    outcome = await registry(context)["read_work_note"].handler({})
    assert outcome.status == "blocked" and outcome.data["error"] == "note_source_revoked"


async def test_cancelled_work_note_does_not_commit(context, monkeypatch):
    original = AutonomousRepository.write_intention
    def late_write(repository, *args, **kwargs):
        before_commit = kwargs["before_commit"]
        def revoke():
            context.guard.allowed = False
            before_commit()
        kwargs["before_commit"] = revoke
        original(repository, *args, **kwargs)
    monkeypatch.setattr(AutonomousRepository, "write_intention", late_write)
    with pytest.raises(RuntimeError, match="revoked"):
        await registry(context)["save_work_note"].handler({"note": "late"})
    with session_scope(context.router.engine) as session:
        assert session.query(AutonomousIntention).count() == 0


async def test_observe_actual_group_pixels_saves_owned_artifact(context):
    path = context.router.runtime.settings.data_dir / "image_cache" / str(GROUP) / "synthetic.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "blue").save(path)
    with session_scope(context.router.engine) as session:
        row = session.get(Message, context.observation.source_message_ids[0])
        row.raw_json = {"message": [{"type": "image", "data": {"local_path": str(path)}}]}
    tool = registry(context)["observe_group_images"]
    outcome = await tool.handler({"source_ids": list(context.observed_sources)})
    assert outcome.status == "ok" and outcome.data["actual_pixels"]
    artifact_id = outcome.data["images"][0]["artifact_id"]
    assert context.store.get(artifact_id).is_file()
    assert Path(outcome.images[0].local_path).is_file()
    assert outcome.images[0].source_user_id == AUTHOR
    assert outcome.images[0].source_message_id == "synthetic-source"
    assert not list(context.store.root.glob(".observed-*"))
    denied = await tool.handler({"source_ids": [999999]})
    assert denied.status == "blocked" and denied.data["error"] == "image_source_out_of_scope"


async def test_dynamic_registry_permissions_and_public_persona(context):
    context.router.runtime.social_policy["services"]["memory_write"] = False
    assert "memory_write" not in registry(context)
    context.observation = SocialObservation(SocialJob("qzone_publish", "qzone_publish", ACCOUNT),
        datetime.now(UTC), "public")
    context.router.runtime.persona["relationships"] = [{"private_fact": "secret"}]
    tools = registry(context)
    assert set(tools) == {"read_persona", "save_work_note", "read_work_note"}
    assert "secret" not in (await tools["read_persona"].handler({})).data["expression_style"]


async def test_wakeup_tools_delegate_to_durable_scheduler(context):
    requests = []
    async def schedule(args):
        requests.append(args)
        return {"task_id": "durable-goal", "status": "waiting"}
    context.schedule_wakeup = schedule
    context.list_wakeups = lambda args: {"tasks": requests}
    context.cancel_wakeup = lambda args: {"task_id": args["task_id"], "status": "cancelled"}
    tools = registry(context)
    result = await tools["schedule_wakeup"].handler({"goal": "提醒讨论观星", "delay_seconds": 30})
    assert result.data == {"task_id": "durable-goal", "status": "waiting"}
    assert (await tools["list_wakeups"].handler({})).data["tasks"] == requests
    assert (await tools["cancel_wakeup"].handler({"task_id": "durable-goal"})).data["status"] == "cancelled"


async def test_persona_requests_are_scoped_idempotent_and_paid_paused(context):
    profile = {"name": "synthetic", "live_refresh": True, "source_group_id": GROUP, "source_user_id": AUTHOR}
    context.router.runtime.personas["synthetic"] = profile
    tools = registry(context)
    first = await tools["request_persona_refresh"].handler({"persona_key": "synthetic", "request_key": "refresh-one"})
    again = await tools["request_persona_refresh"].handler({"persona_key": "synthetic", "request_key": "refresh-one"})
    assert first.data["task_id"] == again.data["task_id"] and first.data["status"] == "queued"
    assert context.task_service.status(group_id=GROUP + 1, task_id=first.data["task_id"]) == {"error": "task_out_of_scope"}
    blocked = context.task_service.process_next(SimpleNamespace())
    assert blocked["status"] == "blocked" and blocked["reason"] == "paid_calls_paused"
    state = await tools["persona_task_status"].handler({"task_id": first.data["task_id"]})
    assert state.data["status"] == "blocked"
    assert (await tools["request_persona_refresh"].handler({"persona_key": "default", "request_key": "x"})).status == "blocked"


async def test_persona_task_processes_real_live_sync_with_only_fake_model(context, monkeypatch):
    from app.providers.llm_client import LlmClient
    profile = {"name": "synthetic", "identity": "群成员", "core_traits": ["直接"], "speaking_style": {"tone": "casual"},
        "self_concept": "群友", "speech_habits": ["短句"], "style_avoid": [], "relationships": [], "address_rules": [],
        "facts": [], "external_relations": [], "live_refresh": True, "source_group_id": GROUP, "source_user_id": AUTHOR}
    context.router.runtime.personas["synthetic"] = profile
    task = await registry(context)["request_persona_refresh"].handler({"persona_key": "synthetic", "request_key": "offline-live-sync"})
    settings = SimpleNamespace(bot_qq=ACCOUNT, data_dir=context.router.runtime.settings.data_dir,
        llm_base_url="http://127.0.0.1:1", llm_api_key="synthetic", llm_model="synthetic", llm_fallback_model="")
    manager = SimpleNamespace(personas=context.router.runtime.personas, load_style_banks=lambda: None,
        prewarm_examples=lambda *args: None)
    service = PersonaLiveSyncService(engine=context.router.engine, settings=settings,
        personas=context.router.runtime.personas, manager=manager, refresh_threshold=1)
    generated = []
    def synthetic_generate(client, prompt):
        generated.append(prompt)
        return yaml.safe_dump({key: value for key, value in profile.items()
            if key not in {"live_refresh", "source_group_id", "source_user_id"}}, allow_unicode=True)
    monkeypatch.setattr(LlmClient, "generate_text", synthetic_generate)
    result = context.task_service.process_next(service, allow_paid=True)
    assert result["task_id"] == task.data["task_id"]
    assert result["status"] == "completed" and len(generated) == 1
    assert (settings.data_dir / "personas" / "synthetic.live.yaml").exists()


def test_persona_restart_never_replays_interrupted_task(context):
    value = context.task_service.enqueue(group_id=GROUP, persona_key="synthetic", source_user_id=AUTHOR,
        source_message_ids=tuple(context.observed_sources), request_key="interrupted", guard=context.guard)
    with session_scope(context.router.engine) as session:
        session.get(Job, value["task_id"]).status = "running"
    assert context.task_service.recover() == 1
    assert context.task_service.status(group_id=GROUP, task_id=value["task_id"])["reason"] == "interrupted_not_replayed"


def test_persona_worker_recovers_expired_request_without_reexecution(context):
    value = context.task_service.enqueue(group_id=GROUP, persona_key="synthetic", source_user_id=AUTHOR,
        source_message_ids=tuple(context.observed_sources), request_key="expired", guard=context.guard)
    with session_scope(context.router.engine) as session:
        row = session.get(Job, value["task_id"])
        row.status, row.lease_until = "running", datetime(2000, 1, 1)
    assert context.task_service.process_next(SimpleNamespace()) is None
    status = context.task_service.status(group_id=GROUP, task_id=value["task_id"])
    assert status["status"] == "failed" and status["reason"] == "interrupted_not_replayed"


async def test_persona_worker_disabled_preserves_queue_and_enabled_uses_real_service(context, monkeypatch):
    from app.providers.llm_client import LlmClient
    profile = {"name": "synthetic", "identity": "群成员", "core_traits": ["直接"], "speaking_style": {"tone": "casual"},
        "self_concept": "群友", "speech_habits": ["短句"], "style_avoid": [], "relationships": [], "address_rules": [],
        "facts": [], "external_relations": [], "live_refresh": True, "source_group_id": GROUP, "source_user_id": AUTHOR}
    context.router.runtime.personas["synthetic"] = profile
    task = await registry(context)["request_persona_refresh"].handler({"persona_key": "synthetic", "request_key": "worker"})
    settings = SimpleNamespace(bot_qq=ACCOUNT, data_dir=context.router.runtime.settings.data_dir,
        llm_base_url="http://127.0.0.1:1", llm_api_key="synthetic", llm_model="synthetic", llm_fallback_model="")
    manager = SimpleNamespace(personas=context.router.runtime.personas, load_style_banks=lambda: None,
        prewarm_examples=lambda *args: None)
    service = PersonaLiveSyncService(engine=context.router.engine, settings=settings,
        personas=context.router.runtime.personas, manager=manager, refresh_threshold=1)
    generated = []
    def fake_model(client, prompt):
        generated.append(prompt)
        return yaml.safe_dump(profile, allow_unicode=True)
    monkeypatch.setattr(LlmClient, "generate_text", fake_model)
    permission = {"enabled": False}
    worker = PersonaTaskWorker(context.task_service, service,
        permission_check=lambda: permission["enabled"], interval_seconds=0.01)
    try:
        assert await worker.start() is False
        await asyncio.sleep(0.02)
        assert generated == []
        assert context.task_service.status(group_id=GROUP, task_id=task.data["task_id"])["status"] == "queued"
        permission["enabled"] = True
        assert await worker.start() is True
        for _ in range(100):
            if context.task_service.status(group_id=GROUP, task_id=task.data["task_id"])["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        assert context.task_service.status(group_id=GROUP, task_id=task.data["task_id"])["status"] == "completed"
        assert len(generated) == 1
    finally:
        await worker.stop()


def test_persona_execution_revoked_during_model_does_not_publish_profile(context, monkeypatch):
    from app.providers.llm_client import LlmClient
    profile = {"name": "synthetic", "identity": "群成员", "core_traits": ["直接"], "speaking_style": {"tone": "casual"},
        "self_concept": "群友", "speech_habits": ["短句"], "style_avoid": [], "relationships": [], "address_rules": [],
        "facts": [], "external_relations": [], "live_refresh": True, "source_group_id": GROUP, "source_user_id": AUTHOR}
    task = context.task_service.enqueue(group_id=GROUP, persona_key="synthetic", source_user_id=AUTHOR,
        source_message_ids=tuple(context.observed_sources), request_key="revoked-worker", guard=context.guard)
    settings = SimpleNamespace(bot_qq=ACCOUNT, data_dir=context.router.runtime.settings.data_dir,
        llm_base_url="http://127.0.0.1:1", llm_api_key="synthetic", llm_model="synthetic", llm_fallback_model="")
    personas = {"synthetic": profile}
    service = PersonaLiveSyncService(engine=context.router.engine, settings=settings, personas=personas,
        manager=SimpleNamespace(personas=personas, load_style_banks=lambda: None, prewarm_examples=lambda *args: None),
        refresh_threshold=1)
    def fake_model(client, prompt):
        context.guard.allowed = False
        return yaml.safe_dump(profile, allow_unicode=True)
    monkeypatch.setattr(LlmClient, "generate_text", fake_model)
    result = context.task_service.process_next(service, allow_paid=True, execution_guard=context.guard.require_valid)
    assert result["task_id"] == task["task_id"] and result["status"] == "failed"
    assert not (settings.data_dir / "personas" / "synthetic.live.yaml").exists()


def test_persona_worker_and_timer_serialize_same_profile_and_recheck_threshold(context, monkeypatch):
    from app.providers.llm_client import LlmClient
    profile = {"name": "synthetic", "identity": "群成员", "core_traits": ["直接"], "speaking_style": {"tone": "casual"},
        "self_concept": "群友", "speech_habits": ["短句"], "style_avoid": [], "relationships": [], "address_rules": [],
        "facts": [], "external_relations": [], "live_refresh": True, "source_group_id": GROUP, "source_user_id": AUTHOR}
    task = context.task_service.enqueue(group_id=GROUP, persona_key="synthetic", source_user_id=AUTHOR,
        source_message_ids=tuple(context.observed_sources), request_key="serialized-worker", guard=context.guard)
    settings = SimpleNamespace(bot_qq=ACCOUNT, data_dir=context.router.runtime.settings.data_dir,
        llm_base_url="http://127.0.0.1:1", llm_api_key="synthetic", llm_model="synthetic", llm_fallback_model="")
    personas = {"synthetic": profile}
    service = PersonaLiveSyncService(engine=context.router.engine, settings=settings, personas=personas,
        manager=SimpleNamespace(personas=personas, load_style_banks=lambda: None, prewarm_examples=lambda *args: None),
        refresh_threshold=1)
    generated, entered, release = [], threading.Event(), threading.Event()
    def fake_model(client, prompt):
        generated.append(prompt)
        entered.set()
        assert release.wait(3)
        return yaml.safe_dump(profile, allow_unicode=True)
    monkeypatch.setattr(LlmClient, "generate_text", fake_model)
    service._sync_examples("synthetic", AUTHOR, GROUP)
    with ThreadPoolExecutor(2) as pool:
        timer = pool.submit(service._maybe_refresh_profile, "synthetic", AUTHOR, GROUP)
        assert entered.wait(3)
        requested = pool.submit(context.task_service.process_next, service, allow_paid=True)
        release.set()
        timer.result(timeout=5)
        result = requested.result(timeout=5)
    assert len(generated) == 1
    assert result["task_id"] == task["task_id"] and result["status"] == "blocked"
    assert result["reason"] == "not_due_or_refresh_declined"
