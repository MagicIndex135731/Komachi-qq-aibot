from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path

from sqlalchemy import bindparam, text

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_group_message_event, resolve_message_type
from app.adapters.sender import Sender
from app.agent.contracts import GroupAgentOptions
from app.agent.observation import GroupObservationService
from app.agent.participation import GroupParticipationPlanner
from app.agent.runtime import GroupAgentRuntime
from app.providers.llm_client import LlmClient
from app.config import AppSettings, load_runtime_config
from app.core.time_utils import ASIA_SHANGHAI
from app.core.context_builder import ContextBuilder
from app.core.group_history_backfill import backfill_recent_group_history
from app.core.member_memory_backfill import MemberFactRefreshService
from app.core.persona_live_sync import PersonaLiveSyncService
from app.core.persona_switch import PersonaManager, PersonaSwitchService
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.main import (
    build_group_image_llm_client,
    build_group_image_reference_planner_client,
    build_group_image_service,
    build_image_reference_search_client,
    build_llm_client,
    build_memory_runtime,
    build_proactive_judge_client,
    build_usage_recorder,
    build_web_search_client,
    create_runtime_banner,
    should_enable_memory_in_group,
    should_ingest_group_message,
    sync_history_archives,
)
from app.providers.semantic_embeddings import EmbeddingProvider
from app.runtime_heartbeat import RuntimeHeartbeat
from app.runtime.social import build_social_agent
from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import MessageRepository, RetrievalDocumentRepository


def _write_group_ready_marker(*, log_dir: Path, state: str) -> None:
    """Persist gateway readiness for external start/status scripts.

    ``connected`` is written as soon as the OneBot websocket is up (the bot is
    already consuming messages at that point); ``ready`` is written after the
    startup backfill and startup-window mention replay complete. The marker is
    refreshed on every reconnect, so start/status scripts can treat a fresh
    marker as proof the bot is accepting messages.
    """
    payload = {
        "pid": os.getpid(),
        "state": state,
        "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "group.ready.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _prewarm_memory_embedding(
    *,
    log_dir: Path,
    provider: EmbeddingProvider | None,
    requested_device: str,
) -> None:
    """Load and verify the reply process' embedding runtime before readiness."""
    marker_path = log_dir / "memory.embedding.ready.json"
    marker_path.unlink(missing_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if provider is None:
        raise RuntimeError("memory embedding provider is missing")
    identity = provider.identity
    if identity.provider == "disabled":
        payload = {
            "pid": os.getpid(),
            "state": "disabled",
            "provider": identity.provider,
            "model": identity.model,
            "dimensions": identity.dimensions,
            "accelerator": "disabled",
            "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
        }
    else:
        if not provider.available:
            raise RuntimeError("memory embedding provider is unavailable")
        logging.info(
            "memory_embedding_prewarm state=starting provider=%s model=%s requested_device=%s",
            identity.provider,
            identity.model,
            requested_device,
        )
        vector = provider.embed_query("小町记忆检索启动预热")
        if vector is None or len(vector) != identity.dimensions:
            raise RuntimeError("memory embedding prewarm returned an invalid vector")
        accelerator = str(
            getattr(
                provider,
                "active_accelerator",
                "remote" if identity.provider == "openai_compatible" else "unknown",
            )
        )
        if (
            identity.provider == "local"
            and requested_device.strip().lower() == "cuda"
            and accelerator != "cuda"
        ):
            raise RuntimeError("memory embedding prewarm did not activate CUDA")
        payload = {
            "pid": os.getpid(),
            "state": "ready",
            "provider": identity.provider,
            "model": identity.model,
            "dimensions": identity.dimensions,
            "accelerator": accelerator,
            "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
        }

    temporary_path = marker_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(marker_path)
    logging.info(
        "memory_embedding_prewarm state=%s provider=%s model=%s dimensions=%s accelerator=%s",
        payload["state"],
        payload["provider"],
        payload["model"],
        payload["dimensions"],
        payload["accelerator"],
    )


def _prewarm_persona_example_vectors(
    manager,
    personas: dict,
    *,
    log_dir: Path,
) -> None:
    """Warm the persona example vector caches for live-refresh personas."""

    marker = Path(log_dir) / "persona.embedding.ready.json"
    marker.unlink(missing_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        prewarmed: list[str] = []
        persona_statuses: list[dict[str, object]] = []
        for persona_key, persona in personas.items():
            if not isinstance(persona, dict) or not persona.get("live_refresh"):
                continue
            try:
                group_id = int(persona.get("source_group_id") or 0)
            except (TypeError, ValueError):
                continue
            if group_id <= 0:
                continue
            count = manager.prewarm_examples(int(group_id), persona_key)
            prewarmed.append(f"{persona_key}={count}")
            status = manager.example_vector_status(int(group_id), persona_key)
            if status:
                persona_statuses.append({"persona_key": persona_key, **status})
            logging.info(
                "persona_embedding_prewarm persona=%s samples=%s",
                persona_key,
                count,
            )
        payload = {
            "state": (
                "failed"
                if any(
                    int(item.get("failed") or 0) > 0
                    or int(item.get("missing") or 0) > 0
                    for item in persona_statuses
                )
                else "ready"
            ),
            "personas": prewarmed,
            "provider": str(getattr(manager.embedding_provider.identity, "provider", "")),
            "model": str(getattr(manager.embedding_provider.identity, "model", "")),
            "embedding_version": str(
                getattr(manager.embedding_provider.identity, "version", "")
            ),
            "dimensions": int(
                getattr(manager.embedding_provider.identity, "dimensions", 0)
            ),
            "document_schema": (
                persona_statuses[0].get("document_schema")
                if persona_statuses
                else "style-situation-v2"
            ),
            "coverage": persona_statuses,
            "pid": os.getpid(),
            "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
        }
    except Exception:
        logging.exception("persona_embedding_prewarm_failed")
        payload = {"state": "failed"}
    marker.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _max_message_id(engine) -> int:
    with session_scope(engine) as session:
        row = session.execute(
            text("SELECT COALESCE(MAX(id), 0) FROM messages")
        ).scalar_one()
        return int(row)


def _startup_window_mention_rows(
    engine,
    *,
    watermark_message_id: int,
    enabled_group_ids: tuple[int, ...],
    bot_qq: int,
    limit: int = 5,
):
    """Return @小町 messages persisted by startup backfill but never replied."""
    if not enabled_group_ids:
        return []
    with session_scope(engine) as session:
        return session.execute(
            text(
                "SELECT id, group_id, user_id, platform_msg_id, raw_json, "
                "plain_text, reply_to_msg_id, timestamp FROM messages "
                "WHERE id > :wm AND mentioned_bot = 1 AND user_id != :bot "
                "AND group_id IN :groups ORDER BY id LIMIT :limit"
            ).bindparams(bindparam("groups", expanding=True)),
            {
                "wm": int(watermark_message_id),
                "bot": int(bot_qq),
                "groups": tuple(enabled_group_ids),
                "limit": max(1, int(limit)),
            },
        ).all()


async def _replay_startup_window_mentions(
    *,
    engine,
    router: InboundRouter,
    settings: AppSettings,
    runtime,
    watermark_message_id: int,
    agent: GroupAgentRuntime | None = None,
) -> None:
    """补处理启动窗口内被回填为历史、但实际是 @小町 的提问。"""
    enabled_group_ids = tuple(
        int(group_id)
        for group_id in runtime.group_policy.get("groups", {})
        if should_ingest_group_message(
            group_id=int(group_id),
            group_policy=runtime.group_policy,
        )
    )
    rows = _startup_window_mention_rows(
        engine,
        watermark_message_id=watermark_message_id,
        enabled_group_ids=enabled_group_ids,
        bot_qq=settings.bot_qq,
    )
    bot_name = str(runtime.persona.get("name", settings.bot_qq))
    for row in rows:
        raw_json = row.raw_json if isinstance(row.raw_json, dict) else {}
        message_segments: list[dict] = []
        raw_message = raw_json.get("message")
        if isinstance(raw_message, list) and raw_message:
            message_segments = list(raw_message)
        else:
            if row.reply_to_msg_id:
                message_segments.append(
                    {"type": "reply", "data": {"id": row.reply_to_msg_id}}
                )
            if row.plain_text:
                message_segments.append(
                    {"type": "text", "data": {"text": row.plain_text}}
                )
        if not any(
            isinstance(segment, dict) and segment.get("type") == "at"
            for segment in message_segments
        ):
            message_segments.append(
                {"type": "at", "data": {"qq": str(settings.bot_qq)}}
            )
        payload = {
            "post_type": "message",
            "message_id": row.platform_msg_id,
            "group_id": row.group_id,
            "user_id": row.user_id,
            "sender": raw_json.get("sender")
            or {"nickname": str(row.user_id), "card": ""},
            "time": int(datetime.fromisoformat(str(row.timestamp)).timestamp()),
            "message": message_segments,
        }
        event = parse_group_message_event(
            payload,
            bot_qq=settings.bot_qq,
            bot_name=bot_name,
        )
        logging.info(
            "startup_window_replay group_id=%s msg_id=%s",
            event.group_id,
            event.platform_msg_id,
        )
        try:
            if agent is not None:
                await agent.observe(event)
            else:
                await router._handle_persisted_group_message(event)
        except Exception:
            logging.exception(
                "startup_window_replay_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
        await asyncio.sleep(0.5)


def _revoke_group_message_projection(
    engine,
    *,
    group_id: int,
    platform_msg_id: str,
) -> tuple[int | None, int]:
    with session_scope(engine) as session:
        message = MessageRepository(session).mark_group_message_deleted(
            group_id=int(group_id),
            platform_msg_id=str(platform_msg_id),
            reason="group_recall",
        )
        if message is None:
            return None, 0
        session.flush()
        revoked = RetrievalDocumentRepository(
            session
        ).deactivate_raw_message_v3(
            group_id=int(group_id),
            message_id=int(message.id),
        )
        return int(message.id), revoked


async def _handle_group_recall_payload(payload: dict, *, engine) -> bool:
    if (
        payload.get("post_type") != "notice"
        or payload.get("notice_type") != "group_recall"
    ):
        return False
    try:
        group_id = int(payload["group_id"])
        platform_msg_id = str(payload["message_id"])
    except (KeyError, TypeError, ValueError):
        logging.warning("group_recall_invalid_payload")
        return True
    message_id, revoked = await asyncio.to_thread(
        _revoke_group_message_projection,
        engine,
        group_id=group_id,
        platform_msg_id=platform_msg_id,
    )
    logging.info(
        "group_recall_processed group_id=%s message_id=%s found=%s revoked=%s",
        group_id,
        platform_msg_id,
        message_id is not None,
        revoked,
    )
    return True


async def run() -> None:
    # Also covers the legacy app.main alias, which invokes run() directly.
    # QQ media URLs can carry access tokens in their query string.
    for library in ("httpx", "httpcore"):
        logging.getLogger(library).setLevel(logging.WARNING)
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    heartbeat = RuntimeHeartbeat(heartbeat_file=settings.log_dir / "group.heartbeat.json")
    await heartbeat.start()
    from app.core.downloaded_media import DownloadedMediaJanitor
    media_janitor = DownloadedMediaJanitor(settings)
    await media_janitor.start()
    engine = None
    group_image_service = None
    memory_compaction_service = None
    persona_sync_task = None
    persona_task_worker = None
    member_fact_refresh_task = None
    persona_prewarm_task = None
    group_agent = None
    participation_planner = None
    social_agent = None
    social_planner = None
    group_image_started = False
    owner_action_worker = None
    owner_notice_worker = None
    delegated_research_worker = None
    media_worker = None
    try:
        engine = await asyncio.to_thread(build_engine, settings.sqlite_path)
        logging.info("startup_step build_engine done")
        await asyncio.to_thread(create_all, engine)
        logging.info("startup_step create_all done")
        await asyncio.to_thread(sync_history_archives, engine, runtime)
        logging.info("startup_step sync_history done")

        gateway = NapCatGateway(
            ws_url=settings.napcat_ws_url,
            reconnect_forever=True,
        )
        sender = Sender(gateway)
        llm_client = build_llm_client(settings=settings, engine=engine)
        proactive_judge_client = None if settings.group_agent_enabled else build_proactive_judge_client(
            settings=settings,
            llm_client=llm_client,
            engine=engine,
        )
        group_image_llm_client = build_group_image_llm_client(settings=settings, engine=engine, llm_client=llm_client)
        web_search_client = build_web_search_client(settings)
        image_reference_search_client = build_image_reference_search_client(settings)
        image_reference_planner_client = build_group_image_reference_planner_client(
            settings=settings,
            llm_client=llm_client,
        )
        group_image_service = build_group_image_service(
            settings=settings,
            llm_client=group_image_llm_client,
            sender=sender,
            web_search_client=image_reference_search_client,
            image_reference_planner_client=image_reference_planner_client,
        )
        memory_runtime = await asyncio.to_thread(
            build_memory_runtime,
            settings=settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(runtime.persona.get("name", settings.bot_qq)),
            memory_enabled_group_ids=frozenset(
                int(group_id)
                for group_id in runtime.group_policy.get("groups", {})
                if should_enable_memory_in_group(
                    group_id=int(group_id),
                    group_policy=runtime.group_policy,
                )
            ),
        )
        logging.info("startup_step build_memory_runtime done")
        await asyncio.to_thread(
            _prewarm_memory_embedding,
            log_dir=settings.log_dir,
            provider=memory_runtime.embedding_provider,
            requested_device=settings.memory_embedding_device,
        )
        memory_compaction_service = memory_runtime.memory_compaction_service
        persistent_group_engine = engine if hasattr(engine, "connect") else None
        if hasattr(group_image_service, "engine") and getattr(group_image_service, "engine", None) is None:
            group_image_service.engine = persistent_group_engine
        if memory_compaction_service is not None:
            await memory_compaction_service.start()
        persona_manager = PersonaManager(
            engine=engine,
            personas=getattr(runtime, "personas", {}) or {},
            default_persona=runtime.persona,
            embedding_provider=memory_runtime.embedding_provider,
        )
        await asyncio.to_thread(persona_manager.load_state)
        persona_prewarm_task = asyncio.create_task(
            asyncio.to_thread(
                _prewarm_persona_example_vectors,
                persona_manager,
                getattr(runtime, "personas", {}) or {},
                log_dir=settings.log_dir,
            )
        )
        persona_switch_service = PersonaSwitchService(
            manager=persona_manager,
            sender=sender,
            bot_qq=settings.bot_qq,
        )
        persona_sync_service = PersonaLiveSyncService(
            engine=engine,
            settings=settings,
            personas=getattr(runtime, "personas", {}) or {},
            manager=persona_manager,
        )
        persona_sync_task = asyncio.create_task(persona_sync_service.run())
        from app.agent.service_tools import PersonaTaskService, PersonaTaskWorker
        persona_task_worker = PersonaTaskWorker(
            PersonaTaskService(engine, account_id=settings.bot_qq), persona_sync_service,
            permission_check=lambda: getattr(runtime, "social_policy", {}).get("services", {}).get("persona_task_execution") is True,
        )
        await persona_task_worker.start()
        memory_group_ids = {
            int(group_id)
            for group_id in runtime.group_policy.get("groups", {})
            if should_enable_memory_in_group(
                group_id=int(group_id),
                group_policy=runtime.group_policy,
            )
        }
        member_fact_refresh_service = MemberFactRefreshService(
            engine=engine,
            settings=settings,
            usage_recorder=build_usage_recorder(engine),
            group_ids=memory_group_ids,
            bot_qq=settings.bot_qq,
            bot_name=str(runtime.persona.get("name", settings.bot_qq)),
            member_allowlist={
                int(persona["source_user_id"])
                for persona in getattr(runtime, "personas", {}).values()
                if isinstance(persona, dict)
                and persona.get("live_refresh")
                and str(persona.get("source_user_id") or "").isdigit()
            },
        )
        member_fact_refresh_task = asyncio.create_task(
            member_fact_refresh_service.run()
        )
        router = InboundRouter(
            engine=engine,
            runtime=runtime,
            sender=sender,
            llm_client=llm_client,
            proactive_judge_client=proactive_judge_client,
            reply_policy=ReplyPolicy(),
            context_builder=ContextBuilder(),
            web_search_client=web_search_client,
            private_chat_service=None,
            group_image_service=group_image_service,
            memory_compaction_service=memory_compaction_service,
            memory_orchestrator=memory_runtime.memory_orchestrator,
            persona_manager=persona_manager,
            persona_switch_service=persona_switch_service,
        )
        if hasattr(group_image_service, "group_source_validator"):
            observation = GroupObservationService(router)
            group_image_service.group_source_validator = lambda request: (
                gateway.websocket is not None and observation.image_job_allowed(request)
            )
        enabled_group_ids = tuple(int(group_id) for group_id in runtime.group_policy.get("groups", {})
                                  if should_ingest_group_message(group_id=int(group_id), group_policy=runtime.group_policy))
        if settings.group_agent_enabled and enabled_group_ids:
            decision_client = build_proactive_judge_client(
                settings=settings.model_copy(update={
                    "proactive_model_judge_enabled": True,
                    "proactive_judge_max_output_tokens": settings.group_agent_decision_max_output_tokens,
                    "proactive_judge_reasoning_effort": settings.group_agent_decision_reasoning_effort,
                    "llm_timeout_seconds": min(settings.llm_timeout_seconds, settings.group_agent_decision_timeout_seconds),
                }),
                llm_client=llm_client, engine=engine,
            ) or llm_client
            if isinstance(decision_client, LlmClient):
                decision_client.vision_model = (settings.llm_vision_model or "").strip()
            participation_planner = GroupParticipationPlanner(
                router=router, decision_client=decision_client,
                max_workers=settings.group_agent_max_concurrent_turns,
            )
            group_agent = GroupAgentRuntime(
                engine=engine, bot_qq=settings.bot_qq,
                bot_name=str(runtime.persona.get("name", settings.bot_qq)),
                decide=participation_planner.decide, execute=participation_planner.execute,
                options=GroupAgentOptions(
                    quiet_seconds=settings.group_agent_batch_quiet_seconds,
                    max_batch_wait_seconds=settings.group_agent_batch_max_seconds,
                    max_batch_messages=settings.group_agent_batch_max_messages,
                    max_pending_per_group=settings.group_agent_max_pending_per_group,
                    max_message_age_seconds=settings.group_agent_max_message_age_seconds,
                    max_active_groups=max(1, len(enabled_group_ids)),
                    model_concurrency=settings.group_agent_max_concurrent_turns,
                    decision_timeout_seconds=settings.group_agent_decision_timeout_seconds,
                    execution_timeout_seconds=settings.group_agent_turn_timeout_seconds,
                    task_execution_timeout_seconds=settings.agent_task_turn_timeout_seconds,
                ),
            )
            await group_agent.start()
        if settings.social_agent_enabled or (settings.agent_runtime == "harness" and settings.group_agent_enabled):
            social_agent, social_planner = build_social_agent(
                router=router, sender=sender, gateway=gateway, llm_client=llm_client,
            )
        if social_agent is not None:
            if participation_planner is not None and hasattr(social_planner.executor, "execute_group_turn"):
                participation_planner.autonomous_executor = social_planner.executor
            await social_agent.start()
        if (participation_planner is not None and settings.agent_runtime == "harness"
                and participation_planner.autonomous_executor is None):
            from app.agent.autonomous_social import AutonomousSocialExecutor
            from app.adapters.qzone import QzoneAdapter
            from app.adapters.account_profile import AccountProfileAdapter
            participation_planner.autonomous_executor = AutonomousSocialExecutor(
                router=router, sender=sender, qzone_adapter=QzoneAdapter(gateway), decision_client=llm_client,
                image_service=group_image_service, profile_adapter=AccountProfileAdapter(gateway,
                    account_id=settings.bot_qq, allowed_group_ids=tuple(enabled_group_ids),
                    media_roots=((settings.data_dir / "autonomous_artifacts").resolve(),)))
            participation_planner.owns_autonomous_executor = True

        if settings.agent_runtime == 'harness':
            executor = getattr(social_planner, 'executor', None) if social_planner is not None else None
            executor = executor or getattr(participation_planner, 'autonomous_executor', None)
            if executor is not None and hasattr(executor, 'research_jobs'):
                delegated_research_worker = executor.research_jobs
                await delegated_research_worker.start()
                media_worker = executor.media_jobs
                await media_worker.start()
            if executor is not None and hasattr(executor, 'execute_owner_action'):
                from app.agent.owner_actions import OwnerActionService, OwnerActionWorker, OwnerNoticeWorker
                owner_service = OwnerActionService(engine, runtime=runtime)
                owner_action_worker = OwnerActionWorker(owner_service, executor, connected=lambda: gateway.websocket is not None)
                owner_notice_worker = OwnerNoticeWorker(owner_service, sender, surface='group', connected=lambda: gateway.websocket is not None)
                await owner_action_worker.start()
                await owner_notice_worker.start()

        def set_agents_connected(connected: bool) -> None:
            if group_agent is not None:
                group_agent.set_connected(connected)
            if social_agent is not None:
                social_agent.set_connected(connected)

        gateway.on_connection_change = set_agents_connected

        async def handle_payload(payload: dict) -> None:
            if await _handle_group_recall_payload(payload, engine=engine):
                return
            if payload.get("post_type") != "message":
                return
            message_type = resolve_message_type(payload)
            if message_type != "group":
                # Direct chats belong to the xiaomachi-private process, and an
                # unknown value means the bridge speaks a dialect this process
                # cannot parse. Never drop either case without a trace.
                if message_type == "unknown":
                    logging.warning(
                        "inbound_message_unhandled process=group message_type=%r keys=%s",
                        payload.get("message_type"),
                        sorted(payload.keys()),
                    )
                else:
                    logging.info(
                        "inbound_message_ignored process=group message_type=%s msg_id=%s",
                        message_type,
                        payload.get("message_id"),
                    )
                return

            group_id = int(payload["group_id"])
            if group_id == 10001:
                logging.info(
                    "group_payload_received group_id=%s msg_id=%s user_id=%s",
                    group_id,
                    payload.get("message_id"),
                    payload.get("user_id"),
                )
            if not should_ingest_group_message(group_id=group_id, group_policy=runtime.group_policy):
                return

            event = parse_group_message_event(
                payload,
                bot_qq=settings.bot_qq,
                bot_name=str(runtime.persona.get("name", settings.bot_qq)),
            )
            if group_agent is not None:
                # A live event may already exist from history backfill; the
                # durable agent inbox owns live-observation deduplication.
                router.ingest_live_group_message(event)
                if await router.try_handle_context_reset(event,
                        cancel_group=lambda group_id: group_agent.cancel_group(group_id, resume_pending=True)):
                    return
                await group_agent.observe(event)
            else:
                await router.handle_group_message(event)

        async def backfill_group_history_on_connect() -> None:
            nonlocal group_image_started
            set_agents_connected(True)
            if (
                not group_image_started
                and hasattr(group_image_service, "start")
                and getattr(group_image_service, "engine", None) is not None
            ):
                # Recovered drawings may send only after policy and OneBot are ready.
                await group_image_service.start()
                group_image_started = True
            _write_group_ready_marker(log_dir=settings.log_dir, state="connected")
            try:
                watermark_message_id = _max_message_id(engine)
                await backfill_recent_group_history(
                    router=router,
                    gateway=gateway,
                    bot_qq=settings.bot_qq,
                    bot_name=str(runtime.persona.get("name", settings.bot_qq)),
                )
                await _replay_startup_window_mentions(
                    engine=engine,
                    router=router,
                    settings=settings,
                    runtime=runtime,
                    watermark_message_id=watermark_message_id,
                    agent=group_agent,
                )
            except Exception:
                logging.exception("startup_backfill_failed; continuing to accept messages")
            _write_group_ready_marker(log_dir=settings.log_dir, state="ready")
            logging.info("group_ready accepting_messages=True state=ready")

        logging.info(create_runtime_banner(bot_qq=settings.bot_qq, model=f"{settings.llm_model} [group]"))
        await gateway.connect_and_consume(handle_payload, on_connect=backfill_group_history_on_connect)
    finally:
        if delegated_research_worker is not None:
            await delegated_research_worker.stop()
        if media_worker is not None:
            await media_worker.stop()
        if owner_action_worker is not None:
            await owner_action_worker.stop()
        if owner_notice_worker is not None:
            await owner_notice_worker.stop()
        if persona_task_worker is not None:
            await persona_task_worker.stop()
        if social_agent is not None:
            await social_agent.stop()
        if social_planner is not None:
            if hasattr(social_planner, "aclose"):
                await social_planner.aclose()
            else:
                social_planner.close()
        if group_agent is not None:
            await group_agent.stop()
        if participation_planner is not None:
            participation_planner.close()
        if member_fact_refresh_task is not None:
            member_fact_refresh_task.cancel()
        if persona_prewarm_task is not None:
            persona_prewarm_task.cancel()
        if persona_sync_task is not None:
            persona_sync_task.cancel()
        service_tasks = [task for task in (member_fact_refresh_task, persona_prewarm_task, persona_sync_task) if task is not None]
        if service_tasks:
            await asyncio.gather(*service_tasks, return_exceptions=True)
        if group_image_service is not None and hasattr(group_image_service, "stop") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.stop()
        if memory_compaction_service is not None:
            await memory_compaction_service.stop()
        await heartbeat.stop()
        await media_janitor.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
