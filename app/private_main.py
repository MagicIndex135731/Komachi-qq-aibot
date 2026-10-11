from __future__ import annotations

import asyncio
import logging

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_private_message_event, resolve_message_type
from app.adapters.sender import Sender
from app.config import AppSettings, load_runtime_config
from app.core.chat_style import build_reply_split_config
from app.runtime.providers import (
    build_group_image_llm_client,
    build_group_image_reference_planner_client,
    build_image_reference_search_client,
    build_web_search_client,
)
from app.runtime.providers import build_llm_client as _build_llm_client
from app.providers.llm_client import LlmClient
from app.private_chat.service import PrivateChatService
from app.private_chat.access import PrivateChatAccess
from app.private_chat.dispatcher import PrivateMessageDispatcher
from app.private_reminders import PrivateReminderScheduler, load_private_reminders
from app.runtime_heartbeat import RuntimeHeartbeat
from app.storage.db import build_engine, create_all


def build_llm_client(*, settings: AppSettings, engine):
    """Resolve the private process' injected client without the group facade."""
    return _build_llm_client(settings=settings, engine=engine, client_factory=LlmClient)


async def _wait_for_gateway_ready(gateway: NapCatGateway, gateway_task: asyncio.Task) -> None:
    while gateway.websocket is None:
        if gateway_task.done():
            await gateway_task
        await asyncio.sleep(0.1)


async def run() -> None:
    from app.adapters.qzone import QzoneAdapter

    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    create_all(engine)

    gateway = NapCatGateway(
        ws_url=settings.napcat_ws_url,
        reconnect_forever=True,
    )
    heartbeat = RuntimeHeartbeat(heartbeat_file=settings.log_dir / "private.heartbeat.json")
    sender = Sender(gateway)
    llm_client = build_llm_client(settings=settings, engine=engine)
    # Private drawings must not ride the chat transport: ``GROUP_IMAGE_CHAT_*``
    # pins the provider that actually serves the image model.  Without it the
    # private process would ask the chat model (for example DeepSeek) for an image
    # and always answer with the failure notice.
    group_image_llm_client = build_group_image_llm_client(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
    )
    web_search_client = build_web_search_client(settings)
    # Private drawings refine reference-image queries with the same planner and
    # external image search the group image service uses; the chat search client
    # above is only for the text-side grounding path.
    image_reference_search_client = build_image_reference_search_client(settings)
    image_reference_planner_client = build_group_image_reference_planner_client(
        settings=settings,
        llm_client=llm_client,
    )
    private_chat_service = PrivateChatService(
        engine=engine,
        sender=sender,
        llm_client=llm_client,
        image_llm_client=group_image_llm_client,
        owner_qq=settings.owner_qq,
        bot_qq=settings.bot_qq,
        private_chat_qqs=settings.private_chat_whitelist,
        access_policy=PrivateChatAccess(
            owner_qq=settings.owner_qq,
            allowlist=settings.private_chat_whitelist,
            all_friends=settings.private_chat_all_friends,
            friend_reader=QzoneAdapter(gateway, timeout_seconds=8).get_friend_ids,
        ),
        data_dir=settings.data_dir,
        image_model=settings.group_image_model,
        image_size="auto",
        image_quality="high",
        image_background=None,
        image_output_format="png",
        image_output_compression=None,
        image_moderation=None,
        image_queue_capacity=settings.group_image_queue_capacity,
        image_max_attempts=1,
        image_timeout_seconds=settings.group_image_timeout_seconds,
        web_search_client=web_search_client,
        image_reference_search_client=image_reference_search_client,
        image_reference_planner_client=image_reference_planner_client,
        reply_split_config=build_reply_split_config(settings),
        assistant_name=str(runtime.persona.get("name", "小町")),
        persona=runtime.persona,
        safety=runtime.safety,
    )
    if settings.agent_runtime == "harness":
        from app.private_chat.agent_service import PrivateHarnessService
        private_chat_service.agent_handler = PrivateHarnessService(private_chat_service, settings=settings, gateway=gateway, runtime=runtime)
    owner_notice_worker = None
    if settings.agent_runtime == 'harness':
        from app.agent.owner_actions import OwnerActionService, OwnerNoticeWorker
        owner_notice_worker = OwnerNoticeWorker(OwnerActionService(engine, runtime=runtime), sender,
            surface='private', connected=lambda: gateway.websocket is not None)
    reminder_scheduler = PrivateReminderScheduler(
        sender=sender,
        data_dir=settings.data_dir,
        reminders=load_private_reminders(config_dir=settings.config_dir),
        allowed_user_ids={settings.owner_qq},
    )
    dispatcher = PrivateMessageDispatcher(
        engine=engine,
        data_dir=settings.data_dir,
        private_chat_service=private_chat_service,
    )

    async def handle_payload(payload: dict) -> None:
        if payload.get("post_type") == "notice" and payload.get("notice_type") == "friend_recall":
            user_id, message_id = int(payload.get("user_id", 0)), str(payload.get("message_id", ""))
            from app.storage.db import session_scope
            from app.storage.repositories import MessageRepository
            with session_scope(engine) as session:
                source = MessageRepository(session).get_by_platform_msg_id(f"private-inbound-{user_id}-{message_id}")
                if source is not None:
                    source.raw_json = {**source.raw_json, "delivery_state": "deleted"}
            if private_chat_service.agent_handler is not None:
                await private_chat_service.agent_handler.cancel_user(user_id)
            return
        if payload.get("post_type") != "message":
            return
        message_type = resolve_message_type(payload)
        if message_type == "group":
            # The OneBot server broadcasts every event to every connected
            # client, so group traffic normally shows up here as well. Only the
            # explicit diagnostic for group 10001 is recorded; everything else
            # stays at DEBUG to keep the private log quiet.
            if int(payload.get("group_id", 0) or 0) == 10001:
                logging.info(
                    "private_process_observed_group_payload group_id=%s msg_id=%s user_id=%s",
                    payload.get("group_id"),
                    payload.get("message_id"),
                    payload.get("user_id"),
                )
            else:
                logging.debug(
                    "inbound_message_ignored process=private message_type=group group_id=%s msg_id=%s",
                    payload.get("group_id"),
                    payload.get("message_id"),
                )
            return
        if message_type != "private":
            # Never drop an inbound message silently: an unexpected
            # message_type means the bridge speaks a different dialect.
            logging.warning(
                "inbound_message_unhandled process=private message_type=%r keys=%s",
                payload.get("message_type"),
                sorted(payload.keys()),
            )
            return

        event = parse_private_message_event(payload)
        await dispatcher.handle_private_message(event)

    logging.info(f"qq-ai-private starting with owner={settings.owner_qq} model={settings.llm_model}")
    gateway_task = asyncio.create_task(gateway.connect_and_consume(handle_payload))
    try:
        await heartbeat.start()
        await _wait_for_gateway_ready(gateway, gateway_task)
        await private_chat_service.start()
        if owner_notice_worker is not None:
            await owner_notice_worker.start()
        await reminder_scheduler.start()
        await gateway_task
    finally:
        gateway_task.cancel()
        await asyncio.gather(gateway_task, return_exceptions=True)
        await reminder_scheduler.stop()
        if owner_notice_worker is not None:
            await owner_notice_worker.stop()
        await private_chat_service.stop()
        await heartbeat.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
