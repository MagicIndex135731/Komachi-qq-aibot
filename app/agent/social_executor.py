"""Execute a scheduled action after rechecking its current authority and source."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from datetime import UTC, datetime
import logging
from pathlib import Path

from sqlalchemy import func, and_, select

from app.adapters.outbound_guard import GroupDispatchScope, group_dispatch_scope
from app.adapters.sender import OutboundMessage, QQMessageDeliveryUncertainError
from app.adapters.qzone import QzoneRejectedError, QzoneUnavailableError
from app.agent.social_contracts import SocialDecision, SocialObservation, SocialResult
from app.agent.social_policy import (autonomous_social_enabled, configured_social_jobs,
                                     in_quiet_hours, friend_authorized, resolve_friend_ids)
from app.core.memory_source_guard import is_usable_memory_source
from app.core.time_utils import ASIA_SHANGHAI
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository
from app.storage.social_agent import social_provisional_output_clause


logger = logging.getLogger(__name__)


class SocialExecutor:
    def __init__(self, *, router, sender, qzone_adapter, clock=None) -> None:
        self.router = router
        self.sender = sender
        self.qzone = qzone_adapter
        self.clock = clock or (lambda: datetime.now(UTC))

    def policy_check(self, observation: SocialObservation, decision: SocialDecision, *, reserved_action_id=None, read_only=False,
                     owner_authority=None) -> bool:
        try:
            current = next((job for job in configured_social_jobs(self.router.runtime)
                            if job.key == observation.job.key), None)
        except (ValueError, TypeError):
            return False
        now = self.clock()
        manual = owner_authority is not None and owner_authority() is True
        if not manual and not autonomous_social_enabled(self.router.runtime, observation.job.kind):
            return False
        if current != observation.job or observation.suppressed_reason or (not manual and in_quiet_hours(current.quiet_hours, now)):
            return False
        expected = {"group": "send_group", "qzone_publish": "publish_qzone", "qzone_comment": "comment_qzone"}
        if decision.action != expected[current.kind] or not decision.text.strip() or len(decision.text) > 1000:
            return False
        if decision.image_files:
            if not current.allow_images or len(decision.image_files) > 2:
                return False
            root = (self.router.runtime.settings.data_dir / "autonomous_artifacts").resolve()
            if any(not Path(name).resolve().is_relative_to(root) or not Path(name).is_file() for name in decision.image_files):
                return False
        if current.kind == "qzone_publish":
            return not decision.source_message_ids and not observation.source_message_ids and not observation.context
        if current.kind == "qzone_comment":
            return (not decision.source_message_ids and not observation.source_message_ids and not observation.context
                    and friend_authorized(current, decision.owner_id, observation.authorized_friend_ids)
                    and decision.owner_id != self.router.runtime.settings.bot_qq
                    and any(post.owner_id == decision.owner_id and post.post_id == decision.post_id
                            for post in observation.posts))
        if "[CQ:" in decision.text or tuple(decision.source_message_ids) != observation.source_message_ids:
            return False
        with session_scope(self.router.engine) as session:
            messages = MessageRepository(session)
            policy = self.router._resolve_group_policy(groups=GroupRepository(session), group_id=current.scope_id)
            enabled, speak, proactive, interval, quiet, _ = policy
            if not all((enabled, speak, proactive)) or in_quiet_hours(quiet, now):
                return False
            sources = [session.get(Message, source_id) for source_id in observation.source_message_ids]
            if not all(is_usable_memory_source(item, group_id=current.scope_id) for item in sources):
                return False
            # Any newly arrived message revokes an idle-topic plan before dispatch.
            latest_query = session.query(Message).filter(Message.group_id == current.scope_id)
            latest_query = latest_query.filter(~social_provisional_output_clause(self.router.runtime.settings.bot_qq))
            if observation.own_output_action_ids:
                # Only this live wake's durably fenced effects are its own
                # arrivals. A new user message or another consumer still revokes it.
                latest_query = latest_query.filter(~and_(
                    Message.user_id == self.router.runtime.settings.bot_qq,
                    func.coalesce(func.json_extract(Message.raw_json, "$.social_action_id"), "").in_(observation.own_output_action_ids)))
            if reserved_action_id is not None:
                latest_query = latest_query.filter(Message.platform_msg_id != "social:" + str(reserved_action_id))
            # Arrival order is the durable canonical ID, independent of an
            # event's clock. Backfill and delayed events can carry older stamps.
            newest_arrival = latest_query.order_by(Message.id.desc()).first()
            if newest_arrival is not None and int(newest_arrival.id) not in observation.source_message_ids:
                return False
            latest = latest_query.order_by(
                Message.timestamp.desc(), Message.id.desc()).first()
            if latest is not None:
                if int(latest.id) not in observation.source_message_ids:
                    return False
                stamp = latest.timestamp.replace(tzinfo=ASIA_SHANGHAI) if latest.timestamp.tzinfo is None else latest.timestamp
                if not read_only and (now - stamp).total_seconds() < current.min_idle_seconds:
                    return False
            bot_reply = messages.last_bot_reply_at(group_id=current.scope_id,
                                                  bot_user_id=self.router.runtime.settings.bot_qq)
            if bot_reply is not None and not read_only:
                stamp = bot_reply.replace(tzinfo=ASIA_SHANGHAI) if bot_reply.tzinfo is None else bot_reply
                if self.router.runtime.settings.agent_runtime == "harness":
                    from app.agent.observation import GroupObservationService
                    if not GroupObservationService(self.router, clock=self.clock).proactive_send_allowed(
                            current.scope_id, current.key):
                        return False
                elif (now - stamp).total_seconds() < max(float(interval[0]), current.cooldown_seconds):
                    return False
        return True

    async def execute(self, observation: SocialObservation, decision: SocialDecision, guard) -> SocialResult:
        guard.require_valid()
        owner_authority = getattr(guard, "owner_authority", None)
        if not self.policy_check(observation, decision, owner_authority=owner_authority):
            return SocialResult("blocked", reason="authority_or_source_revoked")
        if decision.action == "send_group":
            return await self._send_group(observation, decision, guard)
        if decision.action == "comment_qzone":
            if observation.job.all_friends:
                friends = await resolve_friend_ids(observation.job, self.qzone)
                if decision.owner_id not in friends:
                    return SocialResult("blocked", reason="friend_authority_revoked")
                observation = replace(observation, authorized_friend_ids=friends)
            original = next(post for post in observation.posts if post.owner_id == decision.owner_id
                            and post.post_id == decision.post_id)
            current_posts = await self.qzone.get_posts(original.owner_id, count=20)
            now = self.clock()
            current = next((post for post in current_posts if post.owner_id == original.owner_id
                            and post.post_id == original.post_id), None)
            manual = owner_authority is not None and owner_authority() is True
            age = (now - original.published_at).total_seconds()
            # Keep the owner-directed read and final write policies aligned;
            # friendship, source equality and the live dispatch fence still apply.
            if current != original or age < 0 or (not manual and age > observation.job.max_post_age_seconds):
                return SocialResult("blocked", reason="post_changed_or_unavailable")
        guard.require_valid()
        if not self.policy_check(observation, decision, owner_authority=owner_authority):
            return SocialResult("blocked", reason="authority_revoked")
        # The adapter invokes this guard after its last preflight read, at the
        # real write boundary. A reconnect during a read cannot revive a plan.
        try:
            if decision.action == "publish_qzone":
                receipt = await self.qzone.publish(decision.text, visibility=observation.job.visibility,
                                                  images=tuple("base64://" + base64.b64encode(Path(name).read_bytes()).decode("ascii") for name in decision.image_files),
                                                  before_dispatch=guard.on_dispatch)
            else:
                receipt = await self.qzone.comment(original, decision.text,
                    images=tuple("base64://" + base64.b64encode(Path(name).read_bytes()).decode("ascii") for name in decision.image_files),
                    before_dispatch=guard.on_dispatch,
                    **({"require_friend": True} if observation.job.all_friends else {}))
        except (QzoneRejectedError, QzoneUnavailableError):
            return SocialResult("failed", reason="qzone_definite_rejection")
        return SocialResult("sent", receipt_id=receipt)

    def reply_delivery_state(self, group_id: int, reply_source_id: int) -> str | None:
        """Fence only the exact answered source, never every observed message."""
        if type(reply_source_id) is not int or reply_source_id <= 0:
            return None
        from app.storage.agent_effects import AgentToolEffect
        with session_scope(self.router.engine) as session:
            rows = session.scalars(select(Message).where(Message.group_id == group_id,
                Message.user_id == self.router.runtime.settings.bot_qq,
                func.json_extract(Message.raw_json, '$.reply_source_id') == reply_source_id))
            states = set()
            for row in rows:
                raw = row.raw_json or {}
                state = raw.get('delivery_state')
                if state == 'sent':
                    states.add('sent')
                elif state == 'uncertain':
                    states.add('uncertain')
                elif state == 'reserved':
                    effect = session.get(AgentToolEffect, raw['social_action_id']) if raw.get('social_action_id') else None
                    if effect is not None and effect.status in {'dispatching', 'uncertain', 'sent'}:
                        states.add('uncertain')
            return 'uncertain' if 'uncertain' in states else 'sent' if 'sent' in states else None

    def _reserve_group_output(self, observation, decision, action_id: str, reply_source_id=None) -> str | None:
        provisional_id = "social:" + str(action_id)
        with session_scope(self.router.engine) as session:
            messages = MessageRepository(session)
            if messages.get_by_platform_msg_id(provisional_id) is not None:
                return None
            group_id = observation.job.scope_id
            if reply_source_id is not None:
                source = session.get(Message, reply_source_id) if type(reply_source_id) is int else None
                if (reply_source_id not in observation.source_message_ids or source is None
                        or source.user_id == self.router.runtime.settings.bot_qq
                        or not is_usable_memory_source(source, group_id=group_id)):
                    raise ValueError('reply_source_unavailable')
            groups = GroupRepository(session)
            if groups.get_group(group_id) is None:
                groups.upsert_group(group_id=group_id, group_name="", enabled=True, speak_enabled=True)
            UserRepository(session).upsert_user(user_id=self.router.runtime.settings.bot_qq,
                                               nickname=str(self.router.runtime.persona.get("name", "小町")), group_card="")
            messages.add_group_message(
                platform_msg_id=provisional_id, group_id=group_id, user_id=self.router.runtime.settings.bot_qq,
                timestamp=self.clock(), plain_text=decision.text, msg_type="text", reply_to_msg_id=None,
                mentioned_bot=False, raw_json={"post_type": "message", "message_type": "group",
                    "social_action_id": str(action_id), "delivery_state": "reserved", "delivery_attempts": 0,
                    "source_message_ids": list(observation.source_message_ids),
                    **({'reply_source_id': reply_source_id} if reply_source_id is not None else {})},
            )
            row = messages.get_by_platform_msg_id(provisional_id)
            if decision.image_files:
                row.msg_type = "image"
                row.raw_json = {**row.raw_json, "message": [{"type": "text", "data": {"text": decision.text}},
                    *[{"type": "image", "data": {"local_path": name, "origin": "autonomous_generated"}} for name in decision.image_files]]}
        return provisional_id

    def _finish_group_output(self, provisional_id: str, *, state: str, receipt: str | None = None) -> None:
        with session_scope(self.router.engine) as session:
            messages = MessageRepository(session)
            row = messages.get_by_platform_msg_id(provisional_id)
            if row is None:
                raise RuntimeError("scheduled outbound reservation missing")
            payload = {**row.raw_json, "delivery_state": state, "delivery_attempts": 1}
            if receipt:
                echo = messages.get_by_platform_msg_id(receipt)
                if echo is not None and echo.id != row.id:
                    if echo.group_id != row.group_id or echo.user_id != row.user_id:
                        raise RuntimeError("scheduled receipt belongs to a different scope")
                    echo.raw_json = {**echo.raw_json, **payload}
                    row.raw_json = {**payload, "delivery_state": "deleted", "canonical_platform_msg_id": receipt}
                    row.plain_text = ""
                    return
                row.platform_msg_id = receipt
            if state == "uncertain":
                payload.update(failure_kind="delivery_result_unknown", delivery_reason="gateway_ack_timeout")
            row.raw_json = payload

    async def _send_group(self, observation, decision, guard, *, max_text_chars=300, as_forward=False, video_file=None,
                          reply_source_id=None, media_files=()) -> SocialResult:
        # The model can produce one bounded text action, never CQ control segments.
        if max_text_chars not in {300, 12000} or len(decision.text) > max_text_chars:
            return SocialResult("blocked", reason="group_text_too_long")
        provisional_id = self._reserve_group_output(observation, decision, guard.action_id, reply_source_id)
        if provisional_id is None:
            return SocialResult("blocked", reason="duplicate_group_action")
        if media_files:
            with session_scope(self.router.engine) as session:
                row = MessageRepository(session).get_by_platform_msg_id(provisional_id)
                row.msg_type = 'forward'
                row.raw_json = {**row.raw_json, 'message': [{'type': 'text', 'data': {'text': decision.text}},
                    *[{'type': 'node', 'data': {'uin': str(self.router.runtime.settings.bot_qq),
                        'content': [{'type': kind, 'data': {'local_path': path, 'origin': 'web_download'}}]}}
                      for kind, path in media_files]]}
        elif video_file:
            with session_scope(self.router.engine) as session:
                row = MessageRepository(session).get_by_platform_msg_id(provisional_id)
                video = {'type': 'video', 'data': {'local_path': str(video_file), 'origin': 'web_page_video'}}
                row.msg_type = 'forward' if decision.text else 'video'
                row.raw_json = {**row.raw_json, 'message': ([
                    {'type': 'node', 'data': {'uin': str(self.router.runtime.settings.bot_qq),
                        'content': [{'type': 'text', 'data': {'text': decision.text}}]}},
                    {'type': 'node', 'data': {'uin': str(self.router.runtime.settings.bot_qq), 'content': [video]}}]
                    if decision.text else [video])}
        scope = GroupDispatchScope(observation.job.scope_id,
                                   lambda: guard.valid() and self.policy_check(observation, decision,
                                                                            reserved_action_id=guard.action_id),
                                   guard.on_dispatch)
        # Our provisional row must not turn its own idle/cooldown checks false.
        try:
            with group_dispatch_scope(scope):
                if media_files:
                    receipts = await self.sender.send_media_bundle(group_id=observation.job.scope_id,
                        self_id=self.router.runtime.settings.bot_qq, name=self.router.runtime.persona.get('name', '小町'),
                        text=decision.text, media_files=media_files)
                elif video_file:
                    receipts = await self.sender.send_video(group_id=observation.job.scope_id,
                        text=decision.text, video_file=str(video_file), self_id=self.router.runtime.settings.bot_qq,
                        name=self.router.runtime.persona.get('name', '小町'))
                elif as_forward:
                    receipts = await self.sender.send_image_bundle(group_id=observation.job.scope_id,
                        self_id=self.router.runtime.settings.bot_qq, name=self.router.runtime.persona.get('name', '小町'),
                        text=decision.text, image_files=decision.image_files)
                elif decision.image_files:
                    caption_options = {'max_text_chars': max_text_chars} if max_text_chars != 300 else {}
                    receipts = await self.sender.send_group_media(group_id=observation.job.scope_id, text=decision.text,
                        image_files=decision.image_files, **caption_options)
                else:
                    receipts = await self.sender.send_group_text(OutboundMessage(observation.job.scope_id, decision.text))
            if not receipts:
                raise QQMessageDeliveryUncertainError("scheduled send has no receipt")
            self._finish_group_output(provisional_id, state="sent", receipt=receipts[0])
            return SocialResult("sent", receipt_id=receipts[0])
        except (QQMessageDeliveryUncertainError, asyncio.CancelledError):
            self._finish_group_output(provisional_id, state="uncertain")
            return SocialResult("uncertain", reason="delivery_result_unknown")
        except Exception as exc:
            state = "uncertain" if scope.dispatched and not scope.failed else "deleted"
            self._finish_group_output(provisional_id, state=state)
            logger.warning("social_group_send_failed error_type=%s state=%s", type(exc).__name__, state)
            from app.adapters.sender import QQMessageRejectedError
            reason = exc.code if isinstance(exc, QQMessageRejectedError) else 'group_send_failed'
            return SocialResult("uncertain" if state == "uncertain" else "failed", reason=reason)
