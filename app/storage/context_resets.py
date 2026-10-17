"""Atomic context isolation without deleting canonical messages or write fences."""
from datetime import UTC, datetime
import re
from uuid import uuid4

from sqlalchemy import delete, or_, select

from app.core.memory_source_guard import is_usable_memory_source
from app.storage.models import ContextExcludedMessage, ContextReset, Job, Message


def receipt_identity(receipt):
    return {key: value for key, value in receipt.items() if key in {
        'action_id', 'call_id', 'tool', 'operation', 'status', 'receipt_id', 'replay_allowed'}}


def automatic_context_floor(session, scope):
    """Automatic replay only; explicit history and source masks are independent."""
    return session.scalar(select(ContextReset.automatic_after_id).where(ContextReset.scope == scope)
        .order_by(ContextReset.source_id.desc()).limit(1)) or 0


class ContextResetRepository:
    def __init__(self, engine, *, account_id):
        self.engine, self.account_id = engine, account_id

    def latest_source_id(self, scope):
        from app.storage.db import session_scope
        with session_scope(self.engine) as session:
            return session.scalar(select(ContextReset.source_id).where(
                ContextReset.account_id == self.account_id, ContextReset.scope == scope)
                .order_by(ContextReset.source_id.desc()).limit(1)) or 0

    def reset(self, *, scope, actor_id, source_id, count=None, action='reset'):
        from app.storage.agent_sessions import AgentContextSnapshot, AgentSessionCheckpoint, AgentSessionDependency
        from app.storage.autonomous_agent import AutonomousIntention, AutonomousIntentionWakeup
        from app.storage.db import session_scope
        from app.storage.group_agent import GroupAgentInbox, GroupAgentSession, GroupAgentTurn
        from app.storage.repositories import DevSessionRepository

        match = re.fullmatch(r'(group|private):([1-9][0-9]{0,18})', scope)
        if (not match or action not in {'reset', 'isolate', 'restore'}
                or count is not None and (type(count) is not int or not 1 <= count <= 1000)
                or action != 'reset' and count is not None):
            raise ValueError('invalid_context_reset')
        surface, identifier = match[1], int(match[2])
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql('BEGIN IMMEDIATE')
            prior = session.get(ContextReset, source_id)
            if prior:
                if prior.scope != scope or prior.account_id != self.account_id:
                    raise ValueError('context_reset_scope_mismatch')
                return {'applied': False, 'excluded_count': prior.excluded_count, 'mode': prior.mode}
            source = session.get(Message, source_id)
            if (not is_usable_memory_source(source, author_id=actor_id)
                    or (source.raw_json or {}).get('direction') == 'outbound'
                    or source.group_id != (identifier if surface == 'group' else None)
                    or surface == 'private' and actor_id != identifier):
                raise ValueError('context_reset_source_unavailable')
            if surface == 'group':
                predicate = Message.group_id == identifier
            else:
                predicate = (Message.group_id.is_(None) & or_(
                    (Message.user_id == identifier) & (Message.raw_json['direction'].as_string().is_(None)
                        | (Message.raw_json['direction'].as_string() != 'outbound')),
                    (Message.user_id == self.account_id)
                    & (Message.raw_json['direction'].as_string() == 'outbound')
                    & (Message.raw_json['recipient_user_id'].as_integer() == identifier)))
            # Bound at the committed command: a concurrent newer message is never masked.
            mode = action if action != 'reset' else 'session' if count is None else 'legacy_count'
            target = None
            if action in {'isolate', 'restore'}:
                from app.core.message_content import extract_reply_to_msg_id
                reply_id = extract_reply_to_msg_id((source.raw_json or {}).get('message', []))
                if not reply_id:
                    raise ValueError('context_reset_target_missing')
                identifiers = [reply_id, f'private-inbound-{identifier}-{reply_id}'] if surface == 'private' else [reply_id]
                target = session.scalar(select(Message).where(predicate, Message.id < source_id,
                    or_(Message.platform_msg_id.in_(identifiers),
                        Message.raw_json['message_id'].as_string() == str(reply_id)))
                    .order_by(Message.id.desc()).limit(1))
                if (target is None or session.get(ContextReset, target.id) is not None
                        or action == 'isolate' and not (target.context_excluded or is_usable_memory_source(target))):
                    raise ValueError('context_reset_target_unavailable')
            rows = ([target] if action == 'isolate' else [] if count is None else
                list(session.scalars(select(Message).where(predicate, Message.id < source_id)
                    .order_by(Message.id.desc()).limit(count))))
            # Already masked rows count toward the requested range, without reaching
            # farther back on a repeated reset. Provisional output is never context.
            excluded = [row.id for row in rows if row.context_excluded or is_usable_memory_source(row)]
            now = datetime.now(UTC).replace(tzinfo=None)
            if action == 'restore':
                mask = session.get(ContextExcludedMessage, target.id)
                if mask is not None:
                    session.delete(mask)
            session.add(ContextReset(source_id=source_id, account_id=self.account_id, scope=scope,
                requested_count=count or 0, excluded_count=len(excluded), created_at=now,
                mode=mode, automatic_after_id=source_id if mode == 'session' else 0,
                target_message_id=target.id if target is not None else None))
            session.flush()
            for message_id in [*excluded, source_id]:
                if session.get(ContextExcludedMessage, message_id) is None:
                    session.add(ContextExcludedMessage(message_id=message_id, reset_source_id=source_id))
            scopes = (scope, 'background:' + scope) if surface == 'group' else (scope,)
            checkpoints = tuple(session.scalars(select(AgentSessionCheckpoint).where(
                AgentSessionCheckpoint.account_id == self.account_id,
                AgentSessionCheckpoint.scope.in_(scopes))))
            for row in checkpoints:
                receipts = [receipt_identity(r) for r in row.checkpoint.get('receipts', [])]
                row.items, row.source_message_ids = [], []
                row.checkpoint = {'runtime_id': uuid4().hex, 'receipts': receipts,
                    'last_status': 'context_reset', 'reset_source_id': source_id}
                row.revision += 1
                row.updated_at = now
                session.execute(delete(AgentSessionDependency).where(AgentSessionDependency.session_key == row.key))
            # Snapshots revoke every older child return, even if its source was
            # outside the N masked messages. The action ledger remains authoritative.
            snapshots = tuple(session.scalars(select(AgentContextSnapshot).where(
                AgentContextSnapshot.account_id == self.account_id, AgentContextSnapshot.scope.in_(scopes))))
            for snapshot in snapshots:
                session.execute(delete(AgentSessionDependency).where(
                    AgentSessionDependency.session_key == f'{self.account_id}:snapshot:{snapshot.id}'))
                session.delete(snapshot)
            for row in session.scalars(select(AutonomousIntention).where(
                    AutonomousIntention.account_id == self.account_id, AutonomousIntention.scope.in_(scopes))):
                row.note, row.source_message_ids, row.updated_at = {}, [], now
            for row in session.scalars(select(AutonomousIntentionWakeup).where(
                    AutonomousIntentionWakeup.account_id == self.account_id, AutonomousIntentionWakeup.scope.in_(scopes))):
                row.status, row.next_due, row.lease_until = 'cancelled', None, None
                row.lease_token, row.lease_owner, row.updated_at = '', '', now
                row.revision += 1
            # Restrict candidate jobs by type/key; JSON identity is checked again.
            for job in session.scalars(select(Job).where(Job.status.in_(
                    ('queued', 'running', 'pending', 'result_ready', 'delivering')),
                    or_(Job.job_key.startswith(f'private-agent:{self.account_id}:{identifier}:'),
                        Job.job_key.startswith(f'research:{self.account_id}:{surface}:{identifier}:'),
                        Job.job_key.startswith(f'media:{self.account_id}:{surface}:{identifier}:'),
                        Job.job_key.startswith(f'owner-action:{self.account_id}:'),
                        Job.job_key.startswith(f'owner-notice:{self.account_id}:')))):
                payload = job.payload_json or {}
                matches = (surface == 'private' and job.job_type == 'private_agent_turn'
                    and payload.get('user_id') == identifier and payload.get('account_id') == self.account_id
                    or job.job_type in {'delegated_research', 'agent_media_operation'}
                    and payload.get('account_id') == self.account_id and payload.get('surface') == surface
                    and payload.get('scope_id') == identifier)
                matches = matches and type(payload.get('source_id')) is int and payload['source_id'] <= source_id
                if job.job_type in {'owner_agent_action', 'owner_action_notice'}:
                    origin = session.get(Message, payload.get('source_id')) if type(payload.get('source_id')) is int else None
                    matches = bool(origin and origin.id < source_id and (
                        surface == 'group' and origin.group_id == identifier
                        or surface == 'private' and origin.group_id is None and origin.user_id == identifier))
                if matches:
                    job.status = 'uncertain' if job.status == 'delivering' else 'cancelled'
                    job.last_error_code, job.completed_at = 'context_reset', now
            image_type = 'private_image_generation' if surface == 'private' else 'group_image_generation'
            for job in session.scalars(select(Job).where(Job.job_type == image_type,
                    Job.status.in_(('queued', 'running')))):
                payload = job.payload_json or {}
                if payload.get('user_id' if surface == 'private' else 'group_id') != identifier:
                    continue
                platform_id = payload.get('agent_source_message_id') or (
                    f"private-inbound-{identifier}-{payload.get('trigger_message_id')}" if surface == 'private'
                    else payload.get('trigger_message_id'))
                origin = session.scalar(select(Message).where(Message.platform_msg_id == platform_id))
                if origin is not None and origin.id <= source_id:
                    job.status = 'uncertain' if payload.get('delivery_state') in {'dispatching', 'uncertain'} else 'cancelled'
                    job.last_error_code, job.completed_at = 'context_reset', now
            if surface == 'group':
                row = session.get(GroupAgentSession, identifier)
                if row:
                    row.epoch += 1
                    row.working_note, row.working_note_source_message_ids = '', []
                    row.working_note_updated_at, row.updated_at = None, now
                for inbox in session.scalars(select(GroupAgentInbox).where(
                        GroupAgentInbox.group_id == identifier, GroupAgentInbox.source_message_id <= source_id,
                        GroupAgentInbox.status.in_(('pending', 'processing')))):
                    inbox.status = 'expired'
                    if row:
                        row.cursor = max(row.cursor, inbox.id)
                for turn in session.scalars(select(GroupAgentTurn).where(
                        GroupAgentTurn.group_id == identifier,
                        GroupAgentTurn.status.in_(('deciding', 'ready', 'executing', 'dispatching')))):
                    if any(identifier > source_id for identifier in turn.source_message_ids):
                        continue
                    turn.status = 'uncertain' if turn.status == 'dispatching' else 'expired'
                    turn.reason, turn.updated_at = 'context_reset', now
            else:
                # New daily session excludes the old rolling summary and legacy tasks.
                DevSessionRepository(session).create_owner_session(owner_qq=identifier, session_mode='daily')
            session.flush()
            return {'applied': True, 'excluded_count': len(excluded), 'mode': mode,
                    'reset_at': now.replace(tzinfo=UTC).timestamp(),
                    'target_message_id': target.id if target is not None else None}
