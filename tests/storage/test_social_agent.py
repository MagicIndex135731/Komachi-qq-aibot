from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.agent.social_contracts import SocialDecision, SocialJob
from app.storage.db import session_scope
from app.storage.social_agent import SocialActionRecord, SocialAgentRepository


ACCOUNT = 900000999
NOW = datetime(2026, 10, 9, 6, tzinfo=UTC)


def publish_job(**kwargs):
    return replace(SocialJob("publish", "qzone_publish", ACCOUNT, interval_seconds=60,
                             cooldown_seconds=60, max_daily=2, quiet_hours=None), **kwargs)


def leased(repository, owner="first", now=NOW):
    return repository.acquire(account_id=ACCOUNT, owner=owner, now=now, lease_seconds=300)


def reserve(repository, lease, job, key, now=NOW, decision=None):
    return repository.reserve_action(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                     job=job, decision=decision or SocialDecision("publish_qzone", "public thought"),
                                     dedupe_key=key, now=now)


def test_account_lease_is_exclusive_and_old_owner_is_fenced(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    first = leased(repository)
    assert leased(repository, "second") is None
    action = reserve(repository, first, publish_job(), "first-write")
    successor = leased(repository, "second", NOW + timedelta(seconds=301))
    assert successor.epoch > first.epoch
    assert not repository.mark_dispatching(account_id=ACCOUNT, owner=first.owner, epoch=first.epoch,
                                           action_id=action, job=publish_job(), now=NOW + timedelta(seconds=301))


def test_due_jobs_are_durable_and_advance_before_external_work(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    job = publish_job()
    assert repository.claim_due_jobs(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                     jobs=(job,), now=NOW) == ()
    future = NOW + timedelta(seconds=60)
    due = repository.claim_due_jobs(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                    jobs=(job,), now=future)
    assert [item.job.key for item in due] == [job.key]
    assert repository.claim_due_jobs(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                     jobs=(job,), now=future) == ()
    repository.release(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch, now=future)
    successor = leased(repository, "successor", future)
    assert repository.claim_due_jobs(account_id=ACCOUNT, owner=successor.owner, epoch=successor.epoch,
                                     jobs=(job,), now=future + timedelta(seconds=59)) == ()
    assert len(repository.claim_due_jobs(account_id=ACCOUNT, owner=successor.owner, epoch=successor.epoch,
                                         jobs=(job,), now=future + timedelta(seconds=60))) == 1


def test_concurrent_quota_reservations_are_atomic(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    job = publish_job(max_daily=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        actions = list(pool.map(lambda key: reserve(repository, lease, job, key), ("one", "two")))
    assert sum(action is not None for action in actions) == 1


def test_comment_source_is_permanently_deduplicated_across_job_names(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    job = SocialJob("comments", "qzone_comment", ACCOUNT, cooldown_seconds=1, quiet_hours=None,
                    friend_ids=(900000101,))
    decision = SocialDecision("comment_qzone", "nice picture", "real-post", 900000101)
    action = reserve(repository, lease, job, "ignored-slot", decision=decision)
    assert action is not None
    repository.finish(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch, action_id=action,
                      status="failed", now=NOW)
    assert reserve(repository, lease, replace(job, key="renamed"), "other-slot",
                   NOW + timedelta(seconds=2), decision) is None


def test_dispatch_recovery_retains_unknown_action_without_replay(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    first = leased(repository)
    job = publish_job()
    action = reserve(repository, first, job, "unique-slot")
    assert repository.mark_dispatching(account_id=ACCOUNT, owner=first.owner, epoch=first.epoch,
                                      action_id=action, job=job, now=NOW)
    successor = leased(repository, "successor", NOW + timedelta(seconds=301))
    with session_scope(sqlite_engine) as session:
        assert session.get(SocialActionRecord, action).status == "uncertain"
    assert reserve(repository, successor, job, "unique-slot", NOW + timedelta(seconds=301)) is None
    repository.finish(account_id=ACCOUNT, owner=first.owner, epoch=first.epoch, action_id=action,
                      status="sent", now=NOW + timedelta(seconds=302), receipt_id="late")
    with session_scope(sqlite_engine) as session:
        assert session.get(SocialActionRecord, action).status == "uncertain"


def test_daily_quota_uses_shanghai_midnight_and_cooldown_survives_restart(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    before = NOW.replace(hour=15, minute=50)
    first = leased(repository, now=before)
    job = publish_job(max_daily=1)
    assert reserve(repository, first, job, "today", before)
    assert reserve(repository, first, job, "same-day", before + timedelta(seconds=61)) is None
    after = before + timedelta(minutes=20)
    successor = leased(repository, "successor", after)
    assert reserve(repository, successor, job, "tomorrow", after)
    assert reserve(repository, successor, replace(job, key="renamed"), "quota-bypass", after) is None


def test_dispatch_rechecks_reduced_quota_and_is_single_write(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    job = publish_job(cooldown_seconds=1)
    first = reserve(repository, lease, job, "first")
    assert repository.mark_dispatching(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                      action_id=first, job=job, now=NOW)
    assert not repository.mark_dispatching(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                           action_id=first, job=job, now=NOW)
    later = NOW + timedelta(seconds=2)
    second = reserve(repository, lease, job, "second", later)
    assert second
    assert not repository.mark_dispatching(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                           action_id=second, job=replace(job, max_daily=1), now=later)


def test_due_claim_has_a_hard_batch_bound(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    jobs = tuple(SocialJob(f"group-{index}", "group", 900001000 + index, interval_seconds=1,
                          quiet_hours=None) for index in range(33))
    assert repository.claim_due_jobs(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                     jobs=jobs, now=NOW) == ()
    due = repository.claim_due_jobs(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                    jobs=jobs, now=NOW + timedelta(seconds=1))
    assert len(due) == 16
    assert len({item.job.key for item in due}) == 16


def test_group_source_is_same_group_and_revalidated_inside_dispatch_transaction(sqlite_engine):
    from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    group_id = 900000001
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=900000101, nickname="synthetic", group_card="")
        row = MessageRepository(session).add_group_message(platform_msg_id="social-real-source", group_id=group_id,
            user_id=900000101, timestamp=NOW, plain_text="ordinary source", raw_json={}, msg_type="text",
            mentioned_bot=False, reply_to_msg_id=None)
        session.flush()
        source_id = row.id
    job = SocialJob("group", "group", group_id, quiet_hours=None)
    decision = SocialDecision("send_group", "a context-grounded topic", source_message_ids=(source_id,))
    assert reserve(repository, lease, replace(job, scope_id=group_id + 1), "cross-group", decision=decision) is None
    action = reserve(repository, lease, job, "same-group", decision=decision)
    assert action
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).mark_group_message_deleted(group_id=group_id, platform_msg_id="social-real-source")
    assert not repository.mark_dispatching(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                           action_id=action, job=job, now=NOW)


def test_midnight_dispatch_is_charged_to_the_actual_shanghai_day(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    before = NOW.replace(hour=15, minute=59, second=59)
    lease = leased(repository, now=before)
    job = publish_job(max_daily=1, cooldown_seconds=1)
    action = reserve(repository, lease, job, "before-midnight", before)
    after = before + timedelta(seconds=2)
    assert repository.mark_dispatching(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                      action_id=action, job=job, now=after)
    assert reserve(repository, lease, job, "same-dispatch-day", after + timedelta(seconds=2)) is None


def test_cooldown_starts_at_real_dispatch_after_slow_preparation(sqlite_engine):
    repository = SocialAgentRepository(sqlite_engine)
    lease = leased(repository)
    job = publish_job()
    action = reserve(repository, lease, job, "slow-first")
    assert repository.mark_dispatching(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
                                      action_id=action, job=job, now=NOW + timedelta(seconds=45))
    assert reserve(repository, lease, job, "too-soon-after-send", NOW + timedelta(seconds=61)) is None


@pytest.mark.parametrize("was_dispatched", [False, True])
def test_lease_recovery_finalizes_its_correlated_provisional_group_output(sqlite_engine, was_dispatched):
    from app.core.memory_source_guard import is_usable_memory_source
    from app.storage.models import Message
    from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

    repository = SocialAgentRepository(sqlite_engine)
    first = leased(repository)
    job = SocialJob("group-recovery", "group", 900000001, quiet_hours=None)
    decision = SocialDecision("send_group", "保留合成输出证据")
    action = reserve(repository, first, job, "crash-slot", decision=decision)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=job.scope_id, group_name="synthetic", enabled=True,
                                              speak_enabled=True)
        UserRepository(session).upsert_user(user_id=ACCOUNT, nickname="synthetic bot", group_card="")
        row = MessageRepository(session).add_group_message(
            platform_msg_id="social:" + action, group_id=job.scope_id, user_id=ACCOUNT, timestamp=NOW,
            plain_text=decision.text, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
            raw_json={"social_action_id": action, "delivery_state": "reserved", "delivery_attempts": 0},
        )
        session.flush()
        message_id = row.id
    if was_dispatched:
        assert repository.mark_dispatching(account_id=ACCOUNT, owner=first.owner, epoch=first.epoch,
                                           action_id=action, job=job, now=NOW)

    future = NOW + timedelta(seconds=301)
    successor = leased(repository, "successor", future)

    with session_scope(sqlite_engine) as session:
        record = session.get(SocialActionRecord, action)
        message = session.get(Message, message_id)
        assert record.status == ("uncertain" if was_dispatched else "failed")
        assert message.raw_json["delivery_state"] == ("uncertain" if was_dispatched else "deleted")
        assert message.raw_json["delivery_attempts"] == (1 if was_dispatched else 0)
        assert message.plain_text == decision.text
        assert not is_usable_memory_source(message, group_id=job.scope_id)
    assert reserve(repository, successor, job, "crash-slot", future, decision) is None


def test_recovery_preserves_an_already_confirmed_qq_output(sqlite_engine):
    from app.core.memory_source_guard import is_usable_memory_source
    from app.storage.models import Message
    from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

    repository = SocialAgentRepository(sqlite_engine)
    first = leased(repository)
    job = SocialJob("group-confirmed-recovery", "group", 900000001, quiet_hours=None)
    decision = SocialDecision("send_group", "已确认的合成输出")
    action = reserve(repository, first, job, "confirmed-crash-slot", decision=decision)
    assert repository.mark_dispatching(account_id=ACCOUNT, owner=first.owner, epoch=first.epoch,
                                       action_id=action, job=job, now=NOW)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=job.scope_id, group_name="synthetic", enabled=True,
                                              speak_enabled=True)
        UserRepository(session).upsert_user(user_id=ACCOUNT, nickname="synthetic bot", group_card="")
        message = MessageRepository(session).add_group_message(
            platform_msg_id="actual-confirmed-qq-receipt", group_id=job.scope_id, user_id=ACCOUNT, timestamp=NOW,
            plain_text=decision.text, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
            raw_json={"social_action_id": action, "delivery_state": "sent", "delivery_attempts": 1},
        )
        session.flush()
        message_id = message.id

    assert leased(repository, "successor", NOW + timedelta(seconds=301))

    with session_scope(sqlite_engine) as session:
        message = session.get(Message, message_id)
        assert session.get(SocialActionRecord, action).status == "uncertain"
        assert message.platform_msg_id == "actual-confirmed-qq-receipt"
        assert message.raw_json["delivery_state"] == "sent"
        assert message.plain_text == decision.text
        assert is_usable_memory_source(message, group_id=job.scope_id)
