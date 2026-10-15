from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from app.storage.proactive import ProactiveLimits, ProactiveRepository

ACCOUNT, GROUP = 900000101, 900000001


def setup_repo(engine):
    instant = [datetime(2026, 10, 15, tzinfo=UTC)]
    repo = ProactiveRepository(engine, account_id=ACCOUNT, clock=lambda: instant[0])
    return repo, instant


def claim(repo, start=1, *, count=8, **kwargs):
    return repo.claim(GROUP, candidates=[(i, str(i)) for i in range(start, start + count)],
        bootstrap_cursor=start-1, max_source_id=start+count-1, **kwargs)


def test_silent_cursor_advances_only_captured_messages_and_survives_restart(sqlite_engine):
    repo, now = setup_repo(sqlite_engine)
    permit, _, _ = claim(repo)
    assert permit and repo.reserve(permit, 'model')
    repo.finish(permit, status='silent')
    repo = ProactiveRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: now[0])
    now[0] += timedelta(minutes=10)
    assert claim(repo, 9, count=7)[1] == 'insufficient_new_messages'
    assert claim(repo, 9)[0]


def test_model_budget_includes_followups_retries_and_compaction(sqlite_engine):
    repo, now = setup_repo(sqlite_engine)
    permit, _, _ = claim(repo)
    assert [repo.reserve(permit, 'model') for _ in range(7)] == [True] * 6 + [False]
    repo.finish(permit, status='failed')
    now[0] += timedelta(minutes=20)
    assert claim(repo, 9)[1] == 'proactive_model_limit'
    now[0] += timedelta(minutes=41)
    assert claim(repo, 9)[0]


def test_ordinary_timer_or_duplicate_text_cannot_wake_model(sqlite_engine):
    repo, _ = setup_repo(sqlite_engine)
    assert claim(repo, count=1)[0] is None
    assert claim(repo, count=2)[0] is None
    assert repo.claim(GROUP, candidates=[(i, 'same') for i in range(1, 31)],
        bootstrap_cursor=0, max_source_id=30)[0] is None


def test_silence_backoff_and_failure_does_not_consume_sources(sqlite_engine):
    repo, now = setup_repo(sqlite_engine)
    for iteration, minutes in enumerate([10, 20, 30, 30]):
        permit, _, _ = claim(repo, 1 + 8*iteration)
        assert permit
        repo.finish(permit, status='silent')
        now[0] += timedelta(minutes=minutes - 1)
        assert claim(repo, 9 + 8*iteration)[1] == 'observation_backoff'
        now[0] += timedelta(minutes=1)
    permit, _, _ = claim(repo, 33)
    repo.finish(permit, status='failed')
    now[0] += timedelta(minutes=31)
    assert claim(repo, 33)[0]


def test_concurrent_foreground_and_patrol_only_one_claim(sqlite_engine):
    repo, _ = setup_repo(sqlite_engine)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: claim(repo)[0], range(4)))
    assert sum(value is not None for value in results) == 1


def test_speech_budget_is_shared_across_turns_and_chunks(sqlite_engine):
    repo, now = setup_repo(sqlite_engine)
    for start in [1, 9, 17]:
        permit, _, _ = claim(repo, start)
        assert repo.reserve(permit, 'speech')
        assert repo.reserve(permit, 'speech')  # same intervention
        repo.finish(permit, status='sent')
        now[0] += timedelta(minutes=5)
        assert claim(repo, start+8)[1] in {'proactive_speech_cooldown', 'proactive_speech_limit'}
        now[0] += timedelta(minutes=5)
    assert claim(repo, 25)[1] == 'proactive_speech_limit'


def test_expired_lease_and_group_isolation(sqlite_engine):
    repo, now = setup_repo(sqlite_engine)
    permit, _, _ = claim(repo)
    now[0] += timedelta(minutes=16)
    assert not repo.reserve(permit, 'model')
    another, _, _ = repo.claim(GROUP+1, candidates=[(i, str(i)) for i in range(1, 9)],
        bootstrap_cursor=0, max_source_id=8)
    assert another


def test_due_goal_can_run_without_new_messages_but_not_bypass_budget(sqlite_engine):
    repo, _ = setup_repo(sqlite_engine)
    permit, _, _ = claim(repo, count=0, due_goal=True)
    assert permit
    assert repo.reserve(permit, 'model', limits=replace(ProactiveLimits(), hourly_models=1))
    assert not repo.reserve(permit, 'model', limits=replace(ProactiveLimits(), hourly_models=1))


def test_real_due_goal_is_not_delayed_by_silent_observation_backoff(sqlite_engine):
    repo, _ = setup_repo(sqlite_engine)
    permit, _, _ = claim(repo, count=0, due_goal=True)
    repo.finish(permit, status='silent')
    assert claim(repo, count=0)[1] == 'observation_backoff'
    assert claim(repo, count=0, due_goal=True)[0]


def test_busy_hour_stays_bounded_without_spending_a_model_on_local_checks(sqlite_engine):
    repo, now = setup_repo(sqlite_engine)
    requests = admissions = 0
    for minute in range(60):
        end = 8 * (minute + 1)
        permit, _, _ = repo.claim(GROUP, candidates=[(i, str(i)) for i in range(max(1, end-29), end+1)],
            bootstrap_cursor=0, max_source_id=end)
        if permit:
            admissions += 1
            requests += int(repo.reserve(permit, 'model'))
            repo.finish(permit, status='silent')
        now[0] += timedelta(minutes=1)
    assert requests == admissions == 3
