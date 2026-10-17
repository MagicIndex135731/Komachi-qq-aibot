"""Explicit per-surface social permissions, independent from chat permissions."""

from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING

from app.agent.social_contracts import SocialJob
from app.core.time_utils import ASIA_SHANGHAI

if TYPE_CHECKING:
    from app.config import RuntimeConfig


def in_quiet_hours(hours: tuple[str, str] | None, now: datetime) -> bool:
    if hours is None:
        return False
    start, end = (value if isinstance(value, time) else time.fromisoformat(value) for value in hours)
    if start == end:
        return False
    local = now.replace(tzinfo=ASIA_SHANGHAI) if now.tzinfo is None else now.astimezone(ASIA_SHANGHAI)
    current = local.time().replace(tzinfo=None)
    return start <= current < end if start < end else current >= start or current < end


def _mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _enabled(value: dict) -> bool:
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("social enabled must be a boolean")
    return enabled


def autonomous_social_enabled(runtime: RuntimeConfig, kind: str) -> bool:
    """Keep autonomous Qzone permission separate from owner-directed capability."""
    name = {'qzone_publish': 'publish', 'qzone_comment': 'comment'}.get(kind)
    if name is None:
        return True
    qzone = _mapping(runtime.social_policy.get('qzone', {}), 'qzone policy')
    config = _mapping(qzone.get(name, {}), f'qzone {name}')
    enabled = config.get('autonomous_enabled', True)
    if type(enabled) is not bool:
        raise ValueError('qzone autonomous_enabled must be a boolean')
    return enabled


def _number(value: dict, key: str, default: float, *, minimum=60, maximum=604800) -> float:
    import math

    number = value.get(key, default)
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError(f"{key} must be numeric")
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{key} outside permitted range")
    return float(number)


def _job(*, key: str, kind, scope_id: int, config: dict) -> SocialJob:
    quiet = config.get("quiet_hours", ["23:00", "08:00"])
    if quiet is not None and (not isinstance(quiet, list) or len(quiet) != 2 or
                              any(not isinstance(value, str) for value in quiet)):
        raise ValueError("quiet_hours must be two time strings or null")
    friends = config.get("friend_ids", [])
    if not isinstance(friends, list) or any(type(value) is not int or value <= 0 for value in friends):
        raise ValueError("friend_ids must be positive integer IDs")
    topics = config.get("topics", [])
    if not isinstance(topics, list) or any(not isinstance(value, str) for value in topics):
        raise ValueError("topics must be strings")
    daily = config.get("max_daily", 2 if kind != "qzone_comment" else 5)
    if type(daily) is not int:
        raise ValueError("max_daily must be an integer")
    persona_key = config.get("persona", "default")
    if not isinstance(persona_key, str) or not persona_key or len(persona_key) > 64:
        raise ValueError("invalid social persona key")
    return SocialJob(
        key=key, kind=kind, scope_id=scope_id,
        interval_seconds=_number(config, "interval_seconds", 1800 if kind == "group" else 3600),
        cooldown_seconds=_number(config, "cooldown_seconds", 3600),
        max_daily=daily,
        min_idle_seconds=_number(config, "min_idle_seconds", 1800, minimum=0),
        max_post_age_seconds=_number(config, "max_post_age_seconds", 86400),
        persona_key=persona_key, public_topics=tuple(topics), friend_ids=tuple(dict.fromkeys(friends)),
        visibility=config.get("visibility", 4), quiet_hours=tuple(quiet) if quiet is not None else None,
        autonomous_tools=config.get("autonomous_tools", False), allow_images=config.get("allow_images", False),
        identity_tools=tuple(config.get("identity_tools", [])), max_steps=config.get("max_steps", 8),
        max_model_calls_daily=config.get("max_model_calls_daily", 60), max_image_calls_daily=config.get("max_image_calls_daily", 2),
        all_friends=config.get("all_friends", False),
        max_daily_per_friend=config.get("max_daily_per_friend", 0),
    )


def configured_social_jobs(runtime: RuntimeConfig) -> tuple[SocialJob, ...]:
    policy = _mapping(getattr(runtime, "social_policy", {}), "social policy")
    enabled = _enabled(policy)
    if not enabled or not runtime.settings.social_agent_enabled:
        return ()
    jobs = []
    groups = _mapping(policy.get("groups", {}), "social groups")
    if len(groups) > 32:
        raise ValueError("at most 32 social groups are supported")
    defaults = runtime.group_policy.get("default_group_behavior", {})
    configured_groups = runtime.group_policy.get("groups", {})
    for group_id, raw in groups.items():
        config = _mapping(raw, "social group")
        if not str(group_id).isdigit() or int(group_id) <= 0:
            raise ValueError("invalid social group ID")
        if not _enabled(config):
            continue
        group = {**defaults, **configured_groups.get(str(group_id), {})}
        if all(group.get(name, False) is True for name in ("enabled", "speak", "proactive_reply")):
            jobs.append(_job(key=f"group:{int(group_id)}", kind="group", scope_id=int(group_id), config=config))
    qzone = _mapping(policy.get("qzone", {}), "qzone policy")
    patrol = _mapping(qzone.get("patrol", {}), "qzone patrol")
    for name, kind in (("publish", "qzone_publish"), ("comment", "qzone_comment")):
        config = _mapping(qzone.get(name, {}), f"qzone {name}")
        if _enabled(config):
            autonomous_social_enabled(runtime, kind)
            job = _job(key=kind, kind=kind, scope_id=runtime.settings.bot_qq, config=config)
            if _enabled(patrol):
                from dataclasses import replace
                interval = patrol.get("interval_seconds", [5400, 10800])
                morning = patrol.get("morning_delay_seconds", [1800, 5400])
                if not isinstance(interval, list) or not isinstance(morning, list):
                    raise ValueError("patrol intervals must be lists")
                job = replace(job, autonomous_tools=True, patrol_interval_seconds=tuple(interval),
                    morning_delay_seconds=tuple(morning),
                    shared_cooldown_seconds=patrol.get("shared_cooldown_seconds", 3600))
            if kind == "qzone_comment" and not job.friend_ids and not job.all_friends:
                raise ValueError("autonomous comments require a friend allowlist")
            if kind == "qzone_publish" and not job.public_topics:
                raise ValueError("autonomous publishing requires public topics")
            jobs.append(job)
    profile = _mapping(policy.get("profile", {}), "profile policy")
    if _enabled(profile):
        job = _job(key="profile", kind="profile", scope_id=runtime.settings.bot_qq, config=profile)
        if not job.autonomous_tools or not job.identity_tools or "card" in job.identity_tools:
            raise ValueError("profile jobs require autonomous nickname/avatar tools; cards belong to group jobs")
        jobs.append(job)
    if getattr(runtime.settings, "agent_runtime", "legacy") == "harness":
        from dataclasses import replace
        jobs = [replace(job, autonomous_tools=True, chat_cadence_only=job.kind == "group") for job in jobs]
    return tuple(jobs)


def configured_agent_jobs(runtime: RuntimeConfig) -> tuple[SocialJob, ...]:
    """Authorized surfaces plus dormant group goals; no invented idle events."""
    jobs = configured_social_jobs(runtime)
    if runtime.settings.agent_runtime != "harness" or not runtime.settings.group_agent_enabled:
        return jobs
    existing = {job.key for job in jobs}
    defaults = runtime.group_policy.get("default_group_behavior", {})
    goals = []
    for raw_id, raw in runtime.group_policy.get("groups", {}).items():
        if not str(raw_id).isdigit() or int(raw_id) <= 0:
            continue
        group = {**defaults, **raw}
        key = f"group:{int(raw_id)}"
        if key in existing or not all(group.get(flag) is True for flag in ("enabled", "speak")):
            continue
        goals.append(SocialJob(key, "group", int(raw_id), autonomous_tools=True,
            chat_cadence_only=True,
            event_goals_only=True, allow_images=True, quiet_hours=None, min_idle_seconds=0,
            max_steps=16, max_daily=50, max_model_calls_daily=500, max_image_calls_daily=10))
    return (*jobs, *goals)


def configured_scheduled_jobs(runtime: RuntimeConfig) -> tuple[SocialJob, ...]:
    """Scheduled work excludes disabled autonomy; owner tools retain capabilities."""
    return tuple(job for job in configured_agent_jobs(runtime)
                 if autonomous_social_enabled(runtime, job.kind))


async def resolve_friend_ids(job: SocialJob, qzone_adapter) -> tuple[int, ...]:
    """A fresh gateway snapshot; never cache failed lookups as permissions."""
    return await qzone_adapter.get_friend_ids() if job.all_friends else job.friend_ids


def friend_authorized(job: SocialJob, owner_id: int, snapshot=()) -> bool:
    return type(owner_id) is int and owner_id > 0 and owner_id in (snapshot if job.all_friends else job.friend_ids)
