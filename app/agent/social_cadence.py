"""Pure cadence and accounting names for autonomous social opportunities."""
from datetime import datetime, timedelta, time
from hashlib import sha256
import json
from urllib.parse import urlsplit

from app.agent.social_policy import in_quiet_hours
from app.core.time_utils import ASIA_SHANGHAI


def cost_bucket(kind: str, job_kind: str, *, interactive=False, owner=False) -> str:
    if owner:
        return "owner:" + kind
    surface = ("group_interactive" if interactive else "group_autonomous") if job_kind == "group" else job_kind
    return kind + ":" + surface


def next_patrol_due(job, now: datetime, random_seconds, *, morning=False, initial=False) -> datetime:
    """Night and missed overnight schedules move to a jittered morning slot."""
    local = now.astimezone(ASIA_SHANGHAI)
    if initial and job.quiet_hours and not in_quiet_hours(job.quiet_hours, local):
        end = time.fromisoformat(job.quiet_hours[1])
        morning_end = local.replace(hour=end.hour, minute=end.minute, second=end.second, microsecond=0)
        morning = morning_end <= local <= morning_end + timedelta(seconds=job.morning_delay_seconds[1])
    due = local + timedelta(seconds=random_seconds(*job.patrol_interval_seconds))
    if job.quiet_hours and (morning or in_quiet_hours(job.quiet_hours, local) or in_quiet_hours(job.quiet_hours, due)):
        start, end = map(time.fromisoformat, job.quiet_hours)
        end_at = local.replace(hour=end.hour, minute=end.minute, second=end.second, microsecond=0)
        if not morning and (local.time().replace(tzinfo=None) >= start or end_at <= local):
            end_at += timedelta(days=1)
        due = end_at + timedelta(seconds=random_seconds(*job.morning_delay_seconds))
        # Daytime quiet periods also need a future time, never yesterday.
        if due <= local:
            due = local + timedelta(seconds=random_seconds(*job.morning_delay_seconds))
    return due


def post_fingerprint(post) -> str:
    images = [urlsplit(url)._replace(query="", fragment="").geturl() for url in post.images]
    return sha256(json.dumps([post.owner_id, post.post_id, post.text, post.published_at.isoformat(), images],
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
