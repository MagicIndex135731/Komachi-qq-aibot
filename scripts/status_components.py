"""Low-cost component status probe for the running Xiaomachi container.

The default mode is read-only and spends no model tokens. ``--deep`` makes one
bounded Responses request, validates its persona YAML, and writes only to a
temporary directory (never to the production persona or database).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import yaml

from app.config import AppSettings
from app.core.persona_live_sync import (
    PERSONA_REFRESH_REASONING_EFFORT,
    _normalize_live_profile_contract,
    write_live_profile,
)
from app.core.style_distill import parse_persona_yaml
from app.providers.llm_client import LlmClient, LlmUsage


PROBE_OUTPUT_TOKEN_CAP = 1200
PROBE_PROMPT = (
    "你是人格画像更新器。这是独立的合成健康检查，不涉及真实用户。"
    "现有画像：名字=状态探针，身份=测试群友；新发言：我喜欢喝茶，说话简短。"
    "只返回一个 YAML 映射，不要解释，必须包含以下字段且类型严格正确："
    "name: 字符串；identity: 字符串；core_traits: 字符串列表；"
    "speaking_style: 映射；self_concept: 字符串；speech_habits: 字符串列表；"
    "style_avoid: 字符串列表；relationships: 列表；address_rules: 列表。"
    "内容尽量短，不要输出其他字段。"
)


def _print(status: str, component: str, detail: str) -> None:
    print(f"{status:5} {component}: {detail}", flush=True)


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    # Only call with literals below, never user-provided table names.
    return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _check_memory_episode_backlog(
    db: sqlite3.Connection, *, tables: set[str], now: datetime | None = None,
) -> None:
    """Show actionable current episode work without affecting health exit status."""
    group_join = ""
    group_filter = ""
    if "groups" in tables:
        group_join = "JOIN groups g ON g.group_id=e.group_id "
        group_filter = "AND g.enabled=1 "
    now_local = (now or datetime.now(UTC)).astimezone(
        ZoneInfo("Asia/Shanghai")
    ).replace(tzinfo=None)
    queued, running, failed, overdue = db.execute(
        "SELECT "
        "COALESCE(SUM(j.status='queued'),0), "
        "COALESCE(SUM(j.status='running'),0), "
        "COALESCE(SUM(j.status='failed'),0), "
        "COALESCE(SUM(j.status='failed' "
        "AND j.completed_at < datetime(?, '-1 day')),0) "
        "FROM jobs j JOIN conversation_episodes e "
        "ON e.id=CAST(json_extract(j.payload_json,'$.episode_id') AS INTEGER) "
        + group_join +
        "WHERE j.job_type='memory_episode_process' "
        "AND j.status IN ('queued','running','failed') "
        "AND e.status IN ('closed','processing','failed') "
        "AND e.is_current=1 AND e.compaction_version=j.target_generation "
        "AND e.group_id=CAST(json_extract(j.payload_json,'$.group_id') AS INTEGER) "
        + group_filter,
        (now_local.isoformat(sep=" "),),
    ).fetchone()
    counts = (int(queued), int(running), int(failed), int(overdue))
    _print(
        "WARN" if any(counts) else "OK",
        "memory_episode_backlog",
        "queued={} running={} failed={} overdue={}".format(*counts),
    )


def check_storage(settings: AppSettings) -> bool:
    path = settings.sqlite_path
    if not path.is_file():
        _print("FAIL", "storage", "database missing")
        return False
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5) as db:
            db.execute("PRAGMA query_only=ON")
            integrity = db.execute("PRAGMA quick_check(1)").fetchone()[0]
            if integrity != "ok":
                _print("FAIL", "storage", "SQLite quick_check failed")
                return False
            tables = {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            required = {
                "messages", "users", "persona_style_examples", "persona_style_sync_state",
                "member_fact_refresh_state", "memory_items", "conversation_episodes",
                "retrieval_documents", "jobs", "usage_records",
            }
            missing = sorted(required - tables)
            if missing:
                _print("FAIL", "storage", "missing tables=" + ",".join(missing))
                return False
            _print("OK", "storage", f"sqlite=ok messages={_table_count(db, 'messages')}")
            _print(
                "OK", "memory",
                f"items={_table_count(db, 'memory_items')} "
                f"episodes={_table_count(db, 'conversation_episodes')} "
                f"retrieval_documents={_table_count(db, 'retrieval_documents')}",
            )
            _print(
                "OK", "member_facts",
                f"refresh_states={_table_count(db, 'member_fact_refresh_state')}",
            )
            job_rows = dict(db.execute(
                "SELECT status, count(*) FROM jobs WHERE job_type = 'memory_compaction' "
                "GROUP BY status"
            ).fetchall())
            _print(
                "OK" if not job_rows.get("failed") else "WARN",
                "memory_compaction",
                " ".join(f"{status}={count}" for status, count in sorted(job_rows.items()))
                or "jobs=0",
            )
            _check_memory_episode_backlog(db, tables=tables)
            vector_failures = int(db.execute(
                "SELECT count(*) FROM retrieval_documents "
                "WHERE status = 'active' AND embedding_eligible = 1 "
                "AND embedding_status = 'failed'"
            ).fetchone()[0])
            _print(
                "OK" if vector_failures == 0 else "WARN", "retrieval_vectors",
                f"active_failed={vector_failures}",
            )
            persona_ok = _check_persona_state(db, settings)
            worker_ok = check_worker_status(settings)
            return persona_ok and worker_ok
    except (sqlite3.Error, OSError) as exc:
        _print("FAIL", "storage", f"{type(exc).__name__}: {exc}")
        return False


def _check_persona_state(db: sqlite3.Connection, settings: AppSettings) -> bool:
    persona_dir = settings.data_dir / "personas"
    files = list(persona_dir.glob("*.live.yaml")) if persona_dir.is_dir() else []
    try:
        for path in files:
            profile = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(profile, dict):
                raise ValueError(f"{path.name}: not a YAML mapping")
            _normalize_live_profile_contract(profile)
        state_count = _table_count(db, "persona_style_sync_state")
        refreshed_count = int(db.execute(
            "SELECT count(*) FROM persona_style_sync_state WHERE last_refresh_at IS NOT NULL"
        ).fetchone()[0])
        if refreshed_count and not files:
            raise ValueError("refresh state exists but no live persona file was written")
        backlog = int(db.execute(
            "SELECT count(*) FROM persona_style_sync_state WHERE new_since_refresh >= 100"
        ).fetchone()[0])
        _print(
            "OK" if backlog == 0 else "WARN", "persona_sync",
            f"states={state_count} refreshed={refreshed_count} "
            f"valid_live_files={len(files)} pending_threshold={backlog}",
        )
        return True
    except (OSError, yaml.YAMLError, ValueError) as exc:
        _print("FAIL", "persona_sync", f"{type(exc).__name__}: {exc}")
        return False


def check_worker_status(settings: AppSettings) -> bool:
    ok = True
    for name in ("persona_sync", "member_facts"):
        path = settings.log_dir / f"{name}.worker.json"
        if not path.is_file():
            _print("WARN", name + "_worker", "marker missing; waiting for first tick")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            updated_at = datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
            if updated_at.tzinfo is None:
                raise ValueError("worker timestamp has no timezone")
            age = (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds()
            interval = float(payload["interval_seconds"])
            state = str(payload["state"])
            if state not in {"running", "idle", "error"} or interval <= 0 or age < -60:
                raise ValueError("invalid worker marker")
            stale = age > interval + 900
            failed = state == "error" or stale
            _print(
                "FAIL" if failed else "OK", name + "_worker",
                f"state={state} age_seconds={age:.0f} interval_seconds={interval:.0f}",
            )
            ok = ok and not failed
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            _print("FAIL", name + "_worker", f"invalid marker ({type(exc).__name__})")
            ok = False
    return ok


def check_provider_config(settings: AppSettings) -> bool:
    text_ok = bool(settings.llm_base_url and settings.llm_api_key and settings.llm_model)
    _print("OK" if text_ok else "FAIL", "text_provider", f"model={settings.llm_model}")
    if settings.group_image_transport == "images":
        image_ok = bool(
            settings.group_image_base_url
            and settings.group_image_api_key
            and settings.group_image_model
        )
    else:
        image_ok = bool(
            (settings.group_image_chat_base_url or settings.llm_base_url)
            and (settings.group_image_chat_api_key or settings.llm_api_key)
            and settings.group_image_model
        )
    _print(
        "OK" if image_ok else "FAIL", "image_provider_config",
        f"transport={settings.group_image_transport} model={settings.group_image_model}",
    )
    search_ok = settings.search_provider.strip().lower() == "ddgs" or bool(
        settings.search_api_key.strip()
    )
    _print(
        "OK" if search_ok else "WARN", "search_provider_config",
        f"provider={settings.search_provider or 'unset'}",
    )
    return text_ok and image_ok


def check_live_persona(settings: AppSettings) -> bool:
    if not settings.llm_api_key or not settings.llm_base_url:
        _print("FAIL", "model_persona_live", "provider credentials missing")
        return False
    usages: list[LlmUsage] = []
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        responses_model=settings.llm_model,
        responses_only=True,
        reasoning_effort=PERSONA_REFRESH_REASONING_EFFORT,
        max_output_tokens=PROBE_OUTPUT_TOKEN_CAP,
        timeout_seconds=min(90.0, settings.llm_timeout_seconds),
        usage_recorder=usages.append,
    )
    # A health check must not multiply billed requests during an outage.
    client.REQUEST_MAX_ATTEMPTS = 1
    try:
        generated = client.generate_text([PROBE_PROMPT], allow_web_search=False)
        profile = _normalize_live_profile_contract(parse_persona_yaml(generated))
        with TemporaryDirectory(prefix="xiaomachi-status-") as temp_dir:
            path = write_live_profile(
                data_dir=Path(temp_dir), persona_key="status_probe", profile=profile
            )
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if _normalize_live_profile_contract(loaded) != profile:
                raise ValueError("persona file round-trip mismatch")
        usage = usages[-1] if usages else None
        token_detail = (
            f" input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}"
            if usage else " usage_unavailable"
        )
        _print(
            "OK", "model_persona_live",
            f"model={settings.llm_model} reasoning={PERSONA_REFRESH_REASONING_EFFORT} "
            "schema=valid temp_file_roundtrip=ok "
            f"output_cap={PROBE_OUTPUT_TOKEN_CAP}{token_detail}",
        )
        return True
    except Exception as exc:
        # Do not print the response, prompt, endpoint, or credentials.
        _print("FAIL", "model_persona_live", type(exc).__name__)
        return False
    finally:
        client.http_client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="one bounded live model request")
    args = parser.parse_args()
    settings = AppSettings()
    config_ok = check_provider_config(settings)
    storage_ok = check_storage(settings)
    ok = config_ok and storage_ok
    if args.deep:
        ok = check_live_persona(settings) and ok
    else:
        _print("SKIP", "model_persona_live", "run status.sh --deep; default costs zero model tokens")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
