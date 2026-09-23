from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime, timedelta
import json
import sqlite3

from app.config import AppSettings
from app.core.persona_live_sync import write_live_profile
from app.core.worker_status import write_worker_status
from scripts import status_components


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings.model_construct(
        data_dir=tmp_path,
        llm_base_url="https://example.invalid/v1",
        llm_api_key="test-key",
        llm_model="gpt-6-sol",
        llm_timeout_seconds=30.0,
    )


def test_live_persona_probe_validates_model_and_temp_file(tmp_path, monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["reasoning_effort"] == "medium"
            assert kwargs["max_output_tokens"] <= 1200
            self.http_client = self

        def generate_text(self, prompts, **kwargs):
            assert "合成健康检查" in prompts[0]
            assert kwargs["allow_web_search"] is False
            return (
                "name: 状态探针\nidentity: 测试群友\ncore_traits: [直接]\n"
                "speaking_style: {tone: 简短}\nself_concept: 群友\n"
                "speech_habits: [短句]\nstyle_avoid: []\n"
                "relationships: []\naddress_rules: []\n"
            )

        def close(self):
            pass

    monkeypatch.setattr(status_components, "LlmClient", FakeClient)
    assert status_components.check_live_persona(_settings(tmp_path))
    assert list(tmp_path.rglob("*.live.yaml")) == []


def test_live_persona_probe_rejects_missing_model_fields(tmp_path, monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **_kwargs):
            self.http_client = self

        def generate_text(self, _prompts, **_kwargs):
            return "name: 状态探针\nidentity: 测试群友\n"

        def close(self):
            pass

    monkeypatch.setattr(status_components, "LlmClient", FakeClient)
    assert not status_components.check_live_persona(_settings(tmp_path))


def test_storage_probe_rejects_invalid_persisted_persona(tmp_path) -> None:
    settings = _settings(tmp_path)
    with sqlite3.connect(settings.sqlite_path) as db:
        for table in (
            "messages", "users", "persona_style_examples", "persona_style_sync_state",
            "member_fact_refresh_state", "memory_items", "conversation_episodes",
            "retrieval_documents", "jobs", "usage_records",
        ):
            if table == "persona_style_sync_state":
                db.execute(
                    f"CREATE TABLE {table} (last_refresh_at TEXT, new_since_refresh INTEGER)"
                )
            elif table == "jobs":
                db.execute(f"CREATE TABLE {table} (job_type TEXT, status TEXT)")
            elif table == "retrieval_documents":
                db.execute(
                    f"CREATE TABLE {table} (status TEXT, embedding_eligible INTEGER, embedding_status TEXT)"
                )
            else:
                db.execute(f"CREATE TABLE {table} (id INTEGER)")
    assert status_components.check_storage(settings)
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    (persona_dir / "broken.live.yaml").write_text("name: broken\n", encoding="utf-8")
    assert not status_components.check_storage(settings)


def test_shared_profile_writer_enforces_contract(tmp_path) -> None:
    try:
        write_live_profile(data_dir=tmp_path, persona_key="probe", profile={"name": "x"})
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("invalid profile was written")
    assert not (tmp_path / "personas" / "probe.live.yaml").exists()


def test_worker_marker_detects_stale_and_failed_scheduler(tmp_path) -> None:
    settings = _settings(tmp_path)
    write_worker_status(settings.log_dir, "persona_sync", "idle", 300)
    write_worker_status(settings.log_dir, "member_facts", "idle", 21600)
    assert status_components.check_worker_status(settings)

    path = settings.log_dir / "persona_sync.worker.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["updated_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not status_components.check_worker_status(settings)

    write_worker_status(settings.log_dir, "persona_sync", "error", 300)
    assert not status_components.check_worker_status(settings)


def test_provider_config_checks_optional_image_transport(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.group_image_transport = "images"
    settings.group_image_base_url = ""
    settings.group_image_api_key = ""
    settings.group_image_model = "gpt-image-2"
    settings.search_provider = "ddgs"
    settings.search_api_key = ""
    assert not status_components.check_provider_config(settings)
    settings.group_image_base_url = "https://example.invalid/v1"
    settings.group_image_api_key = "test-image-key"
    assert status_components.check_provider_config(settings)
