from app.config import AppSettings
from app.main import build_memory_compaction_client
from app.providers.llm_client import LlmClient


def _settings(monkeypatch) -> AppSettings:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    return AppSettings()


def test_build_memory_compaction_client_uses_low_effort(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    main_client = LlmClient(
        base_url="https://example.test",
        api_key="k",
        model="gpt-5.6-luna",
        reasoning_effort="high",
    )

    compaction_client = build_memory_compaction_client(
        settings=settings,
        llm_client=main_client,
    )

    assert compaction_client is not main_client
    assert compaction_client.reasoning_effort == "low"


def test_build_memory_compaction_client_preserves_test_fakes(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    fake = object()

    assert build_memory_compaction_client(settings=settings, llm_client=fake) is fake
