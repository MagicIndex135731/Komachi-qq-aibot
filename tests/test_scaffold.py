from importlib import metadata
from pathlib import Path

import pytest

from app.main import main


ROOT = Path(__file__).resolve().parents[1]


def test_scaffold_files_exist() -> None:
    required_files = [
        "pyproject.toml",
        "app/__init__.py",
        "app/main.py",
        "configs/persona.yaml",
        "configs/groups.yaml",
        "configs/safety.yaml",
    ]

    missing = [path for path in required_files if not (ROOT / path).is_file()]

    assert missing == []


def test_main_requires_required_env(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for key in ("NAPCAT_WS_URL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "BOT_QQ", "OWNER_QQ"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(Exception):
        main()


def test_main_returns_zero(monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    async def fake_run() -> None:
        return None

    monkeypatch.setattr("app.main.run", fake_run)

    assert main() == 0


def test_scaffold_directories_are_tracked() -> None:
    required_placeholders = [
        "app/adapters/.gitkeep",
        "app/admin/.gitkeep",
        "app/core/.gitkeep",
        "app/jobs/.gitkeep",
        "app/providers/.gitkeep",
        "app/storage/.gitkeep",
        "data/logs/.gitkeep",
        "tests/adapters/.gitkeep",
        "tests/admin/.gitkeep",
        "tests/core/.gitkeep",
        "tests/jobs/.gitkeep",
        "tests/providers/.gitkeep",
        "tests/storage/.gitkeep",
    ]

    missing = [path for path in required_placeholders if not (ROOT / path).is_file()]

    assert missing == []


def test_editable_install_metadata_exists() -> None:
    distribution = metadata.distribution("qq-ai-bot")
    package_metadata = distribution.metadata

    assert metadata.version("qq-ai-bot") == "0.1.0"
    assert package_metadata["Name"] == "qq-ai-bot"
    assert package_metadata["Summary"] == "NapCat-based QQ AI bot with image generation and private admin repo control"
    assert Path(distribution.locate_file("pyproject.toml")).resolve() == ROOT / "pyproject.toml"
    assert Path(distribution.locate_file("app/main.py")).resolve() == ROOT / "app" / "main.py"
