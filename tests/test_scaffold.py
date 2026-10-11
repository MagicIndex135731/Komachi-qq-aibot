from importlib import metadata
import json
from pathlib import Path
import tomllib
from urllib.parse import unquote, urlparse

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


def test_editable_install_metadata_exists() -> None:
    distribution = metadata.distribution("qq-ai-bot")
    package_metadata = distribution.metadata

    assert metadata.version("qq-ai-bot") == "0.1.0"
    assert package_metadata["Name"] == "qq-ai-bot"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert package_metadata["Summary"] == project["description"]
    direct_url = json.loads(distribution.read_text("direct_url.json"))
    assert direct_url["dir_info"]["editable"] is True
    installed_path = unquote(urlparse(direct_url["url"]).path)
    if installed_path.startswith("/") and len(installed_path) > 2 and installed_path[2] == ":":
        installed_path = installed_path[1:]
    assert Path(installed_path).resolve() == ROOT
