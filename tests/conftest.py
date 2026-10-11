from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

import pytest

from app.storage.db import build_engine, create_all


def pytest_configure(config):
    """Keep disposable runs in one folder, with no shared basetemp deletion."""
    root = Path(config.rootpath).resolve()
    work = root / "data" / "test-work"
    if config.option.basetemp is None:
        work.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        config.option.basetemp = str(work / f"pytest-{stamp}-{uuid4().hex[:8]}")
    else:
        requested = Path(config.option.basetemp).resolve()
        if requested.is_relative_to(root) and not requested.is_relative_to(work.resolve()):
            raise pytest.UsageError("Repository basetemp must be inside data/test-work/")


@pytest.fixture
def sqlite_engine(tmp_path):
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()
