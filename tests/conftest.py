import pytest

from app.storage.db import build_engine, create_all


@pytest.fixture
def sqlite_engine(tmp_path):
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()
