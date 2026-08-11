import sqlite3
import threading
import time

from app.storage.db import build_engine, create_all


def test_create_all_survives_transient_database_lock(tmp_path) -> None:
    db_path = tmp_path / "bot.db"
    engine = build_engine(db_path)
    create_all(engine)

    blocker = sqlite3.connect(str(db_path), timeout=30)
    blocker.execute("BEGIN IMMEDIATE")

    result: dict[str, object] = {}

    def run_create_all() -> None:
        try:
            create_all(engine)
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)

    thread = threading.Thread(target=run_create_all)
    thread.start()
    time.sleep(0.5)
    blocker.rollback()
    blocker.close()
    thread.join(timeout=40)

    assert thread.is_alive() is False
    assert result.get("ok") is True, result
