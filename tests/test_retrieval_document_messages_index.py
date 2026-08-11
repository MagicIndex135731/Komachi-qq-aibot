import sqlite3

from app.storage.db import _apply_schema_migrations, build_engine, create_all


def _index_names(database) -> set[str]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='retrieval_document_messages'"
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()


def test_create_all_creates_group_document_index(tmp_path):
    database = tmp_path / "bot.db"
    engine = build_engine(database)
    create_all(engine)
    engine.dispose()
    assert "ix_retrieval_document_messages_group_document" in _index_names(database)


def test_migrations_add_group_document_index_to_existing_table(tmp_path):
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE retrieval_document_messages (
            document_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            role VARCHAR(32) NOT NULL,
            group_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (document_id, message_id, role)
        );
        """
    )
    connection.commit()
    connection.close()
    engine = build_engine(database)
    try:
        with engine.begin() as transaction:
            _apply_schema_migrations(transaction)
    finally:
        engine.dispose()
    assert "ix_retrieval_document_messages_group_document" in _index_names(database)
