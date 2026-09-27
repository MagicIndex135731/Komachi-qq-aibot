"""Read-only replay of one persona style retrieval event.

The default JSON contains identifiers and scores only. ``--show-text`` is an
explicit local diagnostic switch and must not be used in routine logs.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sqlite3
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AppSettings, load_runtime_config
from app.core.persona_style_retrieval import (
    DOCUMENT_SCHEMA,
    MAX_SELECTED_EXAMPLES,
    StyleRetrievalTrace,
    build_style_document,
    build_style_retrieval_query,
    rank_style_examples,
)
from app.providers.semantic_embeddings import build_embedding_provider


def _parse_timestamp(value: object) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _load_event(connection: sqlite3.Connection, args) -> sqlite3.Row:
    if args.message_id is not None:
        row = connection.execute(
            "SELECT * FROM messages WHERE id = ? LIMIT 1",
            (int(args.message_id),),
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT * FROM messages WHERE platform_msg_id = ? LIMIT 1",
            (str(args.platform_msg_id),),
        ).fetchone()
    if row is None:
        raise ValueError("target message was not found")
    return row


def replay(args) -> dict[str, object]:
    started = time.perf_counter()
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    persona = runtime.personas.get(args.persona_key)
    if not isinstance(persona, dict):
        raise ValueError(f"unknown persona key: {args.persona_key}")
    user_id = int(persona.get("source_user_id") or 0)
    source_group_id = int(persona.get("source_group_id") or 0)
    if user_id <= 0 or source_group_id <= 0:
        raise ValueError("persona has no source user/group")

    database_path = Path(args.db or settings.sqlite_path)
    connection = _connect_read_only(database_path)
    try:
        event = _load_event(connection, args)
        event_time = _parse_timestamp(event["timestamp"])
        if int(event["group_id"] or 0) != source_group_id:
            raise ValueError("target event group does not match persona source group")
        recent = connection.execute(
            "SELECT platform_msg_id, user_id, plain_text, timestamp FROM messages "
            "WHERE group_id = ? AND (timestamp < ? OR (timestamp = ? AND id < ?)) "
            "ORDER BY timestamp DESC, id DESC LIMIT 10",
            (source_group_id, event["timestamp"], event["timestamp"], int(event["id"])),
        ).fetchall()
        quoted_text = None
        reply_to = str(event["reply_to_msg_id"] or "")
        if reply_to:
            quoted = connection.execute(
                "SELECT plain_text FROM messages "
                "WHERE group_id = ? AND platform_msg_id = ? LIMIT 1",
                (source_group_id, reply_to),
            ).fetchone()
            quoted_text = str(quoted[0] or "") if quoted is not None else None
        query = build_style_retrieval_query(
            current_text=event["plain_text"],
            quoted_text=quoted_text,
            recent_messages=list(reversed(recent)),
            current_timestamp=event_time,
            current_message_id=event["platform_msg_id"],
            bot_user_id=int(settings.bot_qq),
            subject_terms=[persona.get("name"), *(persona.get("aliases") or [])],
        )
        sample_rows = connection.execute(
            "SELECT msg_id, user_id, group_id, text, context_before, context_after, "
            "reply_target, timestamp FROM persona_style_examples "
            "WHERE user_id = ? AND group_id = ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT 1800",
            (user_id, source_group_id, event["timestamp"]),
        ).fetchall()
        documents = []
        for row in sample_rows:
            entry = dict(row)
            for key in ("context_before", "context_after"):
                try:
                    entry[key] = json.loads(entry.get(key) or "[]")
                except (json.JSONDecodeError, TypeError):
                    entry[key] = []
            entry["timestamp"] = _parse_timestamp(entry["timestamp"])
            document = build_style_document(
                entry,
                expected_user_id=user_id,
                expected_group_id=source_group_id,
            )
            if document is not None:
                documents.append(document)

        provider = build_embedding_provider(
            provider=settings.memory_embedding_provider,
            device=settings.memory_embedding_device,
            model=settings.memory_embedding_model,
            dimensions=settings.memory_embedding_dimensions,
            cache_dir=settings.memory_embedding_cache_dir,
            local_files_only=settings.memory_embedding_local_files_only,
            version=settings.memory_embedding_version,
            base_url=settings.memory_embedding_base_url,
            api_key=settings.memory_embedding_api_key,
            timeout_seconds=settings.memory_embedding_timeout_seconds,
        )
        identity = provider.identity
        vector_columns = _table_columns(connection, "persona_example_vectors")
        required_columns = {"embedding_version", "document_schema", "document_hash"}
        vectors: dict[str, list[float]] = {}
        if required_columns <= vector_columns:
            document_by_id = {document.msg_id: document for document in documents}
            vector_rows = connection.execute(
                "SELECT * FROM persona_example_vectors WHERE user_id = ? AND group_id = ?",
                (user_id, source_group_id),
            ).fetchall()
            for row in vector_rows:
                document = document_by_id.get(str(row["msg_id"]))
                if (
                    document is None
                    or str(row["provider"] or "") != identity.provider
                    or str(row["model"] or "") != identity.model
                    or str(row["embedding_version"] or "") != identity.version
                    or int(row["dimensions"] or 0) != identity.dimensions
                    or str(row["document_schema"] or "") != DOCUMENT_SCHEMA
                    or str(row["document_hash"] or "") != document.document_hash
                ):
                    continue
                try:
                    values = [float(value) for value in json.loads(row["vector_json"] or "[]")]
                except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
                    continue
                if len(values) == identity.dimensions and all(math.isfinite(value) for value in values):
                    vectors[document.msg_id] = values
        try:
            query_vector = provider.embed_query(query.semantic_text) if vectors else None
        except Exception:
            query_vector = None
        ranked_documents = [document for document in documents if document.msg_id in vectors]
        if query_vector is None:
            ranked_documents = documents
        trace = StyleRetrievalTrace(quality_rejected=len(sample_rows) - len(documents))
        matches = rank_style_examples(
            query=query,
            documents=ranked_documents,
            vectors_by_id=vectors,
            query_vector=query_vector,
            limit=MAX_SELECTED_EXAMPLES,
            now=event_time,
            trace=trace,
        )
        selected = [
            {
                "msg_id": match.document.msg_id,
                "situation_kind": match.document.situation_kind,
                "semantic": round(match.semantic_score, 6),
                "lexical": round(match.lexical_score, 6),
                "recency": round(match.recency_score, 6),
                "final": round(match.final_score, 6),
                **(
                    {"situation": match.document.situation, "reply": match.document.reply}
                    if args.show_text
                    else {}
                ),
            }
            for match in matches
        ]
        result: dict[str, object] = {
            "message_id": int(event["id"]),
            "platform_msg_id": str(event["platform_msg_id"]),
            "persona_key": args.persona_key,
            "query_source_count": query.fragment_count,
            "query_chars": len(query.semantic_text),
            "candidate_count": len(sample_rows),
            "quality_ready": len(documents),
            "compatible_vectors": len(vectors),
            "selected_count": len(selected),
            "selected": selected,
            "fallback": trace.fallback_mode,
            "reject_reason": trace.rejection_reason,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
        if args.show_text:
            result["query"] = query.semantic_text
        return result
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--message-id", type=int, help="messages.id")
    target.add_argument("--platform-msg-id")
    parser.add_argument("--persona-key", required=True)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--json", action="store_true", help="JSON is always emitted")
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--assert-excludes", default="")
    parser.add_argument("--assert-query-max-lines", type=int, default=4)
    parser.add_argument("--assert-selected-max", type=int, default=MAX_SELECTED_EXAMPLES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = replay(args)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 2
    selected_ids = {str(item["msg_id"]) for item in result["selected"]}
    excluded_ids = {value.strip() for value in args.assert_excludes.split(",") if value.strip()}
    violations = []
    if selected_ids & excluded_ids:
        violations.append("excluded sample selected")
    if int(result["query_source_count"]) > int(args.assert_query_max_lines):
        violations.append("query source limit exceeded")
    if int(result["selected_count"]) > int(args.assert_selected_max):
        violations.append("selection limit exceeded")
    if violations:
        result["assertion_errors"] = violations
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
