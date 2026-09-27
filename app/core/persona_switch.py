"""Per-group persona switching: command parsing, state, and profile sync.

The persona switch only changes which persona profile a group uses and the
QQ-facing avatar/group-card presentation. It never touches knowledge, memory,
or the safety layer.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import time
from datetime import UTC, datetime

from sqlalchemy import text

from app.core.chat_style import retrieve_relevant_facts
from app.core.persona_style_retrieval import (
    DOCUMENT_SCHEMA,
    MAX_SELECTED_EXAMPLES,
    StyleDocument,
    StyleRetrievalQuery,
    StyleRetrievalTrace,
    build_style_document,
    build_style_retrieval_query,
    rank_style_examples,
)
from app.core.memory_fact_ranking import (
    fact_kinds_for_query,
    matching_member_fact_ids,
    memory_query_features,
    preferred_kinds_for_query,
    rank_member_facts,
    select_temporal_current_facts,
    temporal_recency_required,
)
from app.storage.models import (
    MemoryItem,
    MemoryItemSemanticVector,
    PersonaExampleVector,
    User,
)
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupPersonaStateRepository,
    MemoryRepository,
    PersonaStyleExampleRepository,
)


logger = logging.getLogger(__name__)


DEFAULT_PERSONA_KEY = "default"
ACCOUNT_STATE_GROUP_ID = 0

_SWITCH_COMMAND_PATTERN = re.compile(r"^切换人格为\s*[:：]\s*(?P<target>.+?)\s*$")


def persona_aliases(persona: dict) -> set[str]:
    """Return the normalized trigger aliases for one persona profile."""

    aliases: set[str] = set()
    name = str(persona.get("name", "") or "").strip()
    condensed = name.replace(" ", "")
    for token in (name, condensed):
        normalized = token.strip().lower()
        if normalized:
            aliases.add(normalized)
    if (
        condensed
        and any("\u4e00" <= char <= "\u9fff" for char in condensed)
        and len(condensed) >= 2
    ):
        aliases.add(condensed[-2:].lower())
    for alias in persona.get("aliases") or []:
        normalized = str(alias).strip().lower()
        if normalized:
            aliases.add(normalized)
    return aliases


def parse_switch_command(text: str, personas: dict[str, dict]) -> str | None:
    """Resolve ``切换人格为:X`` to a persona key, or None when not a command."""

    match = _SWITCH_COMMAND_PATTERN.match(str(text or "").strip())
    if match is None:
        return None
    target = str(match.group("target") or "").strip().lower()
    if not target:
        return None
    for key, persona in personas.items():
        if target in persona_aliases(persona):
            return key
    return None


class PersonaManager:
    """In-memory per-group persona state backed by SQLite."""

    def __init__(
        self,
        *,
        engine,
        personas: dict[str, dict],
        default_persona: dict,
        embedding_provider=None,
    ) -> None:
        self.engine = engine
        self.personas = {str(key): value for key, value in (personas or {}).items()}
        self.default_persona = default_persona if isinstance(default_persona, dict) else {}
        self.personas.setdefault(DEFAULT_PERSONA_KEY, self.default_persona)
        self._group_keys: dict[int, str] = {}
        self._card_snapshots: dict[int, str] = {}
        self._account_avatar_snapshot: str | None = None
        self._style_banks: dict[tuple[int, int], list[dict]] = {}
        self._member_aliases: dict[int | None, dict[int, str]] = {}
        self._member_aliases_loaded_at: float = 0.0
        self.embedding_provider = embedding_provider
        self._example_vectors: dict[
            tuple[int, int, str, str, str, int, str],
            tuple[dict[str, StyleDocument], dict[str, list[float]]],
        ] = {}
        self._example_prewarm_status: dict[tuple[int, int], dict[str, object]] = {}

    def load_state(self) -> None:
        self._group_keys.clear()
        self._card_snapshots.clear()
        self._account_avatar_snapshot = None
        with session_scope(self.engine) as session:
            repo = GroupPersonaStateRepository(session)
            for group_id, state in repo.load_all().items():
                if group_id == ACCOUNT_STATE_GROUP_ID:
                    self._account_avatar_snapshot = state.avatar_snapshot
                    continue
                # Persona switches are deliberately process-local: every startup
                # begins as Komachi, while display snapshots still survive restarts.
                self._group_keys[group_id] = DEFAULT_PERSONA_KEY
                if state.persona_key != DEFAULT_PERSONA_KEY:
                    repo.set_persona_key(group_id, DEFAULT_PERSONA_KEY)
                if state.card_snapshot is not None:
                    self._card_snapshots[group_id] = state.card_snapshot
        self.load_style_banks()

    def load_style_banks(self) -> None:
        """Load live style examples per member, seeding from baked banks."""

        self._style_banks.clear()
        for persona_key, persona in self.personas.items():
            if persona_key == DEFAULT_PERSONA_KEY:
                continue
            user_id = _as_positive_int(persona.get("source_user_id"))
            group_id = _as_positive_int(persona.get("source_group_id"))
            if user_id is None or group_id is None:
                continue
            with session_scope(self.engine) as session:
                repo = PersonaStyleExampleRepository(session)
                rows = repo.load_active(
                    user_id=user_id,
                    group_id=group_id,
                    limit=1800,
                )
                if not rows:
                    baked = [
                        str(value).strip()
                        for value in (persona.get("example_bank") or [])
                        if str(value).strip()
                    ]
                    if baked:
                        repo.insert_many(
                            [
                                {
                                    "group_id": group_id,
                                    "user_id": user_id,
                                    "msg_id": f"baked-{index}",
                                    "text": text,
                                    "context_before": [],
                                    "reply_target": None,
                                }
                                for index, text in enumerate(baked)
                            ]
                        )
                        rows = repo.load_active(
                            user_id=user_id,
                            group_id=group_id,
                            limit=600,
                        )
                self._style_banks[(user_id, group_id)] = [
                    {
                        "msg_id": row.msg_id,
                        "user_id": row.user_id,
                        "group_id": row.group_id,
                        "text": row.text,
                        "context_before": row.context_before or [],
                        "context_after": row.context_after or [],
                        "reply_target": row.reply_target,
                        "timestamp": row.timestamp,
                    }
                    for row in rows
                ]

    def style_bank(
        self,
        group_id: int,
        *,
        persona_key: str | None = None,
    ) -> list[dict]:
        persona = self._resolve_persona(group_id, persona_key)
        user_id = _as_positive_int(persona.get("source_user_id"))
        source_group_id = _as_positive_int(persona.get("source_group_id"))
        if (
            user_id is not None
            and source_group_id is not None
            and self._style_banks.get((user_id, source_group_id))
        ):
            return list(self._style_banks[(user_id, source_group_id)])
        return [
            {
                "msg_id": f"baked-{index}",
                "user_id": user_id or 0,
                "group_id": source_group_id or int(group_id),
                "text": str(value).strip(),
                "context_before": [],
                "context_after": [],
                "reply_target": None,
            }
            for index, value in enumerate(persona.get("example_bank") or [])
            if str(value).strip()
        ]

    def retrieve_examples(
        self,
        group_id: int,
        query: StyleRetrievalQuery | list[str],
        *,
        limit: int = MAX_SELECTED_EXAMPLES,
        persona_key: str | None = None,
        exclude_texts: tuple[str, ...] | list[str] = (),
    ) -> list[dict]:
        """Read compatible vectors and retrieve; never embed documents here."""

        started_at = time.perf_counter()
        if not isinstance(query, StyleRetrievalQuery):
            first = str(query[0] if query else "").split(":", 1)[-1].strip()
            query = build_style_retrieval_query(current_text=first)
        trace = StyleRetrievalTrace(
            query_fragment_count=query.fragment_count,
            query_chars=len(query.semantic_text),
        )
        bank = self.style_bank(group_id, persona_key=persona_key)
        if not bank or not query.semantic_text:
            trace.rejection_reason = "empty_bank" if not bank else "empty_query"
            self._log_style_retrieval(
                group_id=group_id,
                query=query,
                trace=trace,
                selected_count=0,
                started_at=started_at,
            )
            return []
        persona = self._resolve_persona(group_id, persona_key)
        user_id = _as_positive_int(persona.get("source_user_id"))
        source_group_id = _as_positive_int(persona.get("source_group_id"))
        if user_id is None or source_group_id is None:
            documents = [
                document
                for entry in bank
                if (document := build_style_document(entry)) is not None
            ]
            trace.quality_rejected = len(bank) - len(documents)
            matches = rank_style_examples(
                query=query,
                documents=documents,
                vectors_by_id={},
                query_vector=None,
                limit=limit,
                exclude_texts=exclude_texts,
                trace=trace,
            )
            self._log_style_retrieval(
                group_id=group_id,
                query=query,
                trace=trace,
                selected_count=len(matches),
                started_at=started_at,
            )
            return [match.entry for match in matches]

        documents = self._documents_for_bank(
            bank,
            user_id=user_id,
            group_id=source_group_id,
        )
        trace.quality_rejected = len(bank) - len(documents)
        cache_key = self._example_cache_key(user_id, source_group_id)
        cached = self._example_vectors.get(cache_key)
        documents_by_id: dict[str, StyleDocument] = {}
        vectors_by_id: dict[str, list[float]] = {}
        if cached is not None:
            documents_by_id, vectors_by_id = cached
        elif self.embedding_provider is not None:
            vectors_by_id = self._load_persisted_example_vectors(
                user_id,
                source_group_id,
                {document.msg_id: document for document in documents},
            )
            documents_by_id = {
                document.msg_id: document
                for document in documents
                if document.msg_id in vectors_by_id
            }
            self._example_vectors[cache_key] = (documents_by_id, vectors_by_id)

        query_vector = None
        if self.embedding_provider is not None and vectors_by_id:
            try:
                query_vector = self.embedding_provider.embed_query(query.semantic_text)
            except Exception:
                logger.exception("persona_style_query_embedding_failed group_id=%s", group_id)
        ranked_documents = list(documents_by_id.values()) if query_vector is not None else documents
        matches = rank_style_examples(
            query=query,
            documents=ranked_documents,
            vectors_by_id=vectors_by_id,
            query_vector=query_vector,
            limit=limit,
            exclude_texts=exclude_texts,
            trace=trace,
        )
        self._log_style_retrieval(
            group_id=group_id,
            query=query,
            trace=trace,
            selected_count=len(matches),
            started_at=started_at,
        )
        return [match.entry for match in matches]

    @staticmethod
    def _log_style_retrieval(
        *,
        group_id: int,
        query: StyleRetrievalQuery,
        trace: StyleRetrievalTrace,
        selected_count: int,
        started_at: float,
    ) -> None:
        logger.info(
            "persona_style_retrieval group_id=%s query_msg_id=%s query_source_count=%s "
            "query_chars=%s candidate_count=%s quality_rejected=%s threshold_rejected=%s "
            "selected_count=%s selected_ids=%s selected_scores=%s fallback=%s reason=%s duration_ms=%.1f",
            int(group_id),
            query.current_message_id or "none",
            trace.query_fragment_count,
            trace.query_chars,
            trace.candidate_count,
            trace.quality_rejected,
            trace.threshold_rejected,
            selected_count,
            trace.selected_ids,
            trace.selected_scores,
            trace.fallback_mode,
            trace.rejection_reason or "none",
            (time.perf_counter() - started_at) * 1000.0,
        )

    def _load_persisted_example_vectors(
        self,
        user_id: int,
        group_id: int,
        documents_by_id: dict[str, StyleDocument],
    ) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        identity = getattr(self.embedding_provider, "identity", None)
        provider = str(getattr(identity, "provider", "") or "")
        model = str(getattr(identity, "model", "") or "")
        version = str(getattr(identity, "version", "") or "")
        dimensions = int(getattr(identity, "dimensions", 0) or 0)
        with session_scope(self.engine) as session:
            rows = (
                session.query(PersonaExampleVector)
                .filter(
                    PersonaExampleVector.user_id == int(user_id),
                    PersonaExampleVector.group_id == int(group_id),
                )
                .all()
            )
            for row in rows:
                document = documents_by_id.get(str(row.msg_id))
                if (
                    document is None
                    or str(row.provider or "") != provider
                    or str(row.model or "") != model
                    or str(row.embedding_version or "") != version
                    or int(row.dimensions or 0) != dimensions
                    or str(row.document_schema or "") != DOCUMENT_SCHEMA
                    or str(row.document_hash or "") != document.document_hash
                ):
                    continue
                try:
                    parsed = json.loads(row.vector_json or "[]")
                    values = [float(value) for value in parsed]
                except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
                    parsed = []
                    values = []
                if (
                    isinstance(parsed, list)
                    and len(values) == dimensions
                    and all(math.isfinite(value) for value in values)
                ):
                    vectors[str(row.msg_id)] = values
        return vectors

    def _save_persisted_example_vectors(
        self,
        user_id: int,
        group_id: int,
        documents_by_id: dict[str, StyleDocument],
        vectors: dict[str, list[float]],
    ) -> None:
        if not vectors:
            return
        identity = self.embedding_provider.identity if self.embedding_provider else None
        provider = str(getattr(identity, "provider", "") or "")
        model = str(getattr(identity, "model", "") or "")
        version = str(getattr(identity, "version", "") or "")
        dimensions = int(getattr(identity, "dimensions", 0) or 0)
        with session_scope(self.engine) as session:
            for msg_id, vector in vectors.items():
                document = documents_by_id.get(str(msg_id))
                if document is None:
                    continue
                session.merge(
                    PersonaExampleVector(
                        msg_id=str(msg_id),
                        user_id=int(user_id),
                        group_id=int(group_id),
                        provider=provider,
                        model=model,
                        embedding_version=version,
                        dimensions=dimensions,
                        document_schema=DOCUMENT_SCHEMA,
                        document_hash=document.document_hash,
                        vector_json=json.dumps(
                            [float(value) for value in vector],
                            ensure_ascii=False,
                        ),
                    )
                )

    def _delete_persisted_example_vectors(
        self,
        user_id: int,
        group_id: int,
        msg_ids: set[str],
    ) -> None:
        if not msg_ids:
            return
        with session_scope(self.engine) as session:
            session.query(PersonaExampleVector).filter(
                PersonaExampleVector.user_id == int(user_id),
                PersonaExampleVector.group_id == int(group_id),
                PersonaExampleVector.msg_id.in_(sorted(msg_ids)),
            ).delete(synchronize_session=False)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = sum(a * a for a in left) ** 0.5
        norm_right = sum(b * b for b in right) ** 0.5
        if not norm_left or not norm_right:
            return 0.0
        return dot / (norm_left * norm_right)

    def retrieve_facts(
        self,
        group_id: int,
        context_lines: list[str],
        *,
        limit: int = 5,
        now: datetime | None = None,
        answer_mode: str = "current_fact",
    ) -> list[dict]:
        """Pull valid, intent-compatible facts about the active member."""

        persona = self.active_persona(group_id)
        user_id = _as_positive_int(persona.get("source_user_id"))
        if user_id is None:
            return []
        query_text = str(context_lines[0] if context_lines else "").split(":", 1)[-1].strip()
        allowed_kinds = fact_kinds_for_query(
            query=query_text,
            answer_mode=answer_mode,
        )
        preferred_kinds = preferred_kinds_for_query(
            query=query_text,
            answer_mode=answer_mode,
        )
        with session_scope(self.engine) as session:
            rows = [
                row
                for row in MemoryRepository(session).list_current_group_memories(
                    scope_id=str(int(group_id)),
                    subject_id=str(user_id),
                    as_of=now,
                    limit=500,
                )
                if str(row.memory_kind or "") in allowed_kinds
            ]
            vectors: dict[int, list[float]] = {}
            if rows and self.embedding_provider is not None:
                row_ids = [row.id for row in rows]
                vector_rows = session.query(MemoryItemSemanticVector).filter(
                    MemoryItemSemanticVector.memory_id.in_(row_ids)
                ).all()
                for vector_row in vector_rows:
                    try:
                        parsed = json.loads(vector_row.vector_json or "[]")
                    except (json.JSONDecodeError, TypeError):
                        parsed = []
                    if isinstance(parsed, list) and parsed:
                        vectors[int(vector_row.memory_id)] = [
                            float(value) for value in parsed
                        ]
        query_features = memory_query_features(
            query=query_text,
            intent_query=query_text,
        )
        recency_required = temporal_recency_required(query=query_text)
        ranked_rows = rank_member_facts(
            rows,
            query_features=query_features,
            preferred_kinds=preferred_kinds,
            recency_boost=recency_required,
            limit=len(rows),
        )
        if recency_required:
            ranked_rows = select_temporal_current_facts(
                ranked_rows,
                matching_fact_ids=matching_member_fact_ids(
                    ranked_rows,
                    query_features=query_features,
                ),
                topic_specific=True,
            )
        bank = [
            {
                "memory_id": int(row.id),
                "category": str(row.memory_kind or row.predicate or "fact"),
                "fact": str(row.content or ""),
            }
            for row in ranked_rows
            if str(row.content or "").strip()
        ]
        if self.embedding_provider is not None and vectors:
            semantic_query_text = " ".join(
                str(line).split(":", 1)[-1] for line in context_lines
            )
            query_vector = self.embedding_provider.embed_query(semantic_query_text)
            if query_vector:
                keyword_scores = retrieve_relevant_facts(
                    bank, context_lines, limit=len(bank)
                )
                keyword_rank = {
                    str(item["fact"]): index
                    for index, item in enumerate(keyword_scores)
                }
                scored: list[tuple[float, dict]] = []
                for item in bank:
                    vector = vectors.get(int(item["memory_id"]))
                    semantic = (
                        self._cosine(query_vector, vector)
                        if vector
                        else 0.0
                    )
                    keyword = 1.0 - (
                        keyword_rank.get(str(item["fact"]), len(bank))
                        / max(1, len(bank))
                    )
                    scored.append((0.7 * semantic + 0.3 * keyword, item))
                scored.sort(key=lambda entry: entry[0], reverse=True)
                return [
                    {"category": item["category"], "fact": item["fact"]}
                    for _, item in scored[: max(0, limit)]
                ]
        # ``rows`` have already been kind-filtered, relevance-ranked and (for
        # temporal questions) reduced to the freshest topic candidate.  A
        # second lexical-only pass is lossy for abbreviations and proper names:
        # e.g. a current anime fact containing only ``RW0`` was selected above
        # but then disappeared because it did not literally contain “动画”.
        return [
            {"category": item["category"], "fact": item["fact"]}
            for item in bank[: max(0, limit)]
        ]

    def active_key(self, group_id: int) -> str:
        return self._group_keys.get(int(group_id), DEFAULT_PERSONA_KEY)

    def active_persona(self, group_id: int) -> dict:
        key = self.active_key(group_id)
        return self.personas.get(key) or self.default_persona

    def is_impersonating(self, group_id: int) -> bool:
        """Whether the current group persona represents another member."""

        if self.active_key(group_id) == DEFAULT_PERSONA_KEY:
            return False
        return not bool(self.active_persona(group_id).get("komachi_variant"))

    def _member_alias_map(
        self,
        *,
        max_age_seconds: float = 300.0,
        group_id: int | None = None,
    ) -> dict[int, str]:
        """Map user ids to their latest display name inside one group.

        Group cards are per-group; the shared ``users`` table only keeps one
        card and gets overwritten across groups. Filtering the message sender
        snapshots by ``group_id`` prevents a card from another group (e.g.
        "周奕辰" in group A) leaking into this group's labels.
        """

        group_id = int(group_id) if group_id is not None else None
        now = time.monotonic()
        if now - self._member_aliases_loaded_at > max_age_seconds:
            self._member_aliases_loaded_at = now
            grouped_aliases: dict[int | None, dict[int, str]] = {}
            with session_scope(self.engine) as session:
                for user in session.query(User).all():
                    label = str(user.group_card or "").strip() or str(
                        user.nickname or ""
                    ).strip()
                    if label:
                        grouped_aliases.setdefault(None, {}).setdefault(
                            int(user.user_id), label
                        )
                seen: set[int] = set()
                rows = session.execute(
                    text(
                        "SELECT user_id, raw_json FROM messages "
                        "WHERE raw_json IS NOT NULL AND group_id = :group_id "
                        "ORDER BY id DESC"
                    ),
                    {"group_id": group_id},
                ).fetchall()
                for user_id, raw_json in rows:
                    uid = int(user_id)
                    if uid in seen:
                        continue
                    seen.add(uid)
                    try:
                        payload = json.loads(raw_json or "{}")
                    except (json.JSONDecodeError, TypeError):
                        payload = {}
                    sender = payload.get("sender") if isinstance(payload, dict) else {}
                    sender = sender if isinstance(sender, dict) else {}
                    card = str(sender.get("card") or "").strip()
                    nickname = str(sender.get("nickname") or "").strip()
                    label = card or nickname
                    if label:
                        grouped_aliases.setdefault(group_id, {}).setdefault(
                            uid, label
                        )
                # users table is a cross-group fallback: group-specific sender
                # snapshots win when both exist for the same user.
                if group_id is not None:
                    grouped_aliases[group_id] = {
                        **grouped_aliases.setdefault(None, {}),
                        **grouped_aliases.setdefault(group_id, {}),
                    }
            self._member_aliases = grouped_aliases
        return self._member_aliases.get(group_id) or {}

    def live_persona(self, group_id: int) -> dict:
        """Return the active persona with relationship labels pointing at the
        members' CURRENT group names (nicknames change; QQ ids stay stable)."""

        persona = self.active_persona(group_id)
        live = copy.deepcopy(persona)
        aliases = self._member_alias_map(group_id=group_id)
        alias_to_user: dict[str, int] = {}
        for user_id, label in aliases.items():
            if label:
                alias_to_user.setdefault(label, int(user_id))
        relationships = live.get("relationships")
        if not isinstance(relationships, list):
            return live
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            user_id = rel.get("member_user_id")
            if user_id is None:
                user_id = alias_to_user.get(str(rel.get("member") or ""))
                if user_id is not None:
                    rel["member_user_id"] = user_id
            if user_id is None and str(rel.get("member") or "").isdigit():
                user_id = int(rel["member"])
            if user_id is None:
                continue
            label = aliases.get(int(user_id))
            if label:
                rel["member"] = label
        return live

    def member_label_for_user(self, user_id: int, group_id: int) -> str | None:
        """Latest display name for one user inside one group, or None."""

        return self._member_alias_map(group_id=group_id).get(int(user_id))

    def prewarm_examples(self, group_id: int, persona_key: str) -> int:
        """Build compatible document vectors outside the message reply path."""

        bank = self.style_bank(int(group_id), persona_key=persona_key)
        if not bank or self.embedding_provider is None:
            return 0
        persona = self._resolve_persona(group_id, persona_key)
        user_id = _as_positive_int(persona.get("source_user_id"))
        source_group_id = _as_positive_int(persona.get("source_group_id"))
        if user_id is None or source_group_id is None:
            return 0
        documents = self._documents_for_bank(
            bank,
            user_id=user_id,
            group_id=source_group_id,
        )
        documents_by_id = {document.msg_id: document for document in documents}
        persisted = self._load_persisted_example_vectors(
            user_id,
            source_group_id,
            documents_by_id,
        )
        with session_scope(self.engine) as session:
            persisted_ids = {
                str(row.msg_id)
                for row in session.query(PersonaExampleVector).filter(
                    PersonaExampleVector.user_id == int(user_id),
                    PersonaExampleVector.group_id == int(source_group_id),
                )
            }
        stale_ids = persisted_ids - set(persisted)
        if stale_ids:
            self._delete_persisted_example_vectors(
                user_id,
                source_group_id,
                stale_ids,
            )
        missing = [document for document in documents if document.msg_id not in persisted]
        failed = 0
        if missing:
            try:
                new_vectors = self.embedding_provider.embed_documents(
                    [document.canonical_text for document in missing]
                )
            except Exception:
                logger.exception(
                    "persona_style_document_embedding_failed persona=%s samples=%s",
                    persona_key,
                    len(missing),
                )
                new_vectors = None
            if new_vectors is None:
                failed = len(missing)
            else:
                valid_new: dict[str, list[float]] = {}
                dimensions = int(getattr(self.embedding_provider.identity, "dimensions", 0) or 0)
                for document, vector in zip(missing, new_vectors):
                    try:
                        values = [float(value) for value in vector]
                    except (TypeError, ValueError, OverflowError):
                        failed += 1
                        continue
                    if len(values) != dimensions or not all(math.isfinite(value) for value in values):
                        failed += 1
                        continue
                    valid_new[document.msg_id] = values
                if len(new_vectors) != len(missing):
                    failed += abs(len(missing) - len(new_vectors))
                if valid_new:
                    self._save_persisted_example_vectors(
                        user_id,
                        source_group_id,
                        documents_by_id,
                        valid_new,
                    )
                    persisted.update(valid_new)
        ready_documents = {
            msg_id: document
            for msg_id, document in documents_by_id.items()
            if msg_id in persisted
        }
        self._example_vectors[self._example_cache_key(user_id, source_group_id)] = (
            ready_documents,
            persisted,
        )
        self._example_prewarm_status[(user_id, source_group_id)] = {
            "user_id": user_id,
            "group_id": source_group_id,
            "document_schema": DOCUMENT_SCHEMA,
            "samples": len(bank),
            "quality_ready": len(documents),
            "vectors_ready": len(persisted),
            "missing": max(0, len(documents) - len(persisted)),
            "stale": len(stale_ids),
            "failed": failed,
        }
        return len(persisted)

    def example_vector_status(self, group_id: int, persona_key: str) -> dict[str, object]:
        persona = self._resolve_persona(group_id, persona_key)
        user_id = _as_positive_int(persona.get("source_user_id"))
        source_group_id = _as_positive_int(persona.get("source_group_id"))
        if user_id is None or source_group_id is None:
            return {}
        return dict(self._example_prewarm_status.get((user_id, source_group_id), {}))

    def _documents_for_bank(
        self,
        bank: list[dict],
        *,
        user_id: int,
        group_id: int,
    ) -> list[StyleDocument]:
        return [
            document
            for entry in bank
            if (
                document := build_style_document(
                    entry,
                    expected_user_id=user_id,
                    expected_group_id=group_id,
                )
            )
            is not None
            and document.msg_id
        ]

    def _example_cache_key(
        self,
        user_id: int,
        group_id: int,
    ) -> tuple[int, int, str, str, str, int, str]:
        identity = getattr(self.embedding_provider, "identity", None)
        return (
            int(user_id),
            int(group_id),
            str(getattr(identity, "provider", "") or ""),
            str(getattr(identity, "model", "") or ""),
            str(getattr(identity, "version", "") or ""),
            int(getattr(identity, "dimensions", 0) or 0),
            DOCUMENT_SCHEMA,
        )

    def _resolve_persona(self, group_id: int, persona_key: str | None) -> dict:
        if persona_key is None:
            return self.active_persona(group_id)
        return self.personas.get(persona_key) or self.default_persona

    def active_name(self, group_id: int) -> str:
        name = str(self.active_persona(group_id).get("name", "") or "").strip()
        if name:
            return name
        return str(self.default_persona.get("name", "") or "").strip()

    def default_short_name(self) -> str:
        name = str(self.default_persona.get("name", "") or "").strip()
        condensed = name.replace(" ", "")
        if (
            condensed
            and any("\u4e00" <= char <= "\u9fff" for char in condensed)
            and len(condensed) >= 2
        ):
            return condensed[-2:]
        return name

    def bot_transcript_label(self, group_id: int) -> str:
        """Internal label for bot lines; distinct from the impersonated member."""

        name = self.active_name(group_id)
        if not self.is_impersonating(group_id):
            return name
        short_name = self.default_short_name()
        if not short_name:
            return name
        return f"{name}（{short_name}扮演）"

    def set_persona_key(self, group_id: int, persona_key: str) -> None:
        group_id = int(group_id)
        resolved = persona_key if persona_key in self.personas else DEFAULT_PERSONA_KEY
        self._group_keys[group_id] = resolved
        with session_scope(self.engine) as session:
            GroupPersonaStateRepository(session).set_persona_key(group_id, resolved)

    def card_snapshot(self, group_id: int) -> str | None:
        return self._card_snapshots.get(int(group_id))

    def set_card_snapshot(self, group_id: int, card: str | None) -> None:
        group_id = int(group_id)
        self._card_snapshots[group_id] = card
        with session_scope(self.engine) as session:
            GroupPersonaStateRepository(session).set_card_snapshot(group_id, card)

    def account_avatar_snapshot(self) -> str | None:
        return self._account_avatar_snapshot

    def set_account_avatar_snapshot(self, avatar: str | None) -> None:
        self._account_avatar_snapshot = avatar
        with session_scope(self.engine) as session:
            GroupPersonaStateRepository(session).set_avatar_snapshot(
                ACCOUNT_STATE_GROUP_ID,
                avatar,
            )


class PersonaSwitchService:
    """Orchestrates persona key switches.

    The QQ avatar and group card are intentionally never touched: the bot
    always keeps the 比企谷小町 display name, regardless of which persona is
    impersonated.
    """

    def __init__(self, *, manager: PersonaManager, sender, bot_qq: int) -> None:
        self.manager = manager
        self.sender = sender
        self.bot_qq = int(bot_qq)

    async def switch(self, *, group_id: int, target_key: str) -> str:
        """Apply the switch and return a user-facing confirmation line."""

        group_id = int(group_id)
        target_persona = self.manager.personas.get(target_key) or self.manager.default_persona
        target_name = str(
            target_persona.get("switch_label") or target_persona.get("name", "") or target_key
        ).strip()
        current_key = self.manager.active_key(group_id)
        if current_key == target_key:
            return f"当前已经是{target_name}人格，无需切换。"

        self.manager.set_persona_key(group_id, target_key)
        logger.info(
            "persona_switch group_id=%s persona_key=%s",
            group_id,
            target_key,
        )
        return f"已切换为{target_name}人格。"


def _as_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
