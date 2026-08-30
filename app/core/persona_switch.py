"""Per-group persona switching: command parsing, state, and profile sync.

The persona switch only changes which persona profile a group uses and the
QQ-facing avatar/group-card presentation. It never touches knowledge, memory,
or the safety layer.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from datetime import UTC, datetime

from sqlalchemy import text

from app.core.chat_style import retrieve_relevant_examples, retrieve_relevant_facts
from app.storage.models import (
    MemoryItem,
    MemoryItemSemanticVector,
    PersonaExampleVector,
    User,
)
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupPersonaStateRepository,
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
        self._style_banks: dict[int, list[dict]] = {}
        self._member_aliases: dict[int | None, dict[int, str]] = {}
        self._member_aliases_loaded_at: float = 0.0
        self.embedding_provider = embedding_provider
        self._example_vectors: dict[
            int, tuple[str, list[list[float]] | None, list[dict]]
        ] = {}

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

        for persona_key, persona in self.personas.items():
            if persona_key == DEFAULT_PERSONA_KEY:
                continue
            user_id = _as_positive_int(persona.get("source_user_id"))
            group_id = _as_positive_int(persona.get("source_group_id"))
            if user_id is None or group_id is None:
                continue
            with session_scope(self.engine) as session:
                repo = PersonaStyleExampleRepository(session)
                rows = repo.load_active(user_id=user_id, limit=1800)
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
                        rows = repo.load_active(user_id=user_id, limit=600)
                self._style_banks[user_id] = [
                    {
                        "msg_id": row.msg_id,
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
        if user_id is not None and self._style_banks.get(user_id):
            return list(self._style_banks[user_id])
        return [
            {
                "msg_id": f"baked-{index}",
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
        context_lines: list[str],
        *,
        limit: int = 6,
        persona_key: str | None = None,
    ) -> list[dict]:
        """Retrieve examples by embedding similarity when available."""

        bank = self.style_bank(group_id, persona_key=persona_key)
        if not bank or self.embedding_provider is None:
            return retrieve_relevant_examples(bank, context_lines, limit=limit)
        persona = self._resolve_persona(group_id, persona_key)
        user_id = _as_positive_int(persona.get("source_user_id"))
        cache_key = int(user_id) if user_id is not None else hash(repr(persona.get("name")))
        cached = self._example_vectors.get(cache_key)
        entries_by_id: dict[str, dict] = {}
        vectors_by_id: dict[str, list[float]] = {}
        if cached is not None:
            entries_by_id, vectors_by_id = cached
        elif user_id is not None:
            vectors_by_id = self._load_persisted_example_vectors(user_id)
        bank_ids = {str(entry.get("msg_id") or "") for entry in bank if entry.get("msg_id")}
        stale_ids = (set(entries_by_id) | set(vectors_by_id)) - bank_ids
        for stale_id in stale_ids:
            entries_by_id.pop(stale_id, None)
            vectors_by_id.pop(stale_id, None)
        if stale_ids and user_id is not None:
            self._delete_persisted_example_vectors(user_id, stale_ids)
        missing = [
            entry
            for entry in bank
            if str(entry.get("msg_id") or "") not in entries_by_id
        ]
        if missing:
            texts = [
                " ".join(
                    [
                        entry.get("reply_target") or "",
                        entry.get("text") or "",
                        *[
                            str(item.get("text") or "")
                            for item in (entry.get("context_before") or [])[-2:]
                            if isinstance(item, dict)
                        ],
                        *[
                            str(item.get("text") or "")
                            for item in (entry.get("context_after") or [])[:1]
                            if isinstance(item, dict)
                        ],
                    ]
                )
                for entry in missing
            ]
            new_vectors = self.embedding_provider.embed_documents(texts)
            for entry, vector in zip(missing, new_vectors):
                entries_by_id[str(entry.get("msg_id") or "")] = entry
                vectors_by_id[str(entry.get("msg_id") or "")] = vector
            if user_id is not None and new_vectors:
                self._save_persisted_example_vectors(
                    user_id,
                    group_id,
                    {
                        str(entry.get("msg_id") or ""): vector
                        for entry, vector in zip(missing, new_vectors)
                        if vector
                    },
                )
        self._example_vectors[cache_key] = (entries_by_id, vectors_by_id)
        query_text = " ".join(
            str(line).split(":", 1)[-1] for line in context_lines
        )
        query_vector = self.embedding_provider.embed_query(query_text)
        if not vectors_by_id or query_vector is None:
            return retrieve_relevant_examples(bank, context_lines, limit=limit)
        now = datetime.now(UTC)
        scored: list[tuple[float, dict]] = []
        for msg_id, vector in vectors_by_id.items():
            if not vector:
                continue
            entry = entries_by_id[msg_id]
            base = self._cosine(query_vector, vector)
            age_days = 0.0
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, datetime):
                ts = timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            decay = 1.0 / (1.0 + age_days / 45.0)
            scored.append((base * decay, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        picked: list[dict] = []
        picked_vectors: list[list[float]] = []
        for _, entry in scored:
            if len(picked) >= max(0, limit):
                break
            entry_vector = vectors_by_id.get(str(entry.get("msg_id") or ""))
            if any(
                self._cosine(entry_vector, existing) > 0.92
                for existing in picked_vectors
                if entry_vector
            ):
                continue
            picked.append(entry)
            if entry_vector:
                picked_vectors.append(entry_vector)
        return picked

    def _load_persisted_example_vectors(
        self,
        user_id: int,
    ) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        with session_scope(self.engine) as session:
            rows = (
                session.query(PersonaExampleVector)
                .filter(PersonaExampleVector.user_id == int(user_id))
                .all()
            )
            for row in rows:
                try:
                    parsed = json.loads(row.vector_json or "[]")
                except (json.JSONDecodeError, TypeError):
                    parsed = []
                if isinstance(parsed, list) and parsed:
                    vectors[str(row.msg_id)] = [float(value) for value in parsed]
        return vectors

    def _save_persisted_example_vectors(
        self,
        user_id: int,
        group_id: int,
        vectors: dict[str, list[float]],
    ) -> None:
        if not vectors:
            return
        identity = self.embedding_provider.identity if self.embedding_provider else None
        provider = str(getattr(identity, "provider", "") or "")
        model = str(getattr(identity, "model", "") or "")
        dimensions = int(getattr(identity, "dimensions", 0) or 0)
        with session_scope(self.engine) as session:
            for msg_id, vector in vectors.items():
                session.merge(
                    PersonaExampleVector(
                        msg_id=str(msg_id),
                        user_id=int(user_id),
                        group_id=int(group_id),
                        provider=provider,
                        model=model,
                        dimensions=dimensions,
                        vector_json=json.dumps(
                            [float(value) for value in vector],
                            ensure_ascii=False,
                        ),
                    )
                )

    def _delete_persisted_example_vectors(
        self,
        user_id: int,
        msg_ids: set[str],
    ) -> None:
        if not msg_ids:
            return
        with session_scope(self.engine) as session:
            session.query(PersonaExampleVector).filter(
                PersonaExampleVector.user_id == int(user_id),
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
    ) -> list[dict]:
        """Pull topic-relevant facts about the active member from shared memory."""

        persona = self.active_persona(group_id)
        user_id = _as_positive_int(persona.get("source_user_id"))
        if user_id is None:
            return []
        with session_scope(self.engine) as session:
            rows = list(
                session.query(MemoryItem)
                .filter(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == str(int(group_id)),
                    MemoryItem.subject_id == str(user_id),
                    MemoryItem.memory_kind.in_(("fact", "relationship")),
                    MemoryItem.status == "active",
                )
                .all()
            )
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
        bank = [
            {
                "memory_id": int(row.id),
                "category": str(row.predicate or "fact"),
                "fact": str(row.content or ""),
            }
            for row in rows
            if str(row.content or "").strip()
        ]
        if self.embedding_provider is not None and vectors:
            query_text = " ".join(
                str(line).split(":", 1)[-1] for line in context_lines
            )
            query_vector = self.embedding_provider.embed_query(query_text)
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
        return retrieve_relevant_facts(bank, context_lines, limit=limit)

    def active_key(self, group_id: int) -> str:
        return self._group_keys.get(int(group_id), DEFAULT_PERSONA_KEY)

    def active_persona(self, group_id: int) -> dict:
        key = self.active_key(group_id)
        return self.personas.get(key) or self.default_persona

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
        """Build the example vector cache for one persona (memory-only)."""

        bank = self.style_bank(int(group_id), persona_key=persona_key)
        if not bank or self.embedding_provider is None:
            return 0
        self.retrieve_examples(
            int(group_id),
            ["预热示例向量"],
            limit=1,
            persona_key=persona_key,
        )
        return len(bank)

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
        if self.active_key(group_id) == DEFAULT_PERSONA_KEY:
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
        target_name = str(target_persona.get("name", "") or "").strip() or target_key
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
