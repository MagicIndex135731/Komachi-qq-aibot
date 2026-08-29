"""Per-group persona switching: command parsing, state, and profile sync.

The persona switch only changes which persona profile a group uses and the
QQ-facing avatar/group-card presentation. It never touches knowledge, memory,
or the safety layer.
"""

from __future__ import annotations

import copy
import logging
import re
import time

from sqlalchemy import text

from app.core.chat_style import retrieve_relevant_examples, retrieve_relevant_facts
from app.storage.models import MemoryItem, User
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
        self._member_aliases: dict[int, str] = {}
        self._member_aliases_loaded_at: float = 0.0
        self.embedding_provider = embedding_provider
        self._example_vectors: dict[
            int, tuple[str, list[list[float]] | None, list[dict]]
        ] = {}

    def load_state(self) -> None:
        with session_scope(self.engine) as session:
            repo = GroupPersonaStateRepository(session)
            for group_id, state in repo.load_all().items():
                if group_id == ACCOUNT_STATE_GROUP_ID:
                    self._account_avatar_snapshot = state.avatar_snapshot
                    continue
                self._group_keys[group_id] = state.persona_key or DEFAULT_PERSONA_KEY
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
                rows = repo.load_active(user_id=user_id, limit=600)
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
                        "text": row.text,
                        "context_before": row.context_before or [],
                        "reply_target": row.reply_target,
                    }
                    for row in rows
                ]

    def style_bank(self, group_id: int) -> list[dict]:
        persona = self.active_persona(group_id)
        user_id = _as_positive_int(persona.get("source_user_id"))
        if user_id is not None and self._style_banks.get(user_id):
            return list(self._style_banks[user_id])
        return [
            {
                "text": str(value).strip(),
                "context_before": [],
                "reply_target": None,
            }
            for value in (persona.get("example_bank") or [])
            if str(value).strip()
        ]

    def retrieve_examples(
        self,
        group_id: int,
        context_lines: list[str],
        *,
        limit: int = 6,
    ) -> list[dict]:
        """Retrieve examples by embedding similarity when available."""

        bank = self.style_bank(group_id)
        if not bank or self.embedding_provider is None:
            return retrieve_relevant_examples(bank, context_lines, limit=limit)
        persona = self.active_persona(group_id)
        user_id = _as_positive_int(persona.get("source_user_id"))
        cache_key = int(user_id) if user_id is not None else hash(repr(persona.get("name")))
        signature = "|".join(
            f"{entry.get('text')}\u241f{entry.get('reply_target') or ''}"
            for entry in bank[:80]
        )
        cached = self._example_vectors.get(cache_key)
        if cached is None or cached[0] != signature:
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
                    ]
                )
                for entry in bank
            ]
            vectors = self.embedding_provider.embed_documents(texts)
            cached = (signature, vectors, bank)
            self._example_vectors[cache_key] = cached
        _, vectors, entries = cached
        query_text = " ".join(
            str(line).split(":", 1)[-1] for line in context_lines
        )
        query_vector = self.embedding_provider.embed_query(query_text)
        if not vectors or query_vector is None:
            return retrieve_relevant_examples(bank, context_lines, limit=limit)
        scored = [
            (self._cosine(query_vector, vector), entry)
            for vector, entry in zip(vectors, entries)
            if vector
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[: max(0, limit)]]

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
        bank = [
            {"category": str(row.predicate or "fact"), "fact": str(row.content or "")}
            for row in rows
            if str(row.content or "").strip()
        ]
        return retrieve_relevant_facts(bank, context_lines, limit=limit)

    def active_key(self, group_id: int) -> str:
        return self._group_keys.get(int(group_id), DEFAULT_PERSONA_KEY)

    def active_persona(self, group_id: int) -> dict:
        key = self.active_key(group_id)
        return self.personas.get(key) or self.default_persona

    def _member_alias_map(self, *, max_age_seconds: float = 300.0) -> dict[int, str]:
        now = time.monotonic()
        if now - self._member_aliases_loaded_at > max_age_seconds:
            aliases: dict[int, str] = {}
            with session_scope(self.engine) as session:
                for user in session.query(User).all():
                    label = str(user.group_card or "").strip() or str(
                        user.nickname or ""
                    ).strip()
                    if label:
                        aliases.setdefault(int(user.user_id), label)
                seen: set[int] = set()
                rows = session.execute(
                    text(
                        "SELECT user_id, raw_json FROM messages "
                        "WHERE raw_json IS NOT NULL ORDER BY id DESC"
                    )
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
                        aliases.setdefault(uid, label)
            self._member_aliases = aliases
            self._member_aliases_loaded_at = now
        return self._member_aliases

    def live_persona(self, group_id: int) -> dict:
        """Return the active persona with relationship labels pointing at the
        members' CURRENT group names (nicknames change; QQ ids stay stable)."""

        persona = self.active_persona(group_id)
        live = copy.deepcopy(persona)
        aliases = self._member_alias_map()
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
    """Orchestrates group-card changes around a persona key switch.

    The QQ avatar is intentionally never touched: every persona keeps the
    bot's original avatar.
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

        failures: list[str] = []
        if target_key != DEFAULT_PERSONA_KEY:
            await self._capture_snapshots(group_id=group_id, failures=failures)
            group_card = str(
                target_persona.get("group_card") or target_persona.get("name") or ""
            ).strip()
            if group_card:
                try:
                    await self.sender.set_group_card(
                        group_id=group_id,
                        user_id=self.bot_qq,
                        card=group_card,
                    )
                except Exception:
                    logger.exception(
                        "persona_group_card_apply_failed group_id=%s persona_key=%s",
                        group_id,
                        target_key,
                    )
                    failures.append("群名片切换失败")
        else:
            card_snapshot = self.manager.card_snapshot(group_id)
            try:
                await self.sender.set_group_card(
                    group_id=group_id,
                    user_id=self.bot_qq,
                    card=card_snapshot or "",
                )
            except Exception:
                logger.exception(
                    "persona_group_card_restore_failed group_id=%s",
                    group_id,
                )
                failures.append("群名片恢复失败")

        self.manager.set_persona_key(group_id, target_key)
        logger.info(
            "persona_switch group_id=%s persona_key=%s profile_failures=%s",
            group_id,
            target_key,
            ";".join(failures),
        )
        confirmation = f"已切换为{target_name}人格。"
        if failures:
            confirmation += f"（{'；'.join(failures)}）"
        return confirmation

    async def _capture_snapshots(self, *, group_id: int, failures: list[str]) -> None:
        from app.adapters.sender import extract_group_card

        if self.manager.card_snapshot(group_id) is None:
            try:
                member_info = await self.sender.get_group_member_info(
                    group_id=group_id,
                    user_id=self.bot_qq,
                )
                self.manager.set_card_snapshot(
                    group_id,
                    extract_group_card(member_info),
                )
            except Exception:
                logger.exception("persona_card_snapshot_failed group_id=%s", group_id)
                failures.append("群名片快照失败")


def _as_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
