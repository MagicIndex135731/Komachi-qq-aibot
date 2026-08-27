"""Per-group persona switching: command parsing, state, and profile sync.

The persona switch only changes which persona profile a group uses and the
QQ-facing avatar/group-card presentation. It never touches knowledge, memory,
or the safety layer.
"""

from __future__ import annotations

import logging
import re

from app.storage.db import session_scope
from app.storage.repositories import GroupPersonaStateRepository


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
    ) -> None:
        self.engine = engine
        self.personas = {str(key): value for key, value in (personas or {}).items()}
        self.default_persona = default_persona if isinstance(default_persona, dict) else {}
        self.personas.setdefault(DEFAULT_PERSONA_KEY, self.default_persona)
        self._group_keys: dict[int, str] = {}
        self._card_snapshots: dict[int, str] = {}
        self._account_avatar_snapshot: str | None = None

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

    def active_key(self, group_id: int) -> str:
        return self._group_keys.get(int(group_id), DEFAULT_PERSONA_KEY)

    def active_persona(self, group_id: int) -> dict:
        key = self.active_key(group_id)
        return self.personas.get(key) or self.default_persona

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
    """Orchestrates avatar/group-card changes around a persona key switch."""

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
            source_user_id = _as_positive_int(target_persona.get("source_user_id"))
            if source_user_id is not None:
                try:
                    avatar = await self.sender.get_qq_avatar(user_id=source_user_id)
                    await self.sender.set_qq_avatar(file=avatar)
                except Exception:
                    logger.exception(
                        "persona_avatar_apply_failed group_id=%s persona_key=%s",
                        group_id,
                        target_key,
                    )
                    failures.append("头像切换失败")
        else:
            avatar_snapshot = self.manager.account_avatar_snapshot()
            if avatar_snapshot is not None:
                try:
                    await self.sender.set_qq_avatar(file=avatar_snapshot)
                except Exception:
                    logger.exception(
                        "persona_avatar_restore_failed group_id=%s",
                        group_id,
                    )
                    failures.append("头像恢复失败")
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
        if self.manager.account_avatar_snapshot() is None:
            try:
                avatar = await self.sender.get_qq_avatar(user_id=self.bot_qq)
                self.manager.set_account_avatar_snapshot(avatar)
            except Exception:
                logger.exception("persona_avatar_snapshot_failed group_id=%s", group_id)
                failures.append("头像快照失败")


def _as_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
