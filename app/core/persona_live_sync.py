"""Online persona adaptation: rolling context-rich example bank + periodic refresh."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import yaml
from sqlalchemy import text

from app.core.persona_switch import DEFAULT_PERSONA_KEY
from app.core.message_mentions import (
    bot_text_mention_names,
    collect_bot_display_names,
    message_mentions_bot,
)
from app.core.style_distill import parse_persona_yaml
from app.core.style_distill import merge_persona_lists
from app.storage.db import session_scope
from app.storage.repositories import (
    PersonaStyleExampleRepository,
    PersonaStyleSyncStateRepository,
)


logger = logging.getLogger(__name__)


def _positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _speaker_label(raw_json: object) -> str:
    if not isinstance(raw_json, dict):
        return ""
    sender = raw_json.get("sender")
    if not isinstance(sender, dict):
        return ""
    card = str(sender.get("card") or "").strip()
    nickname = str(sender.get("nickname") or "").strip()
    return card or nickname


class PersonaLiveSyncService:
    def __init__(
        self,
        *,
        engine,
        settings,
        personas: dict[str, dict],
        manager,
        interval_seconds: float = 300.0,
        refresh_threshold: int = 50,
        refresh_cooldown_seconds: float = 86400.0,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.personas = personas
        self.manager = manager
        default_name = str(
            (self.personas.get(DEFAULT_PERSONA_KEY) or {}).get("name", "")
        ).strip()
        self.default_name = default_name
        self.bot_qqs: set[int] = {int(settings.bot_qq)}
        self.bot_text_names: set[str] = set()
        self.interval_seconds = max(30.0, float(interval_seconds))
        self.refresh_threshold = max(1, int(refresh_threshold))
        self.refresh_cooldown_seconds = max(3600.0, float(refresh_cooldown_seconds))

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("persona_live_sync_tick_failed")
            await asyncio.sleep(self.interval_seconds)

    def _tick(self) -> None:
        self._refresh_bot_names()
        for persona_key, persona in self.personas.items():
            if persona_key == DEFAULT_PERSONA_KEY:
                continue
            if not bool(persona.get("live_refresh")):
                continue
            user_id = _positive_int(persona.get("source_user_id"))
            group_id = _positive_int(persona.get("source_group_id"))
            if user_id is None or group_id is None:
                continue
            self._sync_examples(persona_key, user_id, group_id)
            self._maybe_refresh_profile(persona_key, user_id, group_id)

    def _sync_examples(self, persona_key: str, user_id: int, group_id: int) -> int:
        with session_scope(self.engine) as session:
            state_repo = PersonaStyleSyncStateRepository(session)
            state = state_repo.get(group_id=group_id, user_id=user_id)
            watermark = int(state.last_msg_id or 0) if state is not None else 0
            new_rows = session.execute(
                text(
                    "SELECT id, platform_msg_id, group_id, user_id, plain_text, msg_type, "
                    "reply_to_msg_id, raw_json, timestamp "
                    "FROM messages WHERE group_id = :group_id AND id > :watermark "
                    "ORDER BY id"
                ),
                {"group_id": group_id, "watermark": watermark},
            ).mappings().all()
            pre_rows = session.execute(
                text(
                    "SELECT id, platform_msg_id, group_id, user_id, plain_text, msg_type, "
                    "reply_to_msg_id, raw_json, timestamp "
                    "FROM messages WHERE group_id = :group_id AND id <= :watermark "
                    "ORDER BY id DESC LIMIT 20"
                ),
                {"group_id": group_id, "watermark": watermark},
            ).mappings().all()
            rows = list(reversed(pre_rows)) + list(new_rows)
            examples = _build_examples(
                rows,
                user_id=user_id,
                bot_qqs=self.bot_qqs,
                bot_text_names=self.bot_text_names,
            )
            inserted = PersonaStyleExampleRepository(session).insert_many(examples)
            trimmed = PersonaStyleExampleRepository(session).trim_to(user_id=user_id, keep=600)
            last_id = max((int(row["id"]) for row in new_rows), default=watermark)
            state_repo.set_watermark(
                group_id=group_id,
                user_id=user_id,
                last_msg_id=str(last_id),
                new_count=len(examples),
            )
        if inserted or trimmed:
            logger.info(
                "persona_live_bank_sync persona_key=%s user_id=%s inserted=%s trimmed=%s",
                persona_key,
                user_id,
                inserted,
                trimmed,
            )
        if inserted:
            self.manager.load_style_banks()
        return inserted

    def _refresh_bot_names(self) -> None:
        from sqlalchemy import bindparam

        with session_scope(self.engine) as session:
            user_rows = session.execute(
                text("SELECT user_id, nickname, group_card FROM users")
            ).fetchall()
            bot_ids = {int(self.settings.bot_qq)}
            for user_id, nickname, card in user_rows:
                label = str(card or "").strip() or str(nickname or "").strip()
                if "小町" in label:
                    bot_ids.add(int(user_id))
            ids_param = list(bot_ids)
            bot_rows = session.execute(
                text(
                    "SELECT raw_json FROM messages WHERE user_id = :bot_qq "
                    "AND raw_json IS NOT NULL ORDER BY id DESC LIMIT 3000"
                ),
                {"bot_qq": ids_param[0]},
            ).fetchall()
            if len(ids_param) > 1:
                extra_rows = session.execute(
                    text(
                        "SELECT raw_json FROM messages WHERE user_id IN :bot_ids "
                        "AND raw_json IS NOT NULL ORDER BY id DESC LIMIT 3000"
                    ).bindparams(bindparam("bot_ids", expanding=True)),
                    {"bot_ids": ids_param[1:]},
                ).fetchall()
                bot_rows = [*bot_rows, *extra_rows]
            member_rows = session.execute(
                text(
                    "SELECT DISTINCT json_extract(raw_json, '$.sender.card') AS card, "
                    "json_extract(raw_json, '$.sender.nickname') AS nickname "
                    "FROM messages WHERE raw_json IS NOT NULL AND user_id NOT IN :bot_ids"
                ).bindparams(bindparam("bot_ids", expanding=True)),
                {"bot_ids": ids_param},
            ).fetchall()
        bot_display = collect_bot_display_names(row[0] for row in bot_rows)
        member_display: set[str] = set()
        for card, nickname in member_rows:
            for value in (card, nickname):
                cleaned = str(value or "").strip()
                if cleaned:
                    member_display.add(cleaned)
        self.bot_qqs = bot_ids
        self.bot_text_names = bot_text_mention_names(
            bot_qqs=bot_ids,
            default_name=self.default_name,
            bot_display_names=bot_display,
            member_display_names=member_display,
        )

    def _maybe_refresh_profile(self, persona_key: str, user_id: int, group_id: int) -> None:
        now = datetime.now(UTC)
        with session_scope(self.engine) as session:
            state_repo = PersonaStyleSyncStateRepository(session)
            state = state_repo.get(group_id=group_id, user_id=user_id)
            if state is None:
                return
            new_count = int(state.new_since_refresh or 0)
            last_refresh = state.last_refresh_at
            overdue = False
            if last_refresh is not None:
                if last_refresh.tzinfo is None:
                    last_refresh = last_refresh.replace(tzinfo=UTC)
                overdue = (
                    now - last_refresh
                ).total_seconds() >= self.refresh_cooldown_seconds
            if new_count < self.refresh_threshold and not overdue:
                return
            examples = PersonaStyleExampleRepository(session).load_active(
                user_id=user_id, limit=200
            )
            current_profile = self.personas.get(persona_key) or {}

        try:
            live_path = self._write_refreshed_profile(
                persona_key=persona_key,
                current_profile=current_profile,
                examples=examples,
            )
        except Exception:
            logger.exception(
                "persona_live_refresh_failed persona_key=%s user_id=%s",
                persona_key,
                user_id,
            )
            return
        with session_scope(self.engine) as session:
            state_repo = PersonaStyleSyncStateRepository(session)
            state_repo.mark_refreshed(group_id=group_id, user_id=user_id, when=now)
        logger.info(
            "persona_live_refresh persona_key=%s user_id=%s path=%s",
            persona_key,
            user_id,
            live_path,
        )

    def _write_refreshed_profile(
        self,
        *,
        persona_key: str,
        current_profile: dict,
        examples: list,
    ) -> Path:
        from app.providers.llm_client import LlmClient

        sample_block = "\n".join(
            _format_example_line(example) for example in examples
        )
        prompt = (
            "你是人设维护助手。下面是群成员最新的真实发言（含语境：上文/回复对象 + 他说的原话），"
            "以及当前的人格画像 YAML。请根据新语料增量更新画像：修正过时特质、补充新出现的口头禅/"
            "话题/关系与称呼习惯，保持 v2 字段结构（name/identity/core_traits/speaking_style/"
            "self_concept/speech_habits/style_avoid/relationships/address_rules），"
            "外加 external_relations（他反复关注/转发/维护的群外人物：虚拟主播、球星、up主等，"
            "每项含 name/who/relation/attitude/evidence；高频出现的人物必须写全），"
            "并维护 facts（关于他的持久事实列表，每项含 category/fact/evidence；新增语料里的具体事实要补进去），"
            "relationships 每项必须含 member_user_id（群成员QQ号）与 member（该成员当前昵称或群名片），不要把QQ号当作 member 输出，"
            "不要删除仍成立的条目，不要新增语料里没有的事实。只输出一个 ```yaml 代码块。\n"
            f"当前画像：\n{yaml.safe_dump(current_profile, allow_unicode=True, sort_keys=False)}\n"
            f"最新语料：\n{sample_block}"
        )
        client = LlmClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            fallback_model=self.settings.llm_fallback_model,
            responses_only=True,
            responses_model=self.settings.llm_model,
            max_output_tokens=16000,
            timeout_seconds=180.0,
            reasoning_effort=self.settings.llm_reasoning_effort,
        )
        generated = client.generate_text([prompt])
        profile = parse_persona_yaml(generated)
        live_dir = self.settings.data_dir / "personas"
        live_dir.mkdir(parents=True, exist_ok=True)
        live_path = live_dir / f"{persona_key}.live.yaml"
        live_path.write_text(
            yaml.safe_dump(profile, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        merged = _merge_profile(current_profile, profile)
        self.personas[persona_key] = merged
        if hasattr(self.manager, "personas"):
            self.manager.personas[persona_key] = merged
        return live_path


def _build_examples(
    rows: list,
    *,
    user_id: int,
    bot_qqs: set[int],
    bot_text_names: set[str],
) -> list[dict]:
    new_rows = [dict(row) for row in rows]
    by_id = {str(row.get("platform_msg_id")): row for row in new_rows if row.get("plain_text")}
    ordered = [row for row in new_rows if str(row.get("plain_text") or "").strip()]
    examples: list[dict] = []
    for index, row in enumerate(ordered):
        if int(row.get("user_id") or 0) != user_id:
            continue
        if message_mentions_bot(
            row.get("raw_json"),
            bot_qqs=bot_qqs,
            bot_text_names=bot_text_names,
        ):
            # Human-to-AI turns must never become style samples.
            continue
        if str(row.get("msg_type") or "text") != "text":
            continue
        text = str(row.get("plain_text") or "").strip()
        if not text:
            continue
        context_before = []
        for other in ordered[max(0, index - 4) : index]:
            if int(other.get("user_id") or 0) in bot_qqs:
                continue
            label = _speaker_label(other.get("raw_json")) or str(other.get("user_id"))
            context_before.append(
                {"speaker": label, "text": str(other.get("plain_text") or "").strip()}
            )
        reply_target = None
        quoted = by_id.get(str(row.get("reply_to_msg_id") or ""))
        if (
            quoted is not None
            and int(quoted.get("user_id") or 0) not in bot_qqs
        ):
            label = _speaker_label(quoted.get("raw_json")) or str(quoted.get("user_id"))
            reply_target = f"{label}: {str(quoted.get('plain_text') or '').strip()}"
        examples.append(
            {
                "group_id": int(row.get("group_id") or 0),
                "user_id": user_id,
                "msg_id": str(row.get("platform_msg_id")),
                "text": text,
                "context_before": context_before,
                "reply_target": reply_target,
            }
        )
    return examples


def _format_example_line(example) -> str:
    context = (example.context_before or []) if hasattr(example, "context_before") else (example.get("context_before") or [])
    lead = example.reply_target if hasattr(example, "reply_target") else example.get("reply_target")
    if not lead and context:
        lead = str(context[-1].get("text") or "") if isinstance(context[-1], dict) else str(context[-1])
    text = example.text if hasattr(example, "text") else example.get("text")
    if lead:
        return f"上文「{lead}」→ 他回「{text}」"
    return f"他回「{text}」"


def _merge_profile(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in {"facts", "external_relations"} and isinstance(value, list):
            merged[key] = merge_persona_lists(merged.get(key), value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_profile(merged[key], value)
        else:
            merged[key] = value
    return merged
