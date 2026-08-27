"""History-driven persona distillation helpers.

Pure, testable extraction logic used by ``scripts/distill_azha_persona.py``.
The generated persona file and sample transcripts stay local-only; they are
derived from private chat history and must never be committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


_FENCED_YAML_PATTERN = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", re.DOTALL)


def load_history_records(*, history_dir: Path, group_id: int) -> list[dict]:
    """Load all JSONL records for one group, ordered by timestamp."""

    records: list[dict] = []
    for path in sorted(Path(history_dir).rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if int(item.get("group_id", 0) or 0) != int(group_id):
                    continue
                records.append(item)
    records.sort(key=lambda item: str(item.get("timestamp", "")))
    return records


def speaker_label(item: dict) -> str:
    card = str(item.get("group_card") or "").strip()
    nickname = str(item.get("nickname") or "").strip()
    return card or nickname or str(item.get("user_id", "unknown"))


def build_style_samples(
    *,
    records: list[dict],
    user_id: int,
    context_before: int = 6,
    context_after: int = 3,
    max_samples: int = 200,
    min_chars: int = 2,
) -> list[dict]:
    """Extract the target member's messages plus surrounding conversation."""

    target_user_id = int(user_id)
    samples: list[dict] = []
    for index, item in enumerate(records):
        if int(item.get("user_id", 0) or 0) != target_user_id:
            continue
        text = str(item.get("plain_text") or "").strip()
        if len(text) < min_chars:
            continue
        context_before_lines: list[dict] = []
        for other in records[max(0, index - context_before) : index]:
            other_text = str(other.get("plain_text") or "").strip()
            if other_text:
                context_before_lines.append(
                    {"speaker": speaker_label(other), "text": other_text}
                )
        context_after_lines: list[dict] = []
        for other in records[index + 1 : index + 1 + context_after]:
            other_text = str(other.get("plain_text") or "").strip()
            if other_text:
                context_after_lines.append(
                    {"speaker": speaker_label(other), "text": other_text}
                )
        samples.append(
            {
                "id": str(item.get("platform_msg_id") or index),
                "timestamp": item.get("timestamp"),
                "speaker": speaker_label(item),
                "text": text,
                "reply_to_msg_id": item.get("reply_to_msg_id"),
                "context_before": context_before_lines,
                "context_after": context_after_lines,
            }
        )
        if len(samples) >= max_samples:
            break
    return samples


def build_distill_prompt(
    *,
    samples: list[dict],
    target_name: str,
    max_chars: int = 60000,
) -> list[str]:
    """Build the LLM prompt that produces a persona YAML profile."""

    sample_lines: list[str] = []
    budget = max(1000, int(max_chars))
    for sample in samples:
        line = json.dumps(sample, ensure_ascii=False)
        if sum(len(existing) for existing in sample_lines) + len(line) > budget:
            break
        sample_lines.append(line)
    transcript = "\n".join(sample_lines)

    instructions = (
        f"你是人设蒸馏专家。下面是从 QQ 群聊历史中提取的、目标成员（{target_name}）的发言及其前后文。"
        "请从这些样本中蒸馏出该成员稳定的思考方式和说话语气，输出一个 YAML 配置文件（只输出一个 "
        "```yaml 代码块，不要任何其他文字），字段如下：\n"
        "- name: 固定为 {target_name}\n"
        "- identity: 该成员在群里是什么样的人（一句话身份）\n"
        "- core_traits: 3-8 个核心性格特质（列表）\n"
        "- speaking_style: 含 tone（英文描述语气）与 sentence_length（short/medium）\n"
        "- self_concept: 第一人称视角的自我认知（含身份、与群友的关系、典型心态）\n"
        "- speech_habits: 5-12 条具体说话习惯（句式、口头禅、接话方式、攻击/防御风格等）\n"
        "- style_avoid: 3-8 条禁区（绝不出现的语气、句式、正式客服腔等）\n"
        "- example_lines: 6-12 条最能代表其风格的原句（逐字引用样本，不得改写）\n"
        "要求：忠于样本、宁可保守不要脑补；口吻要能直接用于让 AI 扮演该成员；不要输出与样本无关的通用人格。"
        f"\n\n目标成员发言样本（JSONL，含前后文）：\n{transcript}"
    )
    return [instructions]


def parse_persona_yaml(text: str) -> dict:
    match = _FENCED_YAML_PATTERN.search(text or "")
    raw = match.group(1) if match else text
    data = yaml.safe_load(raw or "")
    if not isinstance(data, dict):
        raise ValueError("model output is not a YAML mapping")
    return data


def assemble_persona(
    profile: dict[str, Any],
    *,
    target_name: str,
    group_card: str,
    source_user_id: int | None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize a model-produced profile into the runtime persona schema."""

    persona: dict[str, Any] = {
        "name": target_name,
        "identity": str(profile.get("identity") or "").strip(),
        "core_traits": _as_string_list(profile.get("core_traits")),
        "speaking_style": {},
        "self_concept": str(profile.get("self_concept") or "").strip(),
        "speech_habits": _as_string_list(profile.get("speech_habits")),
        "style_avoid": _as_string_list(profile.get("style_avoid")),
        "example_lines": _as_string_list(profile.get("example_lines")),
        "group_card": group_card,
    }
    speaking_style = profile.get("speaking_style")
    if isinstance(speaking_style, dict):
        persona["speaking_style"] = dict(speaking_style)
    if source_user_id is not None and int(source_user_id) > 0:
        persona["source_user_id"] = int(source_user_id)
    resolved_aliases = [str(alias).strip() for alias in (aliases or []) if str(alias).strip()]
    if resolved_aliases:
        persona["aliases"] = resolved_aliases
    return persona


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []
