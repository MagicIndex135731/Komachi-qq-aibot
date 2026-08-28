"""History-driven persona distillation helpers.

Pure, testable extraction logic used by ``scripts/distill_azha_persona.py``.
The generated persona file and sample transcripts stay local-only; they are
derived from private chat history and must never be committed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml


_FENCED_YAML_PATTERN = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", re.DOTALL)
_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

BANNED_ADDRESS_TERMS = (
    "主人",
    "大人",
    "陛下",
    "殿下",
    "少爷",
    "老爷",
    "小姐",
    "夫君",
    "娘子",
    "您",
)


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
    explicit = str(item.get("speaker") or "").strip()
    if explicit:
        return explicit
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
        text = str(item.get("text") or item.get("plain_text") or "").strip()
        if len(text) < min_chars:
            continue
        context_before_lines: list[dict] = []
        for other in records[max(0, index - context_before) : index]:
            other_text = str(other.get("text") or other.get("plain_text") or "").strip()
            if other_text:
                context_before_lines.append(
                    {"speaker": speaker_label(other), "text": other_text}
                )
        context_after_lines: list[dict] = []
        for other in records[index + 1 : index + 1 + context_after]:
            other_text = str(other.get("text") or other.get("plain_text") or "").strip()
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


def compute_style_stats(records: Sequence[dict]) -> dict:
    """Mechanical, evidence-grounded statistics over a member's corpus."""

    texts = [
        str(record.get("text") or record.get("plain_text") or "").strip()
        for record in records
    ]
    texts = [text for text in texts if text]
    if not texts:
        return {"count": 0}

    lengths = sorted(len(text) for text in texts)

    def percentile(values: list[int], ratio: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, int(len(values) * ratio))
        return values[index]

    def fraction(predicate) -> float:
        return round(sum(1 for text in texts if predicate(text)) / len(texts), 3)

    def has_emoji(text: str) -> bool:
        return any(
            0x1F300 <= ord(char) <= 0x1FAFF or 0x2600 <= ord(char) <= 0x27BF
            for char in text
        )

    replied = sum(
        1 for record in records if bool(record.get("reply_to_msg_id"))
    )
    total_records = len(records)
    reply_fraction = round(replied / total_records, 3) if total_records else 0.0

    repeated = Counter(text for text in texts if len(text) >= 2)
    bigrams: Counter = Counter()
    for text in texts:
        compact = text.replace(" ", "")
        for index in range(len(compact) - 1):
            bigrams[compact[index : index + 2]] += 1

    laugh_prefixes = ("哈哈", "笑死", "草", "哈哈哈", "绷", "难绷", "艹", "草草")
    return {
        "count": len(texts),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "median_len": percentile(lengths, 0.5),
        "p90_len": percentile(lengths, 0.9),
        "max_len": lengths[-1],
        "single_char_fraction": fraction(lambda text: len(text) == 1),
        "ends_with_question_fraction": fraction(lambda text: text.endswith(("？", "?", "吗"))),
        "ends_with_exclaim_fraction": fraction(lambda text: text.endswith(("！", "!", "!!"))),
        "ends_with_tilde_fraction": fraction(lambda text: text.endswith(("~", "～", "…", "。。"))),
        "has_emoji_fraction": fraction(has_emoji),
        "starts_with_laugh_fraction": fraction(
            lambda text: text.startswith(laugh_prefixes)
        ),
        "reply_fraction": reply_fraction,
        "most_repeated_exact": repeated.most_common(20),
        "top_char_bigrams": bigrams.most_common(30),
    }


def compute_relationship_map(
    stream: Sequence[dict],
    *,
    user_id: int,
    exclude_user_ids: set[int] | frozenset[int] = frozenset(),
    max_members: int = 15,
    max_examples: int = 5,
) -> list[dict]:
    """Aggregate the target member's interactions with every other member."""

    target_user_id = int(user_id)
    excluded = {int(value) for value in exclude_user_ids}
    excluded.add(target_user_id)

    by_id: dict[str, dict] = {}
    latest_label: dict[int, str] = {}
    for row in stream:
        platform_id = str(row.get("platform_msg_id") or "")
        if platform_id:
            by_id[platform_id] = row
        member_id = int(row.get("user_id") or 0)
        label = speaker_label(row)
        if label:
            latest_label[member_id] = label

    state: dict[int, dict] = {}

    def member_state(member_id: int) -> dict:
        if member_id not in state:
            state[member_id] = {
                "user_id": member_id,
                "member": latest_label.get(member_id, str(member_id)),
                "interactions": 0,
                "azha_replied_to": 0,
                "replies_to_azha": 0,
                "examples": [],
                "mention_examples": [],
            }
        return state[member_id]

    for row in stream:
        row_user = int(row.get("user_id") or 0)
        reply_to = str(row.get("reply_to_msg_id") or "")
        quoted = by_id.get(reply_to)
        if quoted is None:
            continue
        quoted_user = int(quoted.get("user_id") or 0)
        if row_user == target_user_id and quoted_user not in excluded:
            other = member_state(quoted_user)
            other["azha_replied_to"] += 1
            other["interactions"] += 1
            row_text = str(row.get("text") or row.get("plain_text") or "").strip()
            if (
                row_text
                and len(other["mention_examples"]) < 3
                and row_text not in other["mention_examples"]
            ):
                other["mention_examples"].append(row_text)
            if len(other["examples"]) < max_examples:
                quoted_text = str(quoted.get("text") or quoted.get("plain_text") or "").strip()
                if quoted_text and row_text:
                    other["examples"].append(
                        {
                            "other": speaker_label(quoted),
                            "other_text": quoted_text,
                            "azha_text": row_text,
                        }
                    )
        elif quoted_user == target_user_id and row_user not in excluded:
            other = member_state(row_user)
            other["replies_to_azha"] += 1
            other["interactions"] += 1

    for row in stream:
        row_user = int(row.get("user_id") or 0)
        if row_user != target_user_id:
            continue
        text = str(row.get("text") or row.get("plain_text") or "").strip()
        if not text:
            continue
        for member_id, info in list(state.items()):
            label = info["member"]
            aliases = {label, label[-2:] if len(label) >= 2 else label}
            if any(alias and alias in text for alias in aliases):
                if len(info["mention_examples"]) < 3:
                    info["mention_examples"].append(text)

    ranked = sorted(
        (info for info in state.values() if info["interactions"] > 0),
        key=lambda info: info["interactions"],
        reverse=True,
    )
    return ranked[: max_members]


def build_profile_prompt(
    *,
    samples: list[dict],
    stats: dict[str, Any],
    relationships: list[dict],
    target_name: str,
    max_chars: int = 30000,
) -> list[str]:
    """Stage 1: distill a deep persona profile from the full corpus."""

    sample_lines: list[str] = []
    budget = max(1000, int(max_chars))
    for sample in samples:
        line = json.dumps(sample, ensure_ascii=False)
        if sum(len(existing) for existing in sample_lines) + len(line) > budget:
            break
        sample_lines.append(line)
    transcript = "\n".join(sample_lines)

    stats_block = json.dumps(stats, ensure_ascii=False, indent=2)
    relationships_block = json.dumps(relationships, ensure_ascii=False, indent=2)
    instructions = (
        f"你是人设蒸馏专家。目标是把 QQ 群成员（{target_name}）的真实聊天记录蒸馏成一份可直接驱动 AI 扮演该成员的深度画像。\n"
        "任务要求：只依据下面给出的真实语料（含机械统计和逐条发言样本），不要脑补任何语料之外的设定。\n"
        "输出：只输出一个 ```yaml 代码块，不要任何其他文字。字段如下，越具体越好，禁止泛泛而谈：\n"
        "- name: 固定为 {target_name}\n"
        "- identity: 他在群里是什么样的人（含身份、关系、常见话题，2-3 句）\n"
        "- core_traits: 8-15 个有语料依据的具体特质\n"
        "- speaking_style: 字典，包含 tone（英文风格描述）、sentence_length、emoji_level（none/low/medium/high）、reply_length（短/中/长）、opening_style、closing_style\n"
        "- self_concept: 400-800 字第一人称自我认知：我怎么看自己、怎么看群里这些人、什么话题我会接、什么情绪我怎么反应\n"
        "- speech_habits: 15-30 条，每条都指向真人可验证的口头禅、断句、接话方式、情绪表达、抬杠/自嘲/附和风格；能写具体词就写具体词\n"
        "- style_avoid: 8-15 条禁区，尤其覆盖：客服腔、解释腔、AI 腔、正式书面语、长篇大论、客套\n"
        "- relationships: 按互动数据逐人列出他对每个群友的关系，每项含 member/relation/how_azha_talks/address_terms/notes；有语料依据才写，没依据的不写\n"
        "- address_rules: 3-8 条他的称呼习惯（他怎么叫别人），并明确列出他绝不会用的称呼（如'主人''亲'这类与他不符的称呼）\n"
        "示例原句由程序确定性选取，你不需要输出 example_lines。\n"
        "关键要求：要把'像真人'的细节写满，包括他怎么起句、怎么收尾、什么时候发一个字的消息、怎么用语气词和表情；"
        "画像必须让 AI 在任意群聊话题下都能自然接出他的味道。\n\n"
        f"机械统计（基于全部 {stats.get('count', 0)} 条发言，务必让画像与统计一致）：\n{stats_block}\n\n"
        f"与其他群友的互动证据（用于写 relationships 和 address_rules，只依据这些证据）：\n{relationships_block}\n\n"
        f"发言样本（JSONL，含前后文）：\n{transcript}"
    )
    return [instructions]


def build_refine_prompt(
    *,
    draft_yaml: str,
    fresh_samples: list[dict],
    target_name: str,
    max_chars: int = 40000,
) -> list[str]:
    """Stage 2: validate the draft against held-out samples and revise it."""

    sample_lines: list[str] = []
    budget = max(1000, int(max_chars))
    for sample in fresh_samples:
        line = json.dumps(sample, ensure_ascii=False)
        if sum(len(existing) for existing in sample_lines) + len(line) > budget:
            break
        sample_lines.append(line)
    transcript = "\n".join(sample_lines)
    instructions = (
        f"下面是针对群成员（{target_name}）的初版人格画像，以及一组没有参与蒸馏的验证样本。\n"
        "请做风格校验：对照验证样本，找出画像里不像他、过于泛化、像客服或 AI 的地方，"
        "并输出修订后的完整 YAML（只输出一个 ```yaml 代码块）。修订要求：\n"
        "1. 保留原有字段结构，删除泛化表述，补充能从样本中观察到的具体细节；\n"
        "2. self_concept 和 speech_habits 里必须覆盖他的断句习惯、情绪反应和口头禅；\n"
        "3. 不要新增语料里没有的事实。\n\n"
        f"初版画像：\n{draft_yaml}\n\n"
        f"验证样本（JSONL，含前后文）：\n{transcript}"
    )
    return [instructions]


def select_examples(
    records: Sequence[dict],
    *,
    count: int = 36,
    banned_terms: Sequence[str] = BANNED_ADDRESS_TERMS,
) -> list[str]:
    """Deterministically choose verbatim examples, excluding banned terms."""

    usable: list[str] = []
    seen: set[str] = set()
    for record in records:
        text = str(record.get("text") or record.get("plain_text") or "").strip()
        if not text or text in seen:
            continue
        if any(term in text for term in banned_terms):
            continue
        seen.add(text)
        usable.append(text)
    if not usable:
        return []
    selected: list[str] = []
    step = max(1, len(usable) // max(1, count))
    for index in range(0, len(usable), step):
        selected.append(usable[index])
        if len(selected) >= count:
            break
    return selected


def parse_persona_yaml(text: str) -> dict:
    match = _FENCED_YAML_PATTERN.search(text or "")
    raw = match.group(1) if match else (text or "")
    raw = str(raw).strip()
    raw = re.sub(r"^```(?:yaml|yml)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = yaml.safe_load(raw or "")
    if not isinstance(data, dict):
        raise ValueError("model output is not a YAML mapping")
    return data


def parse_fenced_json(text: str) -> Any:
    match = _FENCED_JSON_PATTERN.search(str(text or ""))
    raw = match.group(1) if match else text
    data = json.loads(raw or "")
    if not isinstance(data, (dict, list)):
        raise ValueError("model output is not a JSON object or array")
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
    relationships = profile.get("relationships")
    if isinstance(relationships, list):
        persona["relationships"] = [
            item for item in relationships if isinstance(item, dict)
        ]
    address_rules = _as_string_list(profile.get("address_rules"))
    if address_rules:
        persona["address_rules"] = address_rules
    if isinstance(profile.get("burst"), dict):
        persona["burst"] = dict(profile["burst"])
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
