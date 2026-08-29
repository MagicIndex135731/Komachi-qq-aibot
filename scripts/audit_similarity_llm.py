"""Style-similarity audit: compare live persona replies against the member's
own historical utterances, judged by DeepSeek flash.

The judge receives the member's real recent messages (with their contexts)
plus the system's replies in similar situations, and scores how close the
simulated voice is to the real one.

Usage (inside xiaomachi-bot container):
    DEEPSEEK_API_KEY=sk-... python scripts/audit_similarity_llm.py \
        --group-id <GROUP_ID> --bot-qq <BOT_QQ> \
        --persona-key <PERSONA_KEY> --requester-qq <REQUESTER_QQ>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import os
import time
from typing import Any

import httpx
import sqlite3

from app.adapters.onebot_models import parse_group_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.context_builder import ContextBuilder
from app.core.persona_switch import PersonaManager, PersonaSwitchService
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("audit_similarity_llm")


class FakeSender(Sender):
    def __init__(self) -> None:
        pass

    async def send_group_message(self, *args: Any, **kwargs: Any) -> str:
        return "fake"

    async def set_group_card(self, *args: Any, **kwargs: Any) -> None:
        return None


def build_event(*, group_id: int, user_id: int, text: str, bot_qq: int, bot_name: str):
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": f"sim-{int(time.time() * 1000)}",
        "group_id": group_id,
        "user_id": user_id,
        "time": int(time.time()),
        "message": [
            {"type": "at", "data": {"qq": str(bot_qq)}},
            {"type": "text", "data": {"text": f" {text}"}},
        ],
        "raw_message": f"[CQ:at,qq={bot_qq}] {text}",
        "sender": {"user_id": user_id, "nickname": "逆蝶蝶", "card": "逆蝶蝶"},
    }
    return parse_group_message_event(payload, bot_qq=bot_qq, bot_name=bot_name)


SIM_QUESTIONS = [
    ("你最近在忙什么", "daily"),
    ("晚上吃什么", "life"),
    ("你觉得这个新模型怎么样", "tech"),
    ("你最喜欢什么动画", "anime"),
    ("如何评价我", "interaction"),
    ("周末要不要一起打游戏", "game"),
    ("刚才那球居然输了，气死我了", "emotion"),
    ("你玩过彩虹六号吗", "niche"),
    ("（你刚说'魔法少女小圆吧'）那《少女歌剧》呢，你咋看", "followup"),
]


def embedding_similarity(provider, samples: list[dict], reply: str) -> float | None:
    """Objective style distance: mean cosine between the reply and the
    member's real utterances, as a second signal alongside the judge."""

    if provider is None or not getattr(provider, "available", False):
        return None
    texts = [sample["text"] for sample in samples[:30] if sample.get("text")]
    if not texts:
        return None
    try:
        sample_vectors = provider.embed_documents(texts)
        query_vector = provider.embed_query(reply or "")
    except Exception:  # noqa: BLE001
        return None
    if not query_vector:
        return None
    scores = []
    for vector in sample_vectors:
        if not vector:
            continue
        dot = sum(a * b for a, b in zip(query_vector, vector))
        norm_q = sum(v * v for v in query_vector) ** 0.5
        norm_s = sum(v * v for v in vector) ** 0.5
        if norm_q and norm_s:
            scores.append(dot / (norm_q * norm_s))
    return round(sum(scores) / len(scores), 3) if scores else None


def load_real_samples(*, db_path, user_id: int, group_id: int, limit: int = 30) -> list[dict]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT text, context_before, reply_target, timestamp "
            "FROM persona_style_examples WHERE user_id=? AND group_id=? "
            "ORDER BY timestamp DESC LIMIT ?",
            (int(user_id), int(group_id), int(limit)),
        ).fetchall()
    samples = []
    for text, ctx, reply, ts in rows:
        ctx_list = []
        try:
            parsed = json.loads(ctx or "[]")
            for item in parsed[-2:]:
                if isinstance(item, dict):
                    ctx_list.append(
                        f"{item.get('speaker','?')}: {str(item.get('text',''))[:40]}"
                    )
        except (json.JSONDecodeError, TypeError):
            pass
        samples.append(
            {
                "text": str(text or ""),
                "context": ctx_list,
                "reply_target": str(reply or ""),
                "timestamp": str(ts or ""),
            }
        )
    return samples


def judge_similarity(*, api_key: str, samples: list[dict], question: str, reply: str) -> dict:
    sample_block = "\n".join(
        f"- 上文「{' / '.join(s['context']) or s['reply_target'] or '无'}」→ 他回「{s['text']}」"
        for s in samples[:15]
    )
    prompt = (
        "你是群成员语言风格评审专家。下面是某个群成员的真实历史发言样本（带当时上下文），"
        "以及AI在相似情境下的模拟发言。请判断模拟发言和本人发言的相似度。\n\n"
        f"【本人真实发言样本】\n{sample_block}\n\n"
        f"【情境】群友问：{question}\n"
        f"【AI模拟发言】{reply}\n\n"
        "请从四个维度分别打分（1-10）：\n"
        "1. 句长与节奏：是否和本人一样短句、断句习惯一致\n"
        "2. 用词与口头禅：是否使用本人常用的词、语气词、网络用语\n"
        "3. 语气与态度：反问、吐槽、接梗、直接程度是否一致\n"
        "4. 情境贴合：在这种情境下本人会不会这样回\n"
        "最后给综合分（1-10，7分以上为像）。\n"
        '严格输出JSON：{"overall": 整数, "dimensions": {"sentence_length": 整数, '
        '"vocabulary": 整数, "tone": 整数, "situation_fit": 整数}, '
        '"pass": true/false, "matches": ["最像的地方"], "gaps": ["不像的地方"], '
        '"reason": "一句话总结"}'
    )
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "minimal",
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    last_error = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()
            parsed = json.loads(content)
            parsed["matches"] = list(parsed.get("matches") or [])
            parsed["gaps"] = list(parsed.get("gaps") or [])
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return {
        "overall": 0,
        "pass": False,
        "dimensions": {},
        "matches": [],
        "gaps": [f"裁判调用失败：{last_error}"],
        "reason": "judge_call_failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
    parser.add_argument("--persona-key", required=True)
    parser.add_argument("--requester-qq", type=int, required=True)
    parser.add_argument(
        "--out",
        default="",
    )
    parser.add_argument("--sample-limit", type=int, default=30)
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY is required")
        return 2

    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    llm_client = build_llm_client(settings=settings, engine=engine)
    llm_client.usage_recorder = lambda usage: None
    memory_runtime = build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        bot_display_name=str(runtime.persona.get("name", settings.bot_qq)),
        memory_enabled_group_ids={int(args.group_id)},
    )
    manager = PersonaManager(
        engine=engine,
        personas=getattr(runtime, "personas", {}) or {},
        default_persona=runtime.persona,
        embedding_provider=memory_runtime.embedding_provider,
    )
    manager.load_state()
    original_key = manager.active_key(int(args.group_id))
    router = InboundRouter(
        engine=engine,
        runtime=runtime,
        sender=FakeSender(),
        llm_client=llm_client,
        proactive_judge_client=None,
        reply_policy=ReplyPolicy(),
        context_builder=ContextBuilder(),
        admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
        web_search_client=None,
        dev_control_service=None,
        group_image_service=None,
        memory_compaction_service=None,
        memory_orchestrator=memory_runtime.memory_orchestrator,
        persona_manager=manager,
        persona_switch_service=PersonaSwitchService(
            manager=manager,
            sender=FakeSender(),
            bot_qq=int(args.bot_qq),
        ),
    )
    persona = runtime.personas.get(args.persona_key) or {}
    member_uid = int(persona.get("source_user_id") or 0)
    samples = load_real_samples(
        db_path=settings.sqlite_path,
        user_id=member_uid,
        group_id=int(args.group_id),
        limit=args.sample_limit,
    )
    print(f"samples loaded: {len(samples)}", flush=True)

    manager._group_keys[int(args.group_id)] = args.persona_key
    results = []
    try:
        for question, tag in SIM_QUESTIONS:
            event = build_event(
                group_id=int(args.group_id),
                user_id=int(args.requester_qq),
                text=question,
                bot_qq=int(args.bot_qq),
                bot_name=str(runtime.persona.get("name", args.bot_qq)),
            )
            try:
                prepared = router._prepare_group_reply(event)
                reply = (
                    router._generate_group_reply_text(event=event, prepared_reply=prepared)
                    if prepared.should_reply
                    else "[未触发回复]"
                )
            except Exception as exc:  # noqa: BLE001
                reply = f"[异常] {type(exc).__name__}: {exc}"
            print(f"[{tag}] Q={question} -> {reply[:80]!r}", flush=True)
            judge = judge_similarity(
                api_key=api_key,
                samples=samples,
                question=question,
                reply=reply,
            )
            sim = embedding_similarity(
                memory_runtime.embedding_provider,
                samples,
                reply,
            )
            results.append(
                {
                    "tag": tag,
                    "question": question,
                    "reply": reply,
                    "embedding_similarity": sim,
                    "judge": judge,
                }
            )
            time.sleep(0.5)
    finally:
        manager._group_keys[int(args.group_id)] = original_key

    overall_scores = [int(r["judge"].get("overall") or 0) for r in results]
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["judge"].get("pass")),
        "average_overall": round(sum(overall_scores) / len(overall_scores), 1)
        if overall_scores
        else 0,
        "scores": overall_scores,
    }
    out = Path(
        args.out
        or f"/workspace/data/personas/{args.persona_key}/eval/similarity_audit.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "persona_key": args.persona_key,
                "member_user_id": member_uid,
                "sample_count": len(samples),
                "summary": summary,
                "real_samples": samples[:15],
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    for result in results:
        judge = result["judge"]
        print(
            f"{result['tag']}|overall={judge.get('overall')}|"
            f"emb={result.get('embedding_similarity')}|"
            f"{'|'.join(judge.get('gaps') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
