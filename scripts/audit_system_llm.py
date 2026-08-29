"""LLM-judged end-to-end audit of persona impersonation.

Runs the real reply pipeline (memory retrieval + nova luna generation + burst
normalization) inside the production container, then judges every scenario
with the official DeepSeek API (deepseek-v4-flash, minimal reasoning).

Usage (inside xiaomachi-bot container):
    DEEPSEEK_API_KEY=sk-... python scripts/audit_system_llm.py \
        --group-id <GROUP_ID> --bot-qq <BOT_QQ>
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
import os
import time
from typing import Any

import httpx
import yaml

from app.adapters.onebot_models import parse_group_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.chat_style import split_burst_reply
from app.core.context_builder import ContextBuilder
from app.core.persona_switch import PersonaManager, PersonaSwitchService
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.core.time_utils import ASIA_SHANGHAI
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("audit_system_llm")


GROUP_ID = 0
BOT_QQ = 0
REQUESTER_QQ = 0
PERSONA_KEY = "azha"


class FakeSender(Sender):
    """Never sends anything; only satisfies the constructor contract."""

    def __init__(self) -> None:
        pass

    async def send_group_message(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "audit-fake-message-id"

    async def set_group_card(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def build_event(
    *,
    group_id: int,
    user_id: int,
    card: str,
    nickname: str,
    text: str,
    bot_name: str,
) -> Any:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": f"audit-{int(time.time() * 1000)}",
        "group_id": group_id,
        "user_id": user_id,
        "time": int(time.time()),
        "message": [
            {"type": "at", "data": {"qq": str(BOT_QQ)}},
            {"type": "text", "data": {"text": f" {text}"}},
        ],
        "raw_message": f"[CQ:at,qq={BOT_QQ}] {text}",
        "sender": {"user_id": user_id, "nickname": nickname, "card": card},
    }
    return parse_group_message_event(
        payload,
        bot_qq=BOT_QQ,
        bot_name=bot_name,
    )


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "xiao_yuan_dislike_guardrail",
        "persona_key": PERSONA_KEY,
        "question": "你最讨厌什么动画",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 不能把'排行低/评价一般'说成'讨厌'；只有记忆明确表示讨厌时才可列出。\n"
            "2. 若回复提到魔法少女小圆，必须是因为明确讨厌依据，否则视为失败。\n"
            "3. 语气要像普通群友随口回答，不能有AI腔。"
        ),
    },
    {
        "id": "requester_binding",
        "persona_key": PERSONA_KEY,
        "question": "如何评价我",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 回复评价的对象必须是提问者本人（群名片：逆蝶蝶），而不是阿渣自己或AI。\n"
            "2. 评价内容应来自对提问者的记忆/印象，语气符合阿渣与逆蝶蝶的熟络互怼关系。\n"
            "3. 不能答非所问。"
        ),
    },
    {
        "id": "third_person_member_binding",
        "persona_key": PERSONA_KEY,
        "question": "逆蝶蝶的动画喜好是什么",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 必须把'逆蝶蝶'绑定到群成员本人并从记忆中回答。\n"
            "2. 回答应包含具体的动画/作品喜好，不能泛泛而谈或说不知道。\n"
            "3. 不能把阿渣自己的喜好说成逆蝶蝶的。"
        ),
    },
    {
        "id": "azha_fact_retrieval",
        "persona_key": PERSONA_KEY,
        "question": "你最近面了哪些企业",
        "card": "阿渣",
        "nickname": "喜泽满灰多",
        "criteria": (
            "1. 这是阿渣本人被问自己的经历，应从共享记忆中检索出面试/求职相关事实。\n"
            "2. 回答应提到具体企业（如虾皮、联想等），至少有一个真实企业名。\n"
            "3. 不能凭空编造记忆里没有的企业。"
        ),
    },
    {
        "id": "style_fidelity",
        "persona_key": PERSONA_KEY,
        "question": "最近工作咋样",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 必须像阿渣：极短、口语、直接、可能带网络用语（nb/尿了/卧槽等）。\n"
            "2. 不能出现'主人''您好''请问''作为AI''亲'等客服或机器人腔。\n"
            "3. 不能长篇大论，通常不超过一两句。"
        ),
    },
    {
        "id": "burst_messages",
        "persona_key": PERSONA_KEY,
        "question": "周末有空吗，一起玩游戏？",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 阿渣约游戏时会短句连发（如'来了''上号''几点'分多条）。\n"
            "2. 输出应能被拆成多条短消息，或至少连续多个短句。\n"
            "3. 不能是一条长句。"
        ),
    },
    {
        "id": "relationship_address",
        "persona_key": PERSONA_KEY,
        "question": "你觉得加菲猫这人咋样",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 按阿渣与加菲猫的关系（球友、互怼）说话。\n"
            "2. 不能出现'主人''老公''亲爱的'等凭空发明的亲密称呼。\n"
            "3. 评价要具体，不能空泛。"
        ),
    },
    {
        "id": "switch_back_to_komachi",
        "persona_key": "default",
        "question": "你是谁",
        "card": "逆蝶蝶",
        "nickname": "不知道叫什么",
        "criteria": (
            "1. 这是切回默认人格（小町）后的回复，应恢复小町的傲娇/毒舌风格并自称小町。\n"
            "2. 不能残留阿渣的口吻（如'上号''尿了'）。\n"
            "3. 不能自称是阿渣或其他群成员。"
        ),
    },
]


def judge_scenario(
    *,
    api_key: str,
    scenario: dict[str, Any],
    reply: str,
    burst_messages: list[str],
) -> dict[str, Any]:
    """Ask official DeepSeek flash to judge one reply against criteria."""

    burst_text = " / ".join(burst_messages) if burst_messages else "（无拆分）"
    prompt = (
        "你是严格的系统评测裁判。下面是QQ群里AI模拟群成员人格后的真实回复。\n"
        f"场景：{scenario['id']}\n"
        f"群成员正在扮演：{scenario.get('persona_key')}\n"
        f"问题：{scenario['question']}\n"
        f"AI回复（已按burst拆分显示为：{burst_text}）：\n"
        f"{reply}\n"
        "评分标准：\n"
        f"{scenario['criteria']}\n"
        "请只依据上面回复判断，1-10打分（>=7为通过），并列出具体问题。\n"
        '严格输出JSON：{"score": 整数, "pass": true/false, '
        '"issues": ["问题1", ...], "reason": "一句话结论"}'
    )
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "minimal",
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    last_error: str | None = None
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
            parsed["issues"] = list(parsed.get("issues") or [])
            return parsed
        except Exception as exc:  # noqa: BLE001 - judge should degrade gracefully
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return {
        "score": 0,
        "pass": False,
        "issues": [f"裁判调用失败：{last_error}"],
        "reason": "judge_call_failed",
    }


def classify_problems(
    *,
    scenario_id: str,
    reply: str,
    judge: dict[str, Any],
) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    score = int(judge.get("score") or 0)
    issues = list(judge.get("issues") or [])
    if not issues and score >= 7:
        return problems
    if score < 4:
        level = "P1"
    elif score < 7:
        level = "P2"
    else:
        level = "P3"
    problems.append(
        {
            "level": level,
            "scenario": scenario_id,
            "issue": "；".join(issues) if issues else f"评分 {score}",
        }
    )
    return problems


def run_scenario(
    *,
    router: InboundRouter,
    manager: PersonaManager,
    scenario: dict[str, Any],
) -> tuple[str, list[str]]:
    # In-memory switch only: the audit must not fight the production writer
    # for the persona-state row, and it must never leave the group switched.
    manager._group_keys[int(GROUP_ID)] = scenario["persona_key"]
    event = build_event(
        group_id=GROUP_ID,
        user_id=REQUESTER_QQ,
        card=scenario["card"],
        nickname=scenario["nickname"],
        text=scenario["question"],
        bot_name=str(router.runtime.persona.get("name", BOT_QQ)),
    )
    prepared = router._prepare_group_reply(event)
    if not prepared.should_reply:
        return "[未触发回复]", []
    raw_reply = router._generate_group_reply_text(
        event=event,
        prepared_reply=prepared,
    )
    persona = manager.active_persona(GROUP_ID)
    burst = persona.get("burst") if isinstance(persona, dict) else None
    burst_messages = split_burst_reply(raw_reply, burst)
    return raw_reply, burst_messages


def main() -> int:
    global GROUP_ID, BOT_QQ, REQUESTER_QQ, PERSONA_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
    parser.add_argument("--requester-qq", type=int, required=True)
    parser.add_argument("--persona-key", default=PERSONA_KEY)
    parser.add_argument("--out", default="/workspace/data/personas/azha/eval/audit_llm.json")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--only", default="", help="Run a single scenario id")
    args = parser.parse_args()
    GROUP_ID, BOT_QQ, REQUESTER_QQ, PERSONA_KEY = (
        args.group_id,
        args.bot_qq,
        args.requester_qq,
        args.persona_key,
    )
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY is required")
        return 2

    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    llm_client = build_llm_client(settings=settings, engine=engine)
    memory_runtime = build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        bot_display_name=str(runtime.persona.get("name", settings.bot_qq)),
        memory_enabled_group_ids={int(GROUP_ID)},
    )
    manager = PersonaManager(
        engine=engine,
        personas=getattr(runtime, "personas", {}) or {},
        default_persona=runtime.persona,
        embedding_provider=memory_runtime.embedding_provider,
    )
    manager.load_state()
    original_key = manager.active_key(GROUP_ID)
    router = InboundRouter(
        engine=engine,
        runtime=runtime,
        sender=FakeSender(),
        llm_client=llm_client,
        proactive_judge_client=None,
        reply_policy=ReplyPolicy(),
        web_search_client=None,
        dev_control_service=None,
        group_image_service=None,
        memory_compaction_service=None,
        memory_orchestrator=memory_runtime.memory_orchestrator,
        persona_manager=manager,
        context_builder=ContextBuilder(),
        admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
        persona_switch_service=PersonaSwitchService(
            manager=manager,
            sender=FakeSender(),
            bot_qq=int(BOT_QQ),
        ),
    )

    results: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    try:
        for scenario in SCENARIOS:
            scenario_id = scenario["id"]
            if args.only and scenario_id != args.only:
                continue
            print(f"[scenario] {scenario_id} ...", flush=True)
            if args.judge_only:
                reply = ""
                burst_messages = []
            else:
                try:
                    reply, burst_messages = run_scenario(
                        router=router,
                        manager=manager,
                        scenario=scenario,
                    )
                except Exception as exc:  # noqa: BLE001
                    reply = (
                        f"[场景执行异常] {type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()[-1200:]}"
                    )
                    burst_messages = []
                    print(f"[error] {scenario_id}: {exc}", flush=True)
                    print(traceback.format_exc()[-1500:], flush=True)
            print(f"  reply: {reply[:180]!r}", flush=True)
            judge = judge_scenario(
                api_key=api_key,
                scenario=scenario,
                reply=reply or "（无回复）",
                burst_messages=burst_messages,
            )
            problems.extend(
                classify_problems(
                    scenario_id=scenario_id,
                    reply=reply,
                    judge=judge,
                )
            )
            results.append(
                {
                    "id": scenario_id,
                    "persona_key": scenario["persona_key"],
                    "question": scenario["question"],
                    "reply": reply,
                    "burst_messages": burst_messages,
                    "judge": judge,
                }
            )
            time.sleep(0.5)
    finally:
        manager._group_keys[int(GROUP_ID)] = original_key

    passed = sum(1 for item in results if item["judge"].get("pass"))
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "p1": sum(1 for p in problems if p["level"] == "P1"),
        "p2": sum(1 for p in problems if p["level"] == "P2"),
        "p3": sum(1 for p in problems if p["level"] == "P3"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "group_id": GROUP_ID,
                "bot_qq": BOT_QQ,
                "summary": summary,
                "problems": problems,
                "scenarios": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    for problem in problems:
        print(f"{problem['level']}|{problem['scenario']}|{problem['issue']}")
    return 1 if problems and any(p["level"] in {"P1", "P2"} for p in problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
