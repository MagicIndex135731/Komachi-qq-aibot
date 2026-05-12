from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re


FOLLOWUP_PHRASES = (
    "你觉得呢",
    "那你怎么看",
    "你刚才说的",
    "所以你的意思是",
)

TAKE_REQUEST_PHRASES = (
    "评价一个",
    "说句实话",
    "分析一个",
    "给个看法",
)

TIME_SENSITIVE_HINTS = (
    "最近",
    "新出的",
    "这季",
    "本季",
    "今年",
    "刚开播",
    "最新",
    "新闻",
    "天气",
    "评分",
    "口碑",
    "热度",
    "排名",
    "票房",
    "官宣",
)

EXPLICIT_SEARCH_HINTS = (
    "查一个",
    "查一查",
    "搜一个",
    "搜一搜",
    "搜索",
    "联网",
    "上网",
    "帮我查",
    "帮我搜",
)

QUERY_REWRITE_REPLACEMENTS = (
    ("联网搜索", " "),
    ("联网查", " "),
    ("搜索一个", " "),
    ("搜一个", " "),
    ("搜一搜", " "),
    ("查一个", " "),
    ("查一查", " "),
    ("评价一个", "评价"),
    ("分析一个", "分析"),
    ("介绍一个", "介绍"),
    ("帮我", " "),
    ("给我", " "),
    ("请你", " "),
    ("麻烦你", " "),
    ("麻烦", " "),
    ("重新", " "),
    ("客观", " "),
)

REFERENCE_MEDIA_HINTS = (
    "动画",
    "动画片",
    "新番",
    "番剧",
    "漫画",
    "电影",
    "剧场版",
    "剧集",
    "电视剧",
    "小说",
    "游戏",
    "作品",
    "角色",
    "作者",
    "导演",
    "声优",
)

REFERENCE_PUBLIC_TOPIC_HINTS = (
    "网上",
    "网传",
    "网友",
    "热搜",
    "新闻",
    "消息",
    "事件",
    "爆料",
    "传闻",
    "舆论",
    "评论区",
)

REFERENCE_DISCUSSION_HINTS = (
    "评价",
    "客观评价",
    "分析",
    "看法",
    "口碑",
    "风评",
    "争议",
    "值不值得看",
    "讲了什么",
    "介绍一个",
    "是不是一部好作品",
    "为什么很多人骂",
    "网上怎么说",
    "大家怎么说",
    "真的假的",
    "是真的吗",
)

CURRENT_DATETIME_HINTS = (
    "今天几号",
    "今天多少号",
    "今天是几号",
    "今天星期几",
    "今天周几",
    "今天礼拜几",
    "现在几点",
    "当前时间",
    "现在时间",
    "当前日期",
    "今天日期",
    "几月几号",
)

BOT_META_DISCUSSION_HINTS = (
    "这么卡",
    "太卡",
    "卡死",
    "又没回",
    "没回",
    "没反应",
    "回复慢",
    "不说话",
    "像gpt",
    "bot",
    "机器人",
    "ai",
)

SEARCH_VERIFICATION_SUBJECT_HINTS = (
    "你",
    "刚刚",
    "刚才",
    "上一条",
    "上条",
    "前面",
    "刚那条",
    "刚那轮",
    "那轮",
    "真的",
)

SMALLTALK_HINTS = (
    "你好",
    "您好",
    "早安",
    "晚安",
    "在吗",
    "在嘛",
    "哈喽",
    "哈哈",
    "谢谢",
    "辛苦",
    "我爱你",
    "想你",
    "贴贴",
    "么么",
    "可爱",
    "吃了吗",
    "你是谁",
    "你叫什么",
    "你叫啥",
    "你叫啥名",
    "你是干嘛的",
    "你是做什么的",
)

GENERAL_SEARCH_HINTS = (
    "推荐",
    "哪家",
    "哪儿",
    "哪里",
    "哪个",
    "什么",
    "怎么",
    "为何",
    "为什么",
    "如何",
    "多少",
    "几",
    "能不能",
    "值不值得",
    "值得吗",
    "评价",
    "测评",
    "口碑",
    "攻略",
    "排名",
    "排行",
    "对比",
    "区别",
    "好吃",
    "好喝",
    "好用",
    "好看",
    "附近",
    "校区",
    "大学",
    "学校",
    "医院",
    "酒店",
    "餐厅",
    "饭店",
    "店",
    "景点",
    "路线",
    "地铁",
    "公交",
    "票价",
    "营业时间",
    "地址",
    "电话",
    "外卖",
    "咖啡",
    "奶茶",
    "火锅",
    "烧烤",
)

SEARCH_QUERY_PUNCTUATION = re.compile(r"[@#\n\r\t,，。！？!?:;；、“”‘’（）()《》【】\[\]<>]+")
SEARCH_QUERY_WHITESPACE = re.compile(r"\s+")
GENERAL_SEARCH_QUESTION_PATTERN = re.compile(
    r"(什么|谁|哪|哪里|哪个|哪家|怎么|为何|为什么|如何|多少|几|能不能|值不值得|值得吗|是不是|是否|好不好)"
)


@dataclass(slots=True)
class AddressDecision:
    is_addressed: bool
    reason: str
    score: int


@dataclass(slots=True)
class SearchDecision:
    should_search: bool
    query: str
    reason: str


def detect_address_intent(
    *,
    text: str,
    bot_names: set[str],
    reply_to_bot: bool,
    quoted_bot: bool,
    bot_recently_participated: bool,
    recent_bot_message_count: int,
) -> AddressDecision:
    normalized = text.strip()
    contains_name = any(name and name in normalized for name in bot_names)
    if contains_name and any(hint in normalized for hint in BOT_META_DISCUSSION_HINTS):
        return AddressDecision(False, "bot_meta_discussion", 0)
    if contains_name:
        return AddressDecision(True, "named_bot", 10)
    if reply_to_bot:
        return AddressDecision(True, "reply_to_bot", 9)
    if quoted_bot:
        return AddressDecision(True, "quoted_bot", 8)
    if bot_recently_participated and any(phrase in normalized for phrase in FOLLOWUP_PHRASES):
        return AddressDecision(True, "recent_followup", 7)
    if recent_bot_message_count > 0 and any(phrase in normalized for phrase in TAKE_REQUEST_PHRASES):
        return AddressDecision(True, "take_request", 6)
    return AddressDecision(False, "not_addressed", 0)


def is_time_sensitive_request(text: str) -> bool:
    normalized = text.strip()
    if any(hint in normalized for hint in TIME_SENSITIVE_HINTS):
        return True
    return bool(
        re.search(
            r"(最近|新出|这季|本季|今年|刚开播|现在|今天).{0,8}(新番|动画|番剧|电影|剧集|上映|开播|播出|天气|新闻|票房|排名|评分|口碑|热度|官宣)",
            normalized,
        )
    )


def is_explicit_search_request(text: str) -> bool:
    normalized = text.strip()
    if is_search_verification_query(normalized):
        return False
    if any(hint in normalized for hint in EXPLICIT_SEARCH_HINTS):
        return True
    return bool(re.search(r"(联网|上网|搜索|查|搜).{0,8}(一个|一查|一搜)?", normalized))


def is_search_verification_query(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if not any(token in normalized for token in ("联网", "上网", "搜索", "搜", "查")):
        return False
    if not any(hint in normalized for hint in SEARCH_VERIFICATION_SUBJECT_HINTS):
        return False
    if not any(token in normalized for token in ("吗", "没", "是不是真的", "真的假的")):
        return False
    return bool(
        re.search(
            r"((你|刚刚|刚才|上一条|上条|前面|刚那条|刚那轮|那轮|真的).{0,8}(联网|上网|搜索|搜|查).{0,6}(了吗|了没|过吗|过没有|没|没有))"
            r"|((联网|上网|搜索|搜|查).{0,6}(了吗|了没|过吗|过没有|没|没有).{0,6}(你|刚刚|刚才|上一条|上条|前面|刚那条|刚那轮|那轮|真的)?)",
            normalized,
        )
    )


def needs_reference_search(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False

    has_media_subject = any(hint in normalized for hint in REFERENCE_MEDIA_HINTS)
    has_public_topic_subject = any(hint in normalized for hint in REFERENCE_PUBLIC_TOPIC_HINTS)
    has_reference_intent = any(hint in normalized for hint in REFERENCE_DISCUSSION_HINTS)

    if has_reference_intent and (has_media_subject or has_public_topic_subject):
        return True

    return bool(
        re.search(
            r"(这部|这个|这作|这番).{0,12}(值不值得看|讲了什么|是不是一部好作品|为什么很多人骂|网上怎么说|大家怎么说)",
            normalized,
        )
    )


def needs_external_lookup_search(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False

    has_local_or_real_world_subject = any(
        hint in normalized
        for hint in (
            "店",
            "餐厅",
            "饭店",
            "校区",
            "大学",
            "学校",
            "医院",
            "酒店",
            "景点",
            "地址",
            "电话",
            "营业时间",
            "人均",
            "外卖",
            "奶茶",
            "咖啡",
            "火锅",
            "烧烤",
        )
    )
    has_lookup_intent = any(
        hint in normalized
        for hint in (
            "推荐",
            "哪家",
            "哪里",
            "附近",
            "好吃",
            "好喝",
            "评价",
            "测评",
            "口碑",
            "攻略",
            "值得",
            "排名",
            "排行",
            "对比",
            "区别",
        )
    )

    if has_local_or_real_world_subject and has_lookup_intent:
        return True

    return bool(
        re.search(
            r"(附近|校区|大学|学校|医院|酒店|景点|店|餐厅|饭店).{0,12}(推荐|好吃|好喝|评价|测评|口碑|攻略|值得|排名|排行|对比|区别)",
            normalized,
        )
    )


def is_general_search_decision_candidate(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if is_search_verification_query(normalized):
        return False
    if any(hint in normalized for hint in SMALLTALK_HINTS):
        return False
    if needs_external_lookup_search(normalized) or needs_reference_search(normalized) or is_time_sensitive_request(normalized):
        return True
    if "?" in normalized or "？" in normalized:
        return True
    if GENERAL_SEARCH_QUESTION_PATTERN.search(normalized):
        return True
    return any(hint in normalized for hint in GENERAL_SEARCH_HINTS)


def build_forced_search_query(text: str, *, bot_names: set[str]) -> str:
    original = text.strip()
    if not original:
        return ""

    normalized = original
    for name in sorted((name.strip() for name in bot_names if name.strip()), key=len, reverse=True):
        normalized = normalized.replace(f"@{name}", " ")
        normalized = normalized.replace(name, " ")
    normalized = re.sub(r"@\S+", " ", normalized)

    for old, new in QUERY_REWRITE_REPLACEMENTS:
        normalized = normalized.replace(old, new)

    normalized = SEARCH_QUERY_PUNCTUATION.sub(" ", normalized)
    normalized = SEARCH_QUERY_WHITESPACE.sub(" ", normalized).strip()
    return normalized or original


def needs_current_datetime_context(text: str) -> bool:
    normalized = text.strip()
    if any(hint in normalized for hint in CURRENT_DATETIME_HINTS):
        return True
    if re.search(r"(今天|现在|当前).{0,6}(几号|日期|星期几|周几|礼拜几|几点|时间)", normalized):
        return True
    if re.search(r"(现在|当前).{0,6}(几年|几几年|哪一年)", normalized):
        return True
    return bool(re.search(r"(今年|明年|去年).{0,6}(几年|几几年|哪一年)", normalized))


def build_current_datetime_facts(now: datetime) -> list[str]:
    local_now = now.astimezone()
    return [
        f"Current local datetime: {local_now.strftime('%Y-%m-%d %H:%M:%S %z')[:-2] + ':' + local_now.strftime('%Y-%m-%d %H:%M:%S %z')[-2:]}",
        f"Current local date: {local_now.strftime('%Y-%m-%d')}",
        f"Current local weekday: {local_now.strftime('%A')}",
    ]


def normalize_relative_time_query(query: str, *, now: datetime) -> str:
    normalized = query.strip()
    if not normalized:
        return normalized

    local_now = now.astimezone()
    replacements = (
        ("今年", f"{local_now.year}年"),
        ("明年", f"{local_now.year + 1}年"),
        ("去年", f"{local_now.year - 1}年"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def parse_search_decision(text: str) -> SearchDecision:
    lines = text.splitlines()
    if len(lines) != 3:
        return SearchDecision(False, "", "malformed")

    first, second, third = lines
    if not first.startswith("SEARCH:"):
        return SearchDecision(False, "", "malformed")
    if not second.startswith("QUERY:"):
        return SearchDecision(False, "", "malformed")
    if not third.startswith("REASON:"):
        return SearchDecision(False, "", "malformed")

    search_value = first.split(":", maxsplit=1)[1].strip()
    if search_value not in {"yes", "no"}:
        return SearchDecision(False, "", "malformed")

    should_search = search_value == "yes"
    query = second.split(":", maxsplit=1)[1].strip()
    reason = third.split(":", maxsplit=1)[1].strip()
    if not should_search:
        return SearchDecision(False, "", reason or "model_declined")
    if not query:
        return SearchDecision(False, "", "empty_query")
    return SearchDecision(True, query, reason or "current-facts-needed")


def build_search_decision_prompt(
    *,
    bot_name: str,
    target_message: str,
    recent_messages: list[str],
    proactive_turn: bool,
    now: datetime,
) -> list[str]:
    mode = "proactive" if proactive_turn else "addressed"
    joined_recent = "\n".join(recent_messages[-8:])
    local_now = now.astimezone()
    return [
        f"System persona: Decide whether {bot_name} needs current web facts before replying.",
        "Safety rules: Reply with exactly three lines in this grammar: SEARCH: yes|no / QUERY: <text> / REASON: <text>.",
        f"Group policy: Search mode={mode}. Prefer search for real-world facts, local places, stores, recommendations, public events, reviews, and recent topics. Decline only for stable commonsense, personal chat, or group-internal talk.",
        f"Current local date: {local_now.strftime('%Y-%m-%d')}. Resolve relative time words like 今天、今年、明年、去年 against this date.",
        f"Recent messages:\n{joined_recent}" if joined_recent else "",
        f"Target message: {target_message}",
    ]
