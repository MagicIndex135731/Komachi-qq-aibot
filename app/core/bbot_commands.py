from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import re

BBOT_TARGET_GROUP_ID = 123456789
BBOT_TARGET_QQ = 234567890
BBOT_ADMIN_DENIED_TEXT = "这个需求对应 BBot 管理员命令，但我现在还没有它的管理员权限，先不能代你执行。"

Matcher = Callable[["BbotCommandContext"], str | None]

_LEADING_MENTION_PATTERN = re.compile(r"^(?:@\S+\s*)+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_UID_PATTERN = re.compile(r"\b(\d{3,20})\b")
_FREQUENCY_PATTERN = re.compile(r"(\d{1,6})\s*秒")
_TRAILING_PARTICLE_PATTERN = re.compile(r"(?:的|吧|呀|啊|呢|吗|嘛|呗)+$")
_NOISE_PHRASES = (
    "帮我",
    "帮忙",
    "麻烦你",
    "麻烦",
    "给我",
    "我想",
    "我想看",
    "看看",
    "看下",
    "看一下",
    "查下",
    "查一下",
    "查查",
    "最新",
    "最近",
    "刚刚",
    "发了什么",
    "发了啥",
    "动态",
    "推文",
    "微博",
    "内容",
    "一下",
)


@dataclass(frozen=True, slots=True)
class BbotCommandContext:
    group_id: int
    mentioned_bot: bool
    raw_text: str
    text: str
    lowered: str


@dataclass(slots=True)
class BbotParsedCommand:
    command_id: str
    command_text: str | None = None
    denied_reason: str | None = None
    cache_platform: str | None = None


@dataclass(frozen=True, slots=True)
class BbotCommandSpec:
    command_id: str
    matcher: Matcher
    admin_required: bool = False
    cache_platform: str | None = None


class BbotCommandResolver:
    def __init__(self, *, extra_command_specs: Iterable[BbotCommandSpec] | None = None) -> None:
        self._command_specs = list(self._build_default_specs())
        if extra_command_specs is not None:
            self._command_specs.extend(extra_command_specs)

    def resolve(self, *, group_id: int, mentioned_bot: bool, plain_text: str) -> BbotParsedCommand | None:
        if group_id != BBOT_TARGET_GROUP_ID or not mentioned_bot:
            return None

        normalized = self._normalize_text(plain_text)
        if not normalized:
            return None
        context = BbotCommandContext(
            group_id=group_id,
            mentioned_bot=mentioned_bot,
            raw_text=plain_text,
            text=normalized,
            lowered=normalized.lower(),
        )

        for spec in self._command_specs:
            command_text = spec.matcher(context)
            if command_text is None:
                continue
            if spec.admin_required:
                return BbotParsedCommand(
                    command_id=spec.command_id,
                    denied_reason=BBOT_ADMIN_DENIED_TEXT,
                    cache_platform=spec.cache_platform,
                )
            return BbotParsedCommand(
                command_id=spec.command_id,
                command_text=command_text,
                cache_platform=spec.cache_platform,
            )
        return None

    def build_outbound_message(self, command_text: str) -> str:
        return f"[CQ:at,qq={BBOT_TARGET_QQ}] {command_text}"

    def _build_default_specs(self) -> list[BbotCommandSpec]:
        return [
            BbotCommandSpec("reload_config", matcher=self._match_reload_config, admin_required=True),
            BbotCommandSpec("restart_bot", matcher=self._match_restart_bot, admin_required=True),
            BbotCommandSpec("set_frequency", matcher=self._match_set_frequency, admin_required=True),
            BbotCommandSpec("enable_feature", matcher=self._match_enable_feature, admin_required=True),
            BbotCommandSpec("disable_feature", matcher=self._match_disable_feature, admin_required=True),
            BbotCommandSpec("add_bilibili_listener", matcher=self._match_bilibili_watch_admin_intent, admin_required=True),
            BbotCommandSpec("add_twitter_listener", matcher=self._match_twitter_watch_admin_intent, admin_required=True),
            BbotCommandSpec("bilibili_listener_list", matcher=self._match_bilibili_listener_list),
            BbotCommandSpec("twitter_listener_list", matcher=self._match_twitter_listener_list),
            BbotCommandSpec("live_list", matcher=self._match_live_list),
            BbotCommandSpec("today_anime", matcher=self._match_today_anime),
            BbotCommandSpec("random_joke", matcher=self._match_random_joke),
            BbotCommandSpec(
                "latest_bilibili_dynamic",
                matcher=self._match_latest_bilibili_dynamic,
                cache_platform="bilibili",
            ),
            BbotCommandSpec(
                "latest_tweet",
                matcher=self._match_latest_tweet,
                cache_platform="twitter",
            ),
        ]

    def _normalize_text(self, value: str) -> str:
        stripped = _LEADING_MENTION_PATTERN.sub("", value.strip())
        collapsed = _WHITESPACE_PATTERN.sub(" ", stripped)
        return collapsed.strip()

    def _match_bilibili_listener_list(self, context: BbotCommandContext) -> str | None:
        if ("b站" in context.text or "bilibili" in context.lowered) and "监听列表" in context.text:
            return "b站监听列表"
        return None

    def _match_twitter_listener_list(self, context: BbotCommandContext) -> str | None:
        if ("推特" in context.text or "twitter" in context.lowered or re.search(r"\bx\b", context.lowered)) and "监听列表" in context.text:
            return "推特监听列表"
        return None

    def _match_live_list(self, context: BbotCommandContext) -> str | None:
        if "直播列表" in context.text or "正在直播" in context.text or "谁在直播" in context.text:
            return "直播列表"
        return None

    def _match_today_anime(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if "今日新番" in text:
            return "今日新番"
        anime_words = ("新番", "动画", "番剧")
        if any(word in text for word in anime_words) and any(word in text for word in ("今天", "今日", "今天的", "今日的")):
            return "今日新番"
        return None

    def _match_random_joke(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if "烂梗" not in text:
            return None
        if "灰泽满" in text:
            return "随机烂梗 灰泽满"
        if "6657" in text:
            return "随机烂梗 6657"
        return "随机烂梗"

    def _match_latest_bilibili_dynamic(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if "动态" not in text:
            return None
        uid_match = _UID_PATTERN.search(text)
        if uid_match is not None:
            return f"最新动态 {uid_match.group(1)}"
        if not any(marker in text for marker in ("最新动态", "b站动态", "B站动态", "up动态", "UP动态")):
            return None
        target = self._extract_subject_before_keyword(text, keyword="动态")
        if target is None:
            return None
        return f"最新动态 {target}"

    def _match_latest_tweet(self, context: BbotCommandContext) -> str | None:
        text = context.text
        lowered = context.lowered
        if "推文" not in text and "twitter" not in lowered and "推特" not in text and not re.search(r"\bx\b", lowered):
            return None
        if "监听" in text:
            return None

        explicit = re.search(r"(?:最新推文)\s*[:：]?\s*(.+)$", text, re.IGNORECASE)
        if explicit is not None:
            target = self._clean_target(explicit.group(1))
            if target:
                return f"最新推文 {target}"

        for keyword, source in (("推文", text), ("推特", text), ("twitter", lowered)):
            target = self._extract_subject_before_keyword(source, keyword=keyword)
            if target is not None:
                return f"最新推文 {target}"
        return None

    def _match_reload_config(self, context: BbotCommandContext) -> str | None:
        return "重载配置" if "重载配置" in context.text else None

    def _match_restart_bot(self, context: BbotCommandContext) -> str | None:
        if "重启bot" in context.lowered or "重启机器人" in context.text:
            return "重启Bot"
        return None

    def _match_set_frequency(self, context: BbotCommandContext) -> str | None:
        if "设置频率" in context.text or _FREQUENCY_PATTERN.search(context.text):
            return "设置频率"
        return None

    def _match_enable_feature(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if ("开启" in text or "打开" in text) and "功能" in text:
            return "开启功能"
        if ("打开" in text or "开启" in text) and any(feature in text for feature in ("每日新番提醒", "随机烂梗播报", "无意义 @ 提醒")):
            return "开启功能"
        return None

    def _match_disable_feature(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if ("关闭" in text or "关掉" in text) and "功能" in text:
            return "关闭功能"
        if ("关闭" in text or "关掉" in text) and any(feature in text for feature in ("每日新番提醒", "随机烂梗播报", "无意义 @ 提醒")):
            return "关闭功能"
        return None

    def _match_bilibili_watch_admin_intent(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if any(prefix in text for prefix in ("添加b站监听", "删除b站监听")):
            return "添加b站监听"
        if "监听" not in text:
            return None
        if "b站" in text or "B站" in text or "up主" in text or "UP主" in text or "uid" in text.lower() or bool(_UID_PATTERN.search(text)):
            return "添加b站监听"
        return None

    def _match_twitter_watch_admin_intent(self, context: BbotCommandContext) -> str | None:
        text = context.text
        if any(prefix in text for prefix in ("添加推特监听", "删除推特监听")):
            return "添加推特监听"
        if "监听" not in text:
            return None
        if "推特" in text or "twitter" in context.lowered or "推文" in text:
            return "添加推特监听"
        return None

    def _extract_subject_before_keyword(self, text: str, *, keyword: str) -> str | None:
        prefix, _, _suffix = text.partition(keyword)
        cleaned = self._clean_target(prefix)
        if not cleaned:
            return None
        return cleaned

    def _clean_target(self, value: str) -> str | None:
        cleaned = value.strip(" \t:：，,。！？!?@")
        if not cleaned:
            return None
        for phrase in _NOISE_PHRASES:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = cleaned.replace("的b站", " ").replace("的B站", " ").replace("的推特", " ").replace("的twitter", " ")
        cleaned = cleaned.replace("在", " ")
        cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip(" \t:：，,。！？!?@")
        cleaned = _TRAILING_PARTICLE_PATTERN.sub("", cleaned).strip(" \t:：，,。！？!?@")
        if not cleaned:
            return None
        return cleaned
