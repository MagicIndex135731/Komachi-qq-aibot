from __future__ import annotations

from pathlib import Path
import re


TEXT_FILE_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".ps1", ".json"}
SPECIAL_TEXT_FILE_NAMES = {".env"}
IGNORED_DIR_NAMES = {"data", ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "is",
    "of",
    "session",
    "the",
    "this",
    "to",
}
CHINESE_QUERY_HINTS = {
    "小町": ("xiaomachi", "bot"),
    "群聊": ("group",),
    "私聊": ("private",),
    "回复": ("reply",),
    "消息": ("message",),
    "图片": ("image",),
    "联网": ("web", "search"),
    "搜索": ("search",),
    "日志": ("log",),
    "配置": ("config",),
    "代码": ("code",),
    "仓库": ("repo",),
    "项目": ("repo", "code"),
    "启动": ("start",),
    "运行": ("runtime", "run"),
    "运作": ("runtime", "run"),
    "正常": ("status", "health"),
    "进程": ("process",),
    "上下文": ("context",),
    "记忆": ("memory",),
    "摘要": ("summary",),
    "开发": ("dev",),
    "检查": ("inspect",),
    "权限": ("permission", "whitelist", "private_chat_qqs", "private_chat_whitelist"),
    "白名单": ("whitelist", "allowlist", "private_chat_qqs", "private_chat_whitelist"),
    "名单": ("whitelist",),
}


def _iter_project_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES and path.name not in SPECIAL_TEXT_FILE_NAMES:
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.relative_to(repo_root).parts):
            continue
        yield path


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    lowered_query = query.lower()
    for token in TOKEN_PATTERN.findall(lowered_query):
        if token in ENGLISH_STOPWORDS or token in {"现在", "这个", "那个", "怎么", "为什么", "一下", "会话", "项目"}:
            continue
        if any("\u4e00" <= char <= "\u9fff" for char in token):
            continue
        tokens.append(token)
    for hint, expansions in CHINESE_QUERY_HINTS.items():
        if hint not in query:
            continue
        tokens.extend(expansions)
    return list(dict.fromkeys(tokens))


def build_repo_context_snippets(
    *,
    repo_root: Path,
    query: str,
    max_files: int = 3,
    max_lines_per_file: int = 4,
) -> list[str]:
    repo_root = repo_root.resolve()
    tokens = _query_tokens(query)
    if not tokens:
        return []

    scored: list[tuple[int, str]] = []
    for path in _iter_project_files(repo_root):
        relative_path = path.relative_to(repo_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError:
            continue

        lines = content.splitlines()
        score = 0
        excerpts: list[str] = []
        lowered_path = relative_path.lower()
        for index, line in enumerate(lines, 1):
            lowered_line = line.lower()
            matched = [token for token in tokens if token in lowered_line]
            if not matched:
                continue
            score += len(matched)
            if len(excerpts) < max_lines_per_file:
                excerpts.append(f"{index}: {line.strip()}")
        path_hits = sum(2 for token in tokens if token in lowered_path)
        score += path_hits
        if relative_path == ".env":
            score += 4
        if relative_path.startswith(("app/", "configs/")):
            score += 3
        if relative_path.startswith("tests/"):
            score -= 2
        if score <= 0:
            continue
        body = "\n".join(excerpts[:max_lines_per_file]) if excerpts else "(filename match only)"
        scored.append((score, f"{relative_path}\n{body}"))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [snippet for _score, snippet in scored[:max_files]]
