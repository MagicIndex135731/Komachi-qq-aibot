from pathlib import Path

from app.dev_control.repo_context import build_repo_context_snippets


def test_repo_context_prefers_project_files_and_skips_runtime_data(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "data").mkdir()
    (repo_root / "app" / "private_chat.py").write_text(
        "def handle_private_chat():\n    return 'project session'\n",
        encoding="utf-8",
    )
    (repo_root / "data" / "private_chat.log").write_text(
        "private chat runtime noise\n",
        encoding="utf-8",
    )

    snippets = build_repo_context_snippets(repo_root=repo_root, query="private chat session")

    assert any("app/private_chat.py" in snippet for snippet in snippets)
    assert all("data/private_chat.log" not in snippet for snippet in snippets)


def test_repo_context_returns_bounded_results(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    for index in range(1, 6):
        (repo_root / "app" / f"feature_{index}.py").write_text(
            f"def feature_{index}():\n    return 'session feature {index}'\n",
            encoding="utf-8",
        )

    snippets = build_repo_context_snippets(repo_root=repo_root, query="session feature", max_files=3)

    assert len(snippets) == 3


def test_repo_context_matches_chinese_runtime_query_with_code_terms(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "group_runtime.py").write_text(
        "def handle_group_reply():\n    return 'group reply runtime'\n",
        encoding="utf-8",
    )

    snippets = build_repo_context_snippets(repo_root=repo_root, query="检查一下小町群聊回复是否正常运作")

    assert any("app/group_runtime.py" in snippet for snippet in snippets)


def test_repo_context_prefers_runtime_config_files_over_tests_for_permission_lookup(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "tests").mkdir()
    (repo_root / "app" / "config.py").write_text(
        "private_chat_qqs = ''\ndef private_chat_whitelist():\n    return private_chat_qqs\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_config.py").write_text(
        "def test_private_chat_whitelist():\n    assert '20002'\n",
        encoding="utf-8",
    )

    snippets = build_repo_context_snippets(repo_root=repo_root, query="帮我查一下到底给没给20002私聊权限")

    assert snippets
    assert "app/config.py" in snippets[0]
