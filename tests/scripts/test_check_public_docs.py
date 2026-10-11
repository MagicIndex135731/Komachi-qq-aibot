from pathlib import Path

from scripts.check_public_docs import audit_public_docs


def test_public_links_and_unicode_heading_are_resolved(tmp_path: Path):
    (tmp_path / "README.md").write_text("[guide](guide.md#你好-world)\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# 你好 World\n", encoding="utf-8")
    assert audit_public_docs(tmp_path, ("README.md",))["status"] == "PASS"


def test_existing_private_document_cannot_be_required_by_readme(tmp_path: Path):
    private = tmp_path / "docs/project/内部.md"
    private.parent.mkdir(parents=True)
    private.write_text("# 内部\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[private](docs/project/内部.md)", encoding="utf-8")
    assert audit_public_docs(tmp_path, ("README.md",))["status"] == "FAIL"


def test_missing_file_and_heading_fail(tmp_path: Path):
    (tmp_path / "README.md").write_text("[missing](missing.md) [heading](guide.md#absent)", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Exists\n", encoding="utf-8")
    assert len(audit_public_docs(tmp_path, ("README.md",))["errors"]) == 2
