from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from app.agent import social_planner
from app.core.persona_engine import render_persona
from scripts import check_public_persona, status_components


def profile(**extra):
    return {"name": "检查小町", "example_lines": [f"例句{i:02d}" for i in range(16)], **extra}


def test_gate_checks_real_profiles_and_the_same_yaml_merge(tmp_path):
    (tmp_path / "persona.yaml").write_text(yaml.safe_dump(profile()), encoding="utf-8")
    (tmp_path / "persona_variants.yaml").write_text(yaml.safe_dump({"personas": {
        "mouthy": {"identity": "新身份"}, "quiet": {"example_lines": []}}}), encoding="utf-8")
    report = check_public_persona.check_config(tmp_path)
    assert report["status"] == "PASS"
    assert [item["rendered_examples"] for item in report["profiles"]] == [16, 16, 0]
    assert report["configured_examples"] == report["rendered_examples"] == 32
    assert "例句" not in json.dumps(report, ensure_ascii=False)


def test_gate_rejects_reverted_filter_even_when_config_has_no_examples(monkeypatch):
    monkeypatch.setattr(social_planner, "PUBLIC_PERSONA_FIELDS",
        tuple(key for key in social_planner.PUBLIC_PERSONA_FIELDS if key != "example_lines"))
    with pytest.raises(check_public_persona.PublicPersonaContractError, match="examples_incomplete"):
        check_public_persona.inspect_public_personas({"default": {"name": "无例句配置"}})


def test_gate_detects_budget_truncation_in_real_config():
    with pytest.raises(check_public_persona.PublicPersonaContractError, match="examples_incomplete"):
        check_public_persona.inspect_public_personas({"default": profile(background="过长" * 2200)})


@pytest.mark.parametrize("source_key", ["source_user_id", "source_group_id"])
def test_gate_detects_member_samples_exposed(monkeypatch, source_key):
    original = social_planner.SocialPlanner._render_public_persona

    def leaking_renderer(public):
        if source_key in public:
            return render_persona(public)
        return original(public)

    monkeypatch.setattr(social_planner.SocialPlanner, "_render_public_persona", staticmethod(leaking_renderer))
    with pytest.raises(check_public_persona.PublicPersonaContractError, match="member_examples_exposed"):
        check_public_persona.inspect_public_personas({"default": profile()})


def test_gate_detects_private_relationships_exposed(monkeypatch):
    monkeypatch.setattr(social_planner.SocialPlanner, "_render_public_persona", staticmethod(render_persona))
    with pytest.raises(check_public_persona.PublicPersonaContractError, match="private_fields_exposed"):
        check_public_persona.inspect_public_personas({"default": profile()})


def test_gate_preserves_sixteen_example_contract_and_skips_member_data():
    report = check_public_persona.inspect_public_personas({"default": profile(
        example_lines=[f"示例{i}" for i in range(24)]),
        "member": profile(source_user_id=900000101, background="私人内容" * 3000)})
    assert report["configured_examples"] == report["rendered_examples"] == 16
    assert report["profile_count"] == 1


def test_gate_cli_returns_failure_without_config_or_persona_contents(tmp_path, monkeypatch, capsys):
    (tmp_path / "persona.yaml").write_text("name: PRIVATE_SECRET_SENTINEL\nbackground: " + "长" * 4100
        + "\nexample_lines: [PRIVATE_EXAMPLE_SENTINEL]\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_public_persona", "--config-dir", str(tmp_path)])
    assert check_public_persona.main() == 1
    report = capsys.readouterr().out
    assert json.loads(report) == {"status": "FAIL", "reason": "public_persona_examples_incomplete"}
    assert "SECRET" not in report and "SENTINEL" not in report


def test_live_status_checks_loaded_runtime_and_fails_on_missing_examples(monkeypatch, capsys):
    runtime = SimpleNamespace(persona=profile(), personas={"member": profile(source_user_id=900000101)})
    monkeypatch.setattr("app.config.load_runtime_config", lambda _settings: runtime)
    assert status_components.check_public_persona(SimpleNamespace())
    assert "examples=16/16" in capsys.readouterr().out
    monkeypatch.setattr(social_planner, "PUBLIC_PERSONA_FIELDS",
        tuple(key for key in social_planner.PUBLIC_PERSONA_FIELDS if key != "example_lines"))
    assert not status_components.check_public_persona(SimpleNamespace())
    assert "FAIL  public_persona: public_persona_examples_incomplete" in capsys.readouterr().out


def test_status_main_cannot_report_success_when_persona_check_fails(monkeypatch):
    monkeypatch.setattr("sys.argv", ["status_components"])
    monkeypatch.setattr(status_components, "AppSettings", lambda: SimpleNamespace())
    monkeypatch.setattr(status_components, "check_provider_config", lambda _settings: True)
    monkeypatch.setattr(status_components, "check_storage", lambda _settings: True)
    monkeypatch.setattr(status_components, "check_public_persona", lambda _settings: False)
    assert status_components.main() == 1


