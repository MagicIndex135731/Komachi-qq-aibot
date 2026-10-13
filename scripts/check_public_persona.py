"""Offline release gate for public persona examples; prints counts, never content."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.agent.social_planner import SocialPlanner
from app.config import _deep_merge_mapping, _read_yaml
from app.core.persona_engine import _normalize_traits


class PublicPersonaContractError(ValueError):
    """A safe error code without persona content or account identifiers."""


def _check_profile(profile: dict) -> dict:
    rendered = SocialPlanner._render_public_persona(profile)
    examples = _normalize_traits(profile.get("example_lines"))[:16]
    member = "source_user_id" in profile or "source_group_id" in profile
    if member:
        examples = []
    if examples:
        expected = "Example replies: " + " ".join(f"「{line}」" for line in examples) + "."
        if expected not in rendered:
            raise PublicPersonaContractError("public_persona_examples_incomplete")
    if len(rendered) > 4000:
        raise PublicPersonaContractError("public_persona_budget_exceeded")
    return {"configured_examples": len(examples), "rendered_examples": len(examples),
        "rendered_chars": len(rendered), "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()}


def inspect_public_personas(profiles: dict[str, dict]) -> dict:
    """Check actual profiles and synthetic coverage even when configuration has no examples."""
    if "default" not in profiles:
        raise PublicPersonaContractError("public_persona_default_missing")
    probe = {"name": "公开人格检查", "example_lines": [f"PUBLIC_EXAMPLE_{i:02d}" for i in range(16)],
        "facts": ["PRIVATE_FACT_SENTINEL"], "relationships": [{"member": "PRIVATE_RELATION_SENTINEL"}],
        "external_relations": [{"name": "PRIVATE_EXTERNAL_SENTINEL"}]}
    _check_profile(probe)
    if any(marker in SocialPlanner._render_public_persona(probe)
            for marker in ("PRIVATE_FACT_SENTINEL", "PRIVATE_RELATION_SENTINEL", "PRIVATE_EXTERNAL_SENTINEL")):
        raise PublicPersonaContractError("public_persona_private_fields_exposed")
    for source_key in ("source_user_id", "source_group_id"):
        member = dict(probe, **{source_key: 900000101})
        rendered = SocialPlanner._render_public_persona(member)
        if any(example in rendered for example in probe["example_lines"]):
            raise PublicPersonaContractError("public_persona_member_examples_exposed")
    reports = []
    for profile in profiles.values():
        if not isinstance(profile, dict):
            raise PublicPersonaContractError("public_persona_invalid_profile")
        # Member profiles are group-scoped data, not public bot configuration.
        if "source_user_id" not in profile and "source_group_id" not in profile:
            reports.append(_check_profile(profile))
    return {"status": "PASS", "profile_count": len(reports),
        "configured_examples": sum(item["configured_examples"] for item in reports),
        "rendered_examples": sum(item["rendered_examples"] for item in reports),
        "profiles": reports, "synthetic_examples": 16, "member_samples_excluded": True}


def check_config(config_dir: Path) -> dict:
    """Load the image's authored YAML with the same merge rules as runtime config."""
    default = _read_yaml(config_dir / "persona.yaml")
    profiles = {"default": default}
    variants_path = config_dir / "persona_variants.yaml"
    if variants_path.is_file():
        variants = _read_yaml(variants_path).get("personas", {})
        if not isinstance(variants, dict) or any(not isinstance(value, dict) for value in variants.values()):
            raise PublicPersonaContractError("public_persona_invalid_variants")
        profiles.update({key: _deep_merge_mapping(default, value) for key, value in variants.items()})
        # Runtime's default public job always reads runtime.persona directly.
        profiles["default"] = default
    return inspect_public_personas(profiles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()
    try:
        report = check_config(args.config_dir)
    except (ValueError, OSError, TypeError) as exc:
        reason = str(exc) if isinstance(exc, PublicPersonaContractError) else type(exc).__name__
        print(json.dumps({"status": "FAIL", "reason": reason}))
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
