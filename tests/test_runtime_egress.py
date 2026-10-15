from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SPEC = spec_from_file_location("runtime_egress", Path(__file__).resolve().parents[1] / "infra/wsl/scripts/runtime_egress.py")
assert SPEC and SPEC.loader
egress = module_from_spec(SPEC)
SPEC.loader.exec_module(egress)


@pytest.mark.parametrize("content,expected", [
    ("", False),
    ("XIAOMACHI_HTTP_PROXY=\nXIAOMACHI_HTTPS_PROXY=\n", False),
    ('XIAOMACHI_HTTP_PROXY=""\nDOCKER_HTTP_PROXY=\n', False),
    ("XIAOMACHI_HTTPS_PROXY=http://127.0.0.1:18797\n", True),
    ("DOCKER_HTTP_PROXY=http://localhost:18797\n", True),
    ("XIAOMACHI_HTTP_PROXY=http://other.invalid:18797\n", False),
    ("XIAOMACHI_HTTP_PROXY=http://localhost:7898\n", False),
    ("XIAOMACHI_HTTP_PROXY=http://localhost:invalid\n", False),
    ("XIAOMACHI_HTTP_PROXY=http://other.invalid:18797\nDOCKER_HTTP_PROXY=http://localhost:18797\n", False),
])
def test_service_is_enabled_only_when_effective_providers_use_it(content, expected):
    assert egress.uses_mihomo(content) is expected


def test_network_modes_preserve_required_docker_and_unrelated_unit_settings():
    original = "[Unit]\nWants=network-online.target xiaomachi-mihomo.service\nAfter=network-online.target docker.service xiaomachi-mihomo.service\nRequires=docker.service\n[Service]\nExecStop=/known/stop.sh\n"
    direct = egress.configure_unit(original, required=False)
    assert "xiaomachi-mihomo" not in direct
    assert "Requires=docker.service" in direct and "ExecStop=/known/stop.sh" in direct
    proxy = egress.configure_unit(direct, required=True)
    assert proxy == egress.configure_unit(proxy, required=True)
    assert egress.configure_unit(proxy, required=False) == direct
