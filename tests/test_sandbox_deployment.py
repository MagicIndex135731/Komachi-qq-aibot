from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WSL = ROOT / "infra/wsl"


def test_optional_sandbox_overlay_keeps_bot_and_private_service_unchanged():
    overlay = yaml.safe_load((WSL / "docker-compose.sandbox.yml").read_text(encoding="utf-8"))
    assert set(overlay["services"]) == {"sandbox-base", "xiaomachi-sandbox"}
    assert overlay["services"]["xiaomachi-sandbox"]["profiles"] == ["sandbox"]
    assert overlay["services"]["sandbox-base"]["profiles"] == ["sandbox-build"]
    for base in ("docker-compose.llbot.yml", "docker-compose.snowluma.yml", "docker-compose.yml"):
        services = yaml.safe_load((WSL / base).read_text(encoding="utf-8"))["services"]
        for name in ("xiaomachi", "xiaomachi-private"):
            assert all("docker.sock" not in str(mount) for mount in services.get(name, {}).get("volumes", []))


def test_gateway_only_has_own_data_and_docker_socket_with_localhost_http():
    service = yaml.safe_load((WSL / "docker-compose.sandbox.yml").read_text(encoding="utf-8"))["services"]["xiaomachi-sandbox"]
    assert service["network_mode"] == "host"
    assert "ports" not in service
    assert "env_file" not in service
    assert service["environment"]["XIAOMACHI_SANDBOX_BIND"] == "127.0.0.1"
    assert service["environment"]["XIAOMACHI_SANDBOX_PORT"] == "8217"
    assert service["environment"]["XIAOMACHI_SANDBOX_TOKEN"] == "${XIAOMACHI_SANDBOX_TOKEN:-}"
    assert service["volumes"] == [
        "sandbox_data:/data", "/var/run/docker.sock:/var/run/docker.sock:ro"
    ]
    assert set(service["environment"]) == {
        "XIAOMACHI_SANDBOX_TOKEN", "XIAOMACHI_SANDBOX_DATA_DIR",
        "XIAOMACHI_SANDBOX_BIND", "XIAOMACHI_SANDBOX_PORT",
    }
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["mem_limit"] == "512m"
    assert service["cpus"] == 1
    assert service["pids_limit"] == 96


def test_build_targets_use_minimal_independent_context_whitelists():
    overlay = yaml.safe_load((WSL / "docker-compose.sandbox.yml").read_text(encoding="utf-8"))
    for name, context, dockerfile in [
        ("sandbox-base", "../..", "infra/wsl/Dockerfile.sandbox-base"),
        ("xiaomachi-sandbox", "../../app", "../infra/wsl/Dockerfile.sandbox-gateway"),
    ]:
        build = overlay["services"][name]["build"]
        assert build == {"context": context, "dockerfile": dockerfile, "network": "host"}
    base = (WSL / "Dockerfile.sandbox-base").read_text(encoding="utf-8")
    gateway = (WSL / "Dockerfile.sandbox-gateway").read_text(encoding="utf-8")
    assert not any(line.startswith(("COPY ", "ADD ")) for line in base.splitlines())
    copies = [line.split()[1] for line in gateway.splitlines() if line.startswith("COPY ")]
    assert copies == ["__init__.py", "agent/__init__.py", "agent/python_sandbox.py", "agent/sandbox_gateway.py"]
    destinations = [line.split()[2] for line in gateway.splitlines() if line.startswith("COPY ")]
    assert destinations == ["./app/" + name for name in copies]
    ignore = (WSL / "Dockerfile.sandbox-gateway.dockerignore").read_text(encoding="utf-8").splitlines()
    assert {line for line in ignore if line.startswith("!") and not line.endswith("/")} == {"!" + name for name in copies}
    assert "**" in ignore
    assert "**" in (WSL / "Dockerfile.sandbox-base.dockerignore").read_text(encoding="utf-8")


def test_readme_has_optional_overlay_operations_and_explicit_http_endpoint():
    text = (WSL / "README.md").read_text(encoding="utf-8")
    assert "docker-compose.sandbox.yml" in text
    assert "--profile sandbox" in text
    assert "http://127.0.0.1:8217" in text


def test_sandbox_supervision_starts_only_prebuilt_compute_and_keeps_secret_out_of_unit():
    text = (WSL / "systemd/xiaomachi-sandbox.service").read_text(encoding="utf8")
    assert "Requires=docker.service" in text
    assert "RemainAfterExit=yes" in text and "WantedBy=multi-user.target" in text
    assert text.count("--env-file /opt/xiaomachi/shared/sandbox.env") == 2
    assert "up -d --no-deps --no-build xiaomachi-sandbox" in text
    assert "stop --timeout 15 xiaomachi-sandbox" in text
    assert "XIAOMACHI_SANDBOX_TOKEN=" not in text
