from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_x_credentials_mount_only_read_only_application_services():
    for name in ('docker-compose.yml', 'docker-compose.llbot.yml', 'docker-compose.snowluma.yml'):
        services = yaml.safe_load((ROOT / 'infra/wsl' / name).read_text())['services']
        for key, service in services.items():
            mounts = [m for m in service.get('volumes', []) if 'xiaomachi-secrets' in str(m)]
            assert mounts == (['./private-secrets:/run/xiaomachi-secrets:ro'] if key in {'xiaomachi', 'xiaomachi-private'} else [])
        assert services['xiaomachi']['network_mode'] == services['xiaomachi-private']['network_mode'] == 'host'


def test_installation_excludes_secrets_and_retires_automatic_browser_only():
    install = (ROOT / 'infra/wsl/scripts/install_linux_runtime.sh').read_text()
    assert "--exclude='infra/wsl/private-secrets'" in install
    assert 'install -d -m 0700 "${shared_dir}/private-secrets"' in install
    assert 'disable --now xiaomachi-x-browser.service' in install
    assert 'xiaomachi-x-login-manual.service' not in install
    assert not (ROOT / 'scripts/x_crawler_gateway.py').exists()
    assert not (ROOT / 'scripts/browser_media.py').exists()
