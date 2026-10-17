from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize('name', ['docker-compose.yml', 'docker-compose.llbot.yml', 'docker-compose.snowluma.yml'])
def test_xhs_credentials_mount_only_in_chat_apps_not_qq_gateway(name):
    root=Path(__file__).resolve().parents[1]
    compose=yaml.safe_load((root/'infra/wsl'/name).read_text('utf-8'))
    for service,definition in compose['services'].items():
        mounted=any(v.startswith('xiaomachi_xhs_session:') for v in definition.get('volumes',[]) if isinstance(v,str))
        assert mounted == (service in {'xiaomachi','xiaomachi-private'})
    assert compose['volumes']['xiaomachi_xhs_session']['external'] is True
    assert compose['volumes']['xiaomachi_xhs_session']['name']=='xiaomachi-xhs-session'
