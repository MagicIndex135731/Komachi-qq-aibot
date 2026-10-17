import pytest

from infra.wsl.scripts.render_mihomo_config import render_page_config
from infra.wsl.scripts.runtime_egress import uses_mihomo


def test_page_group_is_only_japan_rate_one_and_selects_lowest_delay():
    config=render_page_config({'proxies':[
        {'name':'日本01 x1','type':'ss'}, {'name':'JP 02 ×1.0','type':'ss'},
        {'name':'日本03 x10','type':'ss'}, {'name':'日本04 x0.1','type':'ss'},
        {'name':'香港 x1','type':'ss'}]})
    assert [p['name'] for p in config['proxies']]==['日本01 x1','JP 02 ×1.0']
    group=config['proxy-groups'][0]
    assert group['type']=='url-test' and group['tolerance']==0 and group['lazy'] is False
    assert config['rules']==['MATCH,'+group['name']]
    assert config['bind-address']=='127.0.0.1' and not config['allow-lan']
    assert config['tun']['enable'] is False
    assert config['mixed-port']==18797


def test_no_rate_one_japan_fails_before_activation():
    with pytest.raises(ValueError):render_page_config({'proxies':[{'name':'日本 x2'},{'name':'香港 x1'}]})


def test_only_page_proxy_requires_service_without_global_proxy():
    assert uses_mihomo('WEB_PAGE_PROXY_URL=http://127.0.0.1:18797\nXIAOMACHI_HTTPS_PROXY=\n')


def test_container_crawler_proxy_requires_service_only_when_configured():
    assert uses_mihomo('X_CRAWLER_SESSION_FILE=/run/synthetic.json\n')
    assert not uses_mihomo('X_CRAWLER_SESSION_FILE=\nX_CRAWLER_PROXY_URL=http://127.0.0.1:18797\n')


def test_unmarked_rate_one_is_an_explicit_convention_and_excludes_other_rates():
    source={'proxies':[{'name':'日本Y01'},{'name':'日本Y02 | IEPL'}, {'name':'日本Y06 | x0.8'}]}
    with pytest.raises(ValueError):render_page_config(source)
    config=render_page_config(source,unmarked_rate_one=True)
    assert len(config['proxies'])==2


def test_reserved_windows_proxy_port_is_not_a_project_service():
    assert not uses_mihomo("WEB_PAGE_PROXY_URL=http://127.0.0.1:7897\n")


def test_japan_node_cannot_silently_use_excluded_hong_kong_dialer():
    with pytest.raises(ValueError, match='excluded proxy'):
        render_page_config({'proxies':[{'name':'日本 x1','dialer-proxy':'香港 x1'},
                                       {'name':'香港 x1'}]})
