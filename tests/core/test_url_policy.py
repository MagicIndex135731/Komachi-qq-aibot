import pytest

from app.core.url_policy import explicitly_requests_urls, filter_reply_urls, url_reply_policy_instruction


def test_default_policy_removes_markdown_bare_and_scheme_urls() -> None:
    reply = "来源：[官方说明](https://example.com/docs)；备用 www.example.cn 和 b23.tv/abc。"

    filtered = filter_reply_urls(reply, allow_urls=False)

    assert filtered == "来源：官方说明；备用 和。"
    assert "http" not in filtered
    assert ".tv" not in filtered


def test_explicit_link_request_preserves_urls() -> None:
    text = "把官网链接发我"
    reply = "官网：https://example.com"

    assert explicitly_requests_urls(text) is True
    assert filter_reply_urls(reply, allow_urls=explicitly_requests_urls(text)) == reply
    assert "Include only the URLs needed" in url_reply_policy_instruction(text)


def test_general_web_search_request_does_not_count_as_link_request() -> None:
    text = "联网查一下这件事并总结"

    assert explicitly_requests_urls(text) is False
    assert "Do not include URLs" in url_reply_policy_instruction(text)


def test_filter_removes_parenthesized_citations_and_empty_brackets() -> None:
    reply = "第一条事实。([官方来源](https://example.com/a)) 第二条事实。() 第三条。（）"

    filtered = filter_reply_urls(reply, allow_urls=False)

    assert filtered == "第一条事实。 第二条事实。 第三条。"
    assert "(" not in filtered
    assert "（" not in filtered


def test_filter_removes_url_only_line_with_label_but_keeps_answer() -> None:
    reply = "套餐以当天页面为准。\n发我截图，我陪你核对。\n查到的门店页：https://example.com/shop"
    assert filter_reply_urls(reply, allow_urls=False) == "套餐以当天页面为准。\n发我截图，我陪你核对。"
    assert filter_reply_urls(reply, allow_urls=True) == reply


def test_filter_keeps_source_labels_with_meaningful_text_and_plain_headings() -> None:
    reply = "菜品：\n来源：门店说明 https://example.com/shop\n官网：https://example.com/shop"
    assert filter_reply_urls(reply, allow_urls=False) == "菜品：\n来源：门店说明"
@pytest.mark.parametrize('url,expected', [
    ('https://x.com/synthetic/status/900000000000000001', True),
    ('https://cdn.example/photo.JPG?size=orig', True),
    ('https://twitter.com/intent/tweet?url=https://cdn.example/photo.jpg', False),
    ('http://127.0.0.1/photo.jpg', False),
    ('javascript:photo.jpg', False),
])
def test_media_locator_hint_never_grants_private_or_share_url_access(url, expected):
    from app.core.url_policy import media_resource_url
    assert media_resource_url(url) is expected
