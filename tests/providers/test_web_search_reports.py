import json

import httpx
import pytest

from app.providers.web_search import WebSearchClient, WebSearchProviderError


def client(payload, captured):
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=payload)
    return WebSearchClient(provider="wyzlab_alpha", base_url="https://synthetic.invalid/v1", api_key="synthetic",
        timeout_seconds=45, reasoning_effort="medium", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_plaintext_report_and_commands_do_not_discard_body_or_decode_cipher():
    captured = []
    c = client({"output": "full detailed body", "encrypted_output": "opaque",
        "results": [{"type": "image_result"}, *[{"type": "text_result", "title": str(i), "url": f"https://example.com/{i}",
            "ref_id": f"turn0search{i}", "snippet": "short"} for i in range(12)]]}, captured)
    report = c.search_report("details", max_results=8)
    assert len(report.results) == 8 and report.content == "full detailed body"
    assert report.results[0].ref_id == "turn0search0"
    assert captured[0]["reasoning"] == {"effort": "medium"}
    assert captured[0]["settings"]["external_web_access"] is True
    assert "store" not in captured[0]
    assert len(c.search("legacy", max_results=3)) == 3
    c.open_page("https://example.com/page", lineno=150)
    assert captured[-1]["commands"]["open"] == [{"ref_id": "https://example.com/page", "lineno": 150}]
    c.find_page("https://example.com/page", pattern="def patch")
    assert captured[-1]["commands"]["find"] == [{"ref_id": "https://example.com/page", "pattern": "def patch"}]


@pytest.mark.parametrize("payload", [{"output": "Internal Error ()\nUnable to resolve open call"},
    {"error": {"message": "credential in provider error"}}, [], {"output": 42}])
def test_http_200_inner_errors_and_bad_shapes_are_failures(payload):
    with pytest.raises(WebSearchProviderError):
        client(payload, []).open_page("https://example.com/page")


def test_provider_response_size_is_bounded():
    c = client({"output": "x" * (2 * 1024 * 1024)}, [])
    with pytest.raises(WebSearchProviderError, match="too_large"):
        c.search_report("large")
