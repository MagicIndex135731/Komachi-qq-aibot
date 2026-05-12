import httpx

from app.providers.web_search import WebSearchClient, WebSearchResult


def test_web_search_client_maps_tavily_results() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Official site",
                        "content": "Episode 1 aired and discussion focused on pacing.",
                        "url": "https://official.example/anime",
                        "published_date": "2026-05-01",
                    }
                ]
            },
        )

    client = WebSearchClient(
        provider="tavily",
        base_url="https://api.tavily.com/search",
        api_key="search-key",
        timeout_seconds=8.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    results = client.search("latest anime buzz", max_results=3)

    assert captured["url"] == "https://api.tavily.com/search"
    assert "latest anime buzz" in captured["payload"]
    assert len(results) == 1
    assert results[0].title == "Official site"
    assert results[0].snippet == "Episode 1 aired and discussion focused on pacing."
    assert results[0].source == "https://official.example/anime"
    assert results[0].date == "2026-05-01"


def test_web_search_client_returns_empty_results_when_api_key_is_blank() -> None:
    client = WebSearchClient(
        provider="tavily",
        base_url="https://api.tavily.com/search",
        api_key="   ",
        timeout_seconds=8.0,
    )

    assert client.search("latest anime buzz") == []


def test_web_search_client_treats_null_results_as_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"results": None})

    client = WebSearchClient(
        provider="tavily",
        base_url="https://api.tavily.com/search",
        api_key="search-key",
        timeout_seconds=8.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.search("latest anime buzz") == []


def test_web_search_client_maps_ddgs_text_results() -> None:
    captured: dict[str, object] = {}

    class FakeDdgs:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def text(self, query: str, **kwargs):
            captured["query"] = query
            captured["text_kwargs"] = kwargs
            return [
                {
                    "title": "Anime News Network",
                    "body": "Spring season impressions mention strong opening episodes.",
                    "href": "https://www.animenewsnetwork.com/",
                }
            ]

    client = WebSearchClient(
        provider="ddgs",
        base_url="",
        api_key="",
        timeout_seconds=8.0,
        ddgs_factory=FakeDdgs,
        region="wt-wt",
        backend="auto",
    )

    results = client.search("recent anime discussion", max_results=3)

    assert captured["query"] == "recent anime discussion"
    assert captured["kwargs"] == {"timeout": 8}
    assert captured["text_kwargs"] == {
        "region": "wt-wt",
        "safesearch": "moderate",
        "max_results": 3,
        "backend": "auto",
    }
    assert len(results) == 1
    assert results[0].title == "Anime News Network"
    assert results[0].snippet == "Spring season impressions mention strong opening episodes."
    assert results[0].source == "https://www.animenewsnetwork.com/"
    assert results[0].date == ""


def test_web_search_client_retries_ddgs_with_fallback_region_and_backend() -> None:
    calls: list[dict[str, object]] = []

    class FakeDdgs:
        def __init__(self, **kwargs) -> None:
            calls.append({"init": kwargs})

        def text(self, query: str, **kwargs):
            calls.append({"query": query, "kwargs": kwargs})
            if kwargs["region"] == "cn-zh" and kwargs["backend"] == "auto":
                raise RuntimeError("No results found.")
            return [
                {
                    "title": "Bangumi entry",
                    "body": "Fallback search recovered matching results.",
                    "href": "https://bgm.tv/subject/example",
                }
            ]

    client = WebSearchClient(
        provider="ddgs",
        base_url="",
        api_key="",
        timeout_seconds=8.0,
        ddgs_factory=FakeDdgs,
        region="cn-zh",
        backend="auto",
    )

    results = client.search("上伊那牡丹", max_results=2)

    assert calls == [
        {"init": {"timeout": 8}},
        {
            "query": "上伊那牡丹",
            "kwargs": {
                "region": "cn-zh",
                "safesearch": "moderate",
                "max_results": 2,
                "backend": "auto",
            },
        },
        {
            "query": "上伊那牡丹",
            "kwargs": {
                "region": "wt-wt",
                "safesearch": "moderate",
                "max_results": 2,
                "backend": "auto",
            },
        },
    ]
    assert len(results) == 1
    assert results[0].title == "Bangumi entry"
    assert results[0].snippet == "Fallback search recovered matching results."
    assert results[0].source == "https://bgm.tv/subject/example"


def test_web_search_client_reads_page_bodies_from_search_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://site.example/review":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="""
                <html>
                  <head>
                    <title>Review Page</title>
                    <style>body { color: red; }</style>
                  </head>
                  <body>
                    <main>
                      <h1>Episode Guide</h1>
                      <p>Episode 1 introduces the cast.</p>
                      <p>Episode 2 raises the conflict.</p>
                    </main>
                    <script>console.log('ignore me')</script>
                  </body>
                </html>
                """,
            )
        if str(request.url) == "https://site.example/forum":
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=utf-8"},
                text="Fans praised the visual direction.\nSome viewers disliked the pacing.",
            )
        return httpx.Response(404)

    client = WebSearchClient(
        provider="ddgs",
        base_url="",
        api_key="",
        timeout_seconds=8.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    pages = client.read_pages(
        [
            WebSearchResult(
                title="Review listing",
                snippet="Episode by episode breakdown.",
                source="https://site.example/review",
                date="2026-05-09",
            ),
            WebSearchResult(
                title="Forum thread",
                snippet="Mixed audience reactions.",
                source="https://site.example/forum",
                date="2026-05-09",
            ),
        ],
        max_pages=3,
    )

    assert len(pages) == 2
    assert pages[0].title == "Review Page"
    assert "Episode 1 introduces the cast." in pages[0].content
    assert "Episode 2 raises the conflict." in pages[0].content
    assert "console.log" not in pages[0].content
    assert pages[1].title == "Forum thread"
    assert "Fans praised the visual direction." in pages[1].content


def test_web_search_client_skims_more_results_before_selecting_best_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://site.example/wiki":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="""
                <html>
                  <head><title>Blue Archive Wiki</title></head>
                  <body>
                    <main>
                      <p>Character list and school factions.</p>
                      <p>Patch history and item locations.</p>
                    </main>
                  </body>
                </html>
                """,
            )
        if str(request.url) == "https://site.example/review":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="""
                <html>
                  <head><title>Blue Archive Anime Review</title></head>
                  <body>
                    <article>
                      <h1>Blue Archive Anime Review</h1>
                      <p>This review focuses on pacing, direction, and adaptation choices.</p>
                      <p>Episode-by-episode impressions explain why some viewers feel the tone drifts.</p>
                    </article>
                  </body>
                </html>
                """,
            )
        if str(request.url) == "https://site.example/analysis":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="""
                <html>
                  <head><title>Why Fans Criticized the Blue Archive Anime</title></head>
                  <body>
                    <main>
                      <p>Audience criticism centered on pacing, thin emotional buildup, and uneven episode structure.</p>
                      <p>The article compares staff decisions, adaptation cuts, and forum reactions.</p>
                    </main>
                  </body>
                </html>
                """,
            )
        if str(request.url) == "https://site.example/store":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="""
                <html>
                  <head><title>Blue Archive Merchandise Store</title></head>
                  <body>
                    <main>
                      <p>Buy acrylic stands, keychains, and posters.</p>
                    </main>
                  </body>
                </html>
                """,
            )
        return httpx.Response(404)

    client = WebSearchClient(
        provider="ddgs",
        base_url="",
        api_key="",
        timeout_seconds=8.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    pages = client.read_pages(
        [
            WebSearchResult(
                title="Blue Archive character wiki",
                snippet="Background info for every student.",
                source="https://site.example/wiki",
                date="2026-05-09",
            ),
            WebSearchResult(
                title="Blue Archive merchandise",
                snippet="Official goods and event bundles.",
                source="https://site.example/store",
                date="2026-05-09",
            ),
            WebSearchResult(
                title="Blue Archive anime review",
                snippet="Pacing and adaptation review.",
                source="https://site.example/review",
                date="2026-05-09",
            ),
            WebSearchResult(
                title="Why fans criticized the Blue Archive anime",
                snippet="Detailed analysis of complaints.",
                source="https://site.example/analysis",
                date="2026-05-09",
            ),
        ],
        query="blue archive anime review why fans criticized it",
        max_pages=2,
        skim_limit=4,
    )

    assert {page.url for page in pages} == {
        "https://site.example/review",
        "https://site.example/analysis",
    }
    joined = " ".join(page.content.lower() for page in pages)
    assert "pacing" in joined
    assert "criticism" in joined


def test_web_search_client_extracts_main_body_and_suppresses_navigation_noise() -> None:
    client = WebSearchClient(
        provider="ddgs",
        base_url="",
        api_key="",
        timeout_seconds=8.0,
    )

    title, content = client._extract_html_page_text(
        """
        <html>
          <head><title>Anime Review</title></head>
          <body>
            <header>
              <nav>
                <a>Home</a>
                <a>Trending</a>
                <a>Login</a>
                <a>Register</a>
              </nav>
            </header>
            <main>
              <article>
                <h1>Anime Review</h1>
                <p>The series starts with a sharp first episode and a clearly defined emotional hook.</p>
                <p>Later episodes divide viewers because the pacing speeds up and supporting arcs feel compressed.</p>
                <p>Even so, the visual direction and music remain the strongest part of the adaptation.</p>
              </article>
            </main>
            <aside>
              <p>Related topics</p>
              <p>Open app</p>
            </aside>
            <footer>
              <p>Privacy policy</p>
              <p>Sign in to comment</p>
            </footer>
          </body>
        </html>
        """,
        fallback_title="Fallback",
    )

    assert title == "Anime Review"
    assert "sharp first episode" in content
    assert "visual direction and music remain the strongest part" in content
    assert "Trending" not in content
    assert "Login" not in content
    assert "Privacy policy" not in content
