from app.core.web_grounding import build_grounding_notes
from app.providers.web_search import WebPageContent, WebSearchResult


def test_grounding_notes_warn_when_local_lookup_has_no_exact_floor_evidence() -> None:
    notes = build_grounding_notes(
        target_text="西安电子科技大学南校区那家店最好吃，我想不到晚上吃什么直接帮我推荐一个",
        external_lookup=True,
        web_results=[
            WebSearchResult(
                title="西电南校区附近美食",
                snippet="整理了附近常见吃饭去处。",
                source="https://example.test/list",
                date="2026-05-09",
            )
        ],
        web_pages=[
            WebPageContent(
                title="南校区附近美食攻略",
                url="https://example.test/list",
                content="附近常见选择有面食、烧烤和简餐，适合晚饭和夜宵。",
            )
        ],
        recent_bot_replies=[],
    )

    assert any("do not claim an exact floor" in note.lower() for note in notes)


def test_grounding_notes_request_correction_when_new_floor_conflicts_with_previous_bot_claim() -> None:
    notes = build_grounding_notes(
        target_text="这个牛肉拉面具体在哪里",
        external_lookup=True,
        web_results=[
            WebSearchResult(
                title="竹园餐厅二楼牛肉拉面",
                snippet="档口位于竹园餐厅二楼。",
                source="https://example.test/map",
                date="2026-05-09",
            )
        ],
        web_pages=[
            WebPageContent(
                title="地图详情",
                url="https://example.test/map",
                content="牛肉拉面位于竹园餐厅二楼，靠近楼梯口。",
            )
        ],
        recent_bot_replies=["去吃竹园三层的牛肉拉面。"],
    )

    assert any("previous reply said 3层" in note for note in notes)
    assert any("current evidence points to 2层" in note for note in notes)


def test_grounding_notes_warn_when_current_evidence_is_itself_conflicting() -> None:
    notes = build_grounding_notes(
        target_text="这个牛肉拉面具体在哪里",
        external_lookup=True,
        web_results=[
            WebSearchResult(
                title="竹园二楼牛肉拉面",
                snippet="二楼档口。",
                source="https://example.test/map-1",
                date="2026-05-09",
            ),
            WebSearchResult(
                title="竹园三层牛肉拉面",
                snippet="三层窗口。",
                source="https://example.test/map-2",
                date="2026-05-09",
            ),
        ],
        web_pages=[],
        recent_bot_replies=[],
    )

    assert any("conflicting floor details" in note.lower() for note in notes)
