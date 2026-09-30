from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import app.core.member_memory_backfill as member_memory_backfill
from app.core.member_memory_backfill import (
    MemberFactRefreshService,
    build_slices,
    extract_facts_from_lines,
    parse_review_output,
    review_facts,
    upsert_member_facts,
)
from app.storage.db import session_scope
from app.storage.models import (
    ConversationEpisode,
    EpisodeMessage,
    Group,
    MemberFactRefreshState,
    MemoryItem,
    Message,
    User,
)


def test_parse_review_output_extracts_drop_set() -> None:
    text = (
        "```json\n"
        '{"drop": ["他的外公保有记忆"], '
        '"reasons": {"他的外公保有记忆": "从失忆梗反推"}}\n'
        "```"
    )

    assert parse_review_output(text) == {"他的外公保有记忆"}


def test_parse_review_output_tolerates_missing_fence() -> None:
    assert parse_review_output('{"drop": ["x", "y"]}') == {"x", "y"}


@pytest.mark.parametrize("text", ("", "not-json-or-yaml", '{"reasons": {}}'))
def test_parse_review_output_rejects_invalid_contract(text: str) -> None:
    with pytest.raises(ValueError, match="member fact review provider"):
        parse_review_output(text)


def test_build_slices_overlaps_boundaries() -> None:
    slices = build_slices(
        ["一一一一一一", "二二二二二二二二", "三三三三三三三三", "四四四四四四四四"],
        slice_chars=8,
        overlap_lines=2,
    )

    assert slices[-1][:2] == ["二二二二二二二二", "三三三三三三三三"]
    assert "四四四四四四四四" in slices[-1]


def test_fact_extraction_and_review_use_low_reasoning(monkeypatch) -> None:
    efforts: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs):
            efforts.append(kwargs["reasoning_effort"])

        def generate_text(self, messages):
            if "候选：" in messages[0]:
                return '{"drop": []}'
            return '{"facts": []}'

    monkeypatch.setattr(member_memory_backfill, "LlmClient", FakeClient)
    settings = SimpleNamespace(
        llm_base_url="http://example.invalid",
        llm_api_key="test",
        llm_model="test",
        llm_fallback_model="test",
    )

    assert extract_facts_from_lines(settings, ["今天在学日语"]) == []
    candidates = [{"fact": "在学日语", "evidence": "今天在学日语"}]
    assert review_facts(settings, candidates) == candidates
    assert efforts == ["low", "low"]


def test_fact_extraction_reclassifies_current_process_ranking_as_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def generate_text(self, _messages):
            return (
                '{"facts":[{"kind":"preference","category":"工作",'
                '"fact":"该成员认为目前手上的流程中华为方向最好",'
                '"evidence":"目前手上的流程好像还真是华为的方向最好",'
                '"context_evidence":[]}]}'
            )

    monkeypatch.setattr(member_memory_backfill, "LlmClient", FakeClient)
    settings = SimpleNamespace(
        llm_base_url="http://example.invalid",
        llm_api_key="test",
        llm_model="test",
        llm_fallback_model="test",
    )

    facts = extract_facts_from_lines(
        settings,
        ["目前手上的流程好像还真是华为的方向最好"],
    )

    assert facts[0]["kind"] == "decision"


def test_fact_extraction_uses_neighbors_only_for_target_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        (
            "```json\n"
            '{"facts":[{"kind":"current","category":"动漫",'
            '"fact":"他最近在看 Re:Zero","evidence":"看完这集rw0",'
            '"context_evidence":["要不要把re0全看了"]}]}\n```',
            "```json\n"
            '{"facts":[{"kind":"current","category":"动漫",'
            '"fact":"他最近在看 Re:Zero","evidence":"要不要把re0全看了",'
            '"context_evidence":[]}]}\n```',
        )
    )

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def generate_text(self, _messages):
            return next(outputs)

    monkeypatch.setattr(member_memory_backfill, "LlmClient", FakeClient)
    settings = SimpleNamespace(
        llm_base_url="http://example.invalid",
        llm_api_key="test",
        llm_model="test",
        llm_fallback_model="test",
    )

    supported = extract_facts_from_lines(
        settings,
        ["看完这集rw0"],
        context_lines=["要不要把re0全看了", "动画前两季相当好看"],
    )
    unsupported = extract_facts_from_lines(
        settings,
        ["这集剧情怎么样"],
        context_lines=["要不要把re0全看了"],
    )

    assert supported == [
        {
            "kind": "current",
            "category": "动漫",
            "fact": "他最近在看 Re:Zero",
            "evidence": "看完这集rw0",
            "context_evidence": ["要不要把re0全看了"],
        }
    ]
    assert unsupported == []


def test_fact_extraction_prompt_preserves_structured_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def generate_text(self, messages):
            prompts.append(messages[0])
            return (
                '{"facts":[{"kind":"decision","category":"工作",'
                '"fact":"他选择星河科技","evidence":"星河科技",'
                '"context_evidence":["哪家？"]}]}'
            )

    monkeypatch.setattr(member_memory_backfill, "LlmClient", FakeClient)
    settings = SimpleNamespace(
        llm_base_url="http://example.invalid",
        llm_api_key="test",
        llm_model="test",
        llm_fallback_model="test",
    )

    facts = extract_facts_from_lines(
        settings,
        ["星河科技"],
        source_records=[
            {
                "role": "target",
                "source_msg_id": "answer",
                "user_id": "101",
                "sent_at": "2026-09-27T12:00:40+00:00",
                "content": "星河科技",
            }
        ],
        context_records=[
            {
                "role": "context",
                "source_msg_id": "question",
                "user_id": "102",
                "sent_at": "2026-09-27T12:00:20+00:00",
                "content": "哪家？",
            }
        ],
    )

    assert facts[0]["context_evidence"] == ["哪家？"]
    assert '"source_msg_id":"answer"' in prompts[0]
    assert '"source_msg_id":"question"' in prompts[0]
    assert '"role":"target"' in prompts[0]


def test_fact_extraction_rejects_empty_provider_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyClient:
        def __init__(self, **_kwargs):
            pass

        def generate_text(self, _messages):
            return ""

    monkeypatch.setattr(member_memory_backfill, "LlmClient", EmptyClient)
    settings = SimpleNamespace(
        llm_base_url="http://example.invalid",
        llm_api_key="test",
        llm_model="test",
        llm_fallback_model="test",
    )

    with pytest.raises(ValueError, match="empty text"):
        extract_facts_from_lines(settings, ["我在学日语"])


@pytest.mark.parametrize(
    ("provider_text", "message"),
    (
        ("{not-json", "malformed JSON/YAML"),
        ('{"items": []}', "no facts list"),
    ),
)
def test_fact_extraction_rejects_malformed_or_wrong_top_level_contract(
    monkeypatch: pytest.MonkeyPatch,
    provider_text: str,
    message: str,
) -> None:
    class ContractClient:
        def __init__(self, **_kwargs):
            pass

        def generate_text(self, _messages):
            return provider_text

    monkeypatch.setattr(member_memory_backfill, "LlmClient", ContractClient)
    settings = SimpleNamespace(
        llm_base_url="http://example.invalid",
        llm_api_key="test",
        llm_model="test",
        llm_fallback_model="test",
    )

    with pytest.raises(ValueError, match=message):
        extract_facts_from_lines(settings, ["我在学日语"])


def test_current_fact_upsert_keeps_target_and_neighbor_provenance(
    fact_refresh_database,
) -> None:
    observed_at = datetime(2026, 9, 17, 13, tzinfo=UTC)
    with session_scope(fact_refresh_database) as session:
        session.add(User(user_id=900000102, nickname="neighbor", group_card="neighbor"))
        session.flush()
        session.add_all(
            (
                Message(
                    platform_msg_id="neighbor-source",
                    group_id=900000001,
                    user_id=900000102,
                    timestamp=observed_at,
                    raw_json={},
                    plain_text="要不要把re0全看了",
                    msg_type="text",
                    mentioned_bot=False,
                ),
                Message(
                    platform_msg_id="target-source",
                    group_id=900000001,
                    user_id=900000101,
                    timestamp=observed_at,
                    raw_json={},
                    plain_text="看完这集rw0",
                    msg_type="text",
                    mentioned_bot=False,
                ),
            )
        )

    imported = upsert_member_facts(
        fact_refresh_database,
        group_id=900000001,
        user_id=900000101,
        facts=[
            {
                "kind": "current",
                "category": "动漫",
                "fact": "他最近在看 Re:Zero",
                "evidence": "看完这集rw0",
                "context_evidence": ["要不要把re0全看了"],
            }
        ],
    )

    assert imported == 1
    with session_scope(fact_refresh_database) as session:
        row = session.query(MemoryItem).one()
        assert row.memory_kind == "current"
        assert row.source_msg_ids == ["target-source", "neighbor-source"]
        assert row.valid_from == observed_at.replace(tzinfo=None)
        assert row.valid_until == (observed_at + timedelta(days=14)).replace(
            tzinfo=None
        )


def test_upsert_member_facts_keeps_multiple_kinds_from_one_source(
    fact_refresh_database,
) -> None:
    _seed_member_message(
        fact_refresh_database,
        platform_msg_id="multi-facet-source",
        plain_text="二面的面试官还是我实习时的面试官",
    )
    common = {
        "category": "工作",
        "fact": "该成员进入了二面阶段",
        "evidence": "二面的面试官还是我实习时的面试官",
    }

    imported = upsert_member_facts(
        fact_refresh_database,
        group_id=900000001,
        user_id=900000101,
        facts=[
            {**common, "kind": "event"},
            {**common, "kind": "relationship"},
        ],
    )

    assert imported == 2
    with session_scope(fact_refresh_database) as session:
        rows = session.query(MemoryItem).order_by(MemoryItem.memory_kind).all()
        assert [row.memory_kind for row in rows] == ["event", "relationship"]
        assert all(row.source_msg_ids == ["multi-facet-source"] for row in rows)


def test_offline_refresh_uses_verified_clarification_thread_sources(
    fact_refresh_database,
) -> None:
    started_at = datetime(2026, 9, 27, 5, tzinfo=UTC)
    with session_scope(fact_refresh_database) as session:
        session.add(User(user_id=900000102, nickname="questioner", group_card="questioner"))
        session.flush()
        rows = [
            Message(
                platform_msg_id="clarify-anchor",
                group_id=900000001,
                user_id=900000101,
                timestamp=started_at,
                raw_json={},
                plain_text="我现在主要考虑两家公司",
                msg_type="text",
                mentioned_bot=False,
            ),
            Message(
                platform_msg_id="clarify-question",
                group_id=900000001,
                user_id=900000102,
                timestamp=started_at + timedelta(seconds=10),
                raw_json={},
                plain_text="哪家？",
                msg_type="text",
                mentioned_bot=False,
            ),
            Message(
                platform_msg_id="clarify-followup-1",
                group_id=900000001,
                user_id=900000102,
                timestamp=started_at + timedelta(seconds=20),
                raw_json={},
                plain_text="哪个公司呀？",
                msg_type="text",
                mentioned_bot=False,
            ),
            Message(
                platform_msg_id="clarify-followup-2",
                group_id=900000001,
                user_id=900000102,
                timestamp=started_at + timedelta(seconds=30),
                raw_json={},
                plain_text="哪个方向？",
                msg_type="text",
                mentioned_bot=False,
            ),
            Message(
                platform_msg_id="clarify-answer",
                group_id=900000001,
                user_id=900000101,
                timestamp=started_at + timedelta(seconds=40),
                raw_json={},
                plain_text="星河科技",
                msg_type="text",
                mentioned_bot=False,
            ),
        ]
        session.add_all(rows)
        session.flush()
        episode = ConversationEpisode(
            group_id=900000001,
            start_message_id=int(rows[0].id),
            end_message_id=int(rows[-1].id),
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=40),
            status="closed",
            is_current=False,
            message_count=len(rows),
        )
        session.add(episode)
        session.flush()
        session.add_all(
            EpisodeMessage(
                episode_id=int(episode.id),
                message_id=int(row.id),
                group_id=900000001,
                ordinal=index,
            )
            for index, row in enumerate(rows)
        )
        answer_id = int(rows[-1].id)

    service = _fact_refresh_service(fact_refresh_database)
    with session_scope(fact_refresh_database) as session:
        answer = session.query(Message).filter_by(platform_msg_id="clarify-answer").one()
        context_rows = service._clarification_context_rows(
            group_id=900000001,
            target_rows=[answer],
        )

    context_by_text = {str(row.plain_text): int(row.id) for row in context_rows}
    assert "我现在主要考虑两家公司" in context_by_text
    assert context_by_text["哪家？"] < answer_id - 2

    imported = upsert_member_facts(
        fact_refresh_database,
        group_id=900000001,
        user_id=900000101,
        facts=[
            {
                "kind": "decision",
                "category": "工作",
                "fact": "该成员更倾向星河科技",
                "evidence": "星河科技",
                "context_evidence": ["哪家？"],
            }
        ],
        source_message_ids={answer_id},
        context_source_message_ids=set(context_by_text.values()),
    )

    assert imported == 1
    with session_scope(fact_refresh_database) as session:
        row = session.query(MemoryItem).one()
        assert row.source_msg_ids == ["clarify-answer", "clarify-question"]


def _seed_member_message(
    engine,
    *,
    platform_msg_id: str,
    plain_text: str,
    mentions_bot: bool = False,
) -> int:
    raw_message = (
        [{"type": "at", "data": {"qq": "900000999"}}]
        if mentions_bot
        else [{"type": "text", "data": {"text": plain_text}}]
    )
    with session_scope(engine) as session:
        row = Message(
            platform_msg_id=platform_msg_id,
            group_id=900000001,
            user_id=900000101,
            timestamp=datetime.now(UTC),
            raw_json={"message": raw_message},
            plain_text=plain_text,
            msg_type="text",
            mentioned_bot=mentions_bot,
        )
        session.add(row)
        session.flush()
        return int(row.id)


def _fact_refresh_service(
    engine,
    *,
    threshold: int = 10,
    member_allowlist: set[int] | None = None,
) -> MemberFactRefreshService:
    return MemberFactRefreshService(
        engine=engine,
        settings=SimpleNamespace(),
        group_ids={900000001},
        bot_qq=900000999,
        threshold=threshold,
        cooldown_seconds=86400.0,
        min_member_messages=50,
        member_allowlist=member_allowlist,
    )


@pytest.fixture
def fact_refresh_database(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        session.add(Group(group_id=900000001, group_name="test"))
        session.add(User(user_id=900000101, nickname="member", group_card="member"))
    return sqlite_engine


def test_refresh_failure_keeps_watermark_and_retries_same_messages(
    fact_refresh_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_id = _seed_member_message(
        fact_refresh_database,
        platform_msg_id="member-1",
        plain_text="我喜欢火锅",
    )
    with session_scope(fact_refresh_database) as session:
        session.add(
            MemberFactRefreshState(
                group_id=900000001,
                user_id=900000101,
                last_msg_id="0",
                last_refresh_at=datetime.now(UTC),
            )
        )

    service = _fact_refresh_service(fact_refresh_database)
    monkeypatch.setattr(service, "_refresh_bot_names", lambda: None)
    attempts: list[list[str]] = []

    def fail_extract(_settings, lines, **_kwargs):
        attempts.append(list(lines))
        raise RuntimeError("provider response was malformed")

    monkeypatch.setattr(member_memory_backfill, "extract_facts_from_lines", fail_extract)
    with pytest.raises(RuntimeError, match="malformed"):
        service._refresh_member(900000001, 900000101)

    with session_scope(fact_refresh_database) as session:
        state = session.get(MemberFactRefreshState, (900000001, 900000101))
        assert state is not None
        assert state.last_msg_id == "0"

    monkeypatch.setattr(
        member_memory_backfill,
        "extract_facts_from_lines",
        lambda _settings, lines, **_kwargs: attempts.append(list(lines)) or [],
    )
    monkeypatch.setattr(member_memory_backfill, "review_facts", lambda _settings, facts: facts)
    monkeypatch.setattr(member_memory_backfill, "upsert_member_facts", lambda *_args, **_kwargs: 0)
    service._refresh_member(900000001, 900000101)

    assert attempts == [["我喜欢火锅"], ["我喜欢火锅"]]
    with session_scope(fact_refresh_database) as session:
        state = session.get(MemberFactRefreshState, (900000001, 900000101))
        assert state is not None
        assert state.last_msg_id == str(message_id)


def test_tick_uses_pending_eligible_message_count_and_excludes_bot_mentions(
    fact_refresh_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_scope(fact_refresh_database) as session:
        session.add(
            MemberFactRefreshState(
                group_id=900000001,
                user_id=900000101,
                last_msg_id="0",
                last_refresh_at=datetime.now(UTC),
            )
        )
    for index in range(10):
        _seed_member_message(
            fact_refresh_database,
            platform_msg_id=f"mention-{index}",
            plain_text="@小町 测试",
            mentions_bot=True,
        )
    for index in range(9):
        _seed_member_message(
            fact_refresh_database,
            platform_msg_id=f"eligible-{index}",
            plain_text=f"普通发言 {index}",
        )

    service = _fact_refresh_service(fact_refresh_database, threshold=10)
    monkeypatch.setattr(service, "_refresh_bot_names", lambda: None)
    monkeypatch.setattr(
        member_memory_backfill,
        "_active_members",
        lambda *_args, **_kwargs: [900000101],
    )
    refreshed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        service,
        "_refresh_member",
        lambda group_id, user_id: refreshed.append((group_id, user_id)),
    )

    service._tick()
    assert refreshed == []

    _seed_member_message(
        fact_refresh_database,
        platform_msg_id="eligible-9",
        plain_text="达到阈值的普通发言",
    )
    service._tick()

    assert refreshed == [(900000001, 900000101)]


def test_bounded_replay_is_dry_run_by_default_and_never_moves_watermark(
    fact_refresh_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_id = _seed_member_message(
        fact_refresh_database,
        platform_msg_id="replay-target",
        plain_text="我最近在学日语",
    )
    with session_scope(fact_refresh_database) as session:
        session.add(
            MemberFactRefreshState(
                group_id=900000001,
                user_id=900000101,
                last_msg_id="777",
                last_refresh_at=datetime.now(UTC),
            )
        )
    service = _fact_refresh_service(fact_refresh_database)
    monkeypatch.setattr(service, "_refresh_bot_names", lambda: None)
    monkeypatch.setattr(
        member_memory_backfill,
        "extract_facts_from_lines",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run must not call the model")
        ),
    )

    report = service.replay_member_window(
        group_id=900000001,
        user_id=900000101,
        start_message_id=message_id,
        end_message_id=message_id,
    )

    assert report == {
        "dry_run": True,
        "scanned_messages": 1,
        "eligible_messages": 1,
        "facts": 0,
        "imported": 0,
    }
    with session_scope(fact_refresh_database) as session:
        state = session.get(MemberFactRefreshState, (900000001, 900000101))
        assert state is not None
        assert state.last_msg_id == "777"


def test_bounded_replay_apply_is_idempotent_and_preserves_watermark(
    fact_refresh_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_id = _seed_member_message(
        fact_refresh_database,
        platform_msg_id="replay-apply-target",
        plain_text="我最近在学日语",
    )
    with session_scope(fact_refresh_database) as session:
        session.add(
            MemberFactRefreshState(
                group_id=900000001,
                user_id=900000101,
                last_msg_id="777",
                last_refresh_at=datetime.now(UTC),
            )
        )
    service = _fact_refresh_service(
        fact_refresh_database,
        member_allowlist={900000101},
    )
    monkeypatch.setattr(service, "_refresh_bot_names", lambda: None)
    phrasings = iter(
        (
            ("他最近在学日语", "学习"),
            ("该成员目前正在学习日语", "生活"),
        )
    )

    def extract_with_variable_wording(*_args, **_kwargs):
        fact, category = next(phrasings)
        return [
            {
                "kind": "current",
                "category": category,
                "fact": fact,
                "evidence": "我最近在学日语",
                "context_evidence": [],
            }
        ]

    monkeypatch.setattr(
        member_memory_backfill,
        "extract_facts_from_lines",
        extract_with_variable_wording,
    )
    monkeypatch.setattr(
        member_memory_backfill,
        "review_facts",
        lambda _settings, facts: facts,
    )

    for _ in range(2):
        report = service.replay_member_window(
            group_id=900000001,
            user_id=900000101,
            start_message_id=message_id,
            end_message_id=message_id,
            dry_run=False,
        )
        assert report["facts"] == 1
        assert report["imported"] == 1

    with session_scope(fact_refresh_database) as session:
        rows = session.query(MemoryItem).all()
        assert len(rows) == 1
        assert rows[0].memory_kind == "current"
        assert rows[0].content == "该成员目前正在学习日语"
        state = session.get(MemberFactRefreshState, (900000001, 900000101))
        assert state is not None
        assert state.last_msg_id == "777"


def test_bounded_replay_rejects_member_outside_allowlist(
    fact_refresh_database,
) -> None:
    service = _fact_refresh_service(
        fact_refresh_database,
        member_allowlist={900000999},
    )

    with pytest.raises(ValueError, match="not enabled"):
        service.replay_member_window(
            group_id=900000001,
            user_id=900000101,
            start_message_id=1,
            dry_run=False,
        )


def test_bounded_replay_rejects_window_larger_than_limit(
    fact_refresh_database,
) -> None:
    first_id = _seed_member_message(
        fact_refresh_database,
        platform_msg_id="replay-overflow-1",
        plain_text="第一条",
    )
    _seed_member_message(
        fact_refresh_database,
        platform_msg_id="replay-overflow-2",
        plain_text="第二条",
    )
    service = _fact_refresh_service(fact_refresh_database)

    with pytest.raises(ValueError, match="exceeds max_messages"):
        service.replay_member_window(
            group_id=900000001,
            user_id=900000101,
            start_message_id=first_id,
            max_messages=1,
            dry_run=False,
        )
