from dataclasses import asdict

import pytest

from app.core.memory_compaction import (
    MemoryFact,
    build_memory_compaction_prompt,
    canonical_key,
    derive_explicit_memory_invalidations,
    is_single_value_profile_attribute,
    parse_memory_compaction_response,
    structured_digest,
)


def _fact(
    *,
    source_ids: list[str] | None = None,
    content: str = "Alice likes hotpot.",
    **overrides,
) -> dict:
    payload = {
        "kind": "preference",
        "subject_id": "Alice",
        "predicate": "likes",
        "object_text": "hotpot",
        "content": content,
        "importance": 4,
        "confidence": 0.8,
        "source_msg_ids": source_ids or ["m-1"],
        "valid_until": None,
        "ignored_by_parser": "do not persist",
    }
    payload.update(overrides)
    return payload


def test_parser_filters_fields_and_deduplicates_source_backed_facts() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice prefers hotpot.",
            "facts": [
                _fact(source_ids=["m-1"]),
                _fact(source_ids=["m-2"], content="Alice enjoys hotpot."),
            ],
            "unsafe_top_level": True,
        },
        allowed_source_msg_ids={"m-1", "m-2"},
    )

    assert result.summary == "Alice prefers hotpot."
    assert len(result.facts) == 1
    assert asdict(result.facts[0]) == {
        "kind": "preference",
        "subject_id": "Alice",
        "predicate": "likes",
        "object_text": "hotpot",
        "content": "Alice likes hotpot.",
        "importance": 4,
        "confidence": 0.8,
        "source_msg_ids": ("m-1", "m-2"),
        "valid_until": None,
    }


def test_parser_rejects_article_sized_member_plan_without_model_judging() -> None:
    copied_list = " ".join(
        f"{index}. 这是复制文章中的第{index}条说明"
        for index in range(1, 30)
    )
    result = parse_memory_compaction_response(
        {
            "summary": "copied material",
            "facts": [
                _fact(
                    kind="plan",
                    predicate="打算",
                    object_text="复制文章中的整段计划" * 30,
                    content="成员计划：" + "很长的复制内容" * 60,
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"Alice"},
        source_subject_ids={"m-1": "Alice"},
        source_contents={"m-1": copied_list},
    )

    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_parser_keeps_short_direct_member_plan() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice plans to travel.",
            "facts": [
                _fact(
                    kind="plan",
                    predicate="plans",
                    object_text="travel tomorrow",
                    content="Alice plans to travel tomorrow.",
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"Alice"},
        source_subject_ids={"m-1": "Alice"},
        source_contents={"m-1": "I plan to travel tomorrow."},
    )

    assert len(result.facts) == 1
    assert result.facts[0].kind == "plan"


def test_parser_quality_gate_rejects_preference_fragments() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "digest",
            "facts": [
                _fact(
                    source_ids=["m-1"],
                    predicate="likes",
                    object_text="你",
                    content="阿渣 likes 你",
                ),
                _fact(
                    source_ids=["m-2"],
                    object_text="冰美式",
                    content="阿渣喜欢喝冰美式",
                ),
                _fact(
                    source_ids=["m-3"],
                    kind="taboo",
                    predicate="dislikes",
                    object_text="北京",
                    content="阿渣不喜欢北京",
                ),
            ],
        },
        allowed_source_msg_ids={"m-1", "m-2", "m-3"},
    )

    assert result.rejected_fact_count == 1
    assert [fact.object_text for fact in result.facts] == ["冰美式", "北京"]


def test_parser_discards_hallucinated_source_ids_and_uses_summary_fallback() -> None:
    result = parse_memory_compaction_response(
        '{"summary":"unsafe summary","facts":[{"kind":"fact","subject_id":"group","predicate":"meeting","object_text":"Saturday","content":"Meeting is Saturday.","importance":4,"confidence":0.9,"source_msg_ids":["invented"],"valid_until":null}]}',
        allowed_source_msg_ids={"m-1"},
        fallback_text="Recent chat: no verified memory.",
    )

    assert result.summary == "unsafe summary"
    assert result.facts == ()


def test_parser_bad_json_returns_safe_summary_only_fallback() -> None:
    result = parse_memory_compaction_response("not json", allowed_source_msg_ids={"m-1"}, fallback_text="Recent chat: hello")

    assert result.summary == "Recent chat: hello"
    assert result.facts == ()


def test_parser_discards_unknown_subject_ids() -> None:
    result = parse_memory_compaction_response(
        {"summary": "x", "facts": [_fact()]},
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"10001", "group"},
    )

    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_parser_rejects_user_fact_citing_another_authors_message() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "A statement was made.",
            "facts": [
                {
                    "kind": "preference",
                    "subject_id": "42",
                    "predicate": "likes",
                    "object_text": "hotpot",
                    "content": "Alice likes hotpot.",
                    "importance": 4,
                    "confidence": 0.9,
                    "source_msg_ids": ["m-bob"],
                    "valid_until": None,
                }
            ],
        },
        allowed_source_msg_ids={"m-bob"},
        allowed_subject_ids={"42", "43", "group"},
        source_subject_ids={"m-bob": "43"},
    )

    assert result.facts == ()


def test_strict_parser_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError):
        parse_memory_compaction_response("not json", strict=True)


def test_strict_parser_raises_on_incomplete_schema() -> None:
    with pytest.raises(ValueError):
        parse_memory_compaction_response({"summary": "missing facts"}, strict=True)


def test_strict_parser_raises_on_blank_summary() -> None:
    with pytest.raises(ValueError):
        parse_memory_compaction_response({"summary": "   ", "facts": []}, strict=True)


def test_strict_parser_rejects_invalid_fact_without_losing_valid_summary() -> None:
    result = parse_memory_compaction_response(
        {"summary": "source-backed window summary", "facts": [{"kind": "preference"}]},
        strict=True,
    )

    assert result.summary == "source-backed window summary"
    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_parser_safely_normalizes_missing_content_and_blank_valid_until() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice has a current plan.",
            "facts": [
                {
                    "kind": "current", "subject_id": "42", "predicate": "plans",
                    "object_text": "visit Shanghai", "importance": 4, "confidence": 0.9,
                    "source_msg_ids": ["m-1"], "valid_until": "",
                }
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        strict=True,
    )

    assert result.facts[0].content == "42: plans visit Shanghai"
    assert result.facts[0].valid_until is None


def test_parser_normalizes_noncritical_model_format_variance() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "An event occurred.",
            "facts": [
                {
                    "kind": "event", "subject_id": "42", "predicate": "attended",
                    "object_text": "meeting", "content": "Alice attended the meeting.",
                    "importance": 7.2, "confidence": 1.4, "source_msg_ids": [123],
                    "valid_until": "unknown",
                }
            ],
        },
        allowed_source_msg_ids={"123"},
        allowed_subject_ids={"42"},
        source_subject_ids={"123": "42"},
        strict=True,
    )

    assert result.facts[0].importance == 5
    assert result.facts[0].confidence == 1.0
    assert result.facts[0].source_msg_ids == ("123",)
    assert result.facts[0].valid_until is None


def test_parser_rejects_single_author_personal_fact_mislabeled_as_group() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice made a decision.",
            "facts": [
                {
                    "kind": "decision",
                    "subject_id": "group",
                    "predicate": "decided",
                    "object_text": "resign",
                    "content": "Alice decided to resign.",
                    "importance": 4,
                    "confidence": 0.9,
                    "source_msg_ids": ["m-alice"],
                    "valid_until": None,
                }
            ],
        },
        allowed_source_msg_ids={"m-alice"},
        allowed_subject_ids={"42", "group"},
        source_subject_ids={"m-alice": "42"},
    )

    assert result.facts == ()


def test_canonical_key_normalizes_case_spacing_and_unicode() -> None:
    assert canonical_key("Preference", " A\u3000lice ", "LIKES", "Hotpot") == canonical_key(
        "preference", "a lice", "likes", " hotpot "
    )


def test_prompt_builder_has_localized_schema_and_citable_messages() -> None:
    chinese = build_memory_compaction_prompt(
        language="zh",
        previous_digest="Rolling group memory: old detail",
        messages=[{"platform_msg_id": "m-1", "plain_text": "Alice 喜欢火锅"}],
    )
    english = build_memory_compaction_prompt(
        language="en",
        messages=[{"message_id": "m-2", "content": "Alice likes hotpot"}],
    )

    assert "write summary and fact content in Chinese" in chinese
    assert "If any field is uncertain, omit that fact" in chinese
    assert "or expired as kind" in chinese
    assert "[m-1] Alice 喜欢火锅" in chinese
    assert "Rolling group memory:" not in chinese
    assert "Output exactly one compact JSON object" in english
    assert "[m-2] Alice likes hotpot" in english


def test_prompt_builder_includes_kind_semantic_guidance() -> None:
    chinese = build_memory_compaction_prompt(
        language="zh",
        messages=[{"platform_msg_id": "m-1", "plain_text": "hi"}],
    )
    english = build_memory_compaction_prompt(
        language="en",
        messages=[{"message_id": "m-2", "content": "hi"}],
    )

    assert "kind 语义" in chinese
    assert "decision" in chinese
    assert "称呼规则只能记录提出者本人适用" in chinese
    assert "明确说正在看、追或补某部作品" in chinese
    assert "只讨论剧情、角色或季度" in chinese
    assert "profile 不是兜底分类" in chinese
    assert "喜好、厌恶、观点" in chinese
    assert "Kind semantics" in english
    assert "Addressing rules may only be recorded when the requester" in english
    assert "explicitly says they are currently watching" in english
    assert "Merely discussing a plot, character, or season" in english
    assert "profile is not a fallback category" in english
    assert "40岁再看" in chinese
    assert "emit kind=expired" in chinese


def test_profile_parser_rejects_hypothetical_future_age_inference() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "用户讨论以后重看作品。",
            "facts": [
                _fact(
                    kind="profile",
                    subject_id="42",
                    predicate="age",
                    object_text="40岁",
                    content="42 目前 40 岁。",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        source_contents={"m-1": "等我40岁再看这部作品"},
    )

    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_profile_parser_rejects_chinese_hypothetical_future_age_inference() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "用户讨论以后重看作品。",
            "facts": [
                _fact(
                    kind="profile",
                    subject_id="42",
                    predicate="年龄阶段",
                    object_text="接近四十岁",
                    content="42 接近四十岁。",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        source_contents={"m-1": "等我四十岁再看全金属狂潮"},
    )

    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_profile_parser_accepts_direct_self_reported_age() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "用户直接说明年龄。",
            "facts": [
                _fact(
                    kind="profile",
                    subject_id="42",
                    predicate="age",
                    object_text="28岁",
                    content="42 今年 28 岁。",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        source_contents={"m-1": "我今年28岁"},
    )

    assert len(result.facts) == 1


def test_profile_parser_rejects_active_value_from_explicit_denial() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "用户否认旧国籍标签。",
            "facts": [
                _fact(
                    kind="profile",
                    subject_id="42",
                    predicate="nationality",
                    object_text="日本人",
                    content="42 是日本人。",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        source_contents={"m-1": "我不是日本人，那个说法不对"},
    )

    assert result.facts == ()


@pytest.mark.parametrize(
    ("predicate", "object_text", "content", "source_text"),
    (
        ("nationality", "日本人", "42 是日本人。", "你怎么知道我是日本人？"),
        ("nationality", "日本人", "42 是日本人。", "别人说我是日本人"),
        ("age", "40岁", "42 今年 40 岁。", "谁说我今年40岁？"),
    ),
)
def test_profile_parser_rejects_questions_and_attributed_claims(
    predicate: str,
    object_text: str,
    content: str,
    source_text: str,
) -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "用户讨论一项画像说法。",
            "facts": [
                _fact(
                    kind="profile",
                    subject_id="42",
                    predicate=predicate,
                    object_text=object_text,
                    content=content,
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        source_contents={"m-1": source_text},
    )

    assert result.facts == ()


def test_profile_parser_rejects_content_that_disagrees_with_object() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "用户直接说明国籍。",
            "facts": [
                _fact(
                    kind="profile",
                    subject_id="42",
                    predicate="nationality",
                    object_text="日本人",
                    content="42 是中国人。",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        source_contents={"m-1": "我是日本人"},
    )

    assert result.facts == ()


@pytest.mark.parametrize(
    "predicate",
    ("age", "nationality", "hometown", "年龄", "国籍", "常住地"),
)
def test_single_value_profile_attribute_recognizes_replacement_predicates(
    predicate: str,
) -> None:
    assert is_single_value_profile_attribute(predicate) is True


def test_single_value_profile_attribute_does_not_replace_multi_value_traits() -> None:
    assert is_single_value_profile_attribute("personality_trait") is False


def test_profile_canonical_key_normalizes_single_value_predicate_aliases() -> None:
    assert canonical_key("profile", "42", "居住地", "深圳") == canonical_key(
        "profile", "42", "location", "深圳"
    )
    assert canonical_key("profile", "42", "岁数", "40") == canonical_key(
        "profile", "42", "age", "40"
    )


@pytest.mark.parametrize(
    "predicate",
    (
        "年龄阶段",
        "所在地天气感受",
        "hometown_reference",
        "国籍与常住地",
        "籍贯或地域认同",
        "来自",
    ),
)
def test_single_value_profile_attribute_rejects_composite_or_derived_predicates(
    predicate: str,
) -> None:
    assert is_single_value_profile_attribute(predicate) is False


def test_parser_accepts_exact_source_backed_profile_invalidation() -> None:
    target_key = "profile|42|age|40岁"
    result = parse_memory_compaction_response(
        {
            "summary": "用户否认旧年龄。",
            "facts": [],
            "invalidations": [
                {
                    "target_canonical_key": target_key,
                    "source_msg_ids": ["deny-1"],
                    "reason": "explicit_denial",
                    "valid_until": "2026-08-24T20:39:19+08:00",
                }
            ],
        },
        allowed_source_msg_ids={"deny-1"},
        source_subject_ids={"deny-1": "42"},
        allowed_invalidation_targets={
            target_key: {
                "subject_id": "42",
                "memory_kind": "profile",
            }
        },
    )

    assert len(result.invalidations) == 1
    assert result.invalidations[0].target_canonical_key == target_key
    assert result.invalidations[0].source_msg_ids == ("deny-1",)


@pytest.mark.parametrize(
    ("target_key", "source_subject"),
    (("profile|42|age|unknown", "42"), ("profile|42|age|40岁", "43")),
)
def test_parser_rejects_unknown_or_cross_subject_invalidation(
    target_key: str,
    source_subject: str,
) -> None:
    allowed_key = "profile|42|age|40岁"
    result = parse_memory_compaction_response(
        {
            "summary": "否认。",
            "facts": [],
            "invalidations": [
                {
                    "target_canonical_key": target_key,
                    "source_msg_ids": ["deny-1"],
                    "reason": "explicit_denial",
                    "valid_until": None,
                }
            ],
        },
        allowed_source_msg_ids={"deny-1"},
        source_subject_ids={"deny-1": source_subject},
        allowed_invalidation_targets={
            allowed_key: {
                "subject_id": "42",
                "memory_kind": "profile",
            }
        },
    )

    assert result.invalidations == ()
    assert result.rejected_invalidation_count == 1


def test_prompt_lists_exact_active_correction_catalog() -> None:
    prompt = build_memory_compaction_prompt(
        language="zh",
        messages=[{"source_msg_id": "deny-1", "content": "我不是40岁"}],
        active_correction_targets=[
            {
                "target_canonical_key": "profile|42|age|40岁",
                "memory_kind": "profile",
                "subject_id": "42",
                "predicate": "age",
                "object_text": "40岁",
            }
        ],
    )

    assert "Active correction targets (catalog only, not evidence)" in prompt
    assert '"target_canonical_key": "profile|42|age|40岁"' in prompt


def test_direct_same_subject_denials_derive_exact_catalog_invalidations() -> None:
    invalidations = derive_explicit_memory_invalidations(
        messages=(
            {
                "source_msg_id": "deny-age",
                "user_id": "42",
                "plain_text": "我不是40岁",
            },
            {
                "source_msg_id": "deny-work",
                "user_id": "42",
                "plain_text": "我没看过全金属狂潮，也不看小说",
            },
        ),
        active_correction_targets=(
            {
                "target_canonical_key": "profile|42|age|40岁",
                "memory_kind": "profile",
                "subject_id": "42",
                "object_text": "40岁",
            },
            {
                "target_canonical_key": "preference|42|likes|全金属狂潮小说",
                "memory_kind": "preference",
                "subject_id": "42",
                "object_text": "全金属狂潮小说",
            },
        ),
    )

    assert {item.target_canonical_key for item in invalidations} == {
        "profile|42|age|40岁",
        "preference|42|likes|全金属狂潮小说",
    }


def test_direct_age_denial_matches_chinese_age_object_to_arabic_source() -> None:
    invalidations = derive_explicit_memory_invalidations(
        messages=(
            {
                "source_msg_id": "deny-age",
                "user_id": "42",
                "plain_text": "我什么时候40了，别乱说",
            },
        ),
        active_correction_targets=(
            {
                "target_canonical_key": "profile|42|年龄阶段|四十岁",
                "memory_kind": "profile",
                "subject_id": "42",
                "object_text": "四十岁",
            },
        ),
    )

    assert [item.target_canonical_key for item in invalidations] == [
        "profile|42|年龄阶段|四十岁"
    ]


def test_direct_age_denial_matches_real_near_age_correction_wording() -> None:
    invalidations = derive_explicit_memory_invalidations(
        messages=(
            {
                "source_msg_id": "deny-age",
                "user_id": "42",
                "plain_text": "不对啊年近40哪来的",
            },
        ),
        active_correction_targets=(
            {
                "target_canonical_key": "profile|42|年龄阶段|接近四十岁",
                "memory_kind": "profile",
                "subject_id": "42",
                "predicate": "年龄阶段",
                "object_text": "接近四十岁",
            },
        ),
    )

    assert [item.target_canonical_key for item in invalidations] == [
        "profile|42|年龄阶段|接近四十岁"
    ]


@pytest.mark.parametrize(
    "message",
    (
        "我没有40块钱",
        "没有，我今年就是40岁",
        "我没有说不喜欢全金属狂潮",
        "全金属狂潮没有库存，但我还是喜欢",
    ),
)
def test_deterministic_invalidation_rejects_unrelated_or_negated_denials(
    message: str,
) -> None:
    invalidations = derive_explicit_memory_invalidations(
        messages=(
            {"source_msg_id": "m-1", "user_id": "42", "plain_text": message},
        ),
        active_correction_targets=(
            {
                "target_canonical_key": "profile|42|age|40岁",
                "memory_kind": "profile",
                "subject_id": "42",
                "predicate": "age",
                "object_text": "40岁",
            },
            {
                "target_canonical_key": "preference|42|likes|全金属狂潮",
                "memory_kind": "preference",
                "subject_id": "42",
                "predicate": "likes",
                "object_text": "全金属狂潮",
            },
        ),
    )

    assert invalidations == ()


def test_direct_denial_does_not_cross_subject_or_fuzzy_match_unrelated_fact() -> None:
    invalidations = derive_explicit_memory_invalidations(
        messages=(
            {
                "source_msg_id": "deny-age",
                "user_id": "43",
                "plain_text": "我不是40岁",
            },
        ),
        active_correction_targets=(
            {
                "target_canonical_key": "profile|42|age|40岁",
                "memory_kind": "profile",
                "subject_id": "42",
                "object_text": "40岁",
            },
            {
                "target_canonical_key": "preference|43|likes|科幻小说",
                "memory_kind": "preference",
                "subject_id": "43",
                "object_text": "科幻小说",
            },
        ),
    )

    assert invalidations == ()


def test_addressing_rule_decision_is_remapped_to_preference() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "digest",
            "facts": [
                _fact(
                    kind="decision",
                    subject_id="20001",
                    predicate="称呼规则",
                    object_text="以后叫我主人",
                    content="20001: /grill-me 记录并执行以下要求：将对用户“20001”的回复中的称呼统一改为“主人”",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"20001"},
        source_subject_ids={"m-1": "20001"},
    )

    assert len(result.facts) == 1
    assert result.facts[0].kind == "preference"


def test_cross_person_addressing_rule_is_dropped() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "digest",
            "facts": [
                _fact(
                    kind="preference",
                    subject_id="10001",
                    predicate="称呼规则",
                    object_text="以后叫逆蝶蝶主人",
                    content="10001: 记录并执行：将对用户“20002”的回复中的称呼统一改为“主人”",
                    source_ids=["m-1"],
                )
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"10001"},
        source_subject_ids={"m-1": "10001"},
    )

    assert result.facts == ()


def test_structured_digest_is_deterministic_and_never_nests_rolling_labels() -> None:
    fact = MemoryFact(
        kind="preference",
        subject_id="Alice",
        predicate="likes",
        object_text="hotpot",
        content="Alice likes hotpot.",
        importance=4,
        confidence=0.8,
        source_msg_ids=("m-1",),
    )

    digest = structured_digest("Rolling group memory: Rolling group memory: Alice prefers hotpot.", [fact, fact])

    assert digest == (
        "Memory digest:\n"
        "summary: Alice prefers hotpot.\n"
        "facts:\n"
        "- preference | Alice | likes | hotpot | Alice likes hotpot. | sources=m-1 | valid_until=null"
    )


def test_structured_digest_can_compact_its_own_output_without_promoting_old_facts_into_summary() -> None:
    first = structured_digest("Rolling group memory: Alice prefers hotpot.")

    assert structured_digest(first) == "Memory digest:\nsummary: Alice prefers hotpot.\nfacts:\n- (none)"


def test_structured_digest_is_stable_for_source_order_and_equivalent_fact_order() -> None:
    first = MemoryFact(
        kind="Preference",
        subject_id=" Alice ",
        predicate="LIKES",
        object_text=" hotpot ",
        content="Alice likes hotpot.",
        importance=4,
        confidence=0.8,
        source_msg_ids=("m-2", "m-1"),
    )
    second = MemoryFact(
        kind="preference",
        subject_id="Alice",
        predicate="likes",
        object_text="hotpot",
        content="Alice Likes Hotpot.",
        importance=4,
        confidence=0.8,
        source_msg_ids=("m-1", "m-2"),
    )

    assert structured_digest("summary", [first, second]) == structured_digest("summary", [second, first])


def test_parser_rejects_fact_with_blank_or_non_string_source_id() -> None:
    result = parse_memory_compaction_response(
        {"summary": "test", "facts": [_fact(source_ids=["m-1", " "]), _fact(source_ids=["m-1", None])]},
        allowed_source_msg_ids={"m-1"},
    )

    assert result.facts == ()


def test_structured_digest_sorts_sources_for_a_single_direct_fact() -> None:
    unordered = MemoryFact(
        kind="fact",
        subject_id="group",
        predicate="meeting",
        object_text="Saturday",
        content="Meeting is Saturday.",
        importance=4,
        confidence=0.9,
        source_msg_ids=("m-2", "m-1"),
    )
    ordered = MemoryFact(
        kind="fact",
        subject_id="group",
        predicate="meeting",
        object_text="Saturday",
        content="Meeting is Saturday.",
        importance=4,
        confidence=0.9,
        source_msg_ids=("m-1", "m-2"),
    )

    assert structured_digest("summary", [unordered]) == structured_digest("summary", [ordered])


def test_structured_digest_breaks_complete_canonical_ties_stably() -> None:
    first = MemoryFact("Fact", "same", "P", "O", "same", 4, 0.8, ("m-1",))
    second = MemoryFact("fact", "same", "p", "o", "same", 4, 0.8, ("m-1",))

    assert structured_digest("summary", [first, second]) == structured_digest("summary", [second, first])
