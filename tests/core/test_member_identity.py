from types import SimpleNamespace

from app.core.member_identity import (
    GroupMemberIdentity,
    classify_group_member_reference,
    group_member_identities_from_messages,
    resolve_group_member_reference,
)


def test_exact_member_resolution_groups_card_and_nickname_by_user_id() -> None:
    members = (
        GroupMemberIdentity(user_id=42, nickname="A-Zha", group_card="阿渣"),
        GroupMemberIdentity(user_id=43, nickname="Bob", group_card=""),
    )

    resolved = resolve_group_member_reference(" 阿渣！", members, match_mode="exact")

    assert resolved is not None
    assert resolved.user_id == 42
    assert resolved.matched_alias == "阿渣"


def test_member_resolution_rejects_unknown_duplicate_and_excluded_identity() -> None:
    duplicate = (
        GroupMemberIdentity(user_id=42, nickname="阿渣"),
        GroupMemberIdentity(user_id=43, group_card="阿渣"),
    )

    assert resolve_group_member_reference("阿渣", duplicate, match_mode="exact") is None
    assert resolve_group_member_reference("未知", duplicate, match_mode="exact") is None
    assert (
        resolve_group_member_reference(
            "阿渣",
            (GroupMemberIdentity(user_id=42, nickname="阿渣"),),
            match_mode="exact",
            exclude_user_ids={42},
        )
        is None
    )


def test_member_reference_classification_distinguishes_unbound_resolved_and_ambiguous() -> None:
    members = (
        GroupMemberIdentity(user_id=42, nickname="阿渣"),
        GroupMemberIdentity(user_id=43, group_card="阿渣"),
        GroupMemberIdentity(user_id=44, group_card="加菲猫"),
    )

    duplicate = classify_group_member_reference("阿渣喜欢什么", members, match_mode="contained")
    multiple = classify_group_member_reference("阿渣和加菲猫喜欢什么", members, match_mode="contained")
    unknown = classify_group_member_reference("未知人物喜欢什么", members, match_mode="contained")
    unique = classify_group_member_reference("加菲猫喜欢什么", members, match_mode="contained")

    assert duplicate.status == "ambiguous"
    assert multiple.status == "ambiguous"
    assert unknown.status == "unbound"
    assert unique.status == "resolved"
    assert unique.member is not None
    assert unique.member.user_id == 44


def test_member_identities_use_target_group_sender_snapshots_not_global_user_profile() -> None:
    messages = (
        SimpleNamespace(
            user_id=42,
            raw_json={"sender": {"nickname": "Target", "card": "本群名片"}},
        ),
    )

    members = group_member_identities_from_messages(messages)

    assert resolve_group_member_reference("本群名片", members, match_mode="exact") is not None
    assert resolve_group_member_reference("其他群名片", members, match_mode="exact") is None
