from app.admin.commands import AdminCommandParser, CommandContext


def test_group_allow_command_requires_whitelisted_sender() -> None:
    parser = AdminCommandParser(admin_whitelist={987654321})
    context = CommandContext(sender_qq=111111111, is_private_chat=True, group_id=None)

    command = parser.parse("/bot group allow 10001", context)

    assert command is None


def test_group_allow_command_is_parsed_for_whitelisted_sender() -> None:
    parser = AdminCommandParser(admin_whitelist={987654321})
    context = CommandContext(sender_qq=987654321, is_private_chat=True, group_id=None)

    command = parser.parse("/bot group allow 10001", context)

    assert command is not None
    assert command.name == "group_allow"
    assert command.arguments == {"group_id": 10001}


def test_group_allow_command_is_rejected_in_group_chat() -> None:
    parser = AdminCommandParser(admin_whitelist={987654321})
    context = CommandContext(sender_qq=987654321, is_private_chat=False, group_id=20002)

    command = parser.parse("/bot group allow 10001", context)

    assert command is None
