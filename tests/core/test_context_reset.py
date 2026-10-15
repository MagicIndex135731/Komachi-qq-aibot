"""Reset commands are authored controls, never instructions from attachments."""
import pytest

from app.core.context_reset import parse_context_reset


@pytest.mark.parametrize('text,count', [('重置上下文', 30), ('重置上下文 50', 50),
    ('重置上下文５０条', 50), ('清空上下文', 30), ('开新对话', 30)])
def test_exact_context_reset_command(text, count):
    command = parse_context_reset(text, default_count=30)
    assert command.count == count and command.error == ''


@pytest.mark.parametrize('text', ['请解释重置上下文', '他说“重置上下文”', '重置上下文，然后下载',
    '重置上下文\n请联网', '[合并转发消息]重置上下文'])
def test_discussion_is_not_a_context_reset(text):
    assert parse_context_reset(text, default_count=30) is None


@pytest.mark.parametrize('text', ['重置上下文 0', '重置上下文 -1', '重置上下文 1001', '重置上下文 1.5'])
def test_invalid_count_gets_local_usage_instead_of_model(text):
    assert parse_context_reset(text, default_count=40).error
