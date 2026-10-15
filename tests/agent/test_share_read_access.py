"""Expensive APP reads require a live quoted request, not card text."""
from types import SimpleNamespace

import pytest

from app.agent.shared_links import build_share_access, wrap_share_web_tools, share_context_text
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.storage.db import session_scope
from app.storage.models import Message
from test_service_tools import GROUP, ACCOUNT, AUTHOR, seed
from test_shared_links import card


def setup_access(engine, *, text='看看引用分享并评价', mention=True, quote=True,
                 requester=AUTHOR, private=False, card_author=None, trigger_card=False):
    source = seed(engine, author=card_author or requester, message_id='target-card',
                  raw_json={'message': [card()]})
    segments = [{'type': 'text', 'data': {'text': text}}]
    if mention:
        segments.insert(0, {'type': 'at', 'data': {'qq': str(ACCOUNT)}})
    if quote:
        segments.insert(0, {'type': 'reply', 'data': {'id': 'target-card'}})
    if trigger_card:
        segments.append(card())
    trigger = seed(engine, author=requester, message_id='read-request', raw_json={'message': segments})
    with session_scope(engine) as db:
        db.get(Message, trigger).mentioned_bot = mention
        db.get(Message, trigger).reply_to_msg_id = 'target-card' if quote else None
        if private:
            db.get(Message, source).group_id = None
            db.get(Message, source).platform_msg_id = f'private-inbound-{card_author or requester}-target-card'
            db.get(Message, trigger).group_id = None
    access = build_share_access(SimpleNamespace(engine=engine), group_id=None if private else GROUP,
                                trigger_source_ids=(trigger,), private_user_id=requester if private else None,
                                bot_qq=ACCOUNT)
    return source, trigger, access


@pytest.mark.parametrize('requester', [AUTHOR, AUTHOR + 1])
def test_any_real_group_member_can_request_quote_read(sqlite_engine, requester):
    source, _, access = setup_access(sqlite_engine, requester=requester)
    assert access.authorized_source_ids() == (source,)
    assert access.web_allowed()


@pytest.mark.parametrize('options', [
    {'mention': False}, {'quote': False},
    {'text': '不要打开这个分享'}, {'text': '我不想看这个分享'},
    {'text': '不是让你看这个分享'},
])
def test_card_without_quote_or_address_and_explicit_cancel_are_denied(sqlite_engine, options):
    _, _, access = setup_access(sqlite_engine, **options)
    assert access.authorized_source_ids() == ()
    assert not access.web_allowed()


@pytest.mark.parametrize('text', ['', '这个分享啊', '昨天看过这个分享', '随便说两句'])
def test_group_quote_with_address_needs_no_fixed_read_verb(sqlite_engine, text):
    source, _, access = setup_access(sqlite_engine, text=text, mention=True, quote=True)
    assert access.authorized_source_ids() == (source,)
    assert access.web_allowed()


def test_group_quote_with_textual_komachi_name_is_addressed(sqlite_engine):
    source, _, access = setup_access(sqlite_engine, text='小町你看这个', mention=False, quote=True)
    assert access.authorized_source_ids() == (source,)
    assert access.web_allowed()


def test_private_quote_needs_same_user_but_no_mention(sqlite_engine):
    source, _, access = setup_access(sqlite_engine, private=True, mention=False)
    assert access.authorized_source_ids() == (source,)
    assert access.web_allowed()


def test_foreign_private_card_is_not_readable(sqlite_engine):
    _, _, access = setup_access(sqlite_engine, private=True, mention=False, card_author=AUTHOR + 1)
    assert access.authorized_source_ids() == () and not access.web_allowed()


@pytest.mark.parametrize('change', ['request_recalled', 'request_text_changed', 'quote_changed', 'target_changed', 'foreign_group'])
def test_current_trigger_and_target_fingerprints_are_revalidated(sqlite_engine, change):
    source, trigger, access = setup_access(sqlite_engine)
    assert access.web_allowed()
    if change == 'foreign_group':
        seed(sqlite_engine, group_id=GROUP + 1, message_id='foreign-group-row')
    with session_scope(sqlite_engine) as db:
        request, target = db.get(Message, trigger), db.get(Message, source)
        if change == 'request_recalled':
            request.raw_json = {**request.raw_json, 'delivery_state': 'deleted'}
        elif change == 'request_text_changed':
            request.raw_json = {'message': [{'type': 'text', 'data': {'text': '不要读了'}}]}
        elif change == 'quote_changed':
            request.reply_to_msg_id = 'different-card'
        elif change == 'target_changed':
            target.raw_json = {'message': [card('https://example.com/changed')]}
        else:
            target.group_id = GROUP + 1
    assert access.authorized_source_ids() == () and not access.web_allowed()


async def test_direct_web_call_cannot_bypass_card_gate(sqlite_engine):
    _, _, access = setup_access(sqlite_engine, mention=False)
    calls = []
    async def invoke(args):
        calls.append(args)
        return ToolOutcome({'content': 'body'})
    web = AgentTool('open_web_page', 'open', {}, invoke)
    result = await wrap_share_web_tools((web,), access)[0].handler({'ref_id': 'https://example.com/article'})
    assert result.status == 'blocked' and not calls


async def test_ordinary_web_permission_cannot_open_unrequested_observed_card(sqlite_engine):
    source, trigger, access = setup_access(sqlite_engine, text='今天上海天气怎么样', quote=False)
    calls = []
    async def invoke(args):
        calls.append(args)
        return ToolOutcome({'content': 'body'})
    web = AgentTool('open_web_page', 'open', {}, invoke)
    wrapped = wrap_share_web_tools((web,), access, observed_source_ids={source, trigger})[0]
    denied = await wrapped.handler({'ref_id': 'https://example.com/article'})
    assert denied.data['error'] == 'share_requires_quoted_request' and not calls
    assert (await wrapped.handler({'ref_id': 'https://example.org/weather'})).status == 'ok'


async def test_ordinary_question_ignores_unrelated_old_card_but_cannot_open_it(sqlite_engine):
    source, trigger, access = setup_access(sqlite_engine, text='这个角色是打什么伤害的，是c还是辅啊', mention=False, quote=False)
    calls = []
    async def invoke(args):
        calls.append(args)
        return ToolOutcome({'content': 'body'})
    tool = AgentTool('open_web_page', 'open', {}, invoke)
    wrapped = wrap_share_web_tools((tool,), access, observed_source_ids={source, trigger})[0]
    assert (await wrapped.handler({'ref_id': 'https://example.org/research'})).status == 'ok'
    assert (await wrapped.handler({'ref_id': 'https://example.com/article'})).status == 'blocked'
    assert len(calls) == 1


@pytest.mark.parametrize('text', ['普通群聊', '今天上课比较晚', '昨天看过这个分享', '今天天气真好', '刚才的新闻挺有趣', 'https://example.org/search?query=what'])
async def test_casual_chat_never_grants_autonomous_network(sqlite_engine, text):
    source, trigger, access = setup_access(sqlite_engine, text=text, mention=False, quote=False)
    calls = []
    async def invoke(args):
        calls.append(args)
        return ToolOutcome({})
    tool = AgentTool('search_web', 'search', {}, invoke)
    result = await wrap_share_web_tools((tool,), access, observed_source_ids={source, trigger})[0].handler({'query': 'unrequested'})
    assert result.status == 'blocked' and not calls


@pytest.mark.parametrize('text,mention', [('小町', False), ('聊聊', True), ('请核对这些文件', False), ('这个角色强吗', False)])
def test_current_question_request_or_real_address_allows_lookup(sqlite_engine, text, mention):
    _, _, access = setup_access(sqlite_engine, text=text, mention=mention, quote=False)
    assert access.web_allowed() and not access.authorized_source_ids()


def test_address_only_in_card_never_grants_lookup(sqlite_engine):
    _, _, access = setup_access(sqlite_engine, text='', mention=False, quote=False, trigger_card=True)
    assert not access.web_allowed()


@pytest.mark.parametrize('change', ['deleted', 'changed'])
async def test_question_grant_rechecked_after_provider_returns(sqlite_engine, change):
    _, trigger, access = setup_access(sqlite_engine, text='这个角色强吗', mention=False, quote=False)
    calls = []
    async def invoke(args):
        calls.append(args)
        with session_scope(sqlite_engine) as db:
            row = db.get(Message, trigger)
            row.raw_json = {'delivery_state': 'deleted'} if change == 'deleted' else {'message': [{'type': 'text', 'data': {'text': '不用查了'}}]}
        return ToolOutcome({'content': 'late result'})
    tool = AgentTool('search_web', 'search', {}, invoke)
    result = await wrap_share_web_tools((tool,), access)[0].handler({'query': '角色'})
    assert len(calls) == 1 and result.data['error'] == 'share_request_revoked'


async def test_search_reference_to_unrequested_card_does_not_bypass_guard(sqlite_engine):
    source, trigger, access = setup_access(sqlite_engine, text='联网搜索今天新闻', quote=False)
    calls = []
    async def search(args):
        return ToolOutcome({'sources': [{'ref_id': 'current_web_result', 'source': 'https://example.com/article'}]})
    async def open_page(args):
        calls.append(args)
        return ToolOutcome({'content': 'body'})
    tools = {item.name: item for item in wrap_share_web_tools((
        AgentTool('search_web', 'search', {}, search), AgentTool('open_web_page', 'open', {}, open_page)),
        access, observed_source_ids={source, trigger})}
    assert (await tools['search_web'].handler({'query': '新闻'})).status == 'ok'
    result = await tools['open_web_page'].handler({'ref_id': 'current_web_result'})
    assert result.data['error'] == 'share_requires_quoted_request' and not calls


async def test_current_search_reference_is_resolved_for_media_tools(sqlite_engine):
    _, _, access = setup_access(sqlite_engine, text='把刚才作者的照片发给我', mention=False, quote=False)
    calls = []
    async def search(args):
        return ToolOutcome({'sources': [{'ref_id': 'current-source', 'source': 'https://example.org/article'}]})
    async def media(args):
        calls.append(args)
        return ToolOutcome({'images': []})
    registry = {t.name: t for t in wrap_share_web_tools((
        AgentTool('search_web', 'search', {}, search), AgentTool('list_web_images', 'images', {}, media)), access)}
    await registry['search_web'].handler({'query': '作者照片'})
    await registry['list_web_images'].handler({'ref_id': 'current-source'})
    assert calls == [{'ref_id': 'https://example.org/article'}]


def test_old_quoted_request_never_grants_new_turn(sqlite_engine):
    source, _, old = setup_access(sqlite_engine)
    assert old.authorized_source_ids() == (source,)
    fresh = seed(sqlite_engine, message_id='fresh-chat', raw_json={'message': [{'type': 'text', 'data': {'text': '继续普通聊天'}}]})
    current = build_share_access(SimpleNamespace(engine=sqlite_engine), group_id=GROUP,
                                 trigger_source_ids=(fresh,), bot_qq=ACCOUNT)
    assert current.authorized_source_ids() == () and not current.web_allowed()


def test_card_tracking_variants_and_duplicate_targets_preserve_granted_scope(sqlite_engine):
    source, trigger, access = setup_access(sqlite_engine)
    duplicate = seed(sqlite_engine, message_id='duplicate-target-card', raw_json={'message': [card('https://example.com/article?tracking=other')]})
    ids = {source, trigger, duplicate}
    assert access.known_card_target_allowed('https://example.com/article?tracking=new#anchor', ids)
    assert access.has_ungranted_card_sources(ids)
    assert not access.has_ungranted_card_sources({source, trigger})
    ordinary = seed(sqlite_engine, message_id='ordinary-web-query', raw_json={'message': [
        {'type': 'text', 'data': {'text': '联网搜索天气'}}]})
    ordinary_access = build_share_access(SimpleNamespace(engine=sqlite_engine), group_id=GROUP,
                                         trigger_source_ids=(ordinary,), bot_qq=ACCOUNT)
    assert ordinary_access.web_allowed()
    assert not ordinary_access.known_card_target_allowed('https://example.com/article?tracking=new#anchor', ids)


@pytest.mark.parametrize('text', ['联网搜索 requests 库', '打开 https://example.org/file 获取正文', 'search the web for Python docs', '今天上海天气怎么样', '最近有什么新闻'])
def test_explicit_ordinary_web_request_keeps_read_permission(sqlite_engine, text):
    _, _, access = setup_access(sqlite_engine, text=text, mention=False, quote=False)
    assert access.authorized_source_ids() == () and access.web_allowed()


@pytest.mark.parametrize('private', [False, True])
@pytest.mark.parametrize('text', [
    '帮我去x扒一些linye最新的cos视频或图片',
    '麻烦你到官网把最新的活动照片下载给我',
    '请去作者主页找找最近发的图',
    '可以先找一下公开的照片再发给我',
    '给我把作者最近的视频搬运过来',
    '那把这个十月十三号的图片发给我',
    '把刚才那个作者的照片下载过来',
])
def test_directed_request_accepts_location_and_object_before_action(sqlite_engine, private, text):
    _, _, access = setup_access(sqlite_engine, text=text, mention=False, quote=False, private=private)
    assert access.web_allowed() and access.authorized_source_ids() == ()


@pytest.mark.parametrize('text', [
    '帮我去x聊聊，不要下载图片',
    '昨天去x扒了一些照片',
    '有人说帮我去x扒一些照片',
])
def test_request_wording_in_cancel_or_narrative_does_not_grant_lookup(sqlite_engine, text):
    _, _, access = setup_access(sqlite_engine, text=text, mention=False, quote=False, private=True)
    assert not access.web_allowed()


def test_ordinary_context_contains_short_title_without_url_or_card_directives():
    raw = {'message': [card(), {'type': 'text', 'data': {'text': '一起讨论'}}]}
    text = share_context_text(raw, '[分享卡片：合成标题] 卡片摘要 https://example.com/article')
    assert '合成标题' in text and '一起讨论' in text
    assert 'https://' not in text and '卡片摘要' not in text
    assert share_context_text({'message': [{'type': 'text', 'data': {'text': 'https://example.org'}}]},
                              'https://example.org') == 'https://example.org'
