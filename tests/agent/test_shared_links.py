"""Share discovery and canonical access checks use synthetic identities only."""
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.agent.shared_links import build_share_access, build_share_tools, observed_share_sources
from app.storage.db import session_scope
from app.storage.models import Message
from test_service_tools import context, seed, GROUP, AUTHOR, ACCOUNT


def card(url='https://example.com/article'):
    return {'type': 'share', 'data': {'url': url, 'title': '合成标题', 'content': '卡片摘要'}}


def prepare(context):
    source = next(iter(context.observed_sources))
    with session_scope(context.router.engine) as session:
        session.get(Message, source).raw_json = {'message': [card()]}
    request = seed(context.router.engine, message_id='share-request', raw_json={'message': [
        {'type': 'reply', 'data': {'id': 'synthetic-source'}},
        {'type': 'at', 'data': {'qq': str(ACCOUNT)}},
        {'type': 'text', 'data': {'text': '看看引用分享并评价'}}]})
    with session_scope(context.router.engine) as session:
        row = session.get(Message, request)
        row.reply_to_msg_id, row.mentioned_bot = 'synthetic-source', True
    context.observed_sources.add(request)
    context.target_message_id = 'share-request'
    access = build_share_access(context.router, group_id=GROUP, trigger_source_ids=(request,), bot_qq=ACCOUNT)
    return source, build_share_tools(context, access=access)[0].handler, access


async def test_metadata_discovery_read_and_revalidation(context):
    source, read, access = prepare(context)
    found = observed_share_sources(context.router, GROUP, context.observed_sources, access=access)
    assert found[0]['source_id'] == source and found[0]['title'] == '合成标题'
    assert found[0]['link_count'] == 1 and 'links' not in found[0]
    first = await read({'source_id': source})
    assert first.status == 'ok' and first.data['metadata_only']
    assert first.data['trust'] == 'untrusted_shared_content'
    assert 'search_web' in first.data['next_step']
    with session_scope(context.router.engine) as session:
        session.get(Message, source).raw_json = {'message': [card('https://example.com/changed')]}
    assert (await read({'source_id': source})).data['error'] == 'share_requires_quoted_request'


async def test_unobserved_cross_group_deleted_and_revoked_sources(context):
    source, read, access = prepare(context)
    other = seed(context.router.engine, group_id=GROUP + 1, message_id='foreign-share', raw_json={'message': [card()]})
    assert (await read({'source_id': other})).data['error'] == 'share_source_out_of_scope'
    context.observed_sources.add(other)
    assert (await read({'source_id': other})).data['error'] == 'share_requires_quoted_request'
    assert len(observed_share_sources(context.router, GROUP, context.observed_sources, access=access)) == 1
    with session_scope(context.router.engine) as session:
        session.get(Message, source).raw_json = {'delivery_state': 'deleted', 'message': [card()]}
    assert (await read({'source_id': source})).status == 'blocked'
    assert observed_share_sources(context.router, GROUP, context.observed_sources, access=access) == []
    context.guard.allowed = False
    with pytest.raises(RuntimeError, match='revoked'):
        await read({'source_id': source})


async def test_private_isolation_outbound_and_argument_checks(context):
    source, _, _ = prepare(context)
    request = max(context.observed_sources)
    with session_scope(context.router.engine) as session:
        session.get(Message, source).group_id = None
        session.get(Message, source).platform_msg_id = f'private-inbound-{AUTHOR}-synthetic-source'
        session.get(Message, request).group_id = None
    private = SimpleNamespace(router=context.router, observation=context.observation,
        guard=context.guard, observed_sources={source}, private_user_id=AUTHOR)
    access = build_share_access(context.router, group_id=None, trigger_source_ids=(request,), private_user_id=AUTHOR)
    read = build_share_tools(private, access=access)[0].handler
    assert (await read({'source_id': source})).status == 'ok'
    assert observed_share_sources(context.router, None, {source}, private_user_id=AUTHOR, access=access)
    private.private_user_id = AUTHOR + 1
    assert (await build_share_tools(private, access=access)[0].handler({'source_id': source})).status == 'blocked'
    assert observed_share_sources(context.router, None, {source}, private_user_id=AUTHOR + 1, access=access) == []
    with session_scope(context.router.engine) as session:
        session.get(Message, source).raw_json = {'direction': 'outbound', 'message': [card()]}
    assert (await read({'source_id': source})).status == 'blocked'
    assert (await read({'source_id': True})).data['error'] == 'share_arguments_invalid'
    assert (await read({'source_id': source, 'url': 'https://example.com/fake'})).status == 'blocked'


def test_group_quote_outside_recent_window_and_cross_group_rejection(sqlite_engine, tmp_path):
    from test_observation import make_router
    from app.agent.observation import GroupObservationService
    from app.agent.contracts import AgentBatch
    router, incoming, _ = make_router(sqlite_engine, tmp_path)
    source = seed(sqlite_engine, message_id='old-share', raw_json={'message': [card()]})
    for index in range(25):
        seed(sqlite_engine, message_id=f'recent-{index}')
    with session_scope(sqlite_engine) as session:
        session.get(Message, source).timestamp = incoming.timestamp - timedelta(days=1)
        row = session.query(Message).filter_by(platform_msg_id=incoming.platform_msg_id).one()
        row.reply_to_msg_id = 'old-share'
        row.mentioned_bot = True
        row.raw_json = {'message': [
            {'type': 'reply', 'data': {'id': 'old-share'}},
            {'type': 'at', 'data': {'qq': str(router.runtime.settings.bot_qq)}},
            {'type': 'text', 'data': {'text': '看看这张分享'}}]}
    observation = GroupObservationService(router).build_agent_observation(AgentBatch(GROUP, (incoming,), 'quote-turn'))
    assert source in observation['source_message_ids']
    assert len(observation['source_message_ids']) == 25
    foreign = seed(sqlite_engine, group_id=GROUP + 1, message_id='foreign-share', raw_json={'message': [card()]})
    with session_scope(sqlite_engine) as session:
        session.query(Message).filter_by(platform_msg_id=incoming.platform_msg_id).one().reply_to_msg_id = 'foreign-share'
    assert foreign not in GroupObservationService(router).build_agent_observation(
        AgentBatch(GROUP, (incoming,), 'foreign-quote'))['source_message_ids']
