"""Real native runtime against a synthetic model, never QQ or paid models."""
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from tests.agent.test_autonomous_social import composed, add_source, NOW, GROUP
from tests.agent.test_harness_runtime import model_client, responses_sse


@pytest.mark.parametrize('speak', [False, True])
async def test_owner_persona_observes_and_can_send_without_an_extra_judge(sqlite_engine, tmp_path, speak, record_property):
    def script(body, number):
        if speak and number == 1:
            return responses_sse(name='send_group', arguments={'text':'这片星空倒是很合我的胃口嘛', 'artifact_ids':[]})
        return responses_sse(number=number)
    client, requests, _ = model_client(script)
    async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
        fixture.router.runtime.persona['core_traits'] = ['PERSONA_INTEREST_STARS']
        fixture.router._active_persona = lambda _: fixture.router.runtime.persona
        group = fixture.router.runtime.group_policy['groups'][str(GROUP)]
        group.update(proactive_observation_minimum_messages=1)
        source = add_source(sqlite_engine, platform_id='synthetic-stars', text='今天拍到一片星空')
        job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'group')
        observation = SocialObservation(job, NOW, 'untrusted prior render', ('今天拍到一片星空',), (source,))
        guard = SimpleNamespace(action_id='synthetic-patrol', source_message_ids=(source,),
            require_valid=lambda:None, valid=lambda:True, on_dispatch=lambda:None,
            trace_trigger={'trigger_evidence':{'source_ids':[source]}})
        result = await fixture.executor.run_tools(observation, guard)
        assert requests
        text = json.dumps(requests[0], ensure_ascii=False)
        assert 'PERSONA_INTEREST_STARS' in text
        assert '过去几轮没有说话' in text
        names = [tool['name'] for tool in requests[0]['tools']]
        assert 'send_group' in names and 'search_web' not in names
        assert 'finish_patrol' in names
        assert result.status == ('sent' if speak else 'silent'), result
        assert not fixture.executor.sessions.read(f'background:group:{GROUP}')['checkpoint']
        state = fixture.executor.sessions.read(f'group:{GROUP}')
        assert bool(state['items']) is speak
        assert state['checkpoint']['context_retention_reason'] == ('external_action_retained' if speak else 'idle_history_discarded')
        assert len(requests) <= 2
        from app.agent.context_policy import estimate_request_tokens
        record_property('estimated_cold_input_tokens', estimate_request_tokens(requests[0]))
        record_property('upstream_requests', len(requests))
        record_property('tools', len(requests[0]['tools']))


async def test_scheduled_guard_exposes_same_source_contract(sqlite_engine):
    from app.agent.social_runtime import SocialDispatchGuard
    from app.agent.social_contracts import SocialDecision, SocialJob
    runtime = SimpleNamespace(_generation=1, _connection_generation=1)
    observation = SocialObservation(SocialJob('synthetic', 'group', GROUP), NOW, '', source_message_ids=(11,12))
    guard = SocialDispatchGuard(runtime, observation=observation, decision=SocialDecision(),
        action_id='synthetic', epoch=1, clock=lambda:NOW)
    assert guard.source_message_ids == (11,12)


async def test_idle_patrol_then_new_topic_reuses_owner_identity_and_stable_prefix(sqlite_engine, tmp_path, record_property):
    client, requests, _ = model_client(lambda body, n: responses_sse(number=n))
    async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
        fixture.router.runtime.group_policy['groups'][str(GROUP)].update(proactive_observation_minimum_messages=1)
        identities = []
        for n in range(2):
            source = add_source(sqlite_engine, platform_id=f'synthetic-new-{n}', text=f'一个全新的话题{n}')
            job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'group')
            observation = SocialObservation(job, fixture.clock[0], '', source_message_ids=(source,))
            guard = SimpleNamespace(action_id=f'synthetic-idle-{n}', valid=lambda:True,
                require_valid=lambda:None, on_dispatch=lambda:None,
                trace_trigger={'trigger_evidence':{'source_ids':[source]}})
            result = await fixture.executor.run_tools(observation, guard)
            assert result.status == 'silent'
            state = fixture.executor.sessions.read(f'group:{GROUP}')
            assert state['items'] == []
            assert state['source_message_ids'] == []
            identities.append(state['checkpoint']['session_id'])
            fixture.clock[0] += timedelta(minutes=5)
        assert len(requests) == 2 and identities[0] == identities[1]
        assert requests[0]['tools'] == requests[1]['tools']
        assert requests[0]['instructions'] == requests[1]['instructions']
        assert requests[0]['input'][0] == requests[1]['input'][0]
        assert requests[0]['prompt_cache_key'] == requests[1]['prompt_cache_key']
        from app.agent.context_policy import estimate_request_tokens
        record_property('stable_prefix_verified_locally', True)
        record_property('first_input_estimate', estimate_request_tokens(requests[0]))
        record_property('next_input_estimate', estimate_request_tokens(requests[1]))


async def test_silent_patrol_closes_with_more_than_200_retained_sources(sqlite_engine, tmp_path):
    """Lifetime provenance is larger than the bounded current-turn source list."""
    client, requests, _ = model_client(lambda body, n: responses_sse(number=n))
    async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
        fixture.router.runtime.group_policy['groups'][str(GROUP)].update(proactive_observation_minimum_messages=1)
        scope = f'group:{GROUP}'
        prior = [add_source(sqlite_engine, platform_id=f'synthetic-retained-{n}', text=f'旧消息{n}') for n in range(201)]
        prior_turn_sources = prior[-3:]
        # Establish the correct persona and native history using a normal closed patrol.
        job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'group')
        identities = []
        for n in range(3):
            source = add_source(sqlite_engine, platform_id=f'synthetic-after-retained-{n}', text=f'新话题{n}')
            guard = SimpleNamespace(action_id=f'synthetic-long-history-{n}', valid=lambda:True,
                require_valid=lambda:None, on_dispatch=lambda:None,
                trace_trigger={'trigger_evidence':{'source_ids':[source]}})
            result = await fixture.executor.run_tools(SocialObservation(job, fixture.clock[0], '', source_message_ids=(source,)), guard)
            assert result.status == 'silent'
            state = fixture.executor.sessions.read(scope)
            assert not state['checkpoint'].get('in_flight')
            assert state['checkpoint']['last_status'] == 'silent'
            identities.append(state['checkpoint']['session_id'])
            if n == 0:
                fixture.executor.sessions.save(scope, state['items'], state['checkpoint'], prior_turn_sources,
                    expected_revision=state['revision'], retained_context={'sources':prior})
            else:
                assert set(state['context_source_message_ids']) == set(prior)
                assert state['source_message_ids'] == prior_turn_sources
            fixture.clock[0] += timedelta(minutes=5)
        assert len(set(identities)) == 1 and len(requests) == 3


async def test_public_assessment_rejects_unread_image_but_can_defer_persistently(sqlite_engine, tmp_path):
    from app.agent.public_tools import build_public_tools
    from app.agent.social_contracts import SocialPost
    from app.storage.social_agent import SocialAgentRepository
    from tests.agent.test_autonomous_social import FRIEND, BOT
    client, _, _ = model_client(lambda _, n: responses_sse(number=n))
    async with composed(sqlite_engine, tmp_path, client, kind='qzone_comment', runtime='harness') as fixture:
        job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'qzone_comment')
        post = SocialPost(FRIEND, 'synthetic-pixels', '', NOW, ('https://example.com/image.png',))
        observation = SocialObservation(job, NOW, '小町', posts=(post,))
        guard = SimpleNamespace(require_valid=lambda:None)
        tools = {t.name:t for t in build_public_tools(fixture.executor, observation, guard, object())}
        args = {'owner_id':FRIEND, 'post_id':post.post_id, 'decision':'no_response', 'revisit_minutes':15}
        assert (await tools['assess_qzone_post'].handler(args)).status == 'blocked'
        args['decision'] = 'deferred'
        assert (await tools['assess_qzone_post'].handler(args)).status == 'ok'
        restarted = SocialAgentRepository(sqlite_engine)
        assert not restarted.unreviewed_posts(account_id=BOT, posts=(post,), now=NOW)
        assert restarted.unreviewed_posts(account_id=BOT, posts=(post,), now=NOW+timedelta(minutes=16)) == (post,)


@pytest.mark.parametrize('full_manifest', [False, True])
async def test_observation_profile_cold_cost_comparison(sqlite_engine, tmp_path, monkeypatch, full_manifest, record_property):
    if full_manifest:
        class AllTools:
            def __contains__(self, name):
                return True
        monkeypatch.setattr('app.agent.patrol_context.OBSERVATION_TOOLS', AllTools())
    client, requests, _ = model_client(lambda _, n: responses_sse(number=n))
    async with composed(sqlite_engine, tmp_path, client, runtime='harness', compute=True) as fixture:
        fixture.router.runtime.group_policy['groups'][str(GROUP)].update(proactive_observation_minimum_messages=1)
        source = add_source(sqlite_engine, platform_id='synthetic-cost', text='大家在讨论晚上的星空')
        job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'group')
        guard = SimpleNamespace(action_id='synthetic-cost', valid=lambda:True,
            require_valid=lambda:None, on_dispatch=lambda:None,
            trace_trigger={'trigger_evidence':{'source_ids':[source]}})
        result = await fixture.executor.run_tools(SocialObservation(job, NOW, '', source_message_ids=(source,)), guard)
        assert result.status == 'silent' and len(requests) == 1
        from app.agent.context_policy import estimate_request_tokens
        record_property('profile', 'full_manifest' if full_manifest else 'compact')
        record_property('estimated_total_input', estimate_request_tokens(requests[0]))
        record_property('tools', len(requests[0]['tools']))
        record_property('paid_model_calls', 0)
