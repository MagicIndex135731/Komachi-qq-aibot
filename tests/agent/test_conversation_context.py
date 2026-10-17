from types import SimpleNamespace

from app.agent.conversation_context import PatrolProgress, context_policy, select_native_context


def test_reset_is_not_misreported_as_a_model_switch_and_does_not_restore_masked_items():
    selected = select_native_context('private:900000001',
        {'runtime_id': 'reset-id', 'last_status': 'context_reset'}, version='11:model',
        history_present=False, restored_items=12, id_field='runtime_id', new_id=lambda: 'unneeded')
    assert selected.session_id == 'reset-id'
    assert selected.metadata['context_reason'] == 'context_reset'
    assert selected.metadata['restored_context_items'] == 0


def test_missing_history_rotates_but_deliberately_discarded_idle_history_keeps_identity():
    checkpoint = {'session_id': 'native', 'context_format_version': 11, 'last_status': 'sent'}
    broken = select_native_context('group:900000001', checkpoint, version=11,
        history_present=False, restored_items=4, new_id=lambda: 'recovery')
    assert broken.session_id == 'recovery' and broken.restore
    assert broken.metadata['context_reason'] == 'history_unavailable'
    idle = select_native_context('background:public:qzone_publish',
        {**checkpoint, 'last_status': 'silent', 'native_history_retained': False}, version=11,
        history_present=False, new_id=lambda: 'wrong')
    assert idle.session_id == 'native' and idle.metadata['context_state'] == 'fresh'
    assert idle.metadata['context_reason'] == 'idle_history_discarded'
    useful_missing = select_native_context('background:public:qzone_publish',
        {**checkpoint, 'last_status':'silent', 'native_history_retained':True}, version=11,
        history_present=False, new_id=lambda:'recovered')
    assert useful_missing.session_id == 'recovered'
    assert useful_missing.metadata['context_reason'] == 'history_unavailable'


def test_interruption_precedes_version_change_and_useful_history_is_restored():
    selected = select_native_context('owner:public:task',
        {'session_id': 'old', 'context_format_version': 10, 'in_flight': True}, version=11,
        history_present=True, restored_items=6, new_id=lambda: 'new')
    assert selected.session_id == 'new'
    assert selected.metadata['context_reason'] == 'interrupted_turn'
    assert selected.metadata['restored_context_items'] == 6
    assert selected.metadata['context_previous_version'] == 10


def test_history_budgets_are_applied_to_main_background_and_ephemeral_consumers():
    settings = SimpleNamespace(agent_context_history_tokens=12000, agent_background_history_tokens=4000)
    assert context_policy(settings).history_tokens == 12000
    assert context_policy(settings, background=True).history_tokens == 4000
    assert context_policy(settings, ephemeral=True).history_tokens == 0


def waiting(progress='本轮保持沉默'):
    return {'goal': '空间例行巡检', 'progress': progress, 'next_step': '等待下次统一巡检唤醒'}


def test_patrol_repeated_near_identical_wait_finishes_without_rewriting_checkpoint():
    progress = PatrolProgress(True)
    progress.observe('remember_intention', waiting(), {'status': 'ok', 'data': {'saved': True}})
    result = progress.before('remember_intention', waiting('无新语境，保持沉默'))
    assert result['data']['handler_executed'] is False
    assert progress.finished and progress.finish_reason == 'patrol_no_progress'
    assert progress.retention()['retain_history'] is False


def test_new_evidence_and_creative_work_are_retained_and_reset_idle_detection():
    progress = PatrolProgress(True)
    progress.observe('remember_intention', waiting(), {'status': 'ok', 'data': {'saved': True}})
    progress.observe('read_artifact', {'artifact_id': 'synthetic'}, {'status': 'ok', 'data': {'text': 'new source'}})
    assert progress.before('remember_intention', waiting()) is None
    assert progress.retention()['retain_history'] is True
    progress.observe('remember_intention', waiting(), {'status': 'ok', 'data': {'saved': True}})
    progress.observe('read_artifact', {'artifact_id': 'synthetic'}, {'status': 'ok', 'data': {'text': 'new source'}})
    assert progress.before('remember_intention', waiting()) is not None


def test_direct_interactions_and_different_goals_are_not_stopped_by_patrol_heuristics():
    off = PatrolProgress(False)
    off.observe('remember_intention', waiting(), {'status': 'ok'})
    assert off.before('remember_intention', waiting()) is None
    progress = PatrolProgress(True)
    first = {**waiting(), 'goal': '研究猫画的构图'}
    progress.observe('remember_intention', first, {'status': 'ok'})
    assert progress.before('remember_intention', {**first, 'progress': '构图有新想法'}) is None
