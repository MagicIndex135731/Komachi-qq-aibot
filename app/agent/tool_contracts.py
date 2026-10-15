"""Stable descriptive capability contracts; never an authorization mechanism."""
from dataclasses import replace

# Some internal mutations intentionally do not consume an external-write
# reservation. Describe them accurately without changing the effect boundary.
_BACKGROUND = frozenset({'delegate_research_task', 'request_owner_action', 'request_persona_refresh',
    'generate_image', 'generate_private_image', 'download_web_images', 'download_web_video',
    'download_twitter_media', 'queue_media_batch'})
_LOCAL = frozenset({'remember_intention', 'save_work_note', 'memory_write', 'begin_research_task',
    'update_research_task', 'resume_research_task', 'schedule_wakeup', 'cancel_wakeup',
    'schedule_private_wakeup', 'cancel_private_wakeup', 'report_owner_result', 'set_group_persona', 'set_task_goal'})
_ARTIFACT = frozenset({'execute_python', 'download_web_image', 'search_image_references',
    'observe_group_images', 'observe_forward_images', 'observe_post_images'})


def capability_contract(tool):
    effects = tool.side_effects or (('external_write',) if tool.effect == 'write' else
        ('enqueue_work',) if tool.name in _BACKGROUND else ('local_state',) if tool.name in _LOCAL else
        ('save_artifact',) if tool.effect == 'image' or tool.name in _ARTIFACT else ())
    outputs = ('search_summary_and_source_refs' if tool.name == 'search_web' else
        'primary_page_or_explicit_failure' if tool.name in {'open_web_page', 'find_web_page'} else
        'candidate_handles_not_downloaded' if tool.name == 'list_web_images' else
        'owned_artifact_ids_or_job_id' if tool.name.startswith(('download_', 'generate_')) else
        'actual_delivery_receipt' if tool.name in {'send_group', 'send_private'} else
        'structured_scoped_observation')
    return {'name': tool.name, 'effect_boundary': tool.effect, 'side_effects': list(effects),
        'execution': 'background' if tool.name in _BACKGROUND else tool.execution,
        'authority': 'current_source_and_executor_checks', 'input': tool.parameters,
        'output': outputs + '; handles_require_scope_and_provenance_validation',
        'retry': 'never_replay_sent_or_uncertain; queued_jobs_use_status_not_resubmit'}


def tool_description(tool):
    contract = capability_contract(tool)
    mutations = ','.join(contract['side_effects']) or 'read_only'
    output = contract['output'].split(';')[0]
    produced = f'; returns={output}' if output != 'structured_scoped_observation' else ''
    return tool.description + f" [execution={contract['execution']}; effects={mutations}{produced}; executor_authorized]"


def with_contract_descriptions(tools):
    return tuple(replace(tool, description=tool_description(tool)) for tool in tools)
