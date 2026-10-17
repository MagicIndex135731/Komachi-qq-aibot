"""Offline resource/context cost comparison, not a billed-token/cache benchmark.

python -B -X utf8 scripts/benchmark_agent_resource_context.py --output data/project-tools/qq106/context-cost.json
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent.context_policy import RequestContextPolicy, estimate_text_tokens
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.web_tools import build_web_tools
from app.providers.web_search import WebSearchReport, WebSearchResult


class _Guard:
    def require_valid(self):pass


class _Runner:
    async def run(self, fn, *args, **kwargs):return fn(*args, **kwargs)


class _Page:
    supports_page_tools = True
    def __init__(self):self.calls = 0
    def open_page(self, url, *, lineno=0):
        self.calls += 1
        if url.endswith('/summary'):
            return WebSearchReport([WebSearchResult('合成来源', '', url, '')], '已核对来源的简短公开结论。')
        return WebSearchReport([WebSearchResult('合成原文', '', url, '', 'turn0view0')],
            f'合成原文 ({url})\nciteturn0view0 Total lines: 1000\n' +
            '\n'.join(f'L{i}: '+('合成来源事实与说明 ' * 12) for i in range(1000)))


def _entry(data):
    return {'type': 'function_call_output', 'call_id': 'call-1',
        'output': json.dumps({'status': 'ok', 'data': data}, ensure_ascii=False, separators=(',', ':'))}


async def measure(root):
    from types import SimpleNamespace
    page = _Page()
    workspace = ResearchTaskWorkspace(root, scope='group:900000001', actor_id=900000101,
        source_id=1, request='合成读取任务', valid_source=lambda _: True)
    tools = {t.name: t for t in build_web_tools(page, settings=SimpleNamespace(), runner=_Runner(), guard=_Guard(), workspace=workspace)}
    url = 'https://example.com/synthetic'
    # Explicit 16k reproduces the prior default against the same provider body.
    old = await tools['open_web_page'].handler({'ref_id': url, 'max_chars': 16000})
    new = await tools['open_web_page'].handler({'ref_id': url})
    reuse = await tools['open_web_page'].handler({'ref_id': url})
    count = lambda d: estimate_text_tokens(json.dumps(d, ensure_ascii=False, separators=(',', ':')))
    before, after = count(old.data), count(new.data)
    repeated_without_read = page.calls == 2
    summary = await tools['open_web_page'].handler({'ref_id': url + '/summary'})
    policy = RequestContextPolicy()
    prefix_body = {'instructions': 'fixed persona and tool policy', 'tools': [], 'input': [_entry(summary.data)]}
    first, _ = policy.project(prefix_body, protocol='openai-responses', current_call_ids={'call-1'})
    next_turn, _ = policy.project(prefix_body, protocol='openai-responses', current_call_ids=set())
    prior_policy = deepcopy(prefix_body)
    prior_policy['input'] = [_entry({k: v for k, v in summary.data.items() if k != 'context_retention'})]
    prior_first, _ = policy.project(prior_policy, protocol='openai-responses', current_call_ids={'call-1'})
    prior_next, _ = policy.project(prior_policy, protocol='openai-responses', current_call_ids=set())
    assert before > after and first == next_turn and prior_first != prior_next
    archived = workspace.read_evidence(new.data['evidence_id'], max_chars=32000)
    assert len(archived['content']) > len(new.data['content'])
    return {'measurement': 'offline estimated text tokens; no upstream model API invoked',
        'prior_16000_char_default_estimated_tokens': before,
        'new_6000_char_default_estimated_tokens': after,
        'estimated_packet_reduction_percent': round((1 - after / before) * 100, 1),
        'same_turn_repeat_estimated_tokens': count(reuse.data),
        'repeat_did_not_read_provider_again': repeated_without_read,
        'compact_tool_entry_unchanged_across_turns': first == next_turn,
        'prior_tool_entry_changed_across_turns': prior_first != prior_next,
        'recoverable_external_content_chars': len(archived['content']),
        'default_inline_content_chars': len(new.data['content']),
        'automatic_download_preview_images_before': 2, 'automatic_download_preview_images_after': 0,
        'actual_prompt_tokens': None, 'actual_cached_tokens': None, 'actual_cache_hit_rate': None,
        'limitations': 'Does not measure search relevance, real model completion, billed cost or provider cache availability.'}


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / 'data/test-work';work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qq106-cost-', dir=work) as tmp:
        result = asyncio.run(measure(Path(tmp)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':main()
