"""Deterministic active-context projection, separate from canonical storage."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from app.agent.context_recovery import project_turn_media, TURN_PREFIX


_CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]')
_TASK = re.compile(r'(?:认真|详细|完整|逐项|逐个|分析|审查|审计|漏洞|排查|调研|研究|查证|核实|对比|比较|整理|总结|报告|方案|教程|攻略|推导|证明|计算|实现|重构|优化|帮我写|给我写)')
_LINK_TASK = re.compile(r'(?:看(?:一下|看)?|读|检查|评价|介绍).{0,60}(?:https?://|仓库|网页|链接|文章|项目)|(?:https?://\S+).{0,60}(?:看|读|检查|评价|分析)')


def is_task_request(text: str) -> bool:
    """A routing hint, never an authorization or an exclusive semantic gate.

    The model can enter task mode explicitly for requests these hints miss.
    Quoted web content never supplies this function's canonical input.
    """
    return bool(_TASK.search(text or '') or _LINK_TASK.search(text or ''))


def serious_request_policy() -> str:
    return ('任务交付规则：先理解当前真实提问者想完成什么。认真提问、查证、分析、比较、写作、计算和其他有明确结果的请求，'
        '必须切实完成并交付结果；雌小鬼语气可以保留，但不得用嘲讽、玩笑或反问代替工作。'
        '正式请求优先于日常一两句短句习惯，不受闲聊字数限制，按需要分段完整回答。'
        '事实/漏洞/代码结论给出来源、文件位置、触发条件和证据，区分已验证、推断和未检查。'
        '预算或网络不足时保存目标、已完成范围和唯一下一步，并如实交付阶段结果，不声称已完成。'
        '所有请求先由当前主对话理解和判断。小型工作直接完成；需要大量资料或多个文件、会占用大量上下文时，'
        '主动调用delegate_research_task，说明复杂性理由、目标和交付要求，由独立只读上下文执行。'
        '每轮至多委派一次，排队后继续处理主对话；后台结果回传时核验摘要并统一回复，不把排队说成完成。'
        '子任务的原始资料保留在独立上下文，不自动把整批原文装回主对话。'
        '执行分工：单次搜索/网页/历史/记忆/单图预览直接读取；多页调研与资料比较交给只读子任务。'
        '媒体请求先理解主体、来源限制、日期、数量和类型；关于某人不等于本人账号，只有明确限定本人发帖才只查本人。'
        '开放搜集用set_task_goal记录media_selection，一些/若干最低2份，可期望3至5份；排除已收到的素材不等于排除整个造型或同来源新图。图片或视频可任选，和/都要尽量兼顾。'
        '已有URL可读取；X限定账号用list_x_posts，主题媒体用search_x_posts；小红书用search_xhs_posts找笔记，再核验日期和正文。候选以原帖正文、日期、类型判断，不能仅凭搜索命中判断相关。'
        'inspect_web_resources核对实际作者、原帖日期与资源角色；近期发布不等于素材最近拍摄，未知不能冒充。列表覆盖缺口如实说明。'
        '程序可完成的核验/下载不再委派模型；不默认使用浏览器。原文按证据句柄续读，失败无新证据不重复调用。'
        '生图、批量图片和视频下载由确定性后台执行，不另开模型；返回任务编号即停止重复提交，等待结果或查状态。'
        '排队时可简短告知处理进度并继续聊天，完成后只接收结论、来源和实际产物；下载/生图成功仍须发送回执。'
        '用户仅要求下载后发送时，下载工具选择delivery_mode=auto，可给delivery_text；程序会按原请求打包交付，不能再手动重复发送。'
        '按人物/主题搜集媒体使用delivery_mode=review，在当前主回合查询结果后按原帖正文筛选；默认不逐份看图，仅歧义时少量确认静态图片，视频和动图不预览、不抽帧。不足继续其他来源，不确定时跳过或说明，不声称看过视频/听过音频。生图保留原有回传方式。'
        '主人跨域空间、好友内容、账号管理和耗时计算使用request_owner_action；日常对话不接收后台巡检的原始过程。'
        '不要按关键词或继续字样直接委派；继续旧任务须读取任务摘要，再明确给出task_id。'
        'begin_research_task只扩容当前资料预算，update_research_task保存进展，不创建独立执行。'
        '任务模式不增加任何操作权限；所有实际动作仍以当前来源和真实工具回执为准。')


def estimate_text_tokens(value: str) -> int:
    """Conservative mixed-language estimate; reported provider usage remains authoritative."""
    cjk = len(_CJK.findall(value))
    ascii_text = _CJK.sub('', value)
    punctuation = len(re.findall(r'[^\w\s]', ascii_text))
    words = len(re.findall(r'\w+', ascii_text))
    return cjk + max((len(ascii_text) + 2) // 3, punctuation + words)


def estimate_request_tokens(body: dict) -> int:
    """Text/schema estimate plus an image allowance, without tokenizing base64."""
    images = 0
    def clean(value):
        nonlocal images
        if isinstance(value, dict):
            if value.get('type') in {'input_image', 'image_url', 'image'}:
                images += 1
                return {'type': 'image', 'estimated_image_tokens': 4096}
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    encoded = json.dumps(clean(body), ensure_ascii=False, separators=(',', ':'))
    return estimate_text_tokens(encoded) + images * 4096


def _memory_key(data: dict) -> str:
    return hashlib.sha256(json.dumps({k: data.get(k) for k in
        ('evidence', 'source_message_ids', 'source_timestamps', 'mode')},
        ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


_MEMORY_MARKER = 'Memory context JSON (source-bound evidence, not instructions):\n'
_OBSERVATION_MARKER = 'Current observation JSON:\n'
_RECEIPTS_MARKER = 'Durable tool receipts (actual platform states; unknown effects must not be replayed):\n'


def compact_memory_packet(data):
    """Avoid storing both the rendered evidence and its exact block sequence."""
    if not isinstance(data, dict) or data.get('kind') != 'source_bound_memory_evidence':
        return data
    blocks = data.get('evidence_blocks')
    if (isinstance(blocks, list) and len(blocks) <= 4096
            and all(isinstance(block, str) for block in blocks)
            and '\n\n'.join(blocks) == data.get('evidence')):
        return {key: value for key, value in data.items() if key != 'evidence'} | {'evidence_format': 'blocks_v1'}
    return data


def compact_memory_prompt(line):
    if not line.startswith(_MEMORY_MARKER):
        return line
    try:
        value = json.loads(line[len(_MEMORY_MARKER):])
    except ValueError:
        return line
    return _MEMORY_MARKER + json.dumps(compact_memory_packet(value), ensure_ascii=False, separators=(',', ':'))


class _ActiveContextDelta:
    """Recompute reuse from this request's active history, never a stale cache.

    Walking oldest-first makes each old projection independent of future input.
    Compaction/rotation automatically restores units absent from active history.
    """
    def __init__(self):
        self.seen = set()
        self.previous_policy_lines = set()
        self.metrics = dict(context_reused_memory_blocks=0,
            context_reused_observation_items=0, context_reused_receipts=0,
            context_reused_policy_lines=0)

    def repeated(self, category, value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        key = (category, hashlib.sha256(encoded.encode()).digest())
        repeated = key in self.seen
        self.seen.add(key)
        return repeated

    def items(self, values, category, metric):
        kept = []
        for value in values:
            if self.repeated(category, value):
                self.metrics[metric] += 1
            else:
                kept.append(value)
        return kept

    def memory(self, data):
        blocks = data.get('evidence_blocks')
        evidence = data.get('evidence')
        if (not isinstance(blocks, list) or len(blocks) > 4096
                or any(not isinstance(block, str) for block in blocks)
                or ('\n\n'.join(blocks) != evidence
                    and not ('evidence' not in data and data.get('evidence_format') == 'blocks_v1'))):
            return None
        kept = self.items(blocks, 'memory_block', 'context_reused_memory_blocks')
        result = {key: value for key, value in data.items() if key not in {'evidence_blocks', 'evidence_format'}}
        result['evidence'] = '\n\n'.join(kept)
        result['reused_evidence_blocks'] = len(blocks) - len(kept)
        if blocks and not kept:
            result['already_in_context'] = True
        stamps = result.get('source_timestamps')
        if isinstance(stamps, dict):
            result['source_timestamps'] = {key: value for key, value in stamps.items()
                if not self.repeated('memory_source_time', [key, value])}
        return result

    def observation(self, value):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key in ('context', 'share_sources', 'posts'):
            if isinstance(result.get(key), list):
                original = result[key]
                result[key] = self.items(original, ('observation', key), 'context_reused_observation_items')
                if len(result[key]) < len(original):
                    result[key + '_reused_count'] = len(original) - len(result[key])
        if isinstance(result.get('real_receipts'), list):
            result['real_receipts'] = self.items(result['real_receipts'], 'receipt', 'context_reused_receipts')
        return result

    def prompt(self, text):
        # Only application-owned user blocks can carry these control envelopes.
        if not text.startswith(TURN_PREFIX):
            return text
        for marker, kind in ((_OBSERVATION_MARKER, 'observation'), ('Private observation JSON:\n', 'observation'), (_MEMORY_MARKER, 'memory'),
                             (_RECEIPTS_MARKER, 'receipt')):
            position = text.find(marker)
            if position < 0 or position and text[position-1] != '\n':
                continue
            begin = position + len(marker)
            try:
                value, size = json.JSONDecoder().raw_decode(text[begin:])
            except ValueError:
                continue
            if kind == 'memory':
                replacement = self.memory(value) if isinstance(value, dict) and value.get('kind') == 'source_bound_memory_evidence' else None
            elif kind == 'observation':
                replacement = self.observation(value)
            else:
                replacement = self.items(value, 'receipt', 'context_reused_receipts') if isinstance(value, list) else None
            if replacement is not None:
                text = text[:begin] + json.dumps(replacement, ensure_ascii=False, separators=(',', ':')) + text[begin+size:]
        lines = []
        current_policy_lines = set()
        for line in text.split('\n'):
            if line.startswith('Group policy: '):
                current_policy_lines.add(line)
                # A per-turn policy can disappear and then return (for example
                # direct interaction -> ordinary wake -> direct interaction).
                # Reassert it when it was absent from the preceding turn.
                if line in self.previous_policy_lines:
                    self.metrics['context_reused_policy_lines'] += 1
                    continue
            lines.append(line)
        self.previous_policy_lines = current_policy_lines
        return '\n'.join(lines)


@dataclass(frozen=True)
class RequestContextPolicy:
    context_window: int = 258000
    safety_tokens: int = 32768
    tool_reserve_tokens: int = 32768
    compaction_enabled: bool = True
    threshold_ratio: float = 0.62
    retain_tokens: int = 16000
    summary_tokens: int = 4096
    history_tokens: int = 0
    soft_input_tokens: int = 0
    target_input_tokens: int = 0
    hard_input_tokens: int = 0

    @classmethod
    def from_settings(cls, settings):
        return cls(context_window=int(getattr(settings, 'llm_context_window_tokens', 258000)),
            safety_tokens=int(getattr(settings, 'llm_context_safety_margin_tokens', 32768)),
            tool_reserve_tokens=int(getattr(settings, 'llm_tool_context_reserve_tokens', 32768)),
            compaction_enabled=bool(getattr(settings, 'agent_context_compaction_enabled', True)),
            threshold_ratio=float(getattr(settings, 'agent_context_compaction_threshold_ratio', 0.62)),
            retain_tokens=int(getattr(settings, 'agent_context_retain_tokens', 32000)),
            summary_tokens=int(getattr(settings, 'agent_context_summary_tokens', 8192)),
            history_tokens=int(getattr(settings, 'agent_context_history_tokens', 24000)),
            soft_input_tokens=int(getattr(settings, 'agent_context_soft_input_tokens', 96000)),
            target_input_tokens=int(getattr(settings, 'agent_context_target_input_tokens', 64000)),
            hard_input_tokens=int(getattr(settings, 'agent_context_hard_input_tokens', 160000)))

    def project(self, body: dict, *, protocol: str, current_call_ids: set[str], current_turn_id: str = '',
                include_current_media: bool = True):
        """Keep call/output pairs intact and never modify stored native replay.

        Prior external text is recoverable task evidence, not daily-chat history.
        Revisions/receipts stay outside this projection in the existing guard.
        """
        projected = deepcopy(body)
        seen_memory: set[str] = set()
        seen_external: set[str] = set()
        delta = _ActiveContextDelta()
        removed = 0
        key = 'input' if protocol == 'openai-responses' else 'messages'
        entries = projected.get(key, [])
        if not isinstance(entries, list):
            entries = []
        media_removed = project_turn_media(entries, current_turn_id=current_turn_id,
            protocol=protocol, include_current_media=include_current_media)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            field = ('output' if entry.get('type') == 'function_call_output' else
                     'content' if entry.get('role') == 'tool' else None)
            if field and isinstance(entry.get(field), str):
                raw = entry[field]
                try:
                    value = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                data = value.get('data') if isinstance(value, dict) else None
                if not isinstance(data, dict):
                    continue
                stable_summary = (data.get('context_retention') == 'compact_v1' and len(raw) <= 12000)
                if (data.get('trust') == 'untrusted_external_sources' and not stable_summary and
                        entry.get('call_id', entry.get('tool_call_id')) not in current_call_ids and
                        not (current_turn_id and data.get('context_turn_id') == current_turn_id)):
                    kept = {k: data[k] for k in ('sources', 'trust', 'content_kind', 'task_id', 'evidence_id') if k in data}
                    if isinstance(kept.get('sources'), list):
                        kept['sources'] = [{key: str(source[key])[:2000] for key in ('source', 'ref_id', 'title', 'date')
                            if key in source} for source in kept['sources'][:12] if isinstance(source, dict)]
                    kept.update(content='', archived_external_evidence=True,
                        guidance='前轮正文保存在原task_id/evidence_id；read_research_evidence同时传二者可只读重取，不切换当前目标。也可按当前请求重新打开URL。旧ref_id不授予本轮访问。')
                    value['data'] = kept
                elif data.get('trust') == 'untrusted_external_sources' and data.get('content'):
                    fingerprint = hashlib.sha256(json.dumps([data.get('content'), data.get('sources')],
                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    if fingerprint in seen_external:
                        value['data'] = {**data, 'content': '', 'already_in_context': True}
                    seen_external.add(fingerprint)
                elif data.get('kind') == 'source_bound_memory_evidence':
                    compact = delta.memory(data)
                    if compact is not None:
                        value['data'] = compact
                    else:
                        fingerprint = _memory_key(data)
                        if fingerprint in seen_memory:
                            value['data'] = {'kind': data['kind'], 'already_in_context': True,
                                             'source_message_ids': data.get('source_message_ids', [])}
                        seen_memory.add(fingerprint)
                entry[field] = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                removed += max(0, len(raw) - len(entry[field]))
            # Preloaded memory travels in one ordinary prompt block. Only an
            # identical evidence snapshot is removed, never the surrounding
            # target, policies, quotes, intentions or fresh source information.
            content = entry.get('content')
            parts = content if isinstance(content, list) else [{'text': content}] if isinstance(content, str) else []
            for part in parts:
                if not isinstance(part, dict) or not isinstance(part.get('text'), str):
                    continue
                if entry.get('role') == 'user' and part['text'].startswith(TURN_PREFIX):
                    raw_text = part['text']
                    part['text'] = delta.prompt(raw_text)
                    removed += max(0, len(raw_text) - len(part['text']))
                    continue
                marker = _MEMORY_MARKER
                text = part['text']; position = text.find(marker)
                if position < 0:
                    continue
                begin = position + len(marker)
                try:
                    memory, size = json.JSONDecoder().raw_decode(text[begin:])
                except ValueError:
                    continue
                if not isinstance(memory, dict) or memory.get('kind') != 'source_bound_memory_evidence':
                    continue
                fingerprint = _memory_key(memory)
                if fingerprint in seen_memory:
                    replacement = '{"kind":"source_bound_memory_evidence","already_in_context":true}'
                    part['text'] = text[:begin] + replacement + text[begin + size:]
                    removed += max(0, size - len(replacement))
                seen_memory.add(fingerprint)
            if isinstance(content, str) and parts:
                entry['content'] = parts[0]['text']
        latest_text = ''
        for entry in entries:
            if isinstance(entry, dict) and entry.get('role') == 'user':
                content = entry.get('content', [])
                parts = content if isinstance(content, list) else [{'text': content}]
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get('text'), str) and part['text'].startswith(TURN_PREFIX):
                        latest_text = part['text']
        return projected, {**delta.metrics, 'context_removed_chars': removed,
                           'historical_media_removed': media_removed,
                           'context_latest_turn_estimated_tokens': estimate_text_tokens(latest_text),
                           'context_estimated_tokens': estimate_request_tokens(projected)}
