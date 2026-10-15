"use strict";
const $ = id => document.getElementById(id);
const state = {conversations:[], selected:null, conversation:null, turns:new Map(), visible:12, requests:[], requestKey:null, snapshot:null, paused:false, busy:false, deleting:false, generation:0, inputGeneration:0};
const labels = {running:"执行中",sent:"已发送",ok:"完成",failed:"失败",blocked:"受限",uncertain:"发送结果未知",silent:"静默完成",interrupted:"执行中断"};
const kinds = {group:"群聊",background_group:"独立群巡检",private:"私聊",public:"公开任务",research:"独立资料任务",owner_task:"独立委派任务"};
const triggers = {research_delegation:"主对话委派",research_result:"独立任务结果回传",group_direct:"群聊提问",group_messages:"群聊新消息批次",group_schedule:"后台群观察",group_goal:"到期群目标",public_goal:"到期公开工作目标",private_message:"私聊消息",private_wakeup:"到期私聊提醒",owner_directive:"委派任务指令",owner_wakeup:"到期委派任务",qzone_publish:"空间发布巡检",qzone_comment:"好友动态巡检",profile:"资料维护任务"};
const directTriggers = new Set(["group_direct","private_message","owner_directive"]);
const number = value => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN") : "未知";
const date = value => value ? new Date(value*1000).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false}) : "—";
const clock = value => value ? new Date(value*1000).toLocaleTimeString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false}) : "—";
const json = value => typeof value === "string" ? value : JSON.stringify(value,null,2) ?? "—";
const el = (tag,text,className) => {const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
const statusName = value => labels[value] || value || "未知";
const scopeName = c => `${kinds[c.context_kind] || kinds[c.surface] || "其他"} · ${c.scope_id || "来源未标注"}${["research","owner_task"].includes(c.context_kind)&&c.context_scope?" · 任务 "+c.context_scope.split(":").at(-1).slice(0,8):""}`;
const requestKey = (t,id) => `${t.meta.trace_id}/${id}`;
function requestLabel(id,purpose){const parts=/^[mc](\d+)-a(\d+)$/.exec(id||"");return parts ? `${purpose==="compaction"?"上下文压缩":"模型"}第 ${+parts[1]} 步 · 第 ${+parts[2]} 次尝试` : `${purpose==="compaction"?"上下文压缩":"模型请求"} · ${id||"未知"}`;}
async function api(path){const r=await fetch(path,{cache:"no-store"});if(!r.ok)throw new Error(r.status===404?"记录已过期或尚未写入":"监控读取暂时不可用（HTTP "+r.status+"）");return r.json();}
function error(message){$("connection").textContent=message;$("connection").classList.add("error");}
function detail(node,title,data){const d=el("details");d.append(el("summary",title),el("pre",json(data)));node.append(d);return d;}
function nearBottom(){const s=$("thread-scroll");return s.scrollHeight-s.scrollTop-s.clientHeight<100;}
function followTail(){if($("follow").checked)$("thread-scroll").scrollTop=$("thread-scroll").scrollHeight;}
function sidebar(){
 const surface=$("surface").value,query=$("search").value.trim().toLowerCase();
 const rows=state.conversations.filter(c=>(surface==="all"||c.surface===surface)&&json([c.scope_id,c.actor_id,c.request_preview,c.conversation_id,...c.turns.map(t=>t.context_scope)]).toLowerCase().includes(query));
 const signature=json([state.selected,state.deleting,surface,query,rows.map(c=>[c.conversation_id,c.turn_count,c.native_session_count,c.status,c.request_preview])]);if($("conversations").dataset.signature===signature)return;$("conversations").dataset.signature=signature;
 $("conversation-count").textContent=String(rows.length);$("conversations").replaceChildren();
 for(const c of rows){const item=el("div",undefined,"conversation-item"),b=el("button",undefined,"conversation"+(c.conversation_id===state.selected?" selected":""));b.disabled=state.deleting;b.append(el("div",scopeName(c),"conversation-title"));const row=el("div",undefined,"conversation-row");row.append(el("span",`${c.turn_count} 回合 · ${c.native_session_count} 段执行上下文`),el("span",statusName(c.status)));b.append(row,el("div",c.identity_quality==="native_only"?"旧记录 · 逻辑关联未知":c.request_preview||"等待下一轮执行","conversation-preview"));b.addEventListener("click",()=>selectConversation(c.conversation_id));const remove=el("button","删除","delete-conversation");remove.disabled=state.deleting;remove.title="删除此会话的监控记录";remove.setAttribute("aria-label","删除监控记录："+scopeName(c));remove.addEventListener("click",()=>deleteConversation(c));item.append(b,remove);$("conversations").append(item);}
 if(!rows.length)$("conversations").append(el("p","暂无符合条件的会话","muted small"));
}
function header(c){
 $("scope-label").textContent=c.identity_quality==="logical"?"持续 Agent 会话":"旧版执行记录";
 $("conversation-title").textContent=scopeName(c);
 $("conversation-subtitle").textContent=`${c.turn_count} 个执行回合 · ${c.native_session_count} 段执行上下文 · 会话 ${c.conversation_id.slice(0,12)} · ${c.model||"模型见各次请求"}`;
 $("conversation-status").textContent=statusName(c.status);$("conversation-status").className="chip "+(labels[c.status]?c.status:"");
 $("identity-note").textContent=c.identity_quality==="logical"?"按实际作用域关联；下方标明每轮延续、重建与触发来源。":"旧记录缺少逻辑会话标识，仅按已知执行标识分组；不能据此判定不同列表项是否共享聊天历史。";
 $("load-older").hidden=c.turns.length<=state.visible;
 $("open-input").disabled=!state.requests.length;
 $("delete-selected").disabled=state.deleting;
}
function boundary(meta,previous){
 const native=(meta.session_key||meta.trace_id).slice(0,8),base=`执行标识 ${native}`;
 if(meta.context_state==="continued")return ["延续同一执行会话 · "+base+(previous?"":" · 较早过程可能未采集"),false];
 const reasons={previous_failed:"上一轮失败后重建",interrupted_turn:"上轮执行中断后重建",model_format_changed:"模型或输入格式变更后重建",history_unavailable:"执行历史不可用后重建",session_rotation:"切换执行会话",first_turn:"开始执行会话"};
 if(meta.context_state==="restored")return [`${reasons[meta.context_reason]||"重建执行会话"} · 恢复 ${number(meta.restored_context_items)} 条安全上下文 · ${base}`,true];
 if(meta.context_state==="new")return [`${reasons[meta.context_reason]||"新的执行会话"} · ${base}`,true];
 return [(previous&&previous.session_key===meta.session_key?"相同执行标识（旧记录）":"上下文延续状态未记录")+" · "+base,false];
}
function createTurn(meta,index,previous){
 const t={meta,cursor:0,events:[],streams:{},requests:new Map(),tools:new Map(),node:el("article",undefined,"turn"),index};
 t.node.dataset.trace=meta.trace_id;
 const [text,reset]=boundary(meta,previous);t.boundary=el("div",text,"context-boundary"+(reset?" reset":""));t.node.append(t.boundary);
 const trigger=el("div",undefined,"trigger-message"+(directTriggers.has(meta.trigger_kind)?"":" background")),heading=el("div",undefined,"trigger-heading");
 heading.append(el("span",`第 ${index+1} 回合 · ${triggers[meta.trigger_kind]||"触发原因未记录"}`),el("time",clock(meta.started_at)));trigger.append(heading,el("div",meta.trigger_text||meta.request_preview||"后台唤醒；本轮没有新的用户提问。","trigger-text"));t.node.append(trigger);
 const why=el("div",undefined,"trigger-cause");why.append(el("strong","为什么开始行动"),el("div",meta.trigger_reason||"此回合没有记录触发依据，具体原因未知。","trigger-explanation"));
 if(meta.parent_context_scope)why.append(el("div","父会话："+meta.parent_context_scope+" · 结果交回主对话回复","trigger-evidence"));
 const evidence=meta.trigger_evidence;if(evidence){const lines=[];if(evidence.source_id)lines.push("来源 #"+evidence.source_id);if(evidence.batch_size)lines.push("消息批次 "+evidence.batch_size+" 条");if(evidence.job_id)lines.push("队列任务 #"+evidence.job_id);if(evidence.task_id)lines.push("委派任务 #"+evidence.task_id);if(evidence.job_key)lines.push("调度任务 "+evidence.job_key);if(evidence.due_at){const value=new Date(evidence.due_at);lines.push("到期 "+(Number.isNaN(value.getTime())?evidence.due_at:value.toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false})));}if(evidence.goal)lines.push("工作目标："+evidence.goal);if(lines.length)why.append(el("div",lines.join(" · "),"trigger-evidence"));detail(why,"查看触发依据",{reason_code:meta.trigger_reason_code,...evidence});}t.node.append(why);
 t.node.append(el("div",`${meta.source_id?"关联来源 #"+meta.source_id+" · ":""}${date(meta.started_at)} · 回合 ${meta.trace_id.slice(0,8)}`,"context-note"));
 t.flow=el("div",undefined,"agent-flow");const agent=el("div",undefined,"agent-heading");agent.append(el("span","町","agent-avatar"),el("strong","Agent"),el("span",scopeName(meta),"muted small"));t.flow.append(agent);t.node.append(t.flow);
 t.warning=el("div",undefined,"turn-warning");t.node.append(t.warning);return t;
}
function eventNode(t,event,title,className){const n=el("div",undefined,"event-node "+(className||"")),head=el("div",undefined,"event-head");head.append(el("strong",title),el("time",clock(event.at)));n.append(head);t.flow.append(n);return {node:n,head};}
function renderEvent(t,event){
 const d=event.data||{};
 if(event.kind==="turn.started"||event.kind==="model.progress")return;
 if(event.kind==="model.request"){
  const r=eventNode(t,event,requestLabel(d.request_id,d.purpose),"request"),chip=el("span","请求中","chip running");r.head.insertBefore(chip,r.head.lastChild);r.chip=chip;r.meta=d;r.event=event;r.turn=t;
  r.note=el("div","正在等待模型响应。","event-note");r.node.append(r.note);const actions=el("div",undefined,"step-actions"),b=el("button","查看本次输入","quiet input-button");b.addEventListener("click",()=>openRequest(requestKey(t,d.request_id)));actions.append(b);r.node.append(actions);
  r.summary=el("div",undefined,"reasoning");r.summary.append(el("div","公开思考摘要","reasoning-label"));r.summaryText=el("div",undefined,"reasoning-text");r.summary.append(r.summaryText);r.summary.hidden=true;r.node.append(r.summary);r.reasoning=el("div",undefined,"reasoning-status");r.node.append(r.reasoning);
  r.output=el("div",undefined,"output-text");r.node.append(r.output);r.metrics=el("div",undefined,"step-metrics");r.node.append(r.metrics);t.requests.set(d.request_id,r);return;
 }
 if(event.kind==="model.finished"){const r=t.requests.get(d.request_id);if(r){r.finished=d;detail(r.node,"请求结束详情",d);}else{const r=eventNode(t,event,"模型请求结束（开始事件未采集）","request");detail(r.node,"请求结束详情",d);}return;}
 if(event.kind==="tool.started"){
  const r=eventNode(t,event,"调用工具 · "+(d.name||"未知"),"tool");r.started=d;r.chip=el("span","执行中","chip running");r.head.insertBefore(r.chip,r.head.lastChild);r.note=el("div",json(d.arguments).slice(0,280),"tool-summary");r.node.append(r.note);detail(r.node,"工具参数",d.arguments);t.tools.set(d.call_id||`${d.name}/${event.seq}`,r);return;
 }
 if(event.kind==="tool.completed"){
  let r=t.tools.get(d.call_id);if(!r)r=[...t.tools.values()].findLast(v=>v.started.name===d.name&&!v.finished);if(!r){r=eventNode(t,event,"工具返回 · "+(d.name||"未知"),"tool");r.chip=el("span");r.head.insertBefore(r.chip,r.head.lastChild);}
  r.finished=d;r.chip.textContent=statusName(d.status);r.chip.className="chip "+(labels[d.status]?d.status:"");detail(r.node,"工具结果与回执",d);
  if(d.effect==="write"||d.receipt_id){const sent=d.status==="sent",text=r.started?.arguments?.text;r.node.append(el("div",sent?"已发送 · 实际回执已返回":`对外操作 · ${statusName(d.status)}，请查看回执`,"event-note"+(!sent?" error":"")));if(typeof text==="string")r.node.append(el("div",text,sent?"delivery-text":"output-text"));}return;
 }
 if(event.kind==="turn.completed"){
  const r=eventNode(t,event,"本回合结束 · "+statusName(d.status),"terminal"+(["failed","blocked","uncertain","interrupted"].includes(d.status)?" failed":""));
  const reason={runtime_closed:"执行被关闭",runtime_error:"运行出错",native_runtime_error:"执行器出错",no_reply:"本轮未发送回复",completed:"执行完成"}[d.reason]||d.reason;
  const last=[...t.requests.values()].at(-1)?.finished;
  r.node.append(el("div",last?.http_status===503&&d.status==="failed"?"模型接口持续返回 HTTP 503，有限重试已结束，本轮没有得到模型结果。":reason||"查看详情了解结果。","event-note"));detail(r.node,"结束详情",d);return;
 }
 const r=eventNode(t,event,event.kind==="runtime.phase"?d.phase||"执行阶段":event.kind,"");detail(r.node,"阶段详情",d);
}
function updateRequest(r){
 const stream=r.turn.streams[r.meta.request_id],f=r.finished;
 const endedWithoutResult=!f&&r.turn.meta.status!=="running";
 r.chip.textContent=f ? f.outcome==="http_retry"?"将重试":f.outcome==="http_error"?"请求失败":f.outcome?.endsWith("retry")?"将重试":"已结束" : "请求中";
 r.chip.className="chip "+(f?.http_status>=400?"failed":f?"ok":"running");
 if(f?.provider_rejection_kind){r.chip.textContent="输入被模型接口拒绝";r.chip.className="chip failed";}
 else if(f?.output_disposition==="discarded"&&!f.outcome?.endsWith("retry")){r.chip.textContent="输出已丢弃";r.chip.className="chip failed";}
 if(endedWithoutResult){r.chip.textContent="结果未采集";r.chip.className="chip uncertain";}
 const http=f?.http_status,explanations={503:"上游服务暂不可用",429:"接口限流",401:"接口鉴权失败",403:"接口拒绝请求",400:"请求参数被接口拒绝"};
 r.note.textContent=f?http>=400?`HTTP ${http} · ${explanations[http]||"接口请求失败"}${f.outcome==="http_retry"?" · 按既定策略重试":""}`:({completed:"模型响应结束。",empty_output_retry:"未得到有效输出，继续有限重试。",transport_retry:"连接异常，继续有限重试。"}[f.outcome]||`结果：${f.outcome||"已结束"}`):"正在等待模型响应。";
 r.note.className="event-note"+(http>=400?" error":"");
 if(endedWithoutResult)r.note.textContent="回合已经结束，本次请求的结束事件未采集；不能判断请求结果。";
 if(f?.provider_rejection_kind)r.note.textContent="接口返回明确拒绝；本次生成未交给 agent 执行，未自动重试。已存在的真实发送回执单独保留。";
 r.summary.hidden=!stream?.summary;r.summaryText.textContent=stream?.summary||"";
 r.reasoning.textContent=stream?.summary?"接口提供的公开摘要；详细隐藏思维链不展示。":stream?.reasoning_chars?`已收到推理进度 ${number(stream.reasoning_chars)} 字符；接口未提供公开摘要。`:(f||endedWithoutResult?"本次接口未提供公开思考摘要。":"等待公开思考摘要；也可继续查看工具行动。");
 const disposition=f?.output_disposition;
 const outputLabel=disposition==="discarded"?"模型观测输出（已丢弃，未提交执行）":disposition==="committed"?"模型输出（已确认提交）":!f&&r.meta.model?.startsWith("gemini-")?"模型观测输出（待接口完整确认）":"模型文字输出";
 r.output.textContent=stream?.output?outputLabel+"：\n"+stream.output+(stream.truncated?"\n[观测文本达到上限，后续未记录]":""):"";
 r.metrics.textContent=`实际输入 ${number(f?.input_tokens)} · 缓存输入 ${number(f?.cached_input_tokens)} · 实际输出 ${number(f?.output_tokens)} token${typeof f?.elapsed_ms==="number"?" · "+(f.elapsed_ms/1000).toFixed(1)+" s":""}`;
 if(typeof f?.input_tokens==="number"&&typeof f?.cached_input_tokens==="number")r.metrics.textContent+=` · 未缓存 ${number(f.input_tokens-f.cached_input_tokens)}`;
 if(typeof f?.observation_context_estimated_tokens==="number")r.metrics.textContent+=` · 观察估算 ${number(f.observation_context_estimated_tokens)} / ${number(f.observation_context_budget_tokens)}`;
 else if(typeof f?.live_context_estimated_tokens==="number")r.metrics.textContent+=` · 临时输入估算 ${number(f.live_context_estimated_tokens)} / ${number(f.live_context_budget_tokens)}`;
 if(typeof f?.memory_context_estimated_tokens==="number")r.metrics.textContent+=` · 记忆估算 ${number(f.memory_context_estimated_tokens)} / ${number(f.memory_context_budget_tokens)} · 记忆来源 ${number(f.memory_context_source_count)}`;
 if(f?.context_compaction_threshold_tokens)r.metrics.textContent+=` · 压缩阈值 ${number(f.context_compaction_threshold_tokens)}`;
 if(f?.context_target_input_tokens)r.metrics.textContent+=` · 压缩目标 ${number(f.context_target_input_tokens)}`;
 if(f?.context_history_budget_tokens)r.metrics.textContent+=` · 旧历史预算 ${number(f.context_history_budget_tokens)}`;
}
async function loadTurn(t,generation){
 if(t.loading)return;t.loading=true;try{
 const result=await api(`/api/traces/${t.meta.trace_id}?after=${t.cursor}`);if(generation!==state.generation)return;
 t.meta=result.meta;t.cursor=result.cursor;Object.assign(t.streams,result.streams);
 for(const event of result.events){t.events.push(event);renderEvent(t,event);}for(const r of t.requests.values())updateRequest(r);
 t.warning.textContent=t.meta.dropped_events?`已省略 ${t.meta.dropped_events} 条观测事件；此处不是完整执行记录。`:"";
 }finally{t.loading=false;}
}
function requestOptions(){
 const wanted=state.requestKey;state.requests=[];for(const t of state.turns.values())for(const r of t.requests.values())state.requests.push(r);
 state.requests.sort((a,b)=>a.turn.meta.started_at-b.turn.meta.started_at||a.event.seq-b.event.seq);
 const signature=state.requests.map(r=>requestKey(r.turn,r.meta.request_id)).join("|");
 if($("requests").dataset.signature!==signature){$("requests").replaceChildren();for(const r of state.requests){const o=el("option",`第 ${r.turn.index+1} 回合 · ${requestLabel(r.meta.request_id,r.meta.purpose)}`);o.value=requestKey(r.turn,r.meta.request_id);$("requests").append(o);}$("requests").dataset.signature=signature;}
 if(wanted)$("requests").value=wanted;$("open-input").disabled=!state.requests.length;
}
function usageFooter(stats){
 const c=stats?.context||{},u=stats?.usage||{},percent=v=>typeof v==="number"&&Number.isFinite(v)?(v*100).toFixed(1)+"%":"未知";
 $("context-usage").textContent=`${number(c.input_tokens)} / ${number(c.window_tokens)}`;
 $("context-source").textContent=`${c.source==="reported"?"上游实报":c.source==="estimated"?"估算 · 尚无输入实报":"占用未知"}${c.window_source==="configured_reference"?" · 当前配置参考":""} · ${percent(c.ratio)}${c.input_ceiling!=null?" · 输入预算 "+number(c.input_ceiling):""}`;
 $("context-progress").value=typeof c.ratio==="number"?Math.min(1,Math.max(0,c.ratio)):0;
 $("latest-cache-ratio").textContent=percent(c.cache_ratio);
 $("latest-cache-tokens").textContent=`缓存 ${number(c.cached_input_tokens)} token`;
 $("total-cache-ratio").textContent=percent(u.cache_ratio);
 $("total-cache-tokens").textContent=`已报缓存 ${number(u.cached_input_tokens)} token`;
 $("total-input-tokens").textContent=number(u.input_tokens);
 $("uncached-input-tokens").textContent=`其中已报未缓存 ${number(u.uncached_input_tokens)}`;
 $("total-output-tokens").textContent=number(u.output_tokens);
 $("total-model-requests").textContent=number(u.request_count);
 $("usage-coverage").textContent=stats?`本会话保留记录 · 实报覆盖：输入 ${u.input_known_requests}/${u.request_count}、输出 ${u.output_known_requests}/${u.request_count}、缓存 ${u.cache_known_requests}/${u.request_count} 请求。累计消耗不等于窗口占用；缺失不按零计算。${u.events_complete?"":"部分观测事件缺失或尚未读完。"}`:"正在读取当前会话用量…";
}
async function updateConversation(c){
 const generation=state.generation,following=$("follow").checked&&nearBottom();state.conversation=c;
 const retained=new Set(c.turns.map(meta=>meta.trace_id));let removed=false;for(const [id,t] of state.turns){if(!retained.has(id)){t.node.remove();state.turns.delete(id);removed=true;}}
 if(removed){state.inputGeneration++;state.requestKey=null;state.snapshot=null;$("input-drawer").hidden=true;$("comparison").hidden=true;$("input-body").replaceChildren();for(const id of ["input-meta","raw-input","previous-body","current-body","compare-info"])$(id).textContent="";}
 const ordered=[...c.turns].sort((a,b)=>a.started_at-b.started_at||a.trace_id.localeCompare(b.trace_id)),visible=ordered.slice(-state.visible),shown=new Set(visible.map(m=>m.trace_id));
 // Keep cached cursors and expanded details, but only mount the visible page.
 for(const [id,t] of state.turns){if(!shown.has(id))t.node.remove();}
 for(const meta of visible){if(!state.turns.has(meta.trace_id)){const index=ordered.findIndex(t=>t.trace_id===meta.trace_id),t=createTurn(meta,index,ordered[index-1]);state.turns.set(meta.trace_id,t);}}
 for(const [index,meta] of visible.entries()){const t=state.turns.get(meta.trace_id),at=$("turn-list").children[index];if(at!==t.node)$("turn-list").insertBefore(t.node,at||null);}
 const needed=visible.map(m=>state.turns.get(m.trace_id)).filter(t=>t.cursor<(c.turns.find(m=>m.trace_id===t.meta.trace_id)?.last_seq||0)||!t.cursor||t.meta.status==="running");
 // Bound concurrent reads; event cursors retain expanded cards and growing streams.
 for(let i=0;i<needed.length;i+=4){await Promise.all(needed.slice(i,i+4).map(t=>loadTurn(t,generation)));if(generation!==state.generation)return;}
 requestOptions();header(c);if(following)followTail();
 try{const stats=await api(`/api/conversations/${c.conversation_id}/metrics`);if(generation===state.generation)usageFooter(stats);}catch(e){if(generation===state.generation){usageFooter(null);$("usage-coverage").textContent="用量统计暂不可用；执行记录仍可查看。";}}
}
function clearSelection(){
 state.generation++;state.inputGeneration++;state.selected=null;state.conversation=null;state.turns=new Map();state.requests=[];state.requestKey=null;state.snapshot=null;state.visible=12;
 for(const id of ["turn-list","requests","input-body"])$(id).replaceChildren();
 for(const id of ["input-meta","raw-input","previous-body","current-body","compare-info"])$(id).textContent="";
 $("requests").dataset.signature="";$("input-drawer").hidden=true;$("comparison").hidden=true;$("detail").hidden=true;$("empty").hidden=false;usageFooter(null);
}
async function refreshConversations(){
 const result=await api("/api/conversations");state.conversations=result.conversations;
 if(state.selected&&!state.conversations.some(c=>c.conversation_id===state.selected))clearSelection();
 sidebar();if(!state.selected&&state.conversations.length)await selectConversation(state.conversations[0].conversation_id);
 else if(state.selected)await updateConversation(state.conversations.find(c=>c.conversation_id===state.selected));
}
async function deleteConversation(c){
 if(!c||state.deleting)return;state.deleting=true;sidebar();$("delete-selected").disabled=true;
 try{
  const dialog=$("delete-dialog");dialog.returnValue="cancel";$("delete-description").textContent=`${scopeName(c)} · 当前 ${c.turn_count} 个回合`;
  const confirmed=new Promise(resolve=>dialog.addEventListener("close",()=>resolve(dialog.returnValue==="delete"),{once:true}));dialog.showModal();if(!await confirmed)return;
  state.generation++;state.inputGeneration++;
  const controls=await api("/api/controls");if(!controls.record_deletion)throw new Error("查看器尚未支持删除，请重新打开监控网页。");
  const response=await fetch(`/api/conversations/${c.conversation_id}`,{method:"DELETE",headers:{"X-Monitor-Token":controls.deletion_token},cache:"no-store"});
  if(!response.ok&&response.status!==404)throw new Error(response.status===403?"删除验证已失效，请刷新网页后重试。":"监控记录删除失败，请稍后重试（HTTP "+response.status+"）。");
  const result=response.ok?await response.json():{deleted_turns:0};if(state.selected===c.conversation_id)clearSelection();await refreshConversations();
  $("connection").textContent=result.deleted_turns?`已删除 ${result.deleted_turns} 个回合的监控记录`:"该会话记录已删除或过期";$("connection").classList.remove("error");
 }catch(e){error(e.message);}finally{state.deleting=false;sidebar();$("delete-selected").disabled=false;}
}
async function selectConversation(id){
 if(state.selected===id)return;state.generation++;state.inputGeneration++;state.selected=id;state.turns=new Map();state.visible=12;state.requests=[];state.requestKey=null;state.snapshot=null;$("turn-list").replaceChildren();$("input-drawer").hidden=true;$("comparison").hidden=true;$("requests").dataset.signature="";
 $("empty").hidden=true;$("detail").hidden=false;$("follow").checked=true;sidebar();
 usageFooter(null);
 try{const c=state.conversations.find(c=>c.conversation_id===id)||await api(`/api/conversations/${id}`);await updateConversation(c);followTail();}catch(e){error(e.message);}
}
function inputBlock(title,role,content,index){const d=el("details",undefined,"input-block"),s=el("summary");s.append(el("span",role,"block-role"),el("span",title),el("span",`${json(content).length} 字符`,"block-size"));d.append(s,el("pre",json(content)));if(index===0)d.open=true;return d;}
async function openRequest(key){
 if(!key)return;const generation=++state.inputGeneration,selected=state.selected;state.requestKey=key;$("input-drawer").hidden=false;$("comparison").hidden=true;$("input-meta").textContent="正在读取实际发送的输入…";$("input-body").replaceChildren();$("raw-input").textContent="";$("compare").disabled=true;$("requests").value=key;
 try{const [trace,id]=key.split("/"),snapshot=await api(`/api/traces/${trace}/requests/${id}`);if(generation!==state.inputGeneration||selected!==state.selected)return;state.snapshot=snapshot;const {meta,body}=snapshot;
  const r=state.requests.find(r=>requestKey(r.turn,r.meta.request_id)===key);
  $("input-meta").textContent=`${r?"第 "+(r.turn.index+1)+" 回合 · ":""}${requestLabel(id,meta.purpose)}\n${body.model||"已配置模型"} · ${date(meta.captured_at)}\n密钥、图片二进制和隐藏推理已省略。${meta.truncated?"快照已截断，完整输入不能由本页还原。":"快照未触发大小截断。"}\n与上一请求共有 ${number(meta.shared_prefix_chars)} 个序列化前缀字符；这只是结构对照，实际缓存以返回的 usage 为准。`;
  if(body._truncated)$("input-body").append(inputBlock("截断预览","preview",body.preview,0));else{
   if(body.instructions)$("input-body").append(inputBlock("规则与人格","instructions",body.instructions,-1));
   if(body.tools)$("input-body").append(inputBlock(`工具定义 · ${body.tools.length} 项`,"tools",body.tools,-1));
   const items=body.input||body.messages||[];for(const [i,item] of (Array.isArray(items)?items:[items]).entries()){const role=item?.role||item?.type||"text";$("input-body").append(inputBlock(`${i+1}. ${role}${item?.name?" · "+item.name:""}`,role,item,i));}
  }
  $("raw-input").textContent=json(body);$("compare").disabled=!(meta.previous_request_id&&meta.previous_trace_id);
 }catch(e){if(generation===state.inputGeneration){$("input-meta").textContent=e.message;error(e.message);}}
}
async function compare(){const snapshot=state.snapshot,key=state.requestKey,generation=state.inputGeneration;if(!snapshot?.meta.previous_request_id)return;
 try{const m=snapshot.meta,previous=await api(`/api/traces/${m.previous_trace_id}/requests/${m.previous_request_id}`);if(key!==state.requestKey||generation!==state.inputGeneration)return;const prev=state.turns.get(m.previous_trace_id),current=state.turns.get(key.split("/")[0]);$("comparison").hidden=false;$("compare-info").textContent=`同一执行会话：${prev?"第 "+(prev.index+1)+" 回合":"较早回合 "+m.previous_trace_id.slice(0,8)} · ${requestLabel(m.previous_request_id,previous.meta.purpose)} → ${current?"第 "+(current.index+1)+" 回合":"当前回合"} · ${requestLabel(m.request_id,m.purpose)}；共同前缀 ${number(m.shared_prefix_chars)} 字符。按实际快照对照，缓存命中以 usage 为准。`;$("previous-body").textContent=json(previous.body);$("current-body").textContent=json(snapshot.body);$("comparison").scrollIntoView({block:"start"});}catch(e){error(e.message);}
}
async function poll(){if(state.busy||state.paused||state.deleting)return;state.busy=true;const generation=state.generation;try{const result=await api("/api/conversations");if(state.deleting||generation!==state.generation)return;state.conversations=result.conversations;if(state.selected&&!state.conversations.some(c=>c.conversation_id===state.selected))clearSelection();sidebar();if(!state.selected&&state.conversations.length)await selectConversation(state.conversations[0].conversation_id);else if(state.selected)await updateConversation(state.conversations.find(c=>c.conversation_id===state.selected));
 if(!state.deleting){$("connection").textContent="实时连接 · "+new Date().toLocaleTimeString("zh-CN",{hour12:false});$("connection").classList.remove("error");}}catch(e){if(!state.deleting)error(e.message);}finally{state.busy=false;}}
$("surface").addEventListener("change",sidebar);$("search").addEventListener("input",sidebar);
$("pause").addEventListener("click",()=>{state.paused=!state.paused;$("pause").textContent=state.paused?"恢复更新":"暂停更新";$("connection").textContent=state.paused?"更新已暂停":"正在连接";if(!state.paused)poll();});
$("delete-selected").addEventListener("click",()=>deleteConversation(state.conversation));
$("open-input").addEventListener("click",()=>openRequest(state.requestKey||requestKey(state.requests.at(-1).turn,state.requests.at(-1).meta.request_id)));
$("close-input").addEventListener("click",()=>{$("input-drawer").hidden=true;});$("requests").addEventListener("change",()=>openRequest($("requests").value));$("compare").addEventListener("click",compare);$("close-compare").addEventListener("click",()=>{$("comparison").hidden=true;});
$("follow").addEventListener("change",followTail);
$("thread-scroll").addEventListener("wheel",event=>{if(event.deltaY<0)$("follow").checked=false;},{passive:true});
$("load-older").addEventListener("click",async()=>{const scroller=$("thread-scroll"),oldHeight=scroller.scrollHeight,oldTop=scroller.scrollTop;$("follow").checked=false;state.visible+=12;try{await updateConversation(state.conversation);scroller.scrollTop=oldTop+scroller.scrollHeight-oldHeight;}catch(e){error(e.message);}});
document.addEventListener("keydown",e=>{if(e.key==="Escape")$("input-drawer").hidden=true;});
poll();setInterval(poll,1000);
