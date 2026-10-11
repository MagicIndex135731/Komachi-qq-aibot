"""Compose scheduled social participation only in the group process."""

from app.adapters.qzone import QzoneAdapter
from app.agent.social_contracts import SocialRuntimeOptions
from app.agent.social_executor import SocialExecutor
from app.agent.social_planner import SocialPlanner
from app.agent.social_policy import configured_agent_jobs
from app.agent.social_runtime import SocialRuntime
from app.providers.llm_client import LlmClient
from app.runtime.providers import build_proactive_judge_client
from app.adapters.account_profile import AccountProfileAdapter
from app.agent.autonomous_social import AutonomousSocialExecutor, AutonomousSocialPlanner
from app.agent.mcp_tools import McpToolProvider
from app.agent.python_sandbox import PythonSandbox
from app.agent.sandbox_gateway import HttpPythonSandbox
from copy import deepcopy
import asyncio


async def _close_transports(resources):
    await asyncio.gather(*(resource.aclose() if hasattr(resource, "aclose") else resource.close()
                           for resource in resources), return_exceptions=True)


def _close_transports_after_failed_build(resources):
    if not resources:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_close_transports(resources))
    else:
        loop.create_task(_close_transports(resources))


def build_social_agent(*, router, sender, gateway, llm_client):
    runtime = router.runtime
    settings = runtime.settings
    jobs = configured_agent_jobs(runtime)
    if not jobs:
        return None, None
    autonomous = any(job.autonomous_tools for job in jobs)
    compute = runtime.social_policy.get("compute", {})
    manifests = runtime.social_policy.get("mcp", [])
    # Validate extension settings before allocating pools or HTTP transports.
    if autonomous:
        if not isinstance(compute, dict) or type(compute.get("enabled", False)) is not bool:
            raise ValueError("invalid compute permission")
        if compute.get("enabled") and compute.get("endpoint"):
            if not isinstance(compute["endpoint"], str):
                raise ValueError("invalid sandbox endpoint")
            HttpPythonSandbox(compute["endpoint"], token=settings.autonomous_sandbox_token,
                output_root=(settings.data_dir / "autonomous_artifacts" / "sandbox").resolve(), client=object())
        if not isinstance(manifests, list) or len(manifests) > 4:
            raise ValueError("invalid MCP manifests")
        names_by_scope = {kind: set() for kind in {job.kind for job in jobs}}
        reserved_names = {"remember_intention", "list_artifacts", "read_artifact", "read_context", "search_history", "generate_image", "search_image_references",
                          "send_group", "publish_qzone", "comment_qzone", "set_nickname", "set_group_card", "set_avatar", "execute_python", "search_web"}
        reserved_names.update({"read_persona", "save_work_note", "read_work_note", "memory_search", "memory_read",
            "memory_write", "observe_group_images", "read_forward_messages", "observe_forward_images", "schedule_wakeup", "list_wakeups", "cancel_wakeup",
            "persona_task_status", "request_persona_refresh", "read_friend_feed", "read_friend_posts", "observe_post_images"})
        for manifest in manifests:
            if not isinstance(manifest, dict):
                raise ValueError("invalid MCP manifest")
            if manifest.get("enabled") is True:
                McpToolProvider(manifest, client=object())
                for spec in manifest["tools"]:
                    if spec["name"] in reserved_names:
                        raise ValueError("MCP tool conflicts with a built-in capability")
                    for scope in set(spec["scopes"]) & names_by_scope.keys():
                        if spec["name"] in names_by_scope[scope] or len(names_by_scope[scope]) >= 10:
                            raise ValueError("duplicate MCP capability or scoped tool budget exceeded")
                        names_by_scope[scope].add(spec["name"])
    client, planner, executor = llm_client, None, None
    external = []
    try:
        client = None if settings.agent_runtime == "harness" else build_proactive_judge_client(
            settings=settings.model_copy(update={
                "proactive_model_judge_enabled": True,
                "proactive_judge_max_output_tokens": settings.social_agent_max_output_tokens,
                "llm_timeout_seconds": min(settings.llm_timeout_seconds, settings.social_agent_decision_timeout_seconds),
            }), llm_client=llm_client, engine=router.engine,
        )
        client = client or llm_client
        if isinstance(client, LlmClient):
            client.vision_model = (settings.llm_vision_model or "").strip()
        qzone = QzoneAdapter(gateway)
        planner = AutonomousSocialPlanner(router=router, decision_client=client, qzone_adapter=qzone, max_workers=1)
        # Participation judging remains cheap; action reasoning uses the configured
        # primary model, not a lightweight classifier with limited planning ability.
        executor = AutonomousSocialExecutor(router=router, sender=sender, qzone_adapter=qzone,
            decision_client=llm_client, image_service=getattr(router, "group_image_service", None),
            profile_adapter=AccountProfileAdapter(gateway, account_id=settings.bot_qq,
                allowed_group_ids=tuple(job.scope_id for job in jobs if job.kind == "group"),
                media_roots=((settings.data_dir / "autonomous_artifacts").resolve(),))) if autonomous else SocialExecutor(router=router, sender=sender, qzone_adapter=qzone)
        planner.executor = executor
        if autonomous:
            if compute.get("enabled"):
                output = (settings.data_dir / "autonomous_artifacts" / "sandbox").resolve()
                if compute.get("endpoint"):
                    executor.sandbox = HttpPythonSandbox(compute["endpoint"], token=settings.autonomous_sandbox_token, output_root=output,
                        allowed_input_roots=((settings.data_dir / "autonomous_artifacts").resolve(),))
                else:
                    executor.sandbox = PythonSandbox(output_root=output, enabled=True,
                        allowed_input_roots=((settings.data_dir / "autonomous_artifacts").resolve(),))
                executor.compute_permission = deepcopy(compute)
            for manifest in manifests:
                if manifest.get("enabled") is True:
                    snapshot = deepcopy(manifest)
                    provider = McpToolProvider(manifest, permission_check=lambda snapshot=snapshot: snapshot in runtime.social_policy.get("mcp", []))
                    external.append(provider)
                    executor.register_tool_factory(provider.tools_for)
        original_close = planner.close
        def close():
            original_close()
            if isinstance(executor, AutonomousSocialExecutor):
                executor.close()
            if client is not llm_client and isinstance(client, LlmClient):
                client.http_client.close()
        planner.close = close
        async def aclose():
            close()
            resources = [*external]
            sandbox = getattr(executor, "sandbox", None)
            if sandbox is not None and hasattr(sandbox, "aclose"):
                resources.append(sandbox)
            await _close_transports(resources)
        planner.aclose = aclose
        agent = SocialRuntime(
            engine=router.engine, account_id=settings.bot_qq,
            jobs=lambda: configured_agent_jobs(runtime), observe=planner.observe,
            decide=planner.decide, execute=executor.execute, policy_check=executor.policy_check,
            options=SocialRuntimeOptions(poll_seconds=settings.social_agent_poll_seconds,
                                         decision_timeout_seconds=settings.social_agent_decision_timeout_seconds,
                                         execution_timeout_seconds=settings.agent_background_turn_timeout_seconds if autonomous else 45,
                                         lease_seconds=settings.social_agent_decision_timeout_seconds + (settings.agent_background_turn_timeout_seconds + 180 if autonomous else 180)),
            intention_repository=executor.repository if autonomous and settings.agent_runtime == "harness" else None,
            scope_for_job=executor.data_scope if autonomous and settings.agent_runtime == "harness" else None,
        )
        return agent, planner
    except Exception:
        if planner is not None:
            planner.close()
        if isinstance(executor, AutonomousSocialExecutor):
            executor.close()
        if client is not llm_client and isinstance(client, LlmClient):
            client.http_client.close()
        resources = [*external]
        sandbox = getattr(executor, "sandbox", None)
        if sandbox is not None and hasattr(sandbox, "aclose"):
            resources.append(sandbox)
        _close_transports_after_failed_build(resources)
        raise
