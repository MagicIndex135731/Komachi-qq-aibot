"""Combinable public QQ services, with independent per-target authority."""
from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path

from app.agent.social_contracts import SocialDecision, SocialObservation
from app.agent.social_policy import (autonomous_social_enabled, configured_social_jobs,
                                     in_quiet_hours, resolve_friend_ids)
from app.adapters.qzone import QzoneUnavailableError, QzoneProtocolError, QzoneRejectedError
from app.agent.social_images import post_image_inputs
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.agent.autonomous_social import schema, text_schema
from app.core.message_content import ImageAttachment


def build_public_tools(executor, observation, guard, store):
    if observation.job.kind == "group":
        return ()
    snapshots = {job.kind: job for job in configured_social_jobs(executor.router.runtime) if job.kind != "group"}
    posts = {(post.owner_id, post.post_id): post for post in observation.posts}
    viewed = set()
    tools = []

    def owner_directed():
        owner_authority = getattr(guard, "owner_authority", None)
        return owner_authority is not None and owner_authority() is True

    def authority(kind):
        guard.require_valid()
        job = snapshots.get(kind)
        current = next((j for j in configured_social_jobs(executor.router.runtime) if j.kind == kind), None)
        manual = owner_directed()
        if not manual and not autonomous_social_enabled(executor.router.runtime, kind):
            raise ValueError('public_tool_autonomy_disabled')
        if job is None or current != job or (not manual and in_quiet_hours(job.quiet_hours, executor.clock())):
            raise ValueError("public_tool_authority_revoked")
        if observation.context or observation.source_message_ids:
            raise ValueError("public_tool_received_private_context")
        return job

    def media(ids):
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate_artifact")
        result = []
        for artifact_id in ids:
            path = store.get(artifact_id)
            if path is None or path.suffix not in {".png", ".jpg", ".webp", ".gif"}:
                raise ValueError("public_artifact_unavailable")
            result.append(str(path))
        return tuple(result)

    def dispatch(kind):
        authority(kind)
        guard.on_dispatch()

    class PublicGuard:
        def __init__(self, kind):
            self.kind = kind
        @property
        def action_id(self):
            return guard.action_id
        def valid(self):
            try:
                authority(self.kind)
                return True
            except (ValueError, RuntimeError):
                return False
        def require_valid(self):
            authority(self.kind)
        def on_dispatch(self):
            dispatch(self.kind)
        @property
        def owner_authority(self):
            return getattr(guard, "owner_authority", None)

    def add(name, description, parameters, handler, *, effect="read", job=None):
        if (effect == 'write' and not owner_directed()
                and not autonomous_social_enabled(executor.router.runtime, job.kind)):
            return
        patrol = observation.job.patrol_interval_seconds is not None and not owner_directed()
        if patrol and effect == "write" and job.kind != observation.job.kind:
            return
        tools.append(AgentTool(name, description, parameters, handler, effect,
            scope="public:" + job.kind if job else "",
            daily_limit=(job.max_daily_per_friend or job.max_daily) if job else None,
            cooldown_seconds=job.cooldown_seconds if job else None,
            per_target_daily_limit=bool(job and job.kind == "qzone_comment" and job.max_daily_per_friend),
            operation_daily_limit=job.max_daily if job and job.patrol_interval_seconds is not None else None,
            shared_cooldown_seconds=job.shared_cooldown_seconds if job else 0,
            max_writes_per_wake=1 if job and job.patrol_interval_seconds is not None else None))

    if "qzone_comment" in snapshots:
        job = snapshots["qzone_comment"]
        owner = {"type": "integer", "minimum": 1, "maximum": 2**63 - 1}
        async def read_social(args):
            current = authority('qzone_comment')
            from app.storage.public_social import PublicSocialRepository
            try:
                result = PublicSocialRepository(executor.router.engine, account_id=current.scope_id,
                    clock=executor.clock).read(**args)
            except ValueError:
                return ToolOutcome({'error': 'social_activity_query_invalid'}, 'blocked')
            authority('qzone_comment')
            return ToolOutcome(result)
        history_schema = schema(record_id=text_schema(40), kind={'type': 'string',
            'enum': ['all', 'comment', 'thought', 'observation']}, owner_id=owner,
            date=text_schema(10), before=text_schema(40), limit={'type': 'integer', 'minimum': 1, 'maximum': 10})
        history_schema['required'] = []
        add('read_social_activity', '查询自己近期公开看帖、评论草稿/成功/失败/未知回执和显式打算、笔记。'
            '主人提到早上/之前的评论或想法时先查此工具，不用重扫好友动态。默认5条短摘要；kind筛选、date按上海日期，'
            'record_id读取详情，before分页。仅公开记录，不含群私对话；记录不授权发送，恢复目标后仍须读原帖复验，已发送/未知不重做。',
            history_schema, read_social)
        async def current_friends():
            current = authority("qzone_comment")
            try:
                friends = await resolve_friend_ids(current, executor.qzone)
            except (QzoneUnavailableError, QzoneProtocolError, QzoneRejectedError):
                return current, None
            authority("qzone_comment")
            return current, friends

        def eligible_posts(found, current, friends):
            # Autonomous opportunities avoid old posts. A live owner directive
            # may explicitly request a friend's latest post even days later.
            manual = owner_directed()
            eligible = [p for p in found if p.owner_id in friends and p.owner_id != current.scope_id and
                (age := (executor.clock() - p.published_at).total_seconds()) >= 0 and
                (manual or age <= current.max_post_age_seconds)]
            posts.update(((p.owner_id, p.post_id), p) for p in eligible)
            return ToolOutcome({"posts": [{"owner_id": p.owner_id, "post_id": p.post_id, "text": p.text[:1200],
                "image_count": len(p.images), "published_at": p.published_at.isoformat()} for p in eligible]})

        async def find_friend(args):
            current, friends = await current_friends()
            if friends is None:
                return ToolOutcome({'error': 'friend_lookup_unavailable'}, 'blocked')
            contacts = await executor.qzone.get_friend_contacts()
            current, friends = await current_friends()
            if friends is None:
                return ToolOutcome({'error': 'friend_lookup_unavailable'}, 'blocked')
            query = args['query'].strip().casefold()
            aliases = {}
            for key, profile in executor.router.runtime.personas.items():
                author = profile.get('source_user_id')
                labels = [key, profile.get('name', ''), *profile.get('aliases', [])]
                if author in friends and any(str(label).casefold() == query for label in labels):
                    aliases[author] = str(profile.get('name') or key)
            selected = [dict(row, configured_alias=aliases.get(row['owner_id'], '')) for row in contacts
                if row['owner_id'] in friends and (str(row['owner_id']) == query or row['owner_id'] in aliases
                    or query and any(query in row[field].casefold() for field in ('nickname', 'remark')))]
            return ToolOutcome({'friends': selected[:20], 'ambiguous': len(selected)>1,
                'trust': 'current_friend_roster_and_operator_aliases', 'must_not_guess_if_ambiguous': True})
        add('find_friend', '按QQ号、当前好友昵称/备注或已配置人格别名定位好友；同名须确认，不能猜QQ号',
            schema(query=text_schema(80)), find_friend)

        async def read_feed(args):
            current, friends = await current_friends()
            if friends is None:
                return ToolOutcome({"error": "friend_lookup_unavailable"}, "blocked")
            found = await executor.qzone.get_feeds(count=args["count"], allowed_owner_ids=friends,
                **({"require_friends": True} if current.all_friends else {}))
            current, friends = await current_friends()
            if friends is None:
                return ToolOutcome({"error": "friend_lookup_unavailable"}, "blocked")
            return eligible_posts(found, current, friends)
        add("read_friend_feed", "读取已授权好友的近期空间动态，取得真实作者和tid后自行选择；不预设评论目标",
            schema(count={"type": "integer", "minimum": 1, "maximum": 20}), read_feed)

        async def read_posts(args):
            current, friends = await current_friends()
            if friends is None or args["owner_id"] not in friends or args["owner_id"] == current.scope_id:
                return ToolOutcome({"error": "friend_not_authorized"}, "blocked")
            found = await executor.qzone.get_posts(args["owner_id"], count=12)
            current, friends = await current_friends()
            if friends is None or args["owner_id"] not in friends:
                return ToolOutcome({"error": "friend_authority_revoked"}, "blocked")
            return eligible_posts([p for p in found if p.owner_id == args["owner_id"]], current, friends)
        add("read_friend_posts", "按已授权好友读取真实空间原帖，返回作者与tid；不会访问其他好友", schema(owner_id=owner), read_posts)

        async def view_post(args):
            current, friends = await current_friends()
            if friends is None or args["owner_id"] not in friends:
                return ToolOutcome({"error": "friend_not_authorized"}, "blocked")
            key = (args["owner_id"], args["post_id"])
            post = posts.get(key)
            if post is None:
                return ToolOutcome({"error": "post_not_observed"}, "blocked")
            def materialize():
                with post_image_inputs(post, cache_dir=store.root / "public-observed-media") as images:
                    result = []
                    for image in images:
                        artifact_id = store.add(Path(image.local_path), "observed_public_pixels")
                        result.append((artifact_id, str(store.get(artifact_id))))
                    return result
            found = await executor.blocking.run(materialize)
            current, friends = await current_friends()
            if friends is None or args["owner_id"] not in friends:
                return ToolOutcome({"error": "friend_authority_revoked"}, "blocked")
            if found:
                viewed.add(key)
            return ToolOutcome({"post_id": post.post_id, "artifact_ids": [item[0] for item in found],
                "pixels_attached": bool(found), "image_count": len(post.images),
                "viewed_image_count": len(found), "unseen_image_count": max(0, len(post.images) - len(found)),
                "guidance": "artifact_ids是已读取图片的产物ID，不是评论必需附件；文字评论传artifact_ids=[]，不要为证明看过而附回原图。"},
                images=tuple(ImageAttachment(url="", local_path=item[1], reference_subject=f"好友{post.owner_id}动态{post.post_id}") for item in found))
        add("observe_post_images", "读取已观察原帖的实际图片像素并保存产物，最多9张；返回已读与未读数量，不能猜测未读图片内容", schema(owner_id=owner, post_id=text_schema(256)), view_post)

        async def assess_post(args):
            current, friends = await current_friends()
            key = (args['owner_id'], args['post_id'])
            post = posts.get(key)
            if friends is None or post is None or post.owner_id not in friends:
                return ToolOutcome({'error': 'post_not_observed'}, 'blocked')
            if args['decision'] == 'no_response' and post.images and not post.text.strip() and key not in viewed:
                return ToolOutcome({'error': 'read_pixels_or_defer'}, 'blocked')
            from app.storage.social_agent import SocialAgentRepository
            SocialAgentRepository(executor.router.engine).assess_post(account_id=current.scope_id, post=post,
                status=args['decision'], minutes=args['revisit_minutes'], now=executor.clock(),
                before_commit=guard.require_valid)
            return ToolOutcome({'saved': True, 'candidate_only': True, 'decision': args['decision'],
                'revisit_minutes': args['revisit_minutes']})
        assessment_schema = schema(owner_id=owner, post_id=text_schema(256), decision={'type': 'string',
                'enum': ['deferred', 'no_response']}, revisit_minutes={'type': 'integer', 'minimum': 5, 'maximum': 180},
                thought=text_schema(400, 0), next_step=text_schema(300, 0))
        assessment_schema['required'] = ['owner_id', 'post_id', 'decision', 'revisit_minutes']
        add('assess_qzone_post', '记录实际评估的原帖版本：暂缓deferred或本版本不回应no_response，5至180分钟后可重看；新内容重新判断。'
            '可用thought简记看法/暂缓理由、next_step记录实际后续打算，归入近期公开工作记忆。没读图不能否决纯图；不标记未评估候选，不保存长推理或例行等待意图。',
            assessment_schema, assess_post)

        async def comment(args):
            current, friends = await current_friends()
            if args["artifact_ids"] and not current.allow_images:
                return ToolOutcome({"error": "comment_images_disabled", "write_dispatched": False,
                    "guidance": "当前只允许文字评论，尚未派发；保留评论文字并改为artifact_ids=[]。看图产物不需要附回评论。"}, "blocked")
            key = (args["owner_id"], args["post_id"])
            original = posts.get(key)
            if friends is None or original is None or original.owner_id not in friends or (not original.text.strip() and key not in viewed):
                return ToolOutcome({"error": "post_or_pixels_not_observed"}, "blocked")
            obs = SocialObservation(current, executor.clock(), observation.persona, posts=(original,), authorized_friend_ids=friends)
            actual = SocialDecision("comment_qzone", args["text"], original.post_id, original.owner_id,
                image_files=media(args["artifact_ids"]))
            result = await executor.legacy.execute(obs, actual, PublicGuard("qzone_comment"))
            data = {"result": result.status}
            if result.reason:
                data["reason"] = result.reason
            return ToolOutcome(data, result.status, result.receipt_id)
        add("comment_qzone", "自行选择已读取好友的真实动态评论，派发前重新核对原帖；"
            + ("文字评论的artifact_ids填[]，仅在需要附图时提供产物ID。" if job.allow_images else
               "当前仅允许文字评论，artifact_ids必须为[]；读图不需要把原图附回评论。"), schema(owner_id=owner,
            post_id=text_schema(256), text=text_schema(300), artifact_ids={"type": "array", "items": text_schema(64),
                "maxItems": 2 if job.allow_images else 0}), comment, effect="write", job=job)

    if "qzone_publish" in snapshots:
        job = snapshots["qzone_publish"]
        async def publish(args):
            current = authority("qzone_publish")
            obs = SocialObservation(current, executor.clock(), observation.persona)
            actual = SocialDecision("publish_qzone", args["text"], image_files=media(args["artifact_ids"]))
            result = await executor.legacy.execute(obs, actual, PublicGuard("qzone_publish"))
            return ToolOutcome({"result": result.status}, result.status, result.receipt_id)
        add("publish_qzone", "发布自己的公开表达及公开产物，工具不指定生成顺序", schema(text=text_schema(1000),
            artifact_ids={"type": "array", "items": text_schema(64), "maxItems": 2}), publish, effect="write", job=job)

    if "profile" in snapshots and executor.profile is not None:
        job = snapshots["profile"]
        def profile_dispatch(kind):
            authority("profile")
            if not executor.repository.reserve_cost("profile:" + kind, 1):
                raise ValueError("profile_daily_budget")
            dispatch("profile")
        if "nickname" in job.identity_tools:
            async def nickname(args):
                authority("profile")
                await executor.profile.nickname(args["name"], before_dispatch=lambda: profile_dispatch("nickname"))
                return ToolOutcome({"bridge_acknowledged": True, "readback_verified": False}, "sent")
            add("set_nickname", "修改本人QQ账号昵称，独立权限与实际平台回执", schema(name=text_schema(32)), nickname, effect="write", job=job)
        if "avatar" in job.identity_tools:
            async def avatar(args):
                authority("profile")
                path = media([args["artifact_id"]])[0]
                await executor.profile.avatar(path, before_dispatch=lambda: profile_dispatch("avatar"))
                return ToolOutcome({"bridge_acknowledged": True, "readback_verified": False}, "sent")
            add("set_avatar", "使用本人公开产物设置自己的头像；不会读取群私图片", schema(artifact_id=text_schema(64)), avatar, effect="write", job=job)
    return tuple(tools)
