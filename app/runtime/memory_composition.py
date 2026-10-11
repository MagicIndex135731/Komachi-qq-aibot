"""Compose the bounded, source-linked memory runtime for a group process."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import logging
from time import perf_counter
from typing import Callable
from sqlalchemy import select
from app.config import AppSettings
from app.core.hybrid_memory_retriever import HybridMemoryRetriever
from app.core.legacy_memory_context import (
    GroupMemoryContextRequest,
    LegacyMemoryContext,
    member_label_for_user,
)
from app.core.memory_background_service import (
    CompactionEpisodeDeriver,
    MemoryBackgroundService,
    ShadowEvaluation,
    ShadowJobRequest as BackgroundShadowJobRequest,
    SqlAlchemyMemoryBackgroundStore,
)
from app.core.memory_compaction_service import MemoryCompactionService
from app.core.memory_fact_ranking import (
    PERSON_PORTRAIT_KINDS,
    fact_intent_policy,
    filter_member_query_features,
    is_composite_portrait_query,
    matching_member_fact_ids,
    memory_query_features,
    preferred_kinds_for_query,
    prefer_recent_viewing_event,
    rank_member_facts,
    recent_viewing_event_fallback,
    select_diverse_portrait_facts,
    select_temporal_current_facts,
    temporal_recency_required,
    viewing_fallback_match_ids,
)
from app.core.memory_fact_semantics import SemanticFactRanker
from app.core.memory_context_packer import (
    EvidenceMessage,
    MemoryContextPacker,
    MemoryFact,
    MemorySummary,
    PackedMemoryContext,
)
from app.core.memory_evidence_expander import MemoryEvidenceExpander
from app.core.memory_orchestrator import MemoryOrchestrator, ShadowJobRequest
from app.core.memory_query_resolver import MemoryQueryResolver
from app.core.member_identity import GroupMemberIdentity, group_member_identities_from_messages
from app.core.memory_retrieval_channels import build_memory_retrieval_channels
from app.core.memory_v2_context import MemoryV2ContextProvider
from app.core.time_utils import stored_as_utc
from app.providers.llm_client import LlmClient
from app.providers.semantic_embeddings import EmbeddingProvider, build_embedding_provider
from app.storage.db import (
    ensure_retrieval_vector_generation,
    find_active_retrieval_vector_generation,
)
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import (
    EpisodeRepository,
    MemoryRepository,
    MessageRepository,
    SummaryRepository,
    UserRepository,
)
from app.runtime.providers import (
    build_episode_topic_judge_client,
    build_episode_post_segment_client,
    build_memory_compaction_client,
    _build_query_rewrite_provider,
)

MEMORY_SEGMENTATION_GENERATION = "segment-v2"
MEMORY_COMPACTION_GENERATION = "compact-v2"
MEMORY_CONFIG_GENERATION = "memory-v2"


@dataclass(frozen=True, slots=True)
class MemoryRuntimeComposition:
    memory_orchestrator: MemoryOrchestrator
    memory_compaction_service: MemoryCompactionService | None
    background_service: MemoryBackgroundService | None
    embedding_provider: EmbeddingProvider | None
    embedding_generation: int | None
    v2_provider: MemoryV2ContextProvider
    legacy_provider: LegacyMemoryContext
    build_request: Callable[..., GroupMemoryContextRequest]


def _evidence_messages_from_rows(
    *,
    rows,
    users_by_id: dict[int, object],
    messages: MessageRepository,
    settings: AppSettings,
    bot_display_name: str,
) -> tuple[EvidenceMessage, ...]:
    return tuple(
        EvidenceMessage(
            source_msg_id=str(row.platform_msg_id),
            speaker=member_label_for_user(
                user_id=int(row.user_id),
                users_by_id=users_by_id,
                bot_user_id=settings.bot_qq,
                bot_display_name=bot_display_name,
            ),
            content=str(row.plain_text or ""),
            sent_at=row.timestamp,
            blocked=messages.is_qq_blocked_outbound(row),
            group_id=int(row.group_id) if row.group_id is not None else None,
            reply_to_msg_id=row.reply_to_msg_id,
            is_bot=int(row.user_id) == int(settings.bot_qq),
            user_id=int(row.user_id),
            mentioned_uins=_mentioned_uins(row.raw_json),
            delivery_state=(
                str(row.raw_json.get("delivery_state") or "").strip().casefold()
                if isinstance(row.raw_json, dict)
                else ""
            ),
        )
        for row in rows
    )


def _mentioned_uins(raw_json: object) -> tuple[str, ...]:
    if not isinstance(raw_json, dict):
        return ()
    segments = raw_json.get("message")
    if not isinstance(segments, list):
        return ()
    values: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("type") != "at":
            continue
        data = segment.get("data")
        if not isinstance(data, dict):
            continue
        for key in ("qq", "uin", "target"):
            value = str(data.get(key, "") or "").strip()
            if value and value.casefold() != "all":
                values.append(value)
    return tuple(dict.fromkeys(values))


class _DatabaseShadowEvaluator:
    def __init__(
        self,
        *,
        engine,
        settings: AppSettings,
        bot_display_name: str,
        v2_provider: MemoryV2ContextProvider,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.bot_display_name = bot_display_name
        self.v2_provider = v2_provider

    def load_request(
        self,
        *,
        group_id: int,
        message_id: int,
    ) -> GroupMemoryContextRequest:
        with session_scope(self.engine) as session:
            target = session.get(Message, int(message_id))
            if target is None or int(target.group_id or 0) != int(group_id):
                raise ValueError("shadow source message scope mismatch")
            rows = list(
                session.scalars(
                    select(Message)
                    .where(
                        Message.group_id == int(group_id),
                        Message.id <= int(message_id),
                    )
                    .order_by(Message.id.desc())
                    .limit(max(1, self.settings.memory_recent_snapshot_limit))
                )
            )
            rows.reverse()
            messages = MessageRepository(session)
            users_by_id = UserRepository(session).get_users_by_ids(
                [int(row.user_id) for row in rows]
            )
            recent = _evidence_messages_from_rows(
                rows=rows,
                users_by_id=users_by_id,
                messages=messages,
                settings=self.settings,
                bot_display_name=self.bot_display_name,
            )
            quoted = None
            if target.reply_to_msg_id:
                quoted_row = messages.get_by_platform_msg_id(target.reply_to_msg_id)
                if quoted_row is not None and int(quoted_row.group_id or 0) == int(group_id):
                    quoted_users = UserRepository(session).get_users_by_ids(
                        [int(quoted_row.user_id)]
                    )
                    quoted = _evidence_messages_from_rows(
                        rows=(quoted_row,),
                        users_by_id=quoted_users,
                        messages=messages,
                        settings=self.settings,
                        bot_display_name=self.bot_display_name,
                    )[0]

            target_query = str(target.plain_text or "")
            target_platform_msg_id = str(target.platform_msg_id)
            target_timestamp = target.timestamp
            target_user_id = int(target.user_id)
        return GroupMemoryContextRequest(
            group_id=int(group_id),
            query=target_query,
            recent_messages=recent,
            quoted_message=quoted,
            target_message_id=target_platform_msg_id,
            available_input=max(
                1,
                self.settings.llm_context_window_tokens
                - self.settings.llm_max_output_tokens
                - self.settings.llm_context_safety_margin_tokens
                - (
                    self.settings.llm_tool_context_reserve_tokens
                    if self.settings.llm_builtin_web_search
                    else 0
                ),
            ),
            now=target_timestamp,
            current_user_id=target_user_id,
        )

    def evaluate(
        self,
        request: BackgroundShadowJobRequest | None = None,
        *,
        group_id: int | None = None,
        message_id: int | None = None,
    ) -> ShadowEvaluation:
        if request is not None:
            group_id = request.group_id
            message_id = request.message_id
        if group_id is None or message_id is None:
            raise ValueError("group_id and message_id are required")
        started_at = perf_counter()
        trace = self.v2_provider.evaluate(
            self.load_request(group_id=int(group_id), message_id=int(message_id))
        )
        result = trace.result
        packed = result.packed_context
        route_counts = {}
        if isinstance(packed, PackedMemoryContext):
            route_counts = {
                "recent": len(packed.recent_messages),
                "evidence": len(packed.evidence_segments),
                "facts": len(packed.facts),
                "summaries": len(packed.summaries),
            }
        return ShadowEvaluation(
            source_message_ids=tuple(result.selected_source_msg_ids),
            candidate_scores=trace.candidate_scores,
            route_counts=route_counts,
            token_count=max(0, int(result.estimated_tokens)),
            latency_ms=max(0, int((perf_counter() - started_at) * 1000)),
            rewrite_used=bool(trace.resolved_query.rewrite_used),
            fallback_used=False,
        )


def build_memory_compaction_service(
    *,
    settings: AppSettings,
    engine,
    llm_client,
    background_service: MemoryBackgroundService | None = None,
) -> MemoryCompactionService | None:
    if not settings.memory_compaction_enabled and background_service is None:
        return None
    return MemoryCompactionService(
        engine=engine,
        llm_client=llm_client,
        batch_size=settings.memory_compaction_batch_size,
        max_facts=settings.memory_compaction_max_facts,
        retry_limit=settings.memory_compaction_retry_limit,
        backfill_windows=settings.memory_compaction_backfill_windows,
        excluded_user_ids={settings.bot_qq},
        background_service=background_service,
        shadow_enabled=settings.memory_orchestration_shadow_mode,
        legacy_enabled=background_service is None,
    )


def build_memory_runtime(
    *,
    settings: AppSettings,
    engine,
    llm_client,
    bot_display_name: str,
    raw_message_embedding_generation_override: int | None = None,
    evaluation_candidate_filter: Callable[..., tuple[object, ...]] | None = None,
    memory_enabled_group_ids: frozenset[int] | None = None,
    embedding_provider_factory=build_embedding_provider,
    ensure_vector_generation=ensure_retrieval_vector_generation,
    find_vector_generation=find_active_retrieval_vector_generation,
    llm_client_factory=LlmClient,
) -> MemoryRuntimeComposition:
    legacy = LegacyMemoryContext(
        engine=engine,
        settings=settings,
        bot_user_id=settings.bot_qq,
        bot_display_name=bot_display_name,
    )
    embedding_provider = embedding_provider_factory(
        provider=settings.memory_embedding_provider,
        device=settings.memory_embedding_device,
        model=settings.memory_embedding_model,
        dimensions=settings.memory_embedding_dimensions,
        cache_dir=settings.memory_embedding_cache_dir,
        local_files_only=settings.memory_embedding_local_files_only,
        version=settings.memory_embedding_version,
        base_url=settings.memory_embedding_base_url,
        api_key=settings.memory_embedding_api_key,
        timeout_seconds=settings.memory_embedding_timeout_seconds,
    )

    def load_memory_item_vectors(memory_ids):
        with session_scope(engine) as session:
            return MemoryRepository(session).load_memory_item_semantic_vectors(
                tuple(memory_ids),
                provider=embedding_provider.identity.provider,
                model=embedding_provider.identity.model,
                dimensions=embedding_provider.identity.dimensions,
                version=embedding_provider.identity.version,
            )

    semantic_fact_ranker = SemanticFactRanker(
        embedding_provider,
        vector_loader=load_memory_item_vectors,
    )
    legacy_embedding_generation = None
    raw_message_embedding_generation = None
    if settings.memory_orchestration_v2_enabled and embedding_provider.available:
        identity = embedding_provider.identity
        try:
            if settings.memory_raw_v3_enabled:
                raw_message_embedding_generation = (
                    int(raw_message_embedding_generation_override)
                    if raw_message_embedding_generation_override is not None
                    else find_vector_generation(
                        engine,
                        provider=identity.provider,
                        model=identity.model,
                        dimensions=identity.dimensions,
                        version=identity.version,
                        document_family="raw_message_v3",
                    )
                )
                if raw_message_embedding_generation is None:
                    logging.warning(
                        "memory_raw_v3_vector_generation_unavailable " "fallback=fts_only"
                    )
            else:
                legacy_embedding_generation = ensure_vector_generation(
                    engine,
                    provider=identity.provider,
                    model=identity.model,
                    dimensions=identity.dimensions,
                    version=identity.version,
                )
        except Exception as exc:
            logging.warning(
                "memory_vector_generation_unavailable error_type=%s",
                type(exc).__name__,
            )
    resolver = MemoryQueryResolver(
        _build_query_rewrite_provider(
            client_factory=llm_client_factory,
            settings=settings,
            llm_client=llm_client,
            engine=engine,
        ),
        rewrite_timeout_seconds=settings.memory_query_rewrite_timeout_seconds,
        mention_target_ids=(settings.bot_qq,),
    )
    history_classifier = MemoryQueryResolver()
    retriever = HybridMemoryRetriever(
        channels=build_memory_retrieval_channels(
            engine,
            embedding_provider=embedding_provider,
            vector_generation=(
                int(raw_message_embedding_generation_override)
                if raw_message_embedding_generation_override is not None
                else None
            ),
            raw_message_v3_only=settings.memory_raw_v3_enabled,
            legacy_v2_only=not settings.memory_raw_v3_enabled,
            layered_memory_enabled=settings.memory_layered_memory_enabled,
            excluded_speaker_ids=(settings.bot_qq,),
        ),
        candidate_limit=max(
            settings.memory_fts_candidate_limit,
            settings.memory_vector_candidate_limit,
            (
                (
                    settings.memory_adaptive_max_history_messages
                    if settings.memory_adaptive_context_enabled
                    else settings.memory_max_evidence_messages * 2
                )
                if settings.memory_raw_v3_enabled
                else 1
            ),
        ),
        final_limit=(
            (
                settings.memory_adaptive_max_history_messages
                if settings.memory_adaptive_context_enabled
                else settings.memory_max_evidence_messages
            )
            if settings.memory_raw_v3_enabled
            else settings.memory_final_episode_limit
        ),
        channel_timeout_seconds=settings.memory_retrieval_channel_timeout_seconds,
    )

    def load_episode(*, group_id: int, episode_id: int):
        with session_scope(engine) as session:
            rows = EpisodeRepository(session).list_episode_messages(
                episode_id=episode_id,
                group_id=group_id,
            )
            messages = MessageRepository(session)
            users_by_id = UserRepository(session).get_users_by_ids(
                [int(row.user_id) for row in rows]
            )
            return _evidence_messages_from_rows(
                rows=rows,
                users_by_id=users_by_id,
                messages=messages,
                settings=settings,
                bot_display_name=bot_display_name,
            )

    direct_reply_limit = 2
    direct_reply_scan_limit = 32

    def load_sources(*, group_id: int, source_msg_ids: tuple[str, ...]):
        with session_scope(engine) as session:
            messages = MessageRepository(session)
            rows_by_id = messages.get_group_messages_by_platform_msg_ids(
                group_id=group_id,
                platform_msg_ids=list(source_msg_ids),
            )
            direct_replies = messages.list_direct_group_replies(
                group_id=group_id,
                parent_platform_msg_ids=list(source_msg_ids),
                scan_limit_per_parent=direct_reply_scan_limit,
            )
            loaded_by_id = dict(rows_by_id)
            for row in direct_replies:
                loaded_by_id.setdefault(str(row.platform_msg_id), row)
            rows = tuple(
                row
                for row in loaded_by_id.values()
                if not messages.is_reserved_outbound(row)
                and not messages.is_delivery_uncertain_outbound(row)
            )
            users_by_id = UserRepository(session).get_users_by_ids(
                [int(row.user_id) for row in rows]
            )
            return tuple(
                _evidence_messages_from_rows(
                    rows=rows,
                    users_by_id=users_by_id,
                    messages=messages,
                    settings=settings,
                    bot_display_name=bot_display_name,
                )
            )

    def load_source_context(
        *,
        group_id: int,
        source_msg_ids: tuple[str, ...],
        limit_per_source: int,
        max_gap_seconds: int,
    ):
        with session_scope(engine) as session:
            messages = MessageRepository(session)
            rows = messages.list_preceding_group_message_context(
                group_id=group_id,
                anchor_platform_msg_ids=list(source_msg_ids),
                per_anchor_limit=limit_per_source,
                max_gap_seconds=max_gap_seconds,
                excluded_user_ids={int(settings.bot_qq)},
            )
            users_by_id = UserRepository(session).get_users_by_ids(
                [int(row.user_id) for row in rows]
            )
            return tuple(
                _evidence_messages_from_rows(
                    rows=rows,
                    users_by_id=users_by_id,
                    messages=messages,
                    settings=settings,
                    bot_display_name=bot_display_name,
                )
            )

    def load_thread_context(
        *,
        group_id: int,
        source_msg_ids: tuple[str, ...],
        limit_per_source: int,
        max_gap_seconds: int,
    ):
        with session_scope(engine) as session:
            messages = MessageRepository(session)
            rows = messages.list_bounded_same_episode_message_context(
                group_id=group_id,
                anchor_platform_msg_ids=list(source_msg_ids),
                per_anchor_limit=limit_per_source,
                max_gap_seconds=max_gap_seconds,
                excluded_user_ids={int(settings.bot_qq)},
            )
            users_by_id = UserRepository(session).get_users_by_ids(
                [int(row.user_id) for row in rows]
            )
            return tuple(
                _evidence_messages_from_rows(
                    rows=rows,
                    users_by_id=users_by_id,
                    messages=messages,
                    settings=settings,
                    bot_display_name=bot_display_name,
                )
            )

    def load_facts(*, group_id: int, resolved_query):
        if settings.memory_raw_v3_enabled and not settings.memory_layered_memory_enabled:
            return ()
        with session_scope(engine) as session:
            memories = MemoryRepository(session)
            rows = list(
                memories.search_group_memories_fts(
                    scope_id=str(group_id),
                    query=str(resolved_query.retrieval_query),
                    limit=settings.memory_final_episode_limit,
                    as_of=datetime.now().astimezone(),
                    subject_ids=resolved_query.subject_ids,
                )
            )
            subject_ids = resolved_query.subject_ids
            policy = fact_intent_policy(
                query=str(resolved_query.original_query),
                answer_mode=resolved_query.answer_mode,
            )
            allowed_fact_kinds = (
                frozenset(
                    tuple(getattr(resolved_query, "allowed_fact_kinds", ()) or ())
                    or policy.allowed_kinds
                )
                if subject_ids and resolved_query.answer_mode == "current_fact"
                else None
            )
            if allowed_fact_kinds is not None:
                rows = [row for row in rows if str(row.memory_kind or "") in allowed_fact_kinds]
            boosted_fact_ids: set[int] = set()
            preferred_fact_ids: set[int] = set()
            semantic_scores_by_id: dict[int, float] = {}
            selection_priority_by_id: dict[int, int] = {}
            temporal_current_fact_ids: set[int] | None = None
            recency_boost = temporal_recency_required(query=str(resolved_query.original_query))
            inferred_preferred_kinds = preferred_kinds_for_query(
                query=str(resolved_query.original_query),
                answer_mode=resolved_query.answer_mode,
            )
            composite_portrait = (
                inferred_preferred_kinds == PERSON_PORTRAIT_KINDS
                and is_composite_portrait_query(str(resolved_query.original_query))
            )
            preferred_kinds: tuple[str, ...] = (
                PERSON_PORTRAIT_KINDS
                if composite_portrait
                else tuple(resolved_query.preferred_fact_kinds) or inferred_preferred_kinds
            )
            if subject_ids:
                seen_ids = {row.id for row in rows}
                query_features = memory_query_features(
                    query=str(resolved_query.retrieval_query),
                    entities=resolved_query.entities,
                    topic_terms=resolved_query.topic_terms,
                    intent_query=str(resolved_query.original_query),
                )
                member_aliases: list[str] = []
                for member in (
                    UserRepository(session)
                    .get_users_by_ids([int(subject_id) for subject_id in subject_ids])
                    .values()
                ):
                    member_aliases.extend(
                        str(value)
                        for value in (member.nickname, member.group_card)
                        if str(value or "").strip()
                    )
                query_features = filter_member_query_features(
                    query_features,
                    aliases=member_aliases,
                )
                for subject_id in subject_ids:
                    candidates = memories.list_group_memories_for_subject(
                        scope_id=str(group_id),
                        subject_id=subject_id,
                        limit=max(
                            200,
                            settings.memory_member_fact_supplement_limit,
                        ),
                    )
                    if allowed_fact_kinds is not None:
                        candidates = [
                            row
                            for row in candidates
                            if str(row.memory_kind or "") in allowed_fact_kinds
                        ]
                    semantic_scores: dict[int, float] = {}
                    if settings.memory_fact_semantic_ranking_enabled:
                        # Score every member fact, not just the most recent
                        # slice: older-but-relevant "watching plan" facts can
                        # fall outside a small recency-ordered window and then
                        # lose semantic ranking entirely.
                        semantic_candidates = candidates
                        semantic_scores = semantic_fact_ranker.score(
                            str(resolved_query.retrieval_query),
                            semantic_candidates,
                        )
                        semantic_scores_by_id.update(semantic_scores)
                    ranked_rows = rank_member_facts(
                        candidates,
                        query_features=query_features,
                        limit=(
                            len(candidates)
                            if composite_portrait
                            else settings.memory_member_fact_supplement_limit
                        ),
                        preferred_kinds=preferred_kinds,
                        semantic_scores=semantic_scores,
                        recency_boost=recency_boost,
                    )
                    if composite_portrait:
                        ranked_rows = select_diverse_portrait_facts(
                            ranked_rows,
                            limit=settings.memory_member_fact_supplement_limit,
                        )
                    if recency_boost and resolved_query.answer_mode == "current_fact":
                        temporal_match_features = query_features
                        if resolved_query.topic_terms:
                            temporal_match_features = filter_member_query_features(
                                memory_query_features(
                                    query="",
                                    topic_terms=resolved_query.topic_terms,
                                    intent_query=str(resolved_query.original_query),
                                ),
                                aliases=member_aliases,
                            )
                        matching_ids = matching_member_fact_ids(
                            ranked_rows,
                            query_features=temporal_match_features,
                        )
                        viewing_event = recent_viewing_event_fallback(
                            candidates,
                            query=str(resolved_query.original_query),
                            now=datetime.now().astimezone(),
                        )
                        matching_current = [
                            row
                            for row in ranked_rows
                            if row.id in matching_ids and row.memory_kind == "current"
                        ]
                        if prefer_recent_viewing_event(
                            viewing_event,
                            matched_current=matching_current,
                        ):
                            ranked_rows = [
                                viewing_event,
                                *[row for row in ranked_rows if row.id != viewing_event.id],
                            ]
                            matching_ids = viewing_fallback_match_ids(
                                matching_ids,
                                viewing_event=viewing_event,
                                coverage=str(
                                    getattr(resolved_query, "fact_coverage", "") or policy.coverage
                                ),
                            )
                        ranked_rows = select_temporal_current_facts(
                            ranked_rows,
                            matching_fact_ids=matching_ids,
                            topic_specific=bool(resolved_query.topic_terms),
                            coverage=str(
                                getattr(resolved_query, "fact_coverage", "") or policy.coverage
                            ),
                        )
                        if temporal_current_fact_ids is None:
                            temporal_current_fact_ids = set()
                        temporal_current_fact_ids.update(int(row.id) for row in ranked_rows)
                    if recency_boost:
                        selection_priority_by_id.update(
                            {
                                row.id: len(ranked_rows) - index
                                for index, row in enumerate(ranked_rows)
                            }
                        )
                    for row in ranked_rows:
                        if row.id in seen_ids:
                            continue
                        rows.append(row)
                        seen_ids.add(row.id)
                boosted_fact_ids = matching_member_fact_ids(
                    rows,
                    query_features=query_features,
                )
                if preferred_kinds:
                    preferred_fact_ids.update(
                        row.id for row in rows if row.memory_kind in preferred_kinds
                    )
            elif resolved_query.subject_role == "group" and preferred_kinds:
                seen_ids = {row.id for row in rows}
                for row in memories.list_group_memories_for_subject(
                    scope_id=str(group_id),
                    subject_id="group",
                    limit=max(200, settings.memory_member_fact_supplement_limit),
                ):
                    if row.memory_kind not in preferred_kinds:
                        continue
                    preferred_fact_ids.add(row.id)
                    if row.id not in seen_ids:
                        rows.append(row)
                        seen_ids.add(row.id)
            return tuple(
                MemoryFact(
                    text=str(row.content),
                    source_msg_ids=tuple(
                        dict.fromkeys(
                            [
                                *[str(item) for item in (row.source_msg_ids or []) if str(item)],
                                *([str(row.source_msg_id)] if row.source_msg_id else []),
                            ]
                        )
                    ),
                    score=float(row.confidence or 0.0)
                    + (1.0 if row.id in boosted_fact_ids else 0.0)
                    + (0.5 if row.id in preferred_fact_ids else 0.0)
                    + (0.8 * semantic_scores_by_id.get(row.id, 0.0)),
                    selection_priority=selection_priority_by_id.get(row.id, 0),
                    valid_until=row.valid_until,
                    group_id=group_id,
                    memory_kind=str(row.memory_kind or ""),
                    observed_at=row.last_seen_at or row.valid_from,
                    memory_id=int(row.id),
                )
                for row in rows
                if row.source_msg_id or row.source_msg_ids
                if temporal_current_fact_ids is None or int(row.id) in temporal_current_fact_ids
            )

    def load_summaries(*, group_id: int, resolved_query):
        if (
            settings.memory_raw_v3_enabled and not settings.memory_layered_memory_enabled
        ) or not resolved_query.needs_history:
            return ()
        with session_scope(engine) as session:
            summary_kwargs = {}
            if settings.memory_raw_v3_enabled:
                summary_kwargs["summary_levels"] = (
                    "episode",
                    "semantic_window",
                    "semantic_daily",
                    # Keep reading summaries written by the pre-layered
                    # runtime. The production snapshot still contains these
                    # legacy levels and they remain valid derived evidence.
                    "window",
                    "daily",
                )
            time_range = resolved_query.time_range

            def _relevant(row) -> bool:
                if time_range is None:
                    # No deterministic time range: summaries stay a
                    # supplement for any history-intent question.
                    return True
                start = stored_as_utc(row.start_at) if row.start_at is not None else None
                end = stored_as_utc(row.end_at) if row.end_at is not None else None
                if start is None and end is None:
                    return False
                if end is not None and time_range.start is not None and end <= time_range.start:
                    return False
                if start is not None and time_range.end is not None and start >= time_range.end:
                    return False
                return True

            rows = SummaryRepository(session).list_group_summaries(
                scope_id=str(group_id),
                limit=settings.context_summary_limit * 3,
                require_source_ids=True,
                start_at=(time_range.start if time_range is not None else None),
                end_at=(time_range.end if time_range is not None else None),
                **summary_kwargs,
            )
            relevant_rows = [row for row in rows if _relevant(row)][
                : settings.context_summary_limit
            ]
            return tuple(
                MemorySummary(
                    text=str(row.content),
                    source_msg_ids=tuple(
                        dict.fromkeys(
                            source_id
                            for source_id in (
                                row.source_start_msg_id,
                                row.source_end_msg_id,
                            )
                            if source_id
                        )
                    ),
                    relevant=True,
                    group_id=group_id,
                    summary_id=int(row.id),
                )
                for row in relevant_rows
                if row.source_start_msg_id or row.source_end_msg_id
            )

    def load_members(group_id: int) -> tuple[GroupMemberIdentity, ...]:
        with session_scope(engine) as session:
            rows = MessageRepository(session).list_recent_group_member_messages(
                group_id=None,
                limit=None,
            )
            members = group_member_identities_from_messages(
                rows,
                target_group_id=int(group_id),
            )
        return members

    def validate_source_scope(group_id: int, source_msg_ids: tuple[str, ...]) -> bool:
        expected_ids = {
            str(source_id).strip() for source_id in source_msg_ids if str(source_id).strip()
        }
        if not expected_ids:
            return True
        with session_scope(engine) as session:
            messages = MessageRepository(session)
            scoped = messages.get_group_messages_by_platform_msg_ids(
                group_id=int(group_id),
                platform_msg_ids=list(expected_ids),
            )
        return set(scoped) == expected_ids

    expander = MemoryEvidenceExpander(
        episode_loader=load_episode,
        source_loader=load_sources,
        context_loader=load_source_context,
        thread_context_loader=load_thread_context,
        normal_segment_limit=(
            (
                settings.memory_adaptive_max_history_messages
                if settings.memory_adaptive_context_enabled
                else settings.memory_max_evidence_messages
            )
            if settings.memory_raw_v3_enabled
            else min(4, settings.memory_final_episode_limit)
        ),
        detail_segment_limit=(
            (
                settings.memory_adaptive_max_history_messages
                if settings.memory_adaptive_context_enabled
                else settings.memory_max_evidence_messages
            )
            if settings.memory_raw_v3_enabled
            else min(6, settings.memory_final_episode_limit)
        ),
    )
    packer = MemoryContextPacker(
        normal_budget=settings.memory_normal_context_budget_tokens,
        detail_budget=settings.memory_detail_context_budget_tokens,
        recent_budget=settings.memory_recent_context_budget_tokens,
        history_budget=settings.memory_history_context_budget_tokens,
        context_char_budget=settings.memory_effective_context_budget_chars,
        max_recent_messages=settings.context_recent_limit,
        max_history_messages=settings.memory_max_evidence_messages,
        adaptive_enabled=settings.memory_adaptive_context_enabled,
        recent_protected_min_tokens=settings.memory_recent_protected_min_tokens,
        history_protected_min_tokens=settings.memory_history_protected_min_tokens,
        recent_protected_min_messages=settings.memory_recent_protected_min_messages,
        history_protected_min_messages=settings.memory_history_protected_min_messages,
        adaptive_max_recent_messages=settings.memory_adaptive_max_recent_messages,
        adaptive_max_history_messages=settings.memory_adaptive_max_history_messages,
    )
    v2_provider = MemoryV2ContextProvider(
        resolver=resolver,
        retriever=retriever,
        expander=expander,
        packer=packer,
        source_scope_validator=validate_source_scope,
        fact_loader=load_facts,
        summary_loader=load_summaries,
        member_loader=load_members,
        candidate_filter=evaluation_candidate_filter,
        max_direct_replies_per_source=direct_reply_limit,
        excluded_member_ids={settings.bot_qq},
        historical_no_hit_omit_recent=settings.memory_raw_v3_enabled,
        observability_route=("raw_v3" if settings.memory_raw_v3_enabled else "legacy_v2"),
        adaptive_context_enabled=settings.memory_adaptive_context_enabled,
        compact_candidate_limit=settings.memory_max_evidence_messages,
        recent_intent_candidate_limit=settings.memory_recent_intent_candidate_limit,
    )

    shadow_evaluator = _DatabaseShadowEvaluator(
        engine=engine,
        settings=settings,
        bot_display_name=bot_display_name,
        v2_provider=v2_provider,
    )
    background_service = None
    if settings.memory_orchestration_v2_enabled:

        def load_active_correction_targets(
            group_id: int,
            subject_ids: tuple[str, ...],
        ) -> tuple[dict[str, str], ...]:
            with session_scope(engine) as session:
                repository = MemoryRepository(session)
                rows = tuple(
                    row
                    for subject_id in dict.fromkeys(subject_ids)
                    for row in repository.list_current_group_memories(
                        scope_id=str(group_id),
                        subject_id=subject_id,
                        limit=500,
                    )
                )
                return tuple(
                    {
                        "target_canonical_key": str(row.canonical_key),
                        "memory_kind": str(row.memory_kind),
                        "subject_id": str(row.subject_id),
                        "predicate": str(row.predicate or ""),
                        "object_text": str(row.object_text or ""),
                    }
                    for row in rows
                    if str(row.memory_kind) in {"profile", "preference"}
                    and bool(str(row.canonical_key or "").strip())
                )

        identity = embedding_provider.identity
        background_service = MemoryBackgroundService(
            store=SqlAlchemyMemoryBackgroundStore(
                engine,
                max_attempts=settings.memory_compaction_retry_limit,
                embedding_provider=identity.provider,
                embedding_model=identity.model,
                embedding_version=identity.version,
                embedding_dimensions=identity.dimensions,
                embedding_generation=legacy_embedding_generation,
                raw_message_embedding_enabled=settings.memory_raw_v3_enabled,
                raw_message_embedding_generation=raw_message_embedding_generation,
                memory_enabled_group_ids=memory_enabled_group_ids,
            ),
            deriver=CompactionEpisodeDeriver(
                llm_client=build_memory_compaction_client(
                    client_factory=llm_client_factory,
                    settings=settings,
                    llm_client=llm_client,
                    engine=engine,
                ),
                max_facts=settings.memory_compaction_max_facts,
                correction_target_loader=load_active_correction_targets,
            ),
            worker_id="group-memory-v2",
            segmentation_generation=MEMORY_SEGMENTATION_GENERATION,
            compaction_generation=MEMORY_COMPACTION_GENERATION,
            idle_minutes=settings.memory_episode_idle_minutes,
            max_messages=settings.memory_episode_max_messages,
            max_tokens=settings.memory_episode_max_tokens,
            chunk_max_tokens=settings.memory_chunk_max_tokens,
            chunk_overlap_messages=settings.memory_chunk_overlap_messages,
            chunk_max_messages=settings.memory_chunk_max_messages,
            bot_user_id=settings.bot_qq,
            embedder=embedding_provider,
            shadow_evaluator=shadow_evaluator,
            memory_enabled_group_ids=memory_enabled_group_ids,
            topic_judge_client=build_episode_topic_judge_client(
                client_factory=llm_client_factory,
                settings=settings,
                llm_client=llm_client,
                engine=engine,
            ),
            topic_judge_enabled=settings.memory_episode_topic_judge_enabled,
            topic_judge_context_messages=settings.memory_episode_topic_judge_context_messages,
            topic_judge_start_messages=settings.memory_episode_topic_judge_start_messages,
            topic_judge_interval=settings.memory_episode_topic_judge_interval,
            post_segment_client=build_episode_post_segment_client(
                client_factory=llm_client_factory,
                settings=settings,
                llm_client=llm_client,
                engine=engine,
            ),
            post_segment_enabled=settings.memory_episode_post_segment_enabled,
            post_segment_min_messages=settings.memory_episode_post_segment_min_messages,
            current_ttl_hours=settings.memory_current_default_ttl_hours,
        )

    def enqueue_shadow_sync(request: ShadowJobRequest) -> None:
        if background_service is None:
            return
        with session_scope(engine) as session:
            message = MessageRepository(session).get_by_platform_msg_id(request.current_msg_id)
            if message is None or int(message.group_id or 0) != int(request.group_id):
                raise ValueError("shadow source message scope mismatch")
            message_id = int(message.id)
        background_service.enqueue_shadow(
            BackgroundShadowJobRequest(
                group_id=int(request.group_id),
                message_id=message_id,
                config_generation=request.config_version or MEMORY_CONFIG_GENERATION,
                index_generation=(
                    request.index_generation
                    or embedding_provider.identity.version
                    or embedding_provider.identity.model
                    or embedding_provider.identity.provider
                ),
            )
        )

    def enqueue_shadow(request: ShadowJobRequest) -> None:
        if compaction_service is None:
            return
        compaction_service.submit_shadow_enqueue(lambda: enqueue_shadow_sync(request))

    compaction_service = build_memory_compaction_service(
        settings=settings,
        engine=engine,
        llm_client=build_memory_compaction_client(
            client_factory=llm_client_factory,
            settings=settings,
            llm_client=llm_client,
            engine=engine,
        ),
        background_service=background_service,
    )
    orchestrator = MemoryOrchestrator(
        v2_enabled=settings.memory_orchestration_v2_enabled,
        shadow_mode=settings.memory_orchestration_shadow_mode,
        v2_provider=v2_provider,
        legacy_provider=legacy.build_context,
        recent_provider=legacy.build_recent_context,
        shadow_enqueue=enqueue_shadow,
        strict_scoped_fallback=settings.memory_raw_v3_enabled,
        history_request_predicate=lambda request: history_classifier.resolve(
            request.query,
            recent_messages=request.recent_messages,
            quoted_message=request.quoted_message,
            now=request.now,
            group_id=request.group_id,
            requester_id=getattr(request, "current_user_id", None),
        ).needs_history,
    )
    identity = embedding_provider.identity
    logging.info(
        "memory_runtime route=%s raw_enabled=%s embedding_provider=%s "
        "embedding_model=%s embedding_device=%s embedding_generation=%s "
        "adaptive_enabled=%s layered_enabled=%s tools_enabled=%s",
        "raw_v3" if settings.memory_raw_v3_enabled else "legacy_v2",
        settings.memory_raw_v3_enabled,
        identity.provider,
        identity.model,
        settings.memory_embedding_device,
        (
            raw_message_embedding_generation
            if settings.memory_raw_v3_enabled
            else legacy_embedding_generation
        ),
        settings.memory_adaptive_context_enabled,
        settings.memory_layered_memory_enabled,
        settings.memory_memory_tools_enabled,
    )
    return MemoryRuntimeComposition(
        memory_orchestrator=orchestrator,
        memory_compaction_service=compaction_service,
        background_service=background_service,
        embedding_provider=embedding_provider,
        embedding_generation=(
            raw_message_embedding_generation
            if settings.memory_raw_v3_enabled
            else legacy_embedding_generation
        ),
        v2_provider=v2_provider,
        legacy_provider=legacy,
        build_request=shadow_evaluator.load_request,
    )
