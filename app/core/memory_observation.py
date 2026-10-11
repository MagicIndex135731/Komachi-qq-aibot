"""Small, read-only, source-checked memory hints before agent participation."""

from __future__ import annotations

from app.core.memory_contracts import (
    MemoryObservationRequest,
    MemoryObservationResult,
    ObservationEvidence,
    ObservationFact,
)
from app.core.memory_source_guard import is_usable_memory_source, memory_provenance_ids
from app.core.memory_engine import history_search_terms
from app.core.persona_style_retrieval import lexical_units
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import (
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
)


def observation_lexical_query(topic: str) -> str:
    """Reuse existing Chinese n-grams without letting a long prefix crowd FTS."""
    normalized = str(topic or "").casefold()
    units = lexical_units(normalized)
    candidates = [
        term for term in history_search_terms(normalized, limit=1200) if term in units
    ]
    candidates.sort(
        key=lambda term: (normalized.count(term), normalized.rfind(term)),
        reverse=True,
    )
    return " ".join(candidates[:12])


class MemoryObservationReader:
    """Retrieve a bounded lexical packet without a requester or an LLM call.

    This packet helps decide whether to join. Detailed historical answers
    continue using the existing query resolver and evidence pipeline.
    """

    def __init__(self, *, engine) -> None:
        self._engine = engine

    def read(self, request: MemoryObservationRequest) -> MemoryObservationResult:
        evidence: list[ObservationEvidence] = []
        facts: list[ObservationFact] = []
        batch_ids = frozenset(request.source_message_ids)
        lexical_query = observation_lexical_query(request.topic_query)
        with session_scope(self._engine) as session:
            messages = MessageRepository(session)
            sources = messages.get_group_messages_by_platform_msg_ids(
                group_id=request.group_id, platform_msg_ids=list(batch_ids)
            )
            if set(sources) != batch_ids or any(
                not is_usable_memory_source(row, group_id=request.group_id)
                for row in sources.values()
            ):
                raise ValueError("observation_source_out_of_scope")
            if not lexical_query:
                return MemoryObservationResult(
                    group_id=request.group_id,
                    context_char_budget=request.context_char_budget,
                )
            documents = []
            # Existing raw fallback requires two short-unit hits for a
            # multi-term query. A bounded second query for the strongest unit
            # preserves short topic recall when surrounding batch words differ.
            queries = dict.fromkeys((lexical_query, lexical_query.split()[0]))
            repository = RetrievalDocumentRepository(session)
            for query in queries:
                documents.extend(
                    repository.search_group_documents_fts(
                        group_id=request.group_id,
                        query=query,
                        limit=min(40, request.max_results * 3),
                        document_kinds=("raw_message_v3",),
                        speaker_ids=tuple(
                            str(item) for item in request.focus_subject_ids
                        )
                        or None,
                        end_at=request.now,
                    )
                )
            seen = set(batch_ids)
            for document in documents:
                if str(document.source_table) != "messages":
                    continue
                try:
                    row = session.get(Message, int(document.source_id))
                except (TypeError, ValueError):
                    continue
                if (
                    not is_usable_memory_source(row, group_id=request.group_id)
                    or row.platform_msg_id in seen
                ):
                    continue
                if (
                    request.focus_subject_ids
                    and row.user_id not in request.focus_subject_ids
                ):
                    continue
                seen.add(row.platform_msg_id)
                evidence.append(
                    ObservationEvidence(
                        row.platform_msg_id,
                        int(row.user_id),
                        str(row.plain_text or "")[:500],
                    )
                )
                if len(evidence) >= request.max_results:
                    break
            candidates = MemoryRepository(session).search_group_memories_fts(
                scope_id=str(request.group_id),
                query=lexical_query,
                limit=request.max_results,
                as_of=request.now,
                subject_ids=tuple(str(item) for item in request.focus_subject_ids)
                or None,
            )
            for memory in candidates:
                source_ids = memory_provenance_ids(memory)
                if not source_ids:
                    continue
                provenance = messages.get_group_messages_by_platform_msg_ids(
                    group_id=request.group_id, platform_msg_ids=list(source_ids)
                )
                if set(provenance) != set(source_ids) or any(
                    not is_usable_memory_source(row, group_id=request.group_id)
                    for row in provenance.values()
                ):
                    continue
                facts.append(
                    ObservationFact(
                        int(memory.id),
                        str(memory.subject_id),
                        str(memory.content or "")[:500],
                        source_ids,
                    )
                )
        return MemoryObservationResult(
            group_id=request.group_id,
            evidence=tuple(evidence),
            facts=tuple(facts),
            source_message_ids=tuple(
                dict.fromkeys(
                    [
                        *(item.source_msg_id for item in evidence),
                        *(source for item in facts for source in item.source_msg_ids),
                    ]
                )
            ),
            context_char_budget=request.context_char_budget,
        )
