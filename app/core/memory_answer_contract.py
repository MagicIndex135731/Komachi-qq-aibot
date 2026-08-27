"""Decision-envelope answer contract shared by production and evaluation.

The contract follows the principle that the upstream model owns understanding,
selection, and answer generation, while local code only enforces structural
safety: every referenced evidence/source id must come from the current
accessible packet, every claim must carry at least one reference, expansion is
allowed once, and local code never rewrites the answer text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Sequence


DECISION_ENVELOPE_DECISIONS = ("answer", "abstain", "clarify", "expand")
DECISION_ENVELOPE_LAYERS = ("raw", "facts", "summary", "recent")
ENVELOPE_LINE_PREFIX = "SHADOW_ENVELOPE:"
_MAX_CLAIMS = 24
_MAX_REFERENCES_PER_CLAIM = 16
_MAX_EXPANSION_FACETS = 8


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpansionRequest:
    facets: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionEnvelope:
    decision: str
    answer: str
    claims: tuple[Claim, ...] = ()
    expansion_request: ExpansionRequest | None = None


def parse_decision_envelope(payload: Any) -> DecisionEnvelope:
    """Parse and structurally validate one decision envelope."""

    if not isinstance(payload, dict):
        raise ValueError("envelope is not an object")
    decision = str(payload.get("decision") or "")
    if decision not in DECISION_ENVELOPE_DECISIONS:
        raise ValueError(f"invalid decision: {decision!r}")
    answer = str(payload.get("answer") or "")
    if not answer.strip():
        raise ValueError("envelope answer is empty")
    raw_claims = payload.get("claims") or []
    if not isinstance(raw_claims, list):
        raise ValueError("claims must be a list")
    if len(raw_claims) > _MAX_CLAIMS:
        raise ValueError(f"claims exceed limit {_MAX_CLAIMS}")
    claims: list[Claim] = []
    for claim in raw_claims:
        if not isinstance(claim, dict):
            raise ValueError("claim is not an object")
        text = str(claim.get("text") or "")
        if not text.strip():
            raise ValueError("claim text is empty")
        evidence_ids = tuple(_string_list(claim.get("evidence_ids"), "evidence_ids"))
        source_ids = tuple(_string_list(claim.get("source_ids"), "source_ids"))
        if len(evidence_ids) > _MAX_REFERENCES_PER_CLAIM:
            raise ValueError("claim evidence_ids exceed limit")
        if len(source_ids) > _MAX_REFERENCES_PER_CLAIM:
            raise ValueError("claim source_ids exceed limit")
        claims.append(Claim(text=text.strip(), evidence_ids=evidence_ids, source_ids=source_ids))
    expansion: ExpansionRequest | None = None
    raw_expansion = payload.get("expansion_request")
    if raw_expansion is not None:
        if not isinstance(raw_expansion, dict):
            raise ValueError("expansion_request must be an object or null")
        facets = tuple(_string_list(raw_expansion.get("facets"), "facets"))
        layers = tuple(_string_list(raw_expansion.get("layers"), "layers"))
        if len(facets) > _MAX_EXPANSION_FACETS:
            raise ValueError("expansion_request.facets exceed limit")
        if any(layer not in DECISION_ENVELOPE_LAYERS for layer in layers):
            raise ValueError("expansion_request.layers contains an unknown layer")
        expansion = ExpansionRequest(facets=facets, layers=layers)
    return DecisionEnvelope(
        decision=decision,
        answer=answer.strip(),
        claims=tuple(claims),
        expansion_request=expansion,
    )


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} contains a non-string item")
        result.append(item.strip())
    return result


def extract_answer_envelope(
    value: str,
) -> tuple[str, DecisionEnvelope | None, str | None]:
    """Best-effort extraction of the envelope from a model response.

    The envelope is expected as a top-level ``decision_envelope`` field inside
    the answer JSON object; legacy line-prefixed output is also tolerated.
    Returns the cleaned text, the parsed envelope, and the first parse error.
    """

    if not value:
        return value, None, None
    kept: list[str] = []
    envelope: DecisionEnvelope | None = None
    error: str | None = None
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped.startswith(ENVELOPE_LINE_PREFIX):
            kept.append(line)
            continue
        payload = stripped[len(ENVELOPE_LINE_PREFIX) :].strip()
        try:
            parsed = parse_decision_envelope(json.loads(payload))
            if envelope is None:
                envelope = parsed
        except (ValueError, json.JSONDecodeError) as exc:
            if error is None:
                error = f"{type(exc).__name__}: {exc}"
    clean = "\n".join(kept)
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "decision_envelope" in payload:
        try:
            envelope = parse_decision_envelope(payload["decision_envelope"])
            error = None
        except ValueError as exc:
            if error is None:
                error = f"ValueError: {exc}"
    return clean, envelope, error


def validate_envelope_references(
    envelope: DecisionEnvelope,
    allowed_source_ids: Iterable[str],
    *,
    require_claims: bool = True,
) -> tuple[bool, list[str]]:
    """Mechanical check: claims reference only ids from the accessible packet."""

    allowed = set(str(value) for value in allowed_source_ids if str(value).strip())
    failures: list[str] = []
    if envelope.decision == "abstain":
        if envelope.claims:
            failures.append("abstain envelope must have no claims")
        return not failures, failures
    if not envelope.claims and require_claims:
        failures.append("non-abstain envelope must declare at least one claim")
        return False, failures
    for index, claim in enumerate(envelope.claims):
        referenced = {*claim.evidence_ids, *claim.source_ids}
        if not referenced:
            failures.append(f"claim[{index}] has no evidence/source ids")
            continue
        invalid = sorted(value for value in referenced if value not in allowed)
        if invalid:
            failures.append(
                f"claim[{index}] references ids outside the packet: {invalid}"
            )
    return not failures, failures


def append_envelope_contract(
    prompt_lines: Sequence[str],
    allowed_source_ids: Iterable[str],
    *,
    previous_failure: str | None = None,
    production: bool = False,
) -> list[str]:
    """Append the envelope output contract to a base prompt."""

    allowed = sorted({str(value) for value in allowed_source_ids if str(value).strip()})
    if production:
        schema = (
            "Return exactly one JSON object with fields answer, cited_source_message_ids, "
            "abstained, decision_envelope. answer must be your normal concise reply for this "
            "conversation, whether it is ordinary chat, help, general knowledge, or a memory "
            "question; never add phrases like 'memory evidence is insufficient' to non-memory "
            "questions. cited_source_message_ids may only copy ids exactly from the allowed "
            "list below; use [] when the reply is not based on retrieved memory. abstained is "
            "a memory-specific flag: true only when the question asks about remembered facts "
            "and the retrieved evidence cannot support an answer; for ordinary conversation "
            "always use false. decision_envelope must be one JSON object with fields decision "
            "(one of answer|abstain|clarify|expand), claims (a list of objects each with text, "
            "evidence_ids, source_ids), answer (your final reply text), and expansion_request "
            "(an object with facets and layers lists, or null). For memory-based assertions, "
            "every clause must be represented by one claim whose evidence_ids/source_ids come "
            "from the allowed list; a non-memory reply may use an empty claims list. Only "
            "request expansion when the packet is missing a needed attribute, time range, or "
            "evidence layer."
        )
    else:
        schema = (
            "Return exactly one JSON object with fields answer, cited_source_message_ids, "
            "abstained, decision_envelope. answer must be the concise reply you would send. "
            "cited_source_message_ids may only copy ids exactly from the allowed list below. "
            "abstained must be true only when the retrieved evidence cannot support an answer; "
            "when abstaining, answer must state that memory evidence is insufficient and "
            "cited_source_message_ids must be []. decision_envelope must be one JSON object "
            "with fields decision (one of answer|abstain|clarify|expand), claims (a list of "
            "objects each with text, evidence_ids, source_ids), answer (your final reply text), "
            "and expansion_request (an object with facets and layers lists, or null). Every "
            "substantive factual clause in answer must be represented by one claim whose "
            "evidence_ids/source_ids come from the allowed list; leaving a clause unclaimed or "
            "referencing an id outside the allowed list is a failure. Only request expansion "
            "when the allowed packet is missing a needed attribute, time range, or evidence layer."
        )
    prompt = [*prompt_lines, schema, f"Allowed citation IDs JSON list: {json.dumps(allowed, ensure_ascii=False)}"]
    if previous_failure:
        prompt.append(
            "The previous response failed structural validation: "
            + previous_failure
            + " Regenerate the complete answer and decision_envelope from scratch; do not "
            "preserve or repair the previous text field by field."
        )
    return prompt


def envelope_json(envelope: DecisionEnvelope) -> dict[str, Any]:
    return {
        "decision": envelope.decision,
        "answer": envelope.answer,
        "claims": [
            {
                "text": claim.text,
                "evidence_ids": list(claim.evidence_ids),
                "source_ids": list(claim.source_ids),
            }
            for claim in envelope.claims
        ],
        "expansion_request": (
            {
                "facets": list(envelope.expansion_request.facets),
                "layers": list(envelope.expansion_request.layers),
            }
            if envelope.expansion_request is not None
            else None
        ),
    }
