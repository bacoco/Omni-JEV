"""Pure postprocessing of supplied logits, not neural inference or calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from collections.abc import Sequence

from .contracts import (
    ChoiceAnswer, ChoiceQuestion, Confidence, DecisionRequest, DecisionResponse,
    LevelProbability, NoulAnswer, NoulQuestion, OptionProbability, Provenance,
    Question, QuestionResult, ScoreAnswer, entropy_concentration, validate_response,
)


def masked_softmax(logits: Sequence[float], mask: Sequence[bool]) -> tuple[float, ...]:
    """Normalize one question only. Padding may contain NaN/-inf and is ignored."""
    if not logits or len(logits) != len(mask):
        raise ValueError("logits/mask must be nonempty and have identical lengths")
    if any(type(valid) is not bool for valid in mask) or not any(mask):
        raise ValueError("mask must contain booleans and at least one valid option")
    valid_logits = [value for value, valid in zip(logits, mask) if valid]
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in valid_logits):
        raise ValueError("valid logits must be finite real numbers, not booleans")
    peak = max(valid_logits)
    weights = tuple(math.exp(float(value) - peak) if valid else 0.0 for value, valid in zip(logits, mask))
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


@dataclass(frozen=True)
class LogitRow:
    question_id: str
    candidate_ids: tuple[str | None, ...]
    logits: tuple[float, ...]
    metadata: Provenance
    valid_mask: tuple[bool, ...] | None = None


def decode_question(question: Question, row: LogitRow, *, include_confidence: bool = False) -> QuestionResult:
    """Map named candidates to one typed answer; no implicit boolean threshold."""
    if row.question_id != question.id or len(row.candidate_ids) != len(row.logits):
        raise ValueError("question ID and candidate/logit lengths must match")
    mask = row.valid_mask if row.valid_mask is not None else tuple(True for _ in row.logits)
    probabilities = masked_softmax(row.logits, mask)
    if any((candidate is None) == valid for candidate, valid in zip(row.candidate_ids, mask)):
        raise ValueError("only padding must have a null candidate ID")
    ids = tuple(candidate for candidate, valid in zip(row.candidate_ids, mask) if valid)
    if isinstance(question, NoulQuestion):
        expected = ("no", "yes")
    elif isinstance(question, ChoiceQuestion):
        expected = tuple(o.id for o in question.options)
    else:
        expected = tuple(level.id for level in question.levels)
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("valid candidate IDs must exactly match the question")
    by_id = {candidate: p for candidate, p, valid in zip(row.candidate_ids, probabilities, mask) if valid}
    confidence = None
    if include_confidence and not isinstance(question, NoulQuestion):
        confidence = Confidence(value=entropy_concentration(tuple(by_id[c] for c in expected)))
    if isinstance(question, NoulQuestion):
        answer = NoulAnswer(noul=by_id["yes"])
    elif isinstance(question, ChoiceQuestion):
        items = tuple(OptionProbability(id=c, probability=by_id[c]) for c in expected)
        winner = min(items, key=lambda item: (-item.probability, item.id)).id
        answer = ChoiceAnswer(choice=winner, probabilities=items, confidence=confidence)
    else:
        levels = tuple(LevelProbability(id=c, index=i, probability=by_id[c]) for i, c in enumerate(expected))
        answer = ScoreAnswer(score=math.fsum(p.index * p.probability for p in levels), probabilities=levels, confidence=confidence)
    if not set(question.required_modalities) <= set(row.metadata.processed_modalities):
        raise ValueError("required modality was not processed; use a non-ok result")
    return QuestionResult(question_id=question.id, type=question.type, status="ok", answer=answer, metadata=row.metadata)


def decode_batch(
    request: DecisionRequest,
    rows: Sequence[LogitRow],
    *,
    unavailable: Sequence[QuestionResult] = (),
    include_confidence: bool = False,
) -> DecisionResponse:
    """Isolated normalizations; exactly one scored or unavailable row per question."""
    rows_by_id = {row.question_id: row for row in rows}
    failures = {result.question_id: result for result in unavailable}
    if len(rows_by_id) != len(rows) or len(failures) != len(unavailable):
        raise ValueError("duplicate question IDs")
    if any(result.status == "ok" for result in unavailable):
        raise ValueError("unavailable results must have a non-ok status")
    if rows_by_id.keys() & failures.keys():
        raise ValueError("question cannot be both scored and unavailable")
    if rows_by_id.keys() | failures.keys() != {q.id for q in request.questions}:
        raise ValueError("supply exactly one row for each requested question")
    results = tuple(
        failures[q.id] if q.id in failures else decode_question(q, rows_by_id[q.id], include_confidence=include_confidence)
        for q in request.questions
    )
    response = DecisionResponse(state_id=request.state.id, results=results)
    validate_response(request, response)
    return response
