"""Versioned, independent Omni-JEV contracts; not a TypeSafe wire adapter.

JSON arrays become immutable tuples. Python construction uses tuples explicitly.
All inference/evidence claims must come from a real adapter, not these schemas.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=65536)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Modality = Literal["text", "image", "audio", "video"]
QuestionType = Literal["noul", "choice", "score"]
Status = Literal["ok", "insufficient_evidence", "unsupported_modality", "abstained", "error"]
TOLERANCE = 1e-6


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def unique(values: tuple, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


class TimeWindow(Contract):
    start_seconds: Nonnegative
    end_seconds: Positive

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("time window must have positive length")
        return self


class Source(Contract):
    id: Identifier
    modality: Modality
    available: bool = True
    text: Text | None = None
    uri: Text | None = None
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")] | None = None
    duration_seconds: Positive | None = None
    frame_timestamps: tuple[Nonnegative, ...] = ()
    sample_rate_hz: Annotated[int, Field(gt=0)] | None = None
    has_audio: bool | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.available and ((self.text is None) == (self.uri is None)):
            raise ValueError("available source requires exactly one of text or uri")
        if self.text is not None and self.modality != "text":
            raise ValueError("inline text belongs only to text sources")
        if self.modality not in ("audio", "video"):
            if self.duration_seconds is not None or self.sample_rate_hz is not None:
                raise ValueError("duration/sample rate require audio or video")
        if self.modality != "video" and (self.frame_timestamps or self.has_audio is not None):
            raise ValueError("frame timestamps/has_audio require video")
        if any(b <= a for a, b in zip(self.frame_timestamps, self.frame_timestamps[1:])):
            raise ValueError("frame timestamps must strictly increase")
        if self.duration_seconds is not None and any(t > self.duration_seconds for t in self.frame_timestamps):
            raise ValueError("frame timestamp exceeds source duration")
        return self


class State(Contract):
    id: Identifier
    sources: Annotated[tuple[Source, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def source_ids(self) -> Self:
        unique(tuple(s.id for s in self.sources), "source IDs")
        return self


class SourceRef(Contract):
    source_id: Identifier
    window: TimeWindow | None = None


def check_ref(ref: SourceRef, sources: dict[str, Source]) -> None:
    if ref.source_id not in sources:
        raise ValueError(f"unknown source: {ref.source_id}")
    source = sources[ref.source_id]
    if ref.window is not None:
        if source.modality not in ("audio", "video"):
            raise ValueError("time windows require audio or video")
        if source.duration_seconds is not None and ref.window.end_seconds > source.duration_seconds:
            raise ValueError("time window exceeds source duration")


class QuestionBase(Contract):
    id: Identifier
    instructions: Text
    sources: tuple[SourceRef, ...] = ()
    required_modalities: tuple[Modality, ...] = ()

    @model_validator(mode="after")
    def references_unique(self) -> Self:
        unique(self.sources, "question source references")
        unique(self.required_modalities, "required modalities")
        return self


class BinaryCriteria(Contract):
    yes: Text = "The proposition is true."
    no: Text = "The proposition is false."


class NoulQuestion(QuestionBase):
    type: Literal["noul"] = "noul"
    criteria: BinaryCriteria = Field(default_factory=BinaryCriteria)


class Candidate(Contract):
    id: Identifier
    description: Text


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"] = "choice"
    options: Annotated[tuple[Candidate, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def option_ids(self) -> Self:
        unique(tuple(o.id for o in self.options), "candidate IDs")
        return self


class RubricLevel(Candidate):
    index: Annotated[int, Field(ge=0)]


class ScoreQuestion(QuestionBase):
    type: Literal["score"] = "score"
    levels: Annotated[tuple[RubricLevel, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def ordered_levels(self) -> Self:
        unique(tuple(level.id for level in self.levels), "rubric IDs")
        if tuple(level.index for level in self.levels) != tuple(range(len(self.levels))):
            raise ValueError("rubric indices must be contiguous and ordered from zero")
        return self


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(Contract):
    schema_version: Literal["0.1"] = "0.1"
    state: State
    questions: Annotated[tuple[Question, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def references_valid(self) -> Self:
        unique(tuple(q.id for q in self.questions), "question IDs")
        sources = {s.id: s for s in self.state.sources}
        for question in self.questions:
            for ref in question.sources:
                check_ref(ref, sources)
        return self


class OptionProbability(Contract):
    id: Identifier
    probability: Probability


class LevelProbability(OptionProbability):
    index: Annotated[int, Field(ge=0)]


def check_distribution(items: tuple[OptionProbability, ...]) -> None:
    unique(tuple(item.id for item in items), "probability IDs")
    if not math.isclose(math.fsum(item.probability for item in items), 1.0, abs_tol=TOLERANCE, rel_tol=0):
        raise ValueError("probabilities must sum to one")


def entropy_concentration(probabilities: tuple[float, ...]) -> float:
    """Normalized concentration, NOT calibrated accuracy or Jev confidence."""
    if len(probabilities) == 1:
        return 1.0  # A singleton choice is degenerate, not evidence of correctness.
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


class Confidence(Contract):
    method: Literal["normalized_entropy_v1"] = "normalized_entropy_v1"
    value: Probability


def check_confidence(confidence: Confidence | None, items: tuple[OptionProbability, ...]) -> None:
    if confidence is not None:
        expected = entropy_concentration(tuple(item.probability for item in items))
        if not math.isclose(confidence.value, expected, abs_tol=TOLERANCE, rel_tol=0):
            raise ValueError("confidence does not match its declared method")


class NoulAnswer(Contract):
    type: Literal["noul"] = "noul"
    noul: Probability


class ChoiceAnswer(Contract):
    type: Literal["choice"] = "choice"
    choice: Identifier
    probabilities: Annotated[tuple[OptionProbability, ...], Field(min_length=1)]
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def distribution_valid(self) -> Self:
        check_distribution(self.probabilities)
        # Lexical ID resolves exact ties independently of input option order.
        winner = min(self.probabilities, key=lambda item: (-item.probability, item.id)).id
        if self.choice != winner:
            raise ValueError("choice must be the highest-probability option (lexical ID breaks ties)")
        check_confidence(self.confidence, self.probabilities)
        return self


class ScoreAnswer(Contract):
    type: Literal["score"] = "score"
    score: Nonnegative
    probabilities: Annotated[tuple[LevelProbability, ...], Field(min_length=2)]
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def distribution_valid(self) -> Self:
        check_distribution(self.probabilities)
        if tuple(p.index for p in self.probabilities) != tuple(range(len(self.probabilities))):
            raise ValueError("score probability indices must be contiguous and ordered")
        expected = math.fsum(p.index * p.probability for p in self.probabilities)
        if self.score > len(self.probabilities) - 1 or not math.isclose(self.score, expected, abs_tol=TOLERANCE, rel_tol=0):
            raise ValueError("score must equal the expected zero-based rubric index")
        check_confidence(self.confidence, self.probabilities)
        return self


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class ProcessedSource(Contract):
    source_id: Identifier
    modalities: Annotated[tuple[Modality, ...], Field(min_length=1)]
    windows: tuple[TimeWindow, ...] = ()

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        unique(self.modalities, "processed modalities")
        unique(self.windows, "processed time windows")
        return self


class Provenance(Contract):
    model_revision: Text
    calibration_revision: Text | None = None
    processed_sources: tuple[ProcessedSource, ...] = ()
    evidence_refs: tuple[SourceRef, ...] = ()

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        unique(tuple(s.source_id for s in self.processed_sources), "processed source IDs")
        unique(self.evidence_refs, "evidence references")
        return self

    @property
    def processed_source_ids(self) -> tuple[str, ...]:
        return tuple(s.source_id for s in self.processed_sources)

    @property
    def processed_modalities(self) -> tuple[str, ...]:
        return tuple(sorted({m for s in self.processed_sources for m in s.modalities}))


class QuestionResult(Contract):
    question_id: Identifier
    type: QuestionType
    status: Status
    answer: Answer | None = None
    reason: Text | None = None
    metadata: Provenance

    @model_validator(mode="after")
    def status_matches_answer(self) -> Self:
        if self.status == "ok":
            if self.answer is None or self.answer.type != self.type:
                raise ValueError("ok result requires an answer of its declared type")
            if not self.metadata.processed_source_ids or not self.metadata.processed_modalities:
                raise ValueError("ok result requires processed-source and modality provenance")
        elif self.answer is not None or self.reason is None:
            raise ValueError("non-ok result requires a reason and no answer")
        return self


class DecisionResponse(Contract):
    schema_version: Literal["0.1"] = "0.1"
    state_id: Identifier
    results: Annotated[tuple[QuestionResult, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def result_ids(self) -> Self:
        unique(tuple(result.question_id for result in self.results), "result question IDs")
        return self


def validate_response(request: DecisionRequest, response: DecisionResponse) -> None:
    """Validate cross-message invariants not expressible in standalone JSON Schema."""
    questions = {q.id: q for q in request.questions}
    if response.state_id != request.state.id:
        raise ValueError("response belongs to a different state")
    if {r.question_id for r in response.results} != set(questions):
        raise ValueError("response must cover exactly the requested questions")
    sources = {s.id: s for s in request.state.sources}
    for result in response.results:
        question = questions[result.question_id]
        if result.type != question.type:
            raise ValueError("result type does not match question type")
        processed = set(result.metadata.processed_source_ids)
        if not processed <= sources.keys() or any(not sources[s].available for s in processed):
            raise ValueError("processed sources must exist and be available")
        observations = {s.source_id: s for s in result.metadata.processed_sources}
        for observed in result.metadata.processed_sources:
            source = sources[observed.source_id]
            possible = {source.modality}
            if source.modality == "video" and source.has_audio is not False:
                possible.add("audio")
            if not set(observed.modalities) <= possible:
                raise ValueError("processed modalities are inconsistent with their source")
            for window in observed.windows:
                check_ref(SourceRef(source_id=source.id, window=window), sources)
        for ref in result.metadata.evidence_refs:
            check_ref(ref, sources)
            if ref.source_id not in processed:
                raise ValueError("evidence must refer to a processed source")
        if result.status != "ok":
            continue
        if not {ref.source_id for ref in question.sources} <= processed:
            raise ValueError("a required source was not processed")
        scope = {ref.source_id for ref in question.sources} if question.sources else processed
        scoped_modalities = {m for source_id in scope for m in observations[source_id].modalities}
        if not set(question.required_modalities) <= scoped_modalities:
            raise ValueError("required modality was not processed in the question scope")
        for ref in question.sources:
            if ref.window is not None and not any(
                w.start_seconds <= ref.window.start_seconds and w.end_seconds >= ref.window.end_seconds
                for w in observations[ref.source_id].windows
            ):
                raise ValueError("requested time window is not covered by recorded processing scope")
        answer = result.answer
        if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
            if {p.id for p in answer.probabilities} != {o.id for o in question.options}:
                raise ValueError("choice probabilities do not match requested candidate IDs")
        if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
            if tuple((p.id, p.index) for p in answer.probabilities) != tuple((l.id, l.index) for l in question.levels):
                raise ValueError("score probabilities do not match requested rubric")
