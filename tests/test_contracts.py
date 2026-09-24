"""Contract and pure-math tests; no model, network, or GPU is used."""

import importlib.util
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from omni_jev.contracts import (
    Candidate, ChoiceAnswer, ChoiceQuestion, Confidence, DecisionRequest,
    DecisionResponse, LevelProbability, NoulAnswer, NoulQuestion, OptionProbability,
    ProcessedSource, Provenance, QuestionResult, RubricLevel, ScoreAnswer,
    ScoreQuestion, Source, SourceRef, State, TimeWindow, validate_response,
)
from omni_jev.decoding import LogitRow, decode_batch, decode_question, masked_softmax
from omni_jev.schemas import export_schemas


def metadata(source="text", modalities=("text",)):
    return Provenance(model_revision="synthetic", processed_sources=(ProcessedSource(source_id=source, modalities=modalities),))


def request_for(*questions):
    return DecisionRequest(state=State(id="s", sources=(Source(id="text", modality="text", text="fixture"),)), questions=questions)


def choice():
    return ChoiceQuestion(id="c", instructions="Classify.", options=(Candidate(id="a", description="A"), Candidate(id="b", description="B")))


def score():
    return ScoreQuestion(id="s", instructions="Rate.", levels=tuple(RubricLevel(id=f"l{i}", index=i, description=f"Level {i}") for i in range(3)))


def test_json_roundtrip_and_frozen_models():
    request = request_for(NoulQuestion(id="n", instructions="Present?"), choice(), score())
    assert DecisionRequest.model_validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError):
        request.state.id = "changed"
    assert isinstance(request.questions, tuple)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_reject_invalid_probability(value):
    with pytest.raises(ValidationError):
        NoulAnswer(noul=value)


def test_exact_binary_endpoints_allowed():
    assert NoulAnswer(noul=0).noul == 0.0
    assert NoulAnswer(noul=1).noul == 1.0


@pytest.mark.parametrize("patch", [
    {"instructions": " "}, {"id": "bad id"}, {"surprise": 1},
    {"required_modalities": ("text", "text")}, {"required_modalities": ("smell",)},
])
def test_reject_invalid_question(patch):
    with pytest.raises(ValidationError):
        NoulQuestion(**({"id": "q", "instructions": "Question"} | patch))


def test_duplicate_question_and_source_ids():
    q = NoulQuestion(id="q", instructions="Q")
    with pytest.raises(ValidationError):
        request_for(q, q)
    source = Source(id="x", modality="text", text="X")
    with pytest.raises(ValidationError):
        State(id="x", sources=(source, source))


@pytest.mark.parametrize("options", [(), (Candidate(id="a", description="A"), Candidate(id="a", description="B"))])
def test_choice_rejects_empty_and_duplicate_ids(options):
    with pytest.raises(ValidationError):
        ChoiceQuestion(id="c", instructions="Q", options=options)


@pytest.mark.parametrize("indices", [(), (0,), (1, 2), (0, 2), (1, 0), (0, 0)])
def test_reject_malformed_rubrics(indices):
    with pytest.raises(ValidationError):
        ScoreQuestion(id="s", instructions="Q", levels=tuple(RubricLevel(id=f"l{i}", index=v, description="Level") for i, v in enumerate(indices)))


def test_score_arithmetic_and_distribution_preserved():
    q = score()
    result = decode_question(q, LogitRow("s", ("l2", "l0", "l1"), (math.log(0.5), math.log(0.2), math.log(0.3)), metadata()))
    assert result.answer.score == pytest.approx(1.3)
    assert tuple(p.id for p in result.answer.probabilities) == ("l0", "l1", "l2")
    with pytest.raises(ValidationError):
        ScoreAnswer(score=0.5, probabilities=result.answer.probabilities)


@pytest.mark.parametrize("items", [
    (("a", 0.2), ("b", 0.2)), (("a", 0.5), ("a", 0.5)),
])
def test_bad_distributions(items):
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="a", probabilities=tuple(OptionProbability(id=i, probability=p) for i, p in items))


def test_wrong_winner_rejected():
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="a", probabilities=(OptionProbability(id="a", probability=0.2), OptionProbability(id="b", probability=0.8)))


@pytest.mark.parametrize("logits,mask", [
    ((), ()), ((0.0,), ()), ((0.0,), (False,)), ((0.0,), (1,)),
    ((float("nan"),), (True,)), ((float("inf"),), (True,)), ((True,), (True,)), (("1",), (True,)),
])
def test_bad_softmax_inputs(logits, mask):
    with pytest.raises(ValueError):
        masked_softmax(logits, mask)


def test_padding_and_large_logits():
    assert masked_softmax((1000.0, float("nan"), 1000.0), (True, False, True)) == (0.5, 0.0, 0.5)
    assert masked_softmax((-1e308, 1e308), (True, True)) == (0.0, 1.0)
    base = masked_softmax((0.1, 0.3), (True, True))
    assert masked_softmax((50.1, 50.3), (True, True)) == pytest.approx(base)


def test_candidate_permutations_and_padding():
    q = choice()
    for pairs in itertools.permutations((("a", 0.1), ("b", 0.9), (None, float("nan")))):
        row = LogitRow("c", tuple(i for i, _ in pairs), tuple(v for _, v in pairs), metadata(), tuple(i is not None for i, _ in pairs))
        out = decode_question(q, row).answer
        assert out.choice == "b"
        assert tuple(p.probability for p in out.probabilities) == pytest.approx(masked_softmax((0.1, 0.9), (True, True)))
    reverse = ChoiceQuestion(id="c", instructions="Classify.", options=tuple(reversed(q.options)))
    a = decode_question(q, LogitRow("c", ("a", "b"), (0.0, 0.0), metadata())).answer
    b = decode_question(reverse, LogitRow("c", ("b", "a"), (0.0, 0.0), metadata())).answer
    assert a.choice == b.choice == "a"  # Explicit tie policy, not input order.


@pytest.mark.parametrize("ids,logits,mask", [
    (("a",), (1.0,), None), (("a", "a"), (1.0, 2.0), None),
    (("a", "unknown"), (1.0, 2.0), None), (("a", "b"), (1.0,), None),
    (("a", "b", "padding"), (1.0, 2.0, 3.0), (True, True, False)),
    (("a", None), (1.0, 2.0), None),
])
def test_invalid_candidate_binding(ids, logits, mask):
    with pytest.raises(ValueError):
        decode_question(choice(), LogitRow("c", ids, logits, metadata(), mask))


def test_question_isolation_and_noul_semantics():
    q = NoulQuestion(id="n", instructions="Present?")
    row = LogitRow("n", ("yes", "no"), (2.0, 0.0), metadata())
    single = decode_batch(request_for(q), (row,)).results[0]
    assert single.answer.noul == pytest.approx(1 / (1 + math.exp(-2)))
    other = LogitRow("c", ("a", "b"), (10000.0, -10000.0), metadata())
    batch = decode_batch(request_for(choice(), q), (row, other))
    assert batch.results[1] == single
    assert not hasattr(single.answer, "confidence")


def test_batch_rejects_duplicate_missing_extra_and_wrong_type():
    q = NoulQuestion(id="n", instructions="Present?")
    row = LogitRow("n", ("no", "yes"), (0.0, 1.0), metadata())
    request = request_for(q)
    for rows in ((), (row, row), (replace(row, question_id="other"),)):
        with pytest.raises(ValueError):
            decode_batch(request, rows)
    response = decode_batch(request, (row,))
    with pytest.raises(ValueError):
        validate_response(request_for(ChoiceQuestion(id="n", instructions="Q", options=choice().options)), response)


def test_wrong_candidate_response_and_state_binding():
    q = choice()
    response = decode_batch(request_for(q), (LogitRow("c", ("a", "b"), (0.0, 1.0), metadata()),))
    changed = ChoiceQuestion(id="c", instructions="Q", options=(Candidate(id="x", description="X"),))
    with pytest.raises(ValueError):
        validate_response(request_for(changed), response)
    with pytest.raises(ValueError):
        validate_response(request_for(q), DecisionResponse(state_id="other", results=response.results))


def test_confidence_is_opt_in_named_and_validated():
    row = LogitRow("c", ("a", "b"), (0.0, 0.0), metadata())
    assert decode_question(choice(), row).answer.confidence is None
    out = decode_question(choice(), row, include_confidence=True).answer
    assert out.confidence.method == "normalized_entropy_v1"
    assert out.confidence.value == pytest.approx(0.0)
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice=out.choice, probabilities=out.probabilities, confidence=Confidence(value=0.5))
    one = ChoiceQuestion(id="c", instructions="Q", options=(Candidate(id="a", description="A"),))
    assert decode_question(one, replace(row, candidate_ids=("a",), logits=(0.0,)), include_confidence=True).answer.confidence.value == 1.0


@pytest.mark.parametrize("status", ["insufficient_evidence", "unsupported_modality", "abstained", "error"])
def test_unavailable_has_no_fabricated_answer(status):
    q = NoulQuestion(id="n", instructions="Is there speech?", required_modalities=("audio",))
    failure = QuestionResult(question_id="n", type="noul", status=status, reason="Audio not processed.", metadata=Provenance(model_revision="fixture"))
    response = decode_batch(request_for(q), (), unavailable=(failure,))
    assert response.results[0].answer is None
    with pytest.raises(ValidationError):
        QuestionResult(question_id="n", type="noul", status=status, answer=NoulAnswer(noul=0.0), reason="Missing.", metadata=metadata())
    with pytest.raises(ValidationError):
        QuestionResult(question_id="n", type="noul", status=status, metadata=metadata())


def test_missing_audio_and_source_specific_modality_gate():
    q = NoulQuestion(id="n", instructions="Does the video contain speech?", sources=(SourceRef(source_id="v"),), required_modalities=("audio",))
    request = DecisionRequest(state=State(id="av", sources=(
        Source(id="v", modality="video", uri="fixture://silent", has_audio=False),
        Source(id="a", modality="audio", uri="fixture://other"),
    )), questions=(q,))
    meta = Provenance(model_revision="fixture", processed_sources=(ProcessedSource(source_id="v", modalities=("video",)), ProcessedSource(source_id="a", modalities=("audio",))))
    with pytest.raises(ValueError, match="question scope"):
        decode_batch(request, (LogitRow("n", ("no", "yes"), (0.0, 1.0), meta),))
    with pytest.raises(ValueError, match="not processed"):
        decode_question(q, LogitRow("n", ("no", "yes"), (0.0, 1.0), metadata("v", ("video",))))


def test_absent_source_cannot_be_processed():
    q = NoulQuestion(id="n", instructions="Present?")
    request = DecisionRequest(state=State(id="s", sources=(Source(id="text", modality="text", available=False),)), questions=(q,))
    with pytest.raises(ValueError, match="available"):
        decode_batch(request, (LogitRow("n", ("no", "yes"), (0.0, 0.0), metadata()),))


@pytest.mark.parametrize("kwargs", [
    {"modality": "image"}, {"modality": "image", "text": "x"},
    {"modality": "text", "text": "x", "uri": "fixture://x"},
    {"modality": "image", "uri": "x", "duration_seconds": 1.0},
    {"modality": "audio", "uri": "x", "sample_rate_hz": True},
    {"modality": "image", "uri": "x", "has_audio": False},
    {"modality": "video", "uri": "x", "frame_timestamps": (2.0, 1.0)},
    {"modality": "video", "uri": "x", "duration_seconds": 1.0, "frame_timestamps": (2.0,)},
    {"modality": "audio", "uri": "x", "sha256": "not-a-hash"},
])
def test_invalid_source_metadata(kwargs):
    with pytest.raises(ValidationError):
        Source(id="s", **kwargs)


def test_time_windows_and_source_references():
    for start, end in ((1.0, 1.0), (2.0, 1.0), (-1.0, 1.0)):
        with pytest.raises(ValidationError):
            TimeWindow(start_seconds=start, end_seconds=end)
    with pytest.raises(ValidationError):
        request_for(NoulQuestion(id="n", instructions="Q", sources=(SourceRef(source_id="unknown"),)))
    window = TimeWindow(start_seconds=1.0, end_seconds=2.0)
    with pytest.raises(ValidationError):
        request_for(NoulQuestion(id="n", instructions="Q", sources=(SourceRef(source_id="text", window=window),)))
    q = NoulQuestion(id="n", instructions="Event?", sources=(SourceRef(source_id="v", window=window),))
    request = DecisionRequest(state=State(id="s", sources=(Source(id="v", modality="video", uri="fixture://v", duration_seconds=3.0),)), questions=(q,))
    row = LogitRow("n", ("no", "yes"), (0.0, 1.0), metadata("v", ("video",)))
    with pytest.raises(ValueError, match="time window"):
        decode_batch(request, (row,))
    observed = ProcessedSource(source_id="v", modalities=("video",), windows=(window,))
    updated = replace(row, metadata=Provenance(model_revision="fixture", processed_sources=(observed,)))
    assert decode_batch(request, (updated,)).results[0].status == "ok"


def test_no_claims_without_processed_provenance():
    with pytest.raises(ValidationError):
        QuestionResult(question_id="n", type="noul", status="ok", answer=NoulAnswer(noul=0.1), metadata=Provenance(model_revision="fixture"))


def test_exported_schemas_and_synthetic_example(tmp_path):
    spec = importlib.util.spec_from_file_location("example", Path(__file__).parents[1] / "examples" / "typed_decisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request, response = module.synthetic_example()
    paths = export_schemas(tmp_path)
    assert len(paths) == 2
    for path, value in zip(paths, (request, response)):
        schema = json.loads(path.read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value.model_dump(mode="json"))
    validate_response(request, DecisionResponse.model_validate_json(response.model_dump_json()))
    payload = request.model_dump(mode="json")
    payload["questions"][0]["type"] = "freeform"
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate_json(json.dumps(payload))


def test_rubric_ids_must_be_unique():
    with pytest.raises(ValidationError):
        ScoreQuestion(id="s", instructions="Q", levels=(
            RubricLevel(id="same", index=0, description="Low"),
            RubricLevel(id="same", index=1, description="High"),
        ))


def test_wrong_rubric_in_response_rejected():
    q = score()
    response = decode_batch(request_for(q), (LogitRow("s", ("l0", "l1", "l2"), (0.0, 1.0, 0.0), metadata()),))
    other = ScoreQuestion(id="s", instructions="Q", levels=(
        RubricLevel(id="low", index=0, description="Low"),
        RubricLevel(id="high", index=1, description="High"),
    ))
    with pytest.raises(ValueError, match="rubric"):
        validate_response(request_for(other), response)


def test_video_cannot_claim_audio_when_declared_silent():
    q = NoulQuestion(id="n", instructions="Q")
    request = DecisionRequest(state=State(id="s", sources=(Source(id="v", modality="video", uri="x", has_audio=False),)), questions=(q,))
    with pytest.raises(ValueError, match="inconsistent"):
        decode_batch(request, (LogitRow("n", ("no", "yes"), (0.0, 1.0), metadata("v", ("video", "audio"))),))


def test_evidence_reference_must_have_been_processed():
    q = NoulQuestion(id="n", instructions="Q")
    bad = Provenance(model_revision="fixture", processed_sources=metadata().processed_sources, evidence_refs=(SourceRef(source_id="other"),))
    with pytest.raises(ValueError, match="unknown source"):
        decode_batch(request_for(q), (LogitRow("n", ("no", "yes"), (0.0, 1.0), bad),))


def test_status_answer_type_and_batch_conflicts():
    q = NoulQuestion(id="n", instructions="Q")
    request = request_for(q)
    row = LogitRow("n", ("no", "yes"), (0.0, 1.0), metadata())
    failure = QuestionResult(question_id="n", type="noul", status="error", reason="Test.", metadata=Provenance(model_revision="fixture"))
    with pytest.raises(ValueError):
        decode_batch(request, (row,), unavailable=(failure,))
    with pytest.raises(ValueError):
        decode_batch(request, (), unavailable=(failure, failure))
    ok = decode_question(q, row)
    with pytest.raises(ValueError):
        decode_batch(request, (), unavailable=(ok,))
    with pytest.raises(ValidationError):
        QuestionResult(question_id="n", type="choice", status="ok", answer=NoulAnswer(noul=0.1), metadata=metadata())
