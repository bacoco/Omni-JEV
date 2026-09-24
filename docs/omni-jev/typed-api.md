# Typed API contract v0.1 (OJ-002)

This is executable **contract and postprocessing code**, not a model, HTTP server,
perception system, or TypeSafe API client. It consumes explicitly supplied logits.
No checkpoint, encoder, OCR, transcription system, GPU, or network is required.
The example and tests use **synthetic fixtures**, not predictions on real media.

## Run

Python 3.11 or newer:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/typed_decisions.py
python -m omni_jev.schemas /tmp/omni-jev-schemas
```

The last command exports `request.schema.json` and `response.schema.json` using
JSON Schema Draft 2020-12. Parse incoming JSON with
`DecisionRequest.model_validate_json(payload)` and outgoing JSON with
`DecisionResponse.model_validate_json(payload)`. Call
`validate_response(request, response)` before consuming an external response.
`decode_batch` already performs that final check. `decode_question` alone cannot
validate membership in a full state manifest; it is a lower-level building block.

Schemas cover structure and scalar constraints. Probability sums, uniqueness by
ID, contiguous rubric indices, status/answer relationships, temporal bounds, and
cross-message references additionally require the Python semantic validators.
Do not advertise a successful standalone JSON Schema check as complete validation.

Python constructors use tuples for collections; JSON uses arrays. Models and
nested collections are immutable through the supported interface. Pydantic's
unsafe construction/copy APIs are not validated ingestion paths. Unknown fields,
blank descriptions, coercion from strings/booleans to numbers, and non-finite
probabilities are rejected. Input byte limits and authentication remain the
responsibility of a future transport layer.

## Request and output semantics

A request includes `schema_version: "0.1"`, a state manifest, and an ordered array
of questions. IDs are request-local. Every question has `id`, `type`, natural-
language `instructions`, optional `sources` references and `required_modalities`.
Sources have stable IDs, modality, inline text or an opaque URI, optional SHA-256,
availability, and applicable duration, sample-rate, frame-time, and audio metadata.
This library never fetches a URI. Future adapters must implement their own safe
media loading and verify hashes, declared streams, and actual processing.

| Type | Request-specific fields | Answer |
|---|---|---|
| `noul` | Optional `criteria.yes` / `criteria.no` descriptions | `noul = P(yes)`; not an automatic boolean |
| `choice` | Nonempty `options` with unique IDs and descriptions | Highest-probability `choice` ID and the complete distribution |
| `score` | At least two described `levels`, with unique IDs and contiguous indices from zero | `score = sum(index * probability)` and the complete ordered distribution |

Choice options are mutually exclusive. Independent compatible tags are separate
Noul questions, not one softmax over all tags. No implicit `other` label is added.
Score indices are rubric positions, not physical units. Reordering score levels
changes the question; reordering choice options must preserve ID/probability
association. An exact choice tie is resolved by the lexically smallest ID, making
the postprocessor independent of option order. A singleton choice is allowed but
is a degenerate forced selection, not evidence that the category is correct.

`LogitRow` binds logits to named candidate IDs. Binary candidate IDs are exactly
`no` and `yes`; other types use the request's IDs. Padding requires a `None` ID
and `False` mask. It may contain NaN or infinity and contributes zero probability;
valid logits must be finite real numbers. Reject all-masked rows, mismatched
lengths, missing/extra/duplicate candidates and questions. Normalize each question
independently, never across the batch. Distributions are validated within an
absolute tolerance of `1e-6`; supplied malformed distributions are not silently
renormalized. Stable softmax subtracts the maximum before exponentiation.

The postprocessor's permutation and isolation tests do **not** prove an upstream
neural scorer has the same invariances. OJ-006/OJ-008 must test that separately.
This module is not differentiable training code; training losses belong to OJ-009.

## Evidence, abstention, and provenance

Each result includes its question ID/type, status, typed answer or null, and
`metadata`. Provenance names the model revision and optional calibration revision,
plus actual `processed_sources`. Each processed source records its modalities and
optional processed time windows. This is deliberately per-source: an unrelated
audio file must not satisfy a question about the sound of a silent video.

`ok` requires an answer of the correct type and processing provenance. The other
statuses are `insufficient_evidence`, `unsupported_modality`, `abstained`, and
`error`; they require a nonblank reason and `answer: null`. These are operational
statuses, not a fourth decision primitive. A review/abstention policy is external
and is never selected by an undocumented threshold inside this package.

For example, construct an unavailable result rather than inventing a negative:

```python
from omni_jev.contracts import Provenance, QuestionResult

missing_audio = QuestionResult(
    question_id="spoken_topic",
    type="noul",
    status="insufficient_evidence",
    reason="The video's audio stream was not processed.",
    metadata=Provenance(model_revision="actual-adapter-revision"),
)
# Supply through decode_batch(request, scored_rows, unavailable=(missing_audio,)).
```

A request's required modalities are checked within its referenced source scope,
not against an unrelated input elsewhere in the state. `has_audio: null` means
unknown, not absent; audio may only appear in the result when the adapter explicitly
records processing it. `has_audio: false` forbids that claim. Unavailable or unknown
source IDs cannot be marked as processed.

Times are local source seconds. A temporal request must fit within known duration
and within a recorded processing window. The v0.1 implementation requires a single
recorded window containing each requested window; it does not union disjoint ranges.
Processed time windows describe input scope, **not continuous observation or
millisecond accuracy**. Precise synchronization, frame coverage, cross-source
clock alignment, and content-based evidence sufficiency remain adapter/evaluation
work. No structural validator can verify a model's honest perception of the media.

## Confidence and compatibility boundary

No confidence value is produced by default. Choice and Score can explicitly opt
into `confidence: {"method": "normalized_entropy_v1", "value": ...}`:

```text
value = 1 - H(probabilities) / log(number_of_options)
```

For a singleton choice the convention is 1.0. This is distribution concentration,
not the probability that the answer is correct, not evidence availability, not
calibration, and not a claim of Jev confidence equivalence. For Score this diagnostic
also ignores distances between levels; it is not an ordinal uncertainty measure.
Noul has no separate confidence field. A model/calibration revision is provenance,
not evidence that its probabilities are calibrated. OJ-011 evaluates that property.

The three primitive meanings follow the project design and the official references
below. Omni-JEV deliberately uses its own source manifest, question list, candidate
IDs, result envelope and named confidence method. It only accepts string
instructions/descriptions in v0.1. It does not implement TypeSafe's endpoint,
SDK, structured-instruction forms, account, model names, or exact wire protocol.

References inspected on 2026-09-24:

- https://docs.typesafe.ai/primitives/noul
- https://docs.typesafe.ai/primitives/choice
- https://docs.typesafe.ai/primitives/score
- https://docs.pydantic.dev/latest/concepts/strict_mode/

## Validation evidence and remaining work

Local validation on Python 3.13.5, Pydantic 2.13.4 / pydantic-core 2.46.4,
pytest 9.0.2, and jsonschema 4.26.0: **81 tests passed**, including 70 new contract
cases and 11 unchanged publication tests. The current issue manifest and publisher
files were checked against the default-branch blob hashes before the final run.
An offline editable installation, the synthetic example, and schema export also
completed. These are CPU tests and do not evaluate any neural model.

CI is configured for Python 3.11 and 3.13 on GitHub-hosted runners, with read-only
repository permissions, no secrets, no automatic merge, and no model downloads.
CI results must be read from the PR; this document does not claim they passed.
Dependency ranges are declared in `pyproject.toml`; the exact local environment
above is a validation record, not a full platform-independent dependency lock.

Acceptance mapping for issue #2: schemas/examples, malformed-input checks,
probability and ordinal arithmetic, named-candidate reorder/padding, question
isolation, and documented probability/confidence/evidence/policy separation are
implemented. The backbone audit (#1), real state adapter (#4), data/evaluation
(#3), learned heads and calibration remain separate work. The existing bootstrap
publication reports/checksums remain historical records, not a digest of this PR.
