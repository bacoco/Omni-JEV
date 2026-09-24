# Executive synthesis

**Planning date: 2026-09-24. Status: proposed development program.**

## What we are building

Omni-JEV should evaluate natural-language conditions against a multimodal state and return typed values that software can use without parsing generated explanations. The initial idea was visual tagging; the clarified goal is broader: open-condition decisions, choice among supplied options, and rubric-based ordinal scoring. The input scope now includes text, images, visual documents, silent video, audio, and jointly observed audio-video.

The distinction from a retrieval system is the training target. A high similarity score says that two representations match under a learned retrieval objective. Our model must estimate a decision distribution for a specified question. A retrieval model may already provide useful features or even a strong baseline, but that does not settle decision accuracy, negation, evidence binding, or calibration.

## The proposed starting point

Use one main pretrained multimodal backbone, not an Ovis -> ColBERT -> ColModernVBERT stack. Ovis-Omni is the leading *candidate to test* for the full scope; the Ovis family is presented as supporting common embeddings across modalities [S05, S06]. Exact checkpoint availability, loading, hidden-state access, license terms, and revisions must pass the first issue before implementation depends on them.

Retain its original global representation as a baseline. In parallel, evaluate reusable memory in three forms: one vector per whole state; one native vector per time-aligned segment; and selected contextualized token states with masks and position metadata. The segment representation offers a way to obtain several vectors while preserving the pretrained pooling path. Token extraction is a new representation interface and must not be assumed to be already trained for ColBERT scoring.

For questions, retain the native aligned text encoder as a reference and add a BERT-like branch inspired by GLiClass and Laya. Fine-tune the interface rather than assuming that matching vector dimensions produce aligned semantics. Compare a global scorer, a late-interaction scorer, and a small attention-based scorer using the same backbone and data.

## Output contract

TypeSafe documents three primitives: Noul, Choice, and Score [S01-S04]. Our proposed contract preserves those meanings, without claiming wire compatibility or identical confidence semantics before conformance tests.

| Output | Intended meaning | Application use |
|---|---|---|
| Noul | Estimated probability that a binary proposition is true | Threshold into a boolean; route uncertain cases separately |
| Choice | Distribution over a supplied, mutually exclusive option set; return one option | Branch on a selected class |
| Score | Distribution over described, ordered rubric levels; return their expected index | Rank or apply a policy threshold on the declared scale |

Several compatible tags are several binary questions, not one exclusive Choice. A probability of yes is not a degree of quality. A rubric score is not a cosine similarity. Missing evidence is not a negative answer. An evidence/status envelope can express unavailable inputs or abstention without inventing a fourth decision primitive.

## What to reuse

| Source | Reuse target | Boundary |
|---|---|---|
| Ovis | Pretrained multimodal encoding and native processing | Do not discard the contextualizing backbone before a baseline is measured |
| GLiClass / GLiFormer | Dynamic labels, condition encoding, task/head abstractions | Audit GLiFormer's current vision path and actual pretrained heads first |
| ColBERT / colpali_engine | Projection, masks, normalized token interaction, scoring infrastructure | Retrieval scores and transferred projection weights are not automatically decision-ready |
| Laya | Typed-question representation and request-time option scoring | Calibration and checkpoint compatibility must be measured |
| ColModernVBERT | Compact document-specific comparator | Separate model or later student, not a mandatory stage after Ovis |
| Decider-Vision | Candidate functional visual-decision baseline | Exact model and code availability were not revalidated in this package |

## First experiments, in order

1. Audit exact model artifacts and run native modality smoke tests.
2. Implement the contract and dataset/evaluation harness before tuning a complex model.
3. Measure original global embeddings, text-only/transcription baselines where appropriate, and a functional decision baseline.
4. Compare native text encoding against a BERT-like condition branch.
5. Compare global, segment, and token memories; then late interaction against cross-attention while holding other variables fixed.
6. Adapt the backbone only after a frozen-backbone comparison exists.
7. Measure calibration, temporal counterfactuals, and out-of-domain behavior.
8. Compress or distill only after a reference accuracy/coverage curve exists.

## The decisive research question

At a matched error level, does Omni-JEV automate more cases, reduce total cost, or reduce latency compared with existing approaches? Count media decoding, the first state encoding, each additional question, cache invalidation, and memory storage. Retrieval leaderboard results are background evidence, not the answer to this question.

## Deliverables and reading order

The [design](design.md) fixes the intended interfaces. The [experiment plan](experiments.md) gives controlled comparisons and gates. The [discussion record](discussion-record.md) preserves alternatives and corrections so that the project does not repeatedly revisit settled distinctions. The [source register](source-register.md) distinguishes checked documentation from conversation-only leads. The [backlog](../../planning/omni-jev/ROADMAP.md) turns the plan into independently actionable work.

**No hardware budget, latency target, dataset size, or production error threshold has been supplied.** These remain explicit configuration/measurement decisions rather than invented requirements.
