# Design: typed decisions over reusable multimodal memory

Status: design proposal; not an implemented SDK. Symbols, fields, and module names below are proposed Omni-JEV interfaces unless explicitly attributed to an upstream source.

## 1. Problem statement

Given a state S and a typed question q, estimate a distribution over the answers allowed by q. S may include text, images, audio, video, and references to time-aligned portions of those inputs. Questions may concern one modality or a relationship between several modalities.

The neural model supplies judgments; application code composes rules. For example, separately judge whether a consent box is checked and whether a signature is visible in its designated field, then combine the decisions in code. Direct relational questions remain in scope when decomposition would lose the required correspondence: a checked box must belong to the right label, and a sound must coincide with the right event.

This is not a generic chat model, a free-form extractor, a vector database, or an attempt to prove arbitrary logic from an embedding. Retrieval remains a useful baseline or upstream selection mechanism.

## 2. Three typed output families

### 2.1 Noul

TypeSafe's Noul represents the probability of yes [S02]. Proposed computation: score yes/no descriptions with a shared scorer and normalize over the two options, or use a single binary logit. Return p_yes, not only a thresholded boolean. A deployment wrapper can expose the conventional `noul` field.

Use binary cross-entropy or two-class cross-entropy during supervised training. Keep raw logits for calibration. Thresholds belong to application policy and should be chosen on held-out data, not fixed globally by the model implementation.

### 2.2 Choice

Choose exactly one option from a supplied set and expose the option probabilities [S03]. Each option has a stable request-local identifier and a natural-language description. Do not implement the generalist model as a final linear layer with a permanently fixed business label vocabulary.

Mask padding and normalize only over options belonging to the same question. Reject empty and duplicate-ID candidate sets. Require an explicit `other` option when the application needs an open-set fallback; do not silently add one. Test option-order equivariance: reordering candidates should reorder the distribution rather than change the meaning of the answer.

Multiple compatible tags are separate binary questions or an explicit multilabel wrapper. They must not be accidentally normalized as a mutually exclusive set.

### 2.3 Score

A Score uses ordered, described levels and returns their probability-weighted index [S04]. For K levels with probabilities p_k:

    score = sum(k * p_k for k in range(K))

Keep the full distribution because different distributions can have the same mean. Indices encode the declared rubric, not a measured physical unit. Custom numerical level values would be a separately versioned extension.

Start with categorical cross-entropy and compare an order-aware objective such as ranked probability score. A useful order-aware error term compares cumulative predicted and target distributions across the K-1 thresholds. Do not assume that adding an ordinal loss automatically improves calibration.

### 2.4 Confidence, unavailable evidence, and abstention

A similarity score, answer probability, rubric position, and confidence measure are distinct quantities. Do not define an undocumented `confidence` field as max probability while claiming exact Jev compatibility. The first API issue must document its method or explicitly leave compatibility pending.

Proposed result envelope:

    {
      "question_id": "q1",
      "status": "ok | insufficient_evidence | unsupported_modality | error",
      "answer": "a typed result, or null when no valid answer can be produced",
      "metadata": {
        "model_revision": "...",
        "calibration_revision": "...",
        "available_modalities": ["..."],
        "evidence_refs": []
      }
    }

This is transport/evidence metadata, not a fourth modeled decision primitive. Missing audio must not silently become a confident no. Low estimated model confidence and genuinely missing evidence need separate evaluation. A 0.5 binary output by itself does not identify why a case is uncertain.

## 3. Input and time semantics

A state manifest should record source IDs, content hashes, modality, duration, frame timestamps, audio sampling parameters, and cross-source alignment where applicable. Support multiple images and explicitly aligned audio/video streams. Do not assume that uploading a video file means its audio reached the encoder.

For each question, record referenced sources and optional time windows. Model input sampling limits must be visible: an event between sampled frames cannot be assessed with the same evidence as a fully observed event. For precise synchrony, measure the preprocessing resolution and the smallest offsets the input representation can express.

Multimodal input content is evidence, not an instruction channel. Rendered text saying "answer yes" and spoken attempts to change the task belong in adversarial tests, especially when models inherit instruction-following behavior.

## 4. Proposed architecture

    media/state manifest
            |
    native processor + pretrained multimodal backbone
            |
    reusable state representation ------------------------+
            |                                             |
      global vector                                  memory tokens
            |                                     + masks + positions
            +--------------------------+------------------+
                                       |
    question + option/rubric descriptions                  |
            |                                             |
    native text branch OR BERT-like branch                 |
            +--------------------------+------------------+
                                       |
                 global / late-interaction / attention scorer
                                       |
                     question-local option logits
                                       |
                      typed probability/result layer

One backbone is the initial default. GLiFormer, ColBERT, and ColModernVBERT are not three consecutive neural stages. GLiClass demonstrates BERT-like arbitrary-label classification [S09]; Laya documents request-time option-marker scoring [S10]. These provide reusable design patterns, not proof that pretrained heads can be plugged into Ovis unchanged.

## 5. State adapter contract

The proposed adapter returns:

| Field | Meaning |
|---|---|
| `global_embedding` | Native state-level representation when available |
| `memory` | Selected intermediate states or segment embeddings |
| `valid_mask` | Valid positions, excluding padding and unwanted prompt tokens |
| `modality_ids` | Text, visual, acoustic, or other documented token origins |
| `positions` | Native/derived frame, time, page, and spatial coordinates where reliable |
| `provenance` | Processor, model, revision, precision, extraction layer, pooling, and sampling config |

Infer widths from configuration and inspect tensor shapes; never hard-code prior conversational dimension claims as facts. Preserve each model's native processor, chat/retrieval template, and pooling convention for its baseline. Whether weights are at a root or subfolder must be checked at the pinned revision.

Intermediate tokens are not automatically trained retrieval embeddings. If a backbone is causal, a token cannot necessarily incorporate modalities or instructions that occur later in the sequence. Extracting only visual positions may omit useful later audio/text context. Inspect token order, contextualization, masks, and any recurrent/hybrid behavior before interpreting the memory as a fully fused scene representation.

Position metadata must reflect the actual selected representations. Do not fabricate pixel-level grounding or treat attention maps as faithful explanations without a dedicated evaluation.

## 6. Representation ladder

### A. One native global vector

This maximizes reuse of the published embedding objective and is the mandatory low-complexity baseline. For a frozen embedding comparison, encode candidate descriptions through the corresponding native text path. A trainable head may combine state and question vectors, but its performance must be reported separately from raw cosine ranking.

### B. Several native segment vectors

Encode aligned short audio/video segments, pages, or image regions using the native pooling interface. Keep segment order and timestamps. This yields a multi-vector state without claiming that intermediate token embeddings are already aligned for retrieval.

It may miss long-range context, repeat encoder work, or lose events at segment boundaries. Measure overlap, aggregation cost, and context loss. A sequence of pooled segment vectors is not equivalent to patch-level ColBERT memory.

### C. Contextualized token memory

Retain selected states before an optional task projection. Train an interface that makes them useful for decision questions. Compare pre-projection and final retrieval features only when both are supported by the audited implementation. Keep a global feature alongside local memory in an explicit ablation, not by default in every comparison.

### D. Compressed memory

After a reference exists, independently vary token count K, feature width d, and numerical precision. Compare spatial/temporal pooling against a learned query/resampler bottleneck. Keep a no-additional-compression control. A Q-Former-like module is a candidate method, not a mandatory component or a claim of novelty.

A query-independent compressor preserves cache reuse. Query-conditioned selection can improve a decision but adds per-question cost; distinguish it from the reusable state encoding.

## 7. Condition encoding and interaction

A BERT-like encoder is a strongly motivated project hypothesis, not a logical prerequisite. Compare it with native aligned text embeddings. Train projection/alignment and scorer weights; equal dimensionality is not semantic compatibility.

ColBERT motivates independent encoding followed by contextualized late interaction [S07]. A proposed reference scorer is:

    s(q, S) = sum_i max_j dot(normalize(Q_i), normalize(V_j))

Use masks correctly, test numerical parity with a reference implementation, and train on decision labels. A learned temperature or sigmoid cannot by itself repair missing relation information. MaxSim aggregation does not explicitly enforce time ordering; contextualized vectors may still contain temporal information, so test rather than declare failure.

The alternative is a small cross-attention module in which question/option representations read a reusable state memory. It can score candidates without autoregressive answer generation. Preserve question isolation with block masks or independent batching. Adding an unrelated question should not materially change an existing answer. A single batched forward pass must not be advertised as a single shared state encoding unless profiling confirms actual reuse.

## 8. Training program

Start with a frozen backbone. Train the task interface, a BERT-like branch as needed, and the output heads. Establish what supervised decision training adds beyond a calibrated similarity baseline. Then consider adapters or partial/full fine-tuning under a recorded compute budget.

Use `(state, typed question, candidates/rubric, target, evidence availability)` records. Include semantically close counterexamples, negation, spatial binding, temporal inversion, missing evidence, and paraphrases. For dynamic labels, hold out label and question families as well as images. Match splits across all systems.

Do not treat every unmatched item in a retrieval batch as a negative: several states can satisfy the same proposition. Record multi-positive structure and exclude unresolved or contradictory labels. Teacher-generated labels and synthetic counterfactuals need audits and source-grouped train/test separation.

Proper scoring rules encourage useful probability learning, but neither an RL objective nor softmax proves empirical calibration. Begin with supervised differentiable losses. Consider teacher distillation or reinforcement-learning experiments only after supervised baselines justify their complexity.

## 9. Cache and serving boundaries

A state-memory cache key must include content hashes, model revision, adapter/projection revision, processor and sampling configuration, extraction layer, pooling, and dtype. Backbone fine-tuning invalidates old state features; changing a question-only head may not. Do not re-use a detached training cache after updating the producing encoder.

Bound memory and sequence lengths, and report truncation. Measure media decoding, cold model load, warm state encoding, and incremental question costs separately. Add many-question batching only after result isolation and masking tests pass. No vector database is needed for deciding on a provided state; add retrieval indexing only if corpus search enters the product scope.

## 10. Success and non-goals

Select an architecture on matched-domain error/coverage/cost curves, not on a borrowed embedding leaderboard. Support neither unattended high-stakes actuation nor guarantees of universal logical reasoning in the initial release. The first public result should be a reproducible small benchmark, a model/processor manifest, an honest capability matrix, and typed outputs with documented uncertainty behavior.
