# Experimental program

Status: preregistered planning draft. All results are **not run**. Experiment IDs are stable references, not published findings. No numeric superiority or latency claim is inherited from another model's leaderboard.

## 1. Questions to resolve

1. Does pretrained embedding quality transfer to open-condition decisions?
2. Is one global vector sufficient for the actual task distribution?
3. Does a BERT-like question encoder improve decisions after accounting for alignment/training cost?
4. Does cross-attention improve spatial/temporal binding over discriminatively trained MaxSim?
5. Does reusable state memory reduce total cost when many questions share an input?
6. Can compression or a compact specialist retain a useful accuracy/coverage trade-off?

## 2. Scope and sampling

Use a small, controlled pilot before any large training run. Choose its size after a power/variance review and annotator trial; no budget or target size has been supplied. Include English and French if these are deployment targets, and keep language choice configurable. English repository prose is not a requirement for English-only model behavior.

Sample these task families:

| Family | Example condition | Essential counterexample |
|---|---|---|
| Global visual content | A receipt is shown | Similar layout, different document type |
| Document detail | A signature is in the designated field | Signature elsewhere, or field cropped |
| Spatial binding | The checkbox beside label A is checked | Another box is checked |
| Spoken content | The speaker explains password changes | Related computer discussion without that explanation |
| Environmental audio | An alarm sounds | Similar background sound without an alarm |
| Video sequence | The object is placed before the person leaves | Same events in reverse order |
| Broad audiovisual match | Visible instrument matches audible instrument | Similar setting but different instrument sound |
| Precise audiovisual relation | A sound coincides with the visible event | Same streams with a controlled offset |
| Typed ordinal judgment | Description matches demonstration to a specified degree | Same topic with omitted or contradictory actions |
| Availability | Required evidence is assessable | Muted, missing, occluded, truncated, or corrupted input |

Distinguish "same topic" from "same event" and "same instant" in annotations. Do not assign exact synchrony labels to videos whose sampling cannot represent the offset.

## 3. Leakage prevention and annotation

Split by source document/video/speaker/session/template as applicable before generating crops or counterfactuals. Keep related clips, different audio offsets, and alternate question paraphrases in the same group. Use a separate generalization split holding out whole predicate/label families. Maintain an additional calibration split and never fit thresholds on final test labels.

Record label provenance, annotator agreement, ambiguity, sampling artifacts, and source permissions. Resolve contradictions rather than pretending all examples have binary truth under inadequate evidence. For ordinal rubrics, define concrete levels and measure inter-rater consistency before interpreting small metric differences.

Treat OCR text, transcripts, and visual captions as alternate observations, not ground truth. Compare each pipeline end to end, including errors from its preprocessing. Do not place externally supplied content in an instruction role during inference.

## 4. Baselines and controlled ablations

| ID | Variant | Controlled question | Primary dependency |
|---|---|---|---|
| E00 | Native modality/shape smoke tests | Can the exact artifact execute each claimed input path? | Artifact audit |
| E01 | Native global embeddings + description similarity | How far does unmodified representation reuse go? | E00 |
| E02 | Global embeddings + supervised decision scorer | How much comes from decision supervision alone? | Dataset + E01 |
| E03 | OCR or ASR + a text decision model | Is direct multimodal processing useful for text-dominated predicates? | Dataset + audited text baseline |
| E04 | Audited existing visual/omni decision or structured-output model | What does an already task-capable system achieve? | Availability and cost audit |
| E05 | Native text branch vs BERT-like branch | Is BERT worth its interface and training cost? | E02 |
| E06 | Whole-state vs segment vs token memory | What information granularity is needed? | Adapter + E02 |
| E07 | MaxSim decision scorer vs cross-attention scorer | Does extra interaction capacity improve relational decisions? | E05/E06 |
| E08 | Frozen vs adapter-tuned vs partially/full-tuned backbone | Is adaptation of representation necessary? | E07 |
| E09 | Audio-only, video-only, transcript-only, jointly aligned audio-video | Is the model using the required modalities and timing? | E00 + temporal labels |
| E10 | Separate calibration and abstention policies | Are probabilities and accepted decisions reliable? | Trained variants |
| E11 | Token count, feature width, precision sweeps | What is the smallest useful memory? | Uncompressed reference |
| E12 | Compact document specialist / later student | Can deployment cost be reduced in a limited domain? | Measured teacher/reference |

E04 may include Decider-Vision only after verifying its current files and behavior. E12 may use ColModernVBERT or another audited specialist. A unavailable model is a recorded exclusion, not a reason to invent results or block all other baselines.

For E05, reuse the same image/omni features, supervision, and train/test splits. For E07, match backbone and data, report trainable parameter counts and tuning budgets, and disclose unmatched capacity. For E08, recompute feature caches when their producing weights change. A head-only run and an end-to-end run are separate systems.

## 5. Representation/compression design

Compare native whole-state pooling first. For a segment baseline, retain segment order and timestamps, measure overlap and repeated encoder work, and include a global context feature as a distinct ablation. For token memory, validate the extraction layer and masks before compressing.

Suggested exploratory grids, explicitly not requirements: keep-all versus K in {32, 64, 128, 256}; projected width d in {128, 256, 512}; precision chosen from supported native/quantized formats. Skip nonsensical combinations when a model produces fewer tokens. The effective temporal/spatial resolution must be reported alongside K.

Fit learned pooling/projection/quantization on training data only. Compression should be judged on difficult predicates, not only average retrieval or coarse category performance. Include a documented memory-size calculation and measure actual serialized/runtime memory separately.

## 6. Metrics

### Decision quality

Noul: per-condition false-positive/false-negative rates, macro metrics, precision-recall behavior, and negative log-likelihood/Brier score. Choice: top-1 accuracy, macro recall/F1 where meaningful, candidate-count sensitivity, and option-order consistency. Score: mean absolute error of expected rubric position, an ordinal distribution metric, and agreement statistics appropriate to the annotation scheme.

Report results by modality, task family, language, domain, predicate novelty, clip duration, and evidence quality. Do not average unrelated retrieval benchmarks into a project-level decision score.

### Reliability

Use held-out reliability plots and calibration metrics, not only maximum confidence. Plot risk against coverage after abstention. Report accepted-decision error and review rate at operating thresholds selected without touching the final test set. Treat error bars and grouped bootstrap resampling at the source level seriously; individual frames from the same source are not independent test units.

### Temporal and cross-modal behavior

Measure offset sensitivity with the same underlying media, order-inversion detection, contradictory-audio robustness, and modality removal. Broad semantic judgments should remain stable under irrelevant timing shifts; synchrony judgments should change when the defined tolerance is exceeded. Artificial edits can introduce shortcuts: balance artifacts and manually inspect a sample.

### Cost

Measure cold startup separately from warmed inference. Report media decode/processing, state encoding, incremental per-question work, batch throughput, p50/p95, peak device memory, cached-state bytes, hardware, precision, input resolution, frame rate, clip duration, and number of candidates/questions. Measure 1, 10, and 100 questions per state where feasible. These are test points, not speed claims.

Include a comparison with no cache and with valid cache reuse. Several questions in one batch are not proof of one encoder pass. Count actual calls and inspect traces.

## 7. Training controls

Freeze the backbone first. Tune the interface and, where selected, the condition encoder. Use supervised labels for each question type; include close negatives and positive collisions. Losses for different primitives may require balancing, which should be logged and ablated.

Only then try parameter-efficient or partial/full backbone adaptation. Document initialization, seed, optimizer, training steps, effective batch size, compute budget, selection metric, and early-stopping rule. Audit synthetic/teacher labels before use. Reinforcement learning is a deferred experiment, not a prerequisite inherited from Laya's description.

## 8. Decision gates

| Gate | Evidence needed | Action |
|---|---|---|
| G0: artifact readiness | Pinned artifacts, allowed reuse, native input/shape tests | Admit a model to experiments |
| G1: usable benchmark | Leakage-resistant splits, reviewed annotations, baseline execution | Start architecture comparisons |
| G2: extra representation justified | Global versus richer-memory comparison on held-out conditions | Keep or drop token/segment complexity |
| G3: interaction justified | Matched MaxSim/attention comparison and cost report | Select the decision module |
| G4: multimodal relation credible | Joint-input and controlled temporal counterfactual tests | Claim only the evaluated relations |
| G5: deployment candidate | Calibration + risk/coverage + end-to-end latency/memory | Set a scoped deployment policy |
| G6: compression justified | Compressed model at a matched accepted-error target | Adopt compression or retain reference |

Numerical pass/fail targets must be written before final test evaluation once the application and cost of errors are selected. Do not invent them in advance of that context.

## 9. First reproducible result

Publish an experiment manifest, dataset/split identifiers, environment lock, exact preprocessing, model revisions, predictions, metric script, and limitations. Label author-reported upstream results separately from reproduced measurements. A negative finding such as "global embeddings suffice" is a valid outcome: the project should prefer the least complex model that meets its scoped goals.
