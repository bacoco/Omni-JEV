# OJ-003: leakage-resistant data and CPU evaluation

**Implementation slice:** schema/provenance validation, source-connected grouping, metrics, paired-run comparison and synthetic fixtures. Depends on the typed contracts in PR #14. **Still required for issue #3:** real media, independent annotations, reviewed pilot and deployment-specific operating points. This is not a completed multimodal benchmark and does not load any checkpoint.

## Data interface

`Benchmark` contains an immutable ID, schema version `0.1`, explicitly held-out predicate families, and cases. Each case has exactly one `DecisionRequest` from the existing contracts: the interface is reused rather than duplicated. Multiple questions about the same source are separate cases joined by their lineage keys.

A case records its ID, split (`train`, `validation`, `calibration`, `test`), predicate family, language, source groups, ancestor hashes, annotation origin/note, gold evidence status and request-local target ID. Noul uses `no`/`yes`; Choice uses supplied option IDs; Score uses rubric-level IDs. Insufficient evidence has no target, never a negative label. Available media are not necessarily sufficient evidence: a present but unreadable image can legitimately have an insufficient-evidence annotation.

The validator checks request contracts, target membership, required-source availability and declared modalities. Video audio must be explicitly declared for answerable speech questions. Each prediction wraps the existing `DecisionResponse` and optional measured timing fields. Cross-request validation checks source, option and provenance consistency. This is structural verification, not proof that a model really processed the declared input.

## Split before making derived examples

Use globally namespaced lineage keys such as `acquisition:...`, `document:...`, `session:...`, `speaker:...`, `template:...`. Declare the dependencies that need isolation in the given experiment. All related crops, frames, soundtrack offsets, paraphrases and alternative questions inherit original-source groups. Ancestor hashes connect derivatives to originals. The validator also connects equal source SHA-256 values and identical inline text bytes.

Union-find builds connected components across all lineage links. A component crossing splits is rejected, including a transitive chain through a multi-source example. Source IDs inside requests are local identifiers, not global identities; otherwise a generic ID like `image` would incorrectly join the entire dataset.

Declare test-only predicate families before generating paraphrases. They cannot occur in train, validation **or calibration**. A separate calibration set must not leak the novel test rules into operating-point selection.

This tool validates supplied splits; it does not silently randomize or repair them. It cannot detect omitted ancestry, unreported near-duplicates, or semantic duplicates automatically. External media bytes and their hashes must be checked by the acquisition pipeline; this CPU module does not dereference URIs. A reviewed pilot needs both this validation and actual data inspection.

## Annotation matrix for the real pilot

| Family | Positive/negative contrasts | Evidence-sufficiency check |
|---|---|---|
| Global image category | New class descriptions; unknown content with explicit fallback | Is the image accessible? |
| Document detail | Correct checkbox versus another checkbox; signature in wrong region | Is the relevant crop readable and bound to the correct label? |
| Negation and scope | No signature in one crop versus none on the whole page | Does the observation cover the quantifier's scope? |
| Spoken content | Words spoken versus only visible in subtitles | Was the correct audio track processed? |
| Environmental audio | Event present versus acoustically similar event | Is the recording usable at the required interval? |
| Video order | Same actions in opposite order | Are frame timestamps and sample density adequate? |
| Audiovisual relation | Same topic with aligned versus shifted soundtrack | Is offset meaningful at the sampled temporal resolution? |
| Score rubrics | Distinct described ordered levels; disagreement between reviewers | Is each level operationally defined? |
| Evidence selection | Full media versus a derivative dropping the decisive fact | Is gold retained, not just agreement with an erroneous reference? |
| Adversarial content | Embedded instructions unrelated to the requested task | Does evidence remain separate from policy? |

Annotators must see the actual original evidence and intended observation window. Record provenance, consent/rights, disagreements and adjudication. Do not infer a speaker's intent or an event outside the observable scope. Synthetic edits must be checked for cues that reveal the target without understanding the task. Do not use a teacher's predictions as unreviewed ground truth for its own evaluation.

## Metrics and exact denominators

Input predictions must cover exactly the selected split. Duplicate, missing, unknown or invalid predictions fail evaluation rather than being silently dropped. Log invalid-output failures separately before correcting a run; do not curate away difficult cases. One model/calibration revision pair is required per run.

- `answered_metrics`: top-1 accuracy, NLL (zero probability clipped at `1e-12`), **sum-over-classes** Brier score, top-label ECE and reliability bins, only for emitted answers on answerable cases. The binary sum-over-two-classes Brier is twice the common single-probability binary formulation; the name is deliberate.
- `answerable_top1_success`: correct top-1 answers divided by **all answerable cases**, counting abstention/error/unsupported outputs as no successful answer. This is distinct from accuracy conditional on emission.
- `risk_coverage`: admit complete equal-probability groups together, never exploit their ordering. Report coverage among answered cases and among all answerable cases. These curves are descriptive, not automatically chosen production thresholds.
- `unavailable_evidence`: counts unavailable cases, unsupported fabricated answers and explicit recognition as `insufficient_evidence`. `unsupported_modality` is a runtime limitation, not successful recognition of missing evidence.
- `score`: expected-rubric-index MAE and normalized ranked probability score over the K-1 cumulative thresholds. Separate these from discrete top-1 level accuracy and its confidence statistics.
- `slices`: family/type/language metrics with total/answerable counts and status counts, retaining groups with zero answers.
- `answerable_top1_success_interval`: seeded percentile **cluster** bootstrap over source-connected groups, retaining within-source correlation. Fewer than two groups yields no interval. Small-group intervals are unreliable; no significance claim is made by the tool.
- Timing: p50/p95 of supplied decode, state-encoding, per-question and wall measurements; separate cold/warm wall observations. Components may overlap, so their sum is not forced to equal wall time. Synthetic timestamps test arithmetic only.

For Score, max label probability concerns selecting a discrete level; it is **not** calibrated correctness of the returned expected index. The evaluator does not use the optional entropy diagnostic as a probability of correctness. No temperature fitting, confidence-based action, model selection or paid call occurs here.

## Paired comparisons motivated by the Jev engineering notes

`compare_runs(benchmark, reference, candidate, split=...)` validates two complete runs over the same cases and counts correctness transitions plus answer disagreements. In particular, it distinguishes lost correct decisions from cases where both versions agree and are wrong. It also retains separate unavailable-evidence counts.

Use it for full-versus-compressed evidence and fixed-versus-adaptive selection. Keep backbone, labels and split fixed for each ablation. Review correct-to-incorrect examples; agreement alone does not demonstrate semantic preservation. The helper does not produce a paired significance test or an interval for the difference.

## Run on CPU

From the dataset PR branch (which builds on PR #14):

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_evaluation.py
python -m omni_jev.evaluation datasets/oj003/synthetic-smoke.json
python -m omni_jev.evaluation datasets/oj003/synthetic-smoke.json \
  --predictions datasets/oj003/synthetic-predictions.jsonl --split test
python -m omni_jev.evaluation datasets/oj003/synthetic-smoke.json --require-reviewed
# Last command intentionally fails: fixtures are synthetic, not a reviewed pilot.
```

The checked-in smoke dataset is text-only synthetic contract data, not substitute image/audio/video inputs. Its predictions are hand constructed and marked `synthetic-not-a-model`. No accuracy from this fixture is a model result. Real multimodal fixtures and benchmarks remain part of #3/#10; runtime processing remains #1/#4.

Choose dataset size, target languages, deployment error costs and hardware budgets explicitly before final testing. Those requirements have not been supplied and are not invented by this PR.
