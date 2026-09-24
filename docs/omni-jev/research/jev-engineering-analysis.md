# Jev engineering: complete-text review and adaptations for Omni-JEV

**Reviewed:** 2026-09-24. **Status:** research synthesis and implementation plan, not evidence of a trained Omni-JEV model. Related existing work: #1 (artifact audit), #2 / PR #14 (typed contracts), #3 (evaluation), #4 (state adapter), #7-8 (interaction), #10 (audiovisual tests), #11 (calibration), #12 (compression).

## 1. What was located and what was actually read

The screenshot matches **Jev Engineering for Coding Agents**, a 12-page independent working note subtitled *The TypeSafe Founder's Blueprint for Building with Jev*. Its cover and sources explicitly disclaim affiliation with or endorsement by TypeSafe. It is not a peer-reviewed experimental paper or an official TypeSafe implementation. The compiler's identity is not established here.

A public mirror contains that English PDF, its complete English text extraction, all seven illustrations, and a second document: the original design notes attributed to Diogo Almeida, exported from a public Google Doc. The mirror describes the second PDF as 11 pages. The two documents must not be conflated: a polished synthesis can make tentative design ideas look like established findings.

Pinned source snapshot: `yibie/jev-engineering-zh@3754a0cba2d32d3fdaedffa15b16ee32e1c56ef9`.

- [Synthesis PDF](https://github.com/yibie/jev-engineering-zh/blob/3754a0cba2d32d3fdaedffa15b16ee32e1c56ef9/assets/original.pdf)
- [Complete English synthesis text](https://github.com/yibie/jev-engineering-zh/blob/3754a0cba2d32d3fdaedffa15b16ee32e1c56ef9/assets/original-text.txt)
- [Original-notes PDF export](https://github.com/yibie/jev-engineering-zh/blob/3754a0cba2d32d3fdaedffa15b16ee32e1c56ef9/source-notes/original.pdf)
- [Complete English source notes](https://github.com/yibie/jev-engineering-zh/blob/3754a0cba2d32d3fdaedffa15b16ee32e1c56ef9/source-notes/original.txt)
- [Mirror provenance and original Google Doc link](https://github.com/yibie/jev-engineering-zh/blob/3754a0cba2d32d3fdaedffa15b16ee32e1c56ef9/source-notes/README.md)

**Access boundary:** both English transcriptions were read in full through the GitHub connector, including the complete synthesis sections I-XII, sources, original notes, and three appendices. All seven figure captions were read. The user's cover image was visually inspected; the other figure pixels and binary PDF files could not be retrieved into the local runtime because DNS failed. Consequently this is a complete-text analysis, not a claim of pixel-level inspection of all PDF pages. The live Google Doc was not independently refreshed. No OCR was used and no missing pages were invented.

The source manifest in `audits/papers/jev-engineering-sources.json` records source sizes and Git blob identities supplied by GitHub, separately from locally verified bytes. `python scripts/omni_jev_fetch_papers.py` prints immutable raw URLs. Add `--download --directory /tmp/omni-jev-papers` on a network-enabled machine to retrieve and verify both PDFs. This is optional, does not require a GPU or credentials, and refuses to overwrite files. Upstream originals are not vendored; the mirror's translation license is not assumed to relicense the original notes.

## 2. Section-by-section reading map

The following condenses the source arguments; the proposed Omni-JEV work below is our adaptation, not an upstream performance claim.

| Synthesis section / pages | Source argument | Omni-JEV implication |
|---|---|---|
| I, pp. 2-3 | Some useful agent features require control over context assembly; the decision model sits alongside the model doing the work. | Separate learned judgments, state representation, and execution policy. Do not turn the project into another free-form chat model. |
| II, pp. 3-5 | Cache economics can discourage delegation; fixed tool schemas, generic compaction, fragile sub-agent context and restarts are related symptoms. | Measure the complete observation/decision path and preserve re-fetchable evidence instead of assuming one permanent summary. These are hypotheses about agent designs, not universal facts. |
| III, pp. 5-6 | Reading/searching may dominate coding-session tokens; the token-share table is illustrative. | Measure media decode, encoder work, repeated questions and cache transfer in our workload. Do not import coding-session percentages into image/video workloads. |
| IV, p. 6 | Programmable permissions and intent-to-tool routing can be inserted into an existing agent. | Use typed outputs to inform an independent policy engine; validate selected tools and argument schemas. |
| V, pp. 6-7 | Score context chunks for the current query and choose a visibility level; decide explicitly whether to reuse a prefix. | Introduce query-conditioned evidence selection over reusable representations, with escalation to original media. |
| VI, pp. 7-8 | Purpose-built contexts can improve delegation; parallel work requires explicit state, synchronization and subgoal deduplication. | Reuse one immutable media snapshot across related questions; make writes separate and versioned. |
| VII, p. 8 | Disclose tools in tiers: descriptors, selected schemas, detailed documentation. | Disclose candidate capabilities cheaply, then inspect the selected processor configuration only when needed. |
| VIII, p. 9 | Load instructions when their conditions hold; re-establish them after compaction; structured skills can have scoped behavior. | Always pin hard policies; load task rubrics and domain-specific validation rules conditionally without letting evidence override instructions. |
| IX, pp. 9-10 | Routing should consider data sensitivity, not just price and difficulty. | Filter eligible execution locations before learned routing. Do not inherit vendor/nationality assumptions from the source notes. |
| X, pp. 10-11 | Read-only background tasks can share a retrieval pass instead of repeatedly exploring the same state. | Share evidence memory for tagging, quality checks, scoring and shadow evaluation under access controls. |
| XI, p. 11 | Candidate tools include Headroom, RTK, ast-grep, ast-outline, FastContext and fff. | Reuse patterns where applicable; these are mainly coding-harness tools, not replacements for visual/audio encoders. |
| XII + sources, pp. 11-12 | Treat context as something intentionally assembled. References distinguish illustrative numbers from reported measurements. | Keep empirical gates and honest provenance; preserve the small-model baseline instead of presuming architectural complexity wins. |

The original notes add important nuance: several ideas are explicitly speculative; the routing proportions were suggested illustratively; sharing background retrieval, deduplicating subgoals, and temporary tool-specific prompts are separate proposals. Their informal supplier-security assertions are not evidence of any provider's behavior.

## 3. Audit of the routing arithmetic

Reproduce the note's *illustrative* prices and proportions, not current API prices. Let X be context loaded by the helper, Y helper-generated output, and Z additional input encountered during the work, all in millions of tokens. Under the accounting boundary in the notes:

```
frontier only:        C_f = 25Y + 5Z
frontier/helper/back: C_r = 3X + 20Y + 8Z
extra routed cost:    C_r - C_f = 3X - 5Y + 3Z
```

For X=0.65, Y=0.12, Z=0.23 the arithmetic is 4.15 versus 6.19 in the chosen price units. The helper route is cheaper only when `3X + 3Z < 5Y` within this simplified example. This is not a general statement that routing never works.

The example omits or simplifies cache read/write pricing, repeated turns, provider-specific caching semantics, retry rates, possible output-length changes, and the quality of the resulting work. The original claim is useful as a warning against comparing only marginal token prices, not as a transferable speed or cost benchmark. The initial common context cost is outside the compared incremental paths.

For Omni-JEV, replace that example with an experimentally populated ledger:

```
C_total = decode + initial_state_encoding + memory_IO
        + all_question_encodings + selection_and_scoring
        + re_encoding_or_high_resolution_escalations
        + verification + retries + execution_provider_cost
```

Use distinct monetary, latency, peak-memory and serialized-byte fields; do not add unlike units. Report cold versus warm state processing and amortized cost for 1, 10 and 100 questions. A cheap screening model is beneficial only after its own overhead and its false-negative cost are included.

## 4. Keep the product boundaries clear

The source is about an **agent harness**. Omni-JEV is initially a **trainable decision model and typed interface**. Not every harness feature belongs inside its neural network.

| Layer | Responsibility | Learned? |
|---|---|---|
| Evidence store | Immutable originals, pages/crops/segments, hashes, provenance, permissions | No |
| State adapter | Native processor and pretrained encoder; masks and reliable coordinates | Pretrained/adaptable |
| Decision model | Evaluate conditions or supplied answer/rubric options | Yes |
| Evidence planner | Select representations and request more observation within a budget | Learned proposals plus deterministic constraints |
| Policy/runtime | Permission checks, budgets, executable calls, transactions and auditing | Hard limits in code; optional advisory predictions |

The public output families remain **Noul**, **Choice**, and **Score**. Visibility is a Choice over described options, not a fourth primitive. Evidence sufficiency can be a binary judgment; unavailable input remains status metadata. A Score is an expected index over an ordered described rubric; do not round it to invent a tool or visibility action. Multiple usable tools are not inherently mutually exclusive: evaluate suitability separately when appropriate, then select an eligible action under a policy.

## 5. Main adaptation: a multiscale evidence memory

Keep a graph of evidence rather than one destructive compression result:

```
original file / recording
   -> native global embedding
   -> page, region, frame or time-window references
   -> native segment embeddings or contextual token memory
   -> optional compressed representations with explicit derivation
```

Every derivative should identify the original source hash, producing model and processor revisions, preprocessing parameters, extraction layer, representation type, bounding region or time interval, alignment convention, and validation status. Include tenant/access-scope in any shared cache key. Metadata must distinguish reliable coordinates from merely inferred ones.

Four objects must not be conflated: application state, an encoder feature cache, a selected question-specific memory, and an autoregressive model's KV cache. A feature cache is not automatically a transferable KV cache. A set of tokens taken from a causal or jointly contextualized sequence is not necessarily independently composable. Encoding one crop or audio segment can change its meaning relative to encoding the complete interleaved state.

This preserves our baseline ladder: one native global vector; several native segment vectors; contextual token memory; compressed memory. No new image/audio encoder is required. Ovis remains a candidate subject to the artifact/runtime audit; GLiClass/GLiFormer, Laya, and `colpali_engine` remain code/design sources, not interchangeable pretrained heads.

### Concrete examples

For `is the form signed in the designated box?`, start with the page context and region candidates, then inspect the signature region at enough resolution. A document-level label alone is not sufficient. Keep the page-to-region binding, not just a bag of patches.

For `does the speaker mention the refund while the policy is displayed?`, retain the audio stream, frame/segment timestamps and alignment. A transcript can help verify wording but does not replace acoustic evidence for speaker/timing questions. A visible subtitle is not proof that the speaker said it. A temporally shifted soundtrack is a hard negative even when its topic is unchanged.

For a negative claim, such as `there is no signature anywhere`, a crop with no signature does not prove absence in the complete document. The planner must track the inspected scope and either expand it or abstain. Missing evidence and genuine negative evidence are different targets.

## 6. Query-conditioned selection, without a circular bottleneck

The source's visibility ladder becomes task-specific observation choices, for example:

| Input | Possible observation choices, described at request time |
|---|---|
| Page image | Omit; thumbnail/global vector; selected regions; full-resolution page |
| Video | Omit; sparse frames; aligned segment memory; denser frames for a selected interval |
| Audio | Omit; native global vector; selected waveform segments; longer original interval |
| Derived text | Omit; short supported excerpt; detailed excerpt; complete relevant text |

These choices are not interchangeable information levels. A transcript is not a universally higher-fidelity version of audio; a crop is not always better than a global view. Representations form a graph of capabilities and costs, not a single universally ordered quality scale.

Do not use an expensive Omni-JEV pass to supposedly avoid that same expensive initial pass. Test two modes separately: (a) once-encoded features shared across many questions, saving later interaction work; (b) a genuinely cheaper initial screen that can request additional encoding. Account for both. Maintain a full-input baseline.

A first policy can be deterministic: enforce hard scope requirements, rank candidates from a supplied scorer, select within a measured budget, and escalate when required information is unresolved. Train a planner only after those baselines exist. No claim that a MaxSim heatmap localizes faithful evidence: evaluate region/interval selection explicitly.

When many conditions are asked about one state, compute a shared candidate union but retain separate per-question masks. The union can save work and can also become too large; benchmark it. Do not silently let an unrelated question alter another question's decision.

## 7. Compression must preserve decisions, not merely agreement

The notes suggest checking whether a compressed context retains needed information. Adapt that into *paired* evaluation of complete and reduced evidence:

1. Judge both versions against the same reviewed task labels and source-grouped split.
2. Count correct-to-wrong or correct-to-unanswered changes, not only agreement.
3. Count errors repaired by the candidate, false acceptance on unavailable evidence, and relation-specific regressions.
4. Report risk/coverage and cost together, per predicate family.

Two models can agree because both are wrong. A confidence score from the same network is not an independent proof of successful compression. A teacher label can also be wrong; audit synthetic and distilled labels.

This motivates the implemented `compare_runs` helper in the OJ-003 evaluation PR. Its transitions are measured over supplied typed predictions; it does not claim to perform visual comprehension or semantic preservation verification by itself.

## 8. Policy-first routing and scoped instruction loading

Restrict eligible targets using explicit data policy **before** cost/quality ranking. Policies should describe allowed destinations, retention/training terms, tenant boundary, data class and permitted operations. Model nationality, brand, or size is not a security property. A first-party frontier provider is not inherently permitted to receive secrets. Crops, transcripts and embeddings can still carry sensitive information.

A learned classifier may help flag an unexpected sensitive region, but it must not override a hard deny or grant network access by predicting `allow`. An uncertain decision may escalate to a person; thresholds must be selected for the declared error costs and evaluation distribution. The classifier is not the authority that invents deployment thresholds.

Likewise, conditional instructions should carry scopes and versions. Hard safety/access rules remain mandatory; optional domain rubrics load on demand. Text in an image, spoken audio, retrieved notes, issue comments or a repository file is evidence, not authority to change policy. Test prompt-injection attempts in each modality and in derivative OCR/transcription outputs.

Persist reproducible external observations and explicit decision rationales, not hidden model chain-of-thought or credentials. A prompt-content classifier cannot guarantee security by itself; enforce process, filesystem and network isolation where execution is involved.

## 9. Reusing retrieval, tools and background work

The coding-tool shortlist is useful for developing an orchestration layer, not mandatory for the embedding model. Headroom/RTK suggest compressed-versus-original adequacy checks; ast-grep/ast-outline suggest hierarchical narrowing; FastContext suggests reusable exploration outputs; fff suggests persistent indexes. Their performance and runtime compatibility were not re-audited here, and they should not be added as model dependencies merely because the notes mention them.

For Omni-JEV, the native equivalents are page/region/interval selection, native media processors, optional OCR/ASR and a shared representation store. GLiFormer may help with task and candidate abstractions; Laya with option-conditioned scoring; ColBERT/ColPali with late-interaction code. The source paper adds a control strategy around those pieces, not a new embedding training objective.

A read-only observer may share a snapshot with tagging, evaluation or a reviewer. Read-only tasks still incur compute/network cost and still require access controls. Share the acquisition and representation, not an unversioned mutable global buffer. Return results tagged with source/model/rubric revision. Invalidate them after relevant writes. Deduplicate tasks by source snapshot + normalized task description/identity + model/rubric/policy revision; do not use an embedding match alone as a proof that two tasks are equivalent.

No background worker or external task is started by this document. A later shadow-evaluation path should have explicit consent, retention rules, fixed spending caps and no user-visible side effects. Concurrent writes require separate branches/worktrees or conflict-checked transactions and human-approved merges.

## 10. Experiment extensions

Keep E00-E12 from the original roadmap; add these controlled comparisons after the basic data and encoder gates:

| ID | Question | Controls and failure cases |
|---|---|---|
| E13 | Does query-conditioned memory outperform static compression at equal budget? | Native global, full evidence, uniform segment selection, similarity selection, learned selection; wrong region, reordered events, negation, newly requested detail. |
| E14 | Does routing save end-to-end cost while retaining quality? | Fixed model vs screen/escalation; cold/warm state, 1/10/100 questions, repeated encoding, false negatives, failed calls. |
| E15 | Can many observers share evidence safely and efficiently? | Separate encodes vs shared immutable snapshot; question isolation, stale revision, cross-tenant cache access, duplicate tasks, write conflicts. |
| E16 | Are decisions retained after compression or evidence selection? | Paired correct/wrong transitions, gold labels, missing evidence, multiple unrelated details; no agreement-only success criterion. |

Budget/error thresholds and hardware are not invented. Record them before final test evaluation. Reserve whole source groups before making crops, offsets or paraphrases; reserve predicate families before rewriting questions. Use a separate calibration set, including no leakage of held-out predicate families into threshold fitting.

Important additions to the dataset matrix: negated existence; wrong checkbox-label binding; crop that omits a counterexample; question requiring full-page context; soundtrack shift; voice versus subtitles; sparse frames missing an event; missing audio; adversarial embedded instructions; cached derivative produced by a different processor/model; classification with reordered options; score rubric changed after encoding.

## 11. Delivery order and scope control

**Now, without GPU:** typed schema validation; source/group leakage detection; metric arithmetic and risk/coverage; lineage manifests; paired prediction comparison; documented annotation rules and synthetic fixtures; source retrieval tooling; GitHub PRs with tests. These do not validate any model's semantics.

**On a network-enabled user machine:** resolve and pin real artifacts for #1; verify original bytes and native processors; execute native text/image baselines and record actual tensors; then audio/video/joint paths where genuinely supported. GPU is useful for practical inference/training, not for the dataset and audit logic.

**After a usable baseline:** integrate #4; compare native text and BERT-like branches; train #7/#8 on decision labels; calibrate and profile; only then add adaptive selection or more complex routing. Reuse existing components first. A native global embedding that meets the target is a valid research result.

The relevant work should be published in reviewable slices. The OJ-003 CPU evaluation PR can build on PR #14 without pretending that #14 has merged or that #1's native model gate has passed. Real data collection, independent annotation and human review remain open criteria for #3.

## 12. Claims we explicitly do not carry forward

- No evidence in these documents establishes that Omni-JEV is 200x faster, 400x cheaper, or runs a 300 ms visual-control loop.
- The synthetic token-share table is not a benchmark. The 56.2% / 46.5% figures are attributed in the notes to FastContext's coding traces; they are not verified Omni-JEV measurements, nor a basis for choosing our video budget.
- The figures illustrate architecture, not a released trained model or a validated scheduling algorithm.
- A learned relevance selector does not guarantee complete evidence, safe tool execution, or calibrated probabilities.
- The previously generated project infographic is conceptual. Its latency/scale/cheapness wording and numeric confidence bands are illustrative, not validated product claims. Thresholds belong to application policy, not an unconstrained classifier.
- Multivector memory, retrieval embeddings, attention maps, KV state and faithful explanations are distinct. None should be relabeled as another to imply a capability we have not measured.

**Decision:** retain the source's explicit-state, query-conditioned selection and shared-retrieval principles; adapt them to spatial/temporal evidence and typed decisions. Keep model training and runtime policy separate. Implement the inexpensive validation infrastructure now and leave model claims behind empirical gates.

## Follow-up issues created

- [OJ-013 / #16: query-conditioned multiscale evidence planning](https://github.com/bacoco/Omni-JEV/issues/16)
- [OJ-014 / #17: policy-first cost-aware routing](https://github.com/bacoco/Omni-JEV/issues/17)
- [OJ-015 / #18: shared snapshots, deduplication and audit traces](https://github.com/bacoco/Omni-JEV/issues/18)

These extend rather than replace the initial 13-item publication manifest. Their acceptance criteria remain open.
