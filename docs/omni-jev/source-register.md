# Source register and evidence status

Prepared on **2026-09-24**. Links identify primary project/model documentation unless marked otherwise. Source descriptions are deliberately brief; this package's architecture and experiment plan are proposals, not findings borrowed from a benchmark.

## Evidence labels

- **Checked documentation:** readable primary content was retrieved during preparation; no runtime result is implied.
- **Partial verification:** a primary project/search/paper listing supports the general lead, but the exact artifact was not independently loaded or fully inspected.
- **Conversation lead:** preserved from the preceding discussion; details require verification before use.

A browser cache miss or access error is not evidence that a repository is absent, private, deleted, or unreleased. It is an access limitation. The target `bacoco/Omni-JEV` could not be inspected through the unauthenticated browsing attempt, and the GitHub integration was not connected when this package was prepared. No existing files, branch, settings, issues, or permissions were established. This package must not be represented as already committed or published.

## S01 — TypeSafe Jev interface

**Checked documentation.** The introduction describes typed judgments against a state and the Noul/Choice/Score family. It does not establish Omni-JEV's implementation or latency.

https://docs.typesafe.ai/introduction

## S02 — Noul

**Checked documentation.** Probability of yes, optional true/false criteria, and application-side thresholding. No separate Noul confidence field is documented in this source.

https://docs.typesafe.ai/primitives/noul

## S03 — Choice

**Checked documentation.** One option from a supplied set, option probabilities, and confidence. Exact interoperability is a separate implementation test.

https://docs.typesafe.ai/primitives/choice

## S04 — Score

**Checked documentation.** A rubric position computed as the expected index of described levels; the distribution is retained. Different distributions can have the same expected index.

https://docs.typesafe.ai/primitives/score

## S05 — Ovis-Omni project

**Partial verification.** An indexed primary GitHub README describes text, image, video, and audio embeddings. Direct page retrieval was inconsistent. No weights, precise revision, inference wrapper, parameter tensor, license file, or GPU run was verified for the exact checkpoint in this preparation pass.

https://github.com/ATH-MaaS/Ovis-Omni-Embedding

Requested artifact to audit:
https://huggingface.co/ATH-MaaS/Ovis-Omni-Embedding-3B

## S06 — Ovis family paper listing and visual checkpoint

**Partial verification.** The Hugging Face paper listing identifies the family and lists Ovis-Omni-Embedding-3B and Ovis-VL-Embedding-2B model entries. Exact cards/files for these two endpoints were not retrievable in this pass. Earlier conversation reported weights and architecture details; those reports remain useful leads, not a newly reproduced verification. Do not substitute a third-party mirror without checking provenance.

https://huggingface.co/papers/2609.25165
https://huggingface.co/ATH-MaaS/Ovis-VL-Embedding-2B
https://github.com/ATH-MaaS/Ovis-VL-Embedding

## S07 — ColBERT

**Checked documentation.** Primary paper describing contextualized late interaction over BERT for retrieval. Its objective is not an existing guarantee of typed decision accuracy.

https://arxiv.org/abs/2004.12832

## S08 — ColPali / colpali_engine

**Checked documentation.** Upstream repository for training/inference of visual document retrievers and multi-vector scoring. Reuse should be revision-pinned, including masking and score behavior.

https://github.com/illuin-tech/colpali

## S09 — GLiClass and GLiFormer

**GLiClass checked; GLiFormer a conversation lead for exact implementation details.** GLiClass documentation describes BERT-like arbitrary-label classification and multiple encoder/scorer designs. GLiFormer's linked code and documentation were not retrievable in this pass; inspect its actual vision feature contract and released heads before choosing it as the framework.

https://docs.knowledgator.com/docs/frameworks/gliclass/intro/
https://github.com/Knowledgator/GLiFormer

## S10 — Laya

**Checked model card, not executed code.** The card documents English ModernBERT and multilingual variants, typed questions, and option-marker scoring. Author claims about calibration, generality, and latency are not adopted as Omni-JEV results. Runtime batching and cache semantics require inspection.

https://huggingface.co/convaiinnovations/laya

## S11 — Qwen2.5-Omni processing

**Checked documentation.** Related backbone documentation includes explicit handling of audio from video. This does not establish that an Ovis-specific wrapper accepts identical arguments or works without adaptation.

https://huggingface.co/docs/transformers/model_doc/qwen2_5_omni

## S12 — ColModernVBERT

**Checked model-card endpoint; not executed.** Retain as a compact document retrieval comparison. Verify its precise current feature dimensions, load path, language behavior, and license before reuse.

https://huggingface.co/ModernVBERT/colmodernvbert-merged

## S13 — Decider-Vision

**Conversation lead.** The prior discussion identified a visual typed-decision model and model implementation. Retrieval attempts here failed; do not claim that these artifacts have been loaded or that their head can replace an Ovis embedding head directly.

https://huggingface.co/Mapika/decider-2b-vision
https://github.com/Mapika/decider

## S14 — GitHub publication tooling

**Checked documentation.** The publication helper uses authenticated GitHub CLI API calls. It is dry-run by default and requires an explicit apply flag for writes. The helper was only tested locally with mocked GitHub responses, not against the target repository.

https://cli.github.com/manual/gh_api
https://cli.github.com/manual/gh_issue_create

## Required artifact manifest before a model becomes a dependency

Record source owner, repository/model ID, immutable commit or revision, file list and checksums, model and code licenses, required versions, remote-code requirements, processor/chat template, supported input paths, tensor shapes, sampling limits, pooling rule, device/dtype tested, smoke-test output, and reproducibility limitations. Do not copy benchmark numbers without task definitions and whether they are author-reported or independently reproduced.

## Source policy for future updates

Prefer primary repositories, model cards, papers, and official libraries. Preserve dates and revisions. Keep source claims, project hypotheses, and measured results distinct. Report inability to verify rather than turning a prior conversational claim into an implementation fact. Additional candidates in the discussion record require their own source audit if promoted into experiments.
