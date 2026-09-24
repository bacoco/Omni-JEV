# Initial architecture decision register

Date: 2026-09-24. "Accepted goal" denotes the user's product requirements; "provisional" denotes a design hypothesis awaiting experiments.

| ID | Status | Decision | Consequence / revisit trigger |
|---|---|---|---|
| ADR-001 | Accepted goal | Build a typed multimodal decision model, not another retrieval-only system | Evaluate proposition/choice/rubric outcomes directly |
| ADR-002 | Accepted goal | Preserve Noul, Choice, and Score meanings | Use transport/status metadata for unavailable evidence; do not silently add a fourth primitive |
| ADR-003 | Accepted goal | Reuse pretrained components and existing code aggressively | Audit exact interfaces/licenses and prefer adapters to rewrites |
| ADR-004 | Accepted goal | Repository artifacts are English | Evaluate target languages separately; French is a relevant candidate |
| ADR-005 | Provisional | Evaluate Ovis-Omni first for full modality scope | Switch or add a fallback if G0 fails or measurements favor another backbone |
| ADR-006 | Provisional | Preserve a native global embedding baseline | Retain unless it cannot be reproduced; no token-memory superiority assumption |
| ADR-007 | Provisional | Compare native text and BERT-like condition branches | Select from E05, including alignment/training cost |
| ADR-008 | Provisional | Compare global, segment, and token memories | Select from task quality and full cost, not token count alone |
| ADR-009 | Provisional | Compare MaxSim with small cross-attention | Hold backbone/data constant and test relations explicitly |
| ADR-010 | Accepted engineering principle | Keep reusable state memory independent of later questions where feasible | Profile true reuse; include model/processor revisions in cache keys |
| ADR-011 | Accepted engineering principle | Train supervised baselines before RL or large end-to-end runs | Escalate complexity only on measured error analysis |
| ADR-012 | Accepted engineering principle | Calibrate and measure abstention separately | No automatic reliability claim from normalized outputs |
| ADR-013 | Provisional | ColModernVBERT is a separate compact candidate | Distill only after a reference model and scoped dataset exist |
| ADR-014 | Accepted engineering principle | Do not require a vector database for direct state evaluation | Add indexing only for an explicit corpus-search requirement |
| ADR-015 | Accepted research principle | No "best model" claim without matched decision benchmarks | Source-reported retrieval results remain context only |

## New code expected

A model-specific adapter, typed question/result contracts, the interface between state and condition representations, experiment orchestration, and decision-focused training/evaluation glue. The goal is not zero new code: it is to keep new code at the actual missing interfaces while retaining upstream components where they fit.

## Changes require a recorded reason

An update should identify the experiment or failure motivating it, affected interfaces and caches, new assumptions, and whether the change alters the product contract. A benchmark failure is a valid reason to abandon a favored backbone, including Ovis. A successful simpler baseline is a valid reason not to build a more elaborate model.
