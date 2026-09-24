# Prioritized development backlog

Status: all 13 planning issues have been published. Task issues are #1 through #12; the synthesis is #13. Stable planning IDs remain distinct from GitHub issue numbers. Each task includes scope, dependencies, acceptance criteria, and expected evidence.

The [project synthesis](https://github.com/bacoco/Omni-JEV/issues/13) is the self-contained umbrella issue. Versioned local drafts remain under `issues/`.

| Planning ID | Priority | Issue | Dependencies |
|---|---|---|---|
| OJ-001 | P0 | [Audit and pin candidate models, processors, licenses, and reusable code](https://github.com/bacoco/Omni-JEV/issues/1) | None |
| OJ-002 | P0 | [Specify and test Noul, Choice, and Score contracts](https://github.com/bacoco/Omni-JEV/issues/2) | None |
| OJ-003 | P0 | [Build the leakage-resistant dataset and evaluation protocol](https://github.com/bacoco/Omni-JEV/issues/3) | OJ-002 |
| OJ-004 | P0 | [Implement a native-processing multimodal state adapter](https://github.com/bacoco/Omni-JEV/issues/4) | OJ-001, OJ-002 |
| OJ-005 | P0 | [Establish global-embedding, text-pipeline, and existing-model baselines](https://github.com/bacoco/Omni-JEV/issues/5) | OJ-003, OJ-004 |
| OJ-006 | P1 | [Add a BERT-like dynamic question and option encoder](https://github.com/bacoco/Omni-JEV/issues/6) | OJ-002, OJ-004, OJ-005 |
| OJ-007 | P1 | [Implement discriminatively trained ColBERT-style late interaction](https://github.com/bacoco/Omni-JEV/issues/7) | OJ-006 |
| OJ-008 | P1 | [Compare a small cross-attention decision module with MaxSim](https://github.com/bacoco/Omni-JEV/issues/8) | OJ-006, OJ-007 |
| OJ-009 | P1 | [Build staged supervised fine-tuning and reproducibility tooling](https://github.com/bacoco/Omni-JEV/issues/9) | OJ-007, OJ-008 |
| OJ-010 | P1 | [Evaluate audiovisual relationships and temporal counterfactuals](https://github.com/bacoco/Omni-JEV/issues/10) | OJ-003, OJ-004, OJ-005 |
| OJ-011 | P1 | [Calibrate outputs and define evidence-aware abstention](https://github.com/bacoco/Omni-JEV/issues/11) | OJ-009, OJ-010 |
| OJ-012 | P2 | [Measure compression, cache economics, and compact alternatives](https://github.com/bacoco/Omni-JEV/issues/12) | OJ-009, OJ-010, OJ-011 |

## Suggested phases

**Foundation:** OJ-001 through OJ-005. Establish artifacts, contracts, data, native processing, and useful baselines.

**Decision architecture:** OJ-006 through OJ-009. Test BERT-like conditions, late interaction, attention, and adaptation. OJ-010 can start alongside this phase once its dependencies are met.

**Reliability and efficiency:** OJ-011 and OJ-012. Select calibrated operating points and then test compression/compact models.

## Publishing

The manifest is `issues.json`. The helper at `scripts/omni_jev_publish_issues.py` previews by default and requires `--apply` for writes using an authenticated GitHub CLI. It reads the actual repository and existing issues first, and recognizes its own stable issue markers on serial reruns. It does not push code, modify settings, add labels, assign people, close issues, or overwrite existing issue content. See the root publication guide.
