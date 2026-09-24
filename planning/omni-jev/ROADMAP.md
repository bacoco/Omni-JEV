# Prioritized development backlog

Status: 13 local issue drafts, not published GitHub issues. Stable planning IDs do not represent actual GitHub issue numbers. Each draft includes scope, dependencies, acceptance criteria, and expected evidence.

The [project synthesis](issues/OJ-000.md) is the self-contained umbrella issue.

| Planning ID | Priority | Issue | Dependencies |
|---|---|---|---|
| OJ-001 | P0 | [Audit and pin candidate models, processors, licenses, and reusable code](issues/OJ-001.md) | None |
| OJ-002 | P0 | [Specify and test Noul, Choice, and Score contracts](issues/OJ-002.md) | None |
| OJ-003 | P0 | [Build the leakage-resistant dataset and evaluation protocol](issues/OJ-003.md) | OJ-002 |
| OJ-004 | P0 | [Implement a native-processing multimodal state adapter](issues/OJ-004.md) | OJ-001, OJ-002 |
| OJ-005 | P0 | [Establish global-embedding, text-pipeline, and existing-model baselines](issues/OJ-005.md) | OJ-003, OJ-004 |
| OJ-006 | P1 | [Add a BERT-like dynamic question and option encoder](issues/OJ-006.md) | OJ-002, OJ-004, OJ-005 |
| OJ-007 | P1 | [Implement discriminatively trained ColBERT-style late interaction](issues/OJ-007.md) | OJ-006 |
| OJ-008 | P1 | [Compare a small cross-attention decision module with MaxSim](issues/OJ-008.md) | OJ-006, OJ-007 |
| OJ-009 | P1 | [Build staged supervised fine-tuning and reproducibility tooling](issues/OJ-009.md) | OJ-007, OJ-008 |
| OJ-010 | P1 | [Evaluate audiovisual relationships and temporal counterfactuals](issues/OJ-010.md) | OJ-003, OJ-004, OJ-005 |
| OJ-011 | P1 | [Calibrate outputs and define evidence-aware abstention](issues/OJ-011.md) | OJ-009, OJ-010 |
| OJ-012 | P2 | [Measure compression, cache economics, and compact alternatives](issues/OJ-012.md) | OJ-009, OJ-010, OJ-011 |

## Suggested phases

**Foundation:** OJ-001 through OJ-005. Establish artifacts, contracts, data, native processing, and useful baselines.

**Decision architecture:** OJ-006 through OJ-009. Test BERT-like conditions, late interaction, attention, and adaptation. OJ-010 can start alongside this phase once its dependencies are met.

**Reliability and efficiency:** OJ-011 and OJ-012. Select calibrated operating points and then test compression/compact models.

## Publishing

The manifest is `issues.json`. The helper at `scripts/omni_jev_publish_issues.py` previews by default and requires `--apply` for writes using an authenticated GitHub CLI. It reads the actual repository and existing issues first, and recognizes its own stable issue markers on serial reruns. It does not push code, modify settings, add labels, assign people, close issues, or overwrite existing issue content. See the root publication guide.
