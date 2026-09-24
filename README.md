# Omni-JEV

**Typed decisions over text, images, video, and audio, built by reusing pretrained components.**

This is the initial research and development plan for `bacoco/Omni-JEV`. It is a planning bootstrap, not a trained model release. No checkpoint was executed and no model benchmark was reproduced when preparing this package.

The target is a question-conditioned, non-generative decision model. Encode an input state, evaluate natural-language questions, and return one of three output families: **Noul**, **Choice**, or **Score**. Deterministic application code then implements `if`, `else`, and `case` logic.

## Start here

- [Executive synthesis and reading guide](docs/omni-jev/README.md)
- [Architecture and typed output contract](docs/omni-jev/design.md)
- [Experiments, metrics, and decision gates](docs/omni-jev/experiments.md)
- [Preserved discussion and candidate landscape](docs/omni-jev/discussion-record.md)
- [Source register and verification limits](docs/omni-jev/source-register.md)
- [Initial architecture decisions](docs/omni-jev/decisions.md)
- [Prioritized issue backlog](planning/omni-jev/ROADMAP.md)
- [Publication instructions](PUBLISHING.md)

## Initial direction

Evaluate **Ovis-Omni-Embedding-3B** as the principal candidate for the full modality scope. Keep **Ovis-VL-Embedding-2B** as a visual-only comparison. Preserve the native global embedding baseline before adapting intermediate states into a multi-vector memory. Compare a native text branch with a BERT-like condition encoder; compare discriminatively trained late interaction with a small cross-attention decision module.

Reuse components and interfaces from **GLiClass/GLiFormer**, **ColBERT/colpali_engine**, and **Laya** where their exact versions and licenses permit. Evaluate **ColModernVBERT** as a separate compact document model, not another large model stacked after Ovis. These are provisional experimental choices, not claims of established superiority. See the source register for availability checks and unresolved integration details.

## Fixed product constraints

1. Keep the three typed output families and request-time descriptions of options or rubric levels.
2. Treat text, image, video, and audio as the target input scope, including relationships between modalities.
3. Reuse existing pretrained weights, processing code, training utilities, and evaluation tools wherever justified.
4. Preserve spatial and temporal information until measurements justify compression.
5. Write repository documentation, code, and issue discussions in English. Evaluate language support rather than assuming it from documentation language.

Omni-JEV is an independent project. The name does not imply affiliation with TypeSafe AI or any upstream model provider. An implementation license and compatibility claim have not been selected by this planning package.
