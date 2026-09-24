# Preserved discussion: design evolution and alternatives

This is an English consolidation of the substantive analysis supplied in the project conversation. It is not a verbatim transcript, a record of private reasoning, or a claim that every previously mentioned model was independently verified. It preserves motivations, corrections, competing designs, and experimental implications. See [source-register.md](source-register.md) for evidence status.

## 1. Original motivation: visual conditions rather than another search engine

The original proposal was to extend a Jev-style text judgment model to images. A high-quality image encoder would supply one or several embeddings; text-defined flags/tags would drive software conditions. The central question was whether this adds value beyond ColPali.

The useful distinction was retrieval relevance versus proposition evaluation. A retrieved page can contain all the relevant words or objects without satisfying the relationship in the question. A signature in the wrong region, a checked box next to a different label, and a negated statement are examples. This does not prove that ColPali fails; it identifies tests a decision product needs beyond retrieval scores.

The product was clarified as a discriminative evaluator, not literal extraction of `if/else` syntax and not free-form answer generation. Program logic remains outside the network.

## 2. The user clarified that pretrained encoding is foundational

Using an existing image encoder was never an optional discovery; it is a project constraint. The intended novelty is the decision interface and its fine-tuning, not building a visual encoder from scratch.

A BERT-like branch for textual conditions was proposed explicitly, with GLiClass, GLiFormer, ColBERT, and ColModernVBERT as references. The resulting design is asymmetric when a separate BERT-like model is used: one state encoder and one condition encoder need a learned interaction/alignment. Another legitimate comparison keeps a shared/native text branch.

GLiClass documents several architectures, including separated and fused encoders, with request-time label descriptions [S09]. Its role here is a reusable classification design, not proof that a particular pretrained head already works on Ovis features. GLiFormer's exact vision APIs and current weights remain an audit item.

## 3. Laya and typed option scoring

The user asked to include an open Jev-like project, referred to conversationally as Laia/Laya. The Laya model card documents ModernBERT and a separate multilingual variant, with typed questions and option-marker scoring [S10]. This is an important source of interface and training ideas.

Two cautions were retained. First, batching several question-state sequences is not identical to caching one state representation. Second, a proper-scoring-rule objective does not guarantee calibrated probabilities on a new distribution. The project must inspect actual input construction, heads, masks, and calibration rather than copy claims about speed or certainty.

## 4. Two different fine-tuning targets

One route adapts a ColBERT-like MaxSim model directly with decision supervision. This is more substantive than placing a threshold on an unchanged retrieval score: gradients can change the representations and interface.

The second route uses a small attention-based module where conditions inspect a reusable state memory. Its hypothesis is better handling of binding, negation, conjunctions, and ordered events. It is not accepted as the winner; the controlled comparison is part of the plan.

A third reference is a simple global embedding scorer. Keeping it prevents attributing all gains to multi-vector complexity when task supervision alone may explain them.

## 5. Single vector, multiple vectors, and compression

The discussion separated raw/contextualized token states, retrieval-aligned multi-vector embeddings, and a compressed decision memory. These are not interchangeable. Multiple vectors do not imply MaxSim, a vector database, or a search index.

The provisional preference was to avoid discarding local information before testing fine conditions. This is not a claim that a single vector cannot represent multiple attributes. The global output deserves an explicit baseline, especially when it is the representation directly trained by an embedding checkpoint.

Compression has three separate axes: number of tokens, width of each vector, and numerical precision. A learned bottleneck/resampler, spatial/temporal pooling, a task projection, and quantization should be tested independently. Dropping small visual details during initial resizing cannot be repaired by a later classifier.

A useful extension is one native embedding per aligned segment. This retains more of an upstream model's published interface while giving a sequence of representations. Its costs and cross-segment limitations need measurement.

## 6. Ovis changed the proposed primary backbone

The user supplied the Ovis-VL-Embedding-2B repository and corrected an earlier claim that weights were not released. The discussion then prioritized Ovis, while retaining its complete multimodal path rather than assuming that its vision tower alone preserves the embedding model's capabilities.

The important distinction is native global output versus a newly introduced token-memory output. Prior discussion described last-token pooling and specific widths, but exact tensor paths and dimensions are not hard-coded as established facts in this package. Loading, configuration, file layout, revisions, and licenses must be audited on the actual artifact.

The user's demand to reuse as much existing work as possible led to a composition strategy: Ovis for encoding; GLiClass/GLiFormer for task/label structure where suitable; ColBERT/colpali_engine for late-interaction components; Laya for typed scoring; and separate compact model comparisons. Do not serially stack all complete models.

## 7. Expansion to audio and video

The scope expanded from images to video/text, spoken-content questions, and relations between audio and video. Ovis-Omni is the main candidate for this expanded scope; an official upstream project describes text/image/video/audio embedding support [S05]. Direct inference readiness is still a separate gate.

The conceptual task split matters more than the word "multimodal":

- Video plus a proposition asks whether content or an ordered action is visible.
- Audio plus a proposition may ask about words, topic, music, or environmental sound.
- Audio plus video may ask for broad semantic coherence or precise temporal correspondence.

A transcript-plus-text-model pipeline is a meaningful reference for questions about spoken words, but a transcript restricted to words cannot carry omitted environmental sounds or exact visual synchrony. A frames-only video pipeline does not exercise an audio encoder. Related Qwen2.5-Omni documentation includes explicit audio-from-video processing [S11]; the Ovis wrapper must be tested separately.

## 8. All three Jev outputs remain central

The final clarification was to retain Noul, Choice, and Score, not collapse the product into yes/no tagging. These meanings are checked against TypeSafe's documentation [S01-S04]. Dynamic labels are supplied per request. An ordinal score uses described levels, not an arbitrary similarity value. Several compatible tags use separate binary decisions.

Earlier suggestions of true/false/unknown are incorporated as evidence availability and abstention metadata, rather than silently replacing the three-family contract. A standalone three-class entailment task can still be a training/evaluation dataset, but it is not imposed as a new public primitive.

## 9. Candidate landscape preserved from the conversation

This table records leads, not a current ranking or verified release inventory. A lead can be investigated without becoming a project dependency. Historical performance numbers and release-age claims have deliberately not been frozen into the plan.

| Candidate or method | Reason it was discussed | Current project treatment |
|---|---|---|
| Ovis-Omni-Embedding-3B | Full text/image/audio/video direction | Principal candidate, gated by artifact audit |
| Ovis-VL-Embedding-2B | User-selected visual embedding starting point | Required visual-only comparison if artifact passes |
| ColPali / ColQwen variants | Document multi-vector retrieval | Reusable scoring code and visual baselines |
| ColModernVBERT | Vision + BERT-like compact document encoder | Separate compact comparator; possible later student |
| GLiClass / GLiFormer | Dynamic-label classification and modular encoders | Candidate task/head framework; audit exact current APIs |
| Laya | Open typed-decision model design | Text baseline and request-time option scoring reference |
| Decider / decider-2b-vision | Alleged existing visual typed-decision path | Availability/code check before use; not assumed runnable |
| NeoMME-260M / 800M | Earlier proposed compact document-oriented alternative | Deferred candidate; details not revalidated here |
| SigLIP 2 | Modular vision-language representations | Alternative visual-only track if needed |
| DINOv3 | Fine visual/local features | Specialist exploration; text alignment is not assumed |
| Qwen3-VL-Embedding-2B / 8B | Global multimodal embedding baseline | Optional independent comparison after audit |
| DeepSeek-OCR / DeepSeek-OCR 2 | Text-rich document encoding/compression | OCR-feature alternative, not automatically retrieval-aligned |
| RAM / RAM++ | Existing broad image tagging | Baseline for the coarse tagging subproblem only |
| CLIP / SigLIP-style classification | Text-described categories without task-specific generation | Cheap reference for appropriate image tasks |
| Visual entailment / SNLI-VE / EVE | Image plus assertion as a classification task | Related task framing, not proof of production generality |
| BLIP-2 Q-Former / latent resamplers | Learned compression of token memories | Optional bottleneck experiment after reference quality |
| Nemotron ColEmbed / Vultron retrievers | Earlier high-capacity document candidates | Deferred; exact model/version/license/metrics need audit |

The spoken term "Dinnerformer" was previously interpreted as GLiFormer, but this is not a verified project alias. Preserve it as an unresolved name rather than invent a dependency.

## 10. Corrections that should survive implementation

Embedding width equality does not establish cross-model alignment. Hidden states are not necessarily already trained multi-vector retrieval embeddings. Multi-vector output does not automatically provide spatial or temporal proof. Model scope does not guarantee that the actual preprocessing passes all modalities. A framework's vision classes do not prove that a specific released checkpoint includes trained vision heads.

Native global embeddings, richer states, and task supervision must be separated experimentally. Proper scoring rules and normalized probabilities do not remove the need for calibration tests. Retrieval rankings are not rankings for typed decision quality. Preserve the inexpensive baseline and prefer measured simplicity over assumed architectural sophistication.

## 11. What remains intentionally open

The exact checkpoint revisions, framework choice, device budget, dataset domains and sizes, performance targets, feature layer, compression budget, calibration method, and software license are unresolved. The initial issues convert these unknowns into tests and explicit decisions. The project's first milestone is reproducibility and a discriminating benchmark, not a speculative claim that one newly released embedding model solves all conditions.
