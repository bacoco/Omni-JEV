# OJ-016: prefix-cached causal decision baseline

Implementation issue: [#21](https://github.com/bacoco/Omni-JEV/issues/21).
Depends on the typed contracts in PR #14; does not replace the evaluation work in PR #20.

**Status:** text-only reference implementation and offline tests. No trained checkpoint
has been executed in the preparation environment. This is an independent design,
not a claim about the internal implementation of TypeSafe Jev.

## What this implements

The owner proposed caching the task's instructions, output structure and candidate
meaning, appending a new input, and reading only the answer values. This backend
implements the direct next-token scoring variant:

    instructions + typed task + candidate descriptions -> reusable causal KV prefix
    prefix clone + new inline source text + answer slot -> candidate-token logits
    candidate-local normalization -> Noul / Choice / Score -> JSON built in Python

The fixed prefix is first computed on a cache miss. Subsequent compatible requests
reuse it. The new input still requires a forward pass and attends to the prefix.
Tokens in the cached schema are never retroactively filled with input-dependent values.

The model does not generate JSON keys, braces, explanations or a self-reported
confidence number. It supplies logits for the next token; the scorer selects the
allowed codes and the existing `decoding.py` creates all three typed outputs.
`greedy_code` is the argmax restricted to those codes, not a second generative run.
`values.jsonl` exports just values plus status metadata; complete distributions remain
in the typed prediction files.

This is **not yet** a full grammar-constrained JSON generator, a server, a fine-tuning
pipeline, a vLLM/SGLang integration or an image/audio/video adapter. Those alternatives
can be compared later rather than confused with implemented capabilities.

## Decisions and output semantics

Noul uses two codes for no/yes and returns P(yes | one of those codes). Choice uses
request-time descriptions and a distribution over the supplied alternatives. Score
uses described ordered levels and returns their probability-weighted zero-based index.
The complete distribution is retained. No fixed business-class vocabulary is learned.

Codes A through Z must each be a distinct, reversible, non-special single tokenizer
token. Unsupported tokenizers or more than 26 options fail explicitly. This is an
initial implementation limit, not a theoretical limit or a change to the public contracts.
Choice options are canonically ordered by ID; Score retains rubric order. Exact logit
ties use lexical candidate IDs, as in the existing decoder. Code/label bias still needs
measurement with a trained model: canonicalization is not proof of neutral semantics.

`allowed_token_mass` records how much full-vocabulary next-token probability belongs
to the codes before renormalization. A forced 0.99 inside a tiny allowed mass is not
evidence of 99% correctness. No calibration is fitted and no abstention threshold is
invented. This metric is diagnostic, not a permission to act.

## Cache correctness and limits

- The complete native chat prompt is rendered and tokenized. The reusable part is the
  **exact token-wise common prefix**, not a character-count slice. Boundary BPE merges
  are therefore handled without assuming that encode(A)+encode(B)=encode(A+B).
- Native chat templates are the default. `--mode plain` is an explicit alternative for
  a base model without a suitable template, not a silent fallback. Thinking is requested
  off where the template understands that option; templates that ignore it still require
  task-quality evaluation. Data-dependent template branching is rejected.
- Each stored cache is cloned before continuation because cache objects may mutate.
  A/B/A input sequences must not leak B into the second A. No suffix/document cache is
  retained across states by this implementation.
- Keys include exact prefix token IDs, backend artifact identity and the caller's access
  scope. Model weights/training mode/config changes invalidate the instance. Reload a
  new backend for new weights; deliberately bypassing PyTorch's mutation tracking with
  `.data` is unsupported. Treat loaded models/tokenizers as immutable.
- The process-local LRU is bounded by entries, retained prefix tokens and tensor-storage
  bytes. Retained-cache limits do not bound peak model memory, a temporary new cache or
  its per-request clone. Oversized prefixes either bypass caching or are used transiently
  without retention; the traces expose that decision.
- A single engine serializes its calls. This is not a distributed cache, authentication
  system or production scheduler. A service wrapper must derive access scope from an
  authenticated caller instead of trusting arbitrary tenant strings.
- In this first version **each question re-encodes its source suffix**. Reusing a task
  prefix across many documents is different from encoding one document once for many
  questions. Full prompt tokenization also still occurs to validate the boundary; its
  cost is included in request wall time. No multi-question sharing speedup is claimed.

Input over the declared context budget is rejected, never silently truncated. Only
explicitly scoped inline text is processed. Missing sources yield insufficient evidence;
media or URI-only sources yield unsupported modality. An inline hash mismatch yields
an error. These states contain no fabricated answer. An unrelated image in the request
is not processed when a question explicitly refers only to the text source.

Literal tokenizer role/control delimiters in source text are rejected. Semantic prompt
injection remains an evaluation problem. The engine executes no tools and authorizes no
side effects; prompt instructions are not a security boundary.

## Local model loading

`load_local_backend` hashes the local artifact bytes, refuses symlinks, uses local-only
loading with remote code disabled and loads safetensors only. It refuses partial or
mismatched checkpoint loads. The digest identifies bytes, **not their publisher or
upstream revision**. Verify the origin and license with the OJ-001 audit tooling before
using a real checkpoint. Materialize HF snapshot symlinks into regular files first.

The initial allowlist covers standard, unquantized, full-attention GPT-2, Llama, Qwen2
and Qwen3 configurations without scaled rotary positions. It excludes embedding-only
Ovis checkpoints, multimodal architectures, encoder-decoder models, sliding/hybrid
state and quantized loading. These are candidate loader families, not all runtime-
validated families. MPS and multi-device sharding are not implemented here.

The reference uses eager attention and a single CPU/FP32 or CUDA device. When the
model exposes `logits_to_keep` or `num_logits_to_keep`, only the final position's logits
are requested. Otherwise its native forward may compute all position logits; this is
reported rather than misrepresented as an optimized kernel. The code uses the native
language head; it does not train an embedding projection.

## Run on a user-owned machine

Install into an isolated environment that already has an appropriate CPU/CUDA PyTorch
build. The optional dependency range is an integration candidate, not a fully resolved
runtime lock. The CPU CI recipe explicitly selects PyTorch 2.10.0 and Transformers
5.3.0; those versions exist upstream, but the complete recipe could not be installed
in the preparation environment. Save an environment lock with each actual run.

```bash
python -m pip install -e '.[prefix,test]'
python -m pytest -q tests/test_prefix_cache.py tests/test_prefix_torch.py tests/test_prefix_transformers.py

python -m omni_jev.prefix_benchmark \
  --snapshot /path/to/materialized-causal-checkpoint \
  --requests examples/prefix-cache/requests.jsonl \
  --output-dir prefix-results \
  --device cpu --dtype float32 --repeats 3
```

The requests contain two illustrative messages and three kinds of questions. They are
not an independently reviewed quality dataset. On a GPU, use `--device cuda`; choose
precision and an explicit `--atol` only after inspecting numerical differences. The
output directory must not already exist. No credential or Hub token is required by
this local-only runner. The CLI never downloads a model.

Outputs:

| File | Meaning |
| --- | --- |
| `report.json` | Environment/artifact identities, all paired runs, parity and observed costs |
| `cached-predictions.jsonl` | `{case_id,response}` envelopes, one question per example |
| `uncached-predictions.jsonl` | Matching uncached predictions for the same cases |
| `values.jsonl` | Program-reconstructed values and explicit per-question statuses |

Prediction envelopes match the evaluation module's format; a corresponding reviewed
benchmark manifest is still needed for quality metrics. Sample responses and identifiers
may themselves be sensitive. Reports omit raw source text but must remain access-controlled.

## What the comparison measures

The uncached path and cached path use identical full token IDs and candidate mapping.
Order alternates by repeat; each sample is retained so first-use effects remain visible.
Prefix misses, warm hits, prefix prefill, clone cost, score forward, computed/reused tokens
and request wall time are separate. Model hashing/loading has its own timer. CUDA timing
synchronizes the device. Median timing alone is not a statistical speedup claim.

Numerical parity checks both candidate logits and decoded probabilities, exact prompt
identity and discrete choices. An empty/unscored comparison cannot pass. Unsupported
cases are listed, not dropped as successful model tests. The runner does not compare
against gold labels, so `quality_evaluated` and `calibration_fitted` remain false even
when cache parity passes.

This reference does not yet report GPU peak memory or tokenization as a separate stage.
Retained KV tensor bytes are not total device usage. Add a measured serving-engine
comparison before making throughput or deployment latency claims.

## Validation performed here

See `audits/OJ-016/validation.json`. Standard-library mechanism tests and real PyTorch
causal-attention tests pass offline. The latter use a tiny randomly initialized network,
not a pretrained LLM. The optional native GPT-2/Llama tests were skipped because
Transformers is absent. No real checkpoint or model-quality benchmark ran. Local DNS
and the download tool could not fetch dependencies. Hosted CI must be checked separately.

The pretrained acceptance gate of issue #21 remains open. First validate a compatible
checkpoint, then use labeled holdouts to compare cache/no-cache and the earlier
BERT/multivector alternative. Caching should change cost, not the intended decision.

## Primary references and reuse

Implementation reuses the project's existing contracts and decoder, the PyTorch model
interface and Transformers cache/loading/chat APIs. No upstream source or weight is
vendored and no new project-wide license is selected.

- [Transformers prefix-cache example](https://huggingface.co/docs/transformers/en/kv_cache#prefill-a-cache-prefix-caching): prefill and independent continuation copies.
- [Causal cache explanation](https://huggingface.co/docs/transformers/en/cache_explanation): suffix attention, positions and cached past state.
- [Chat templating](https://huggingface.co/docs/transformers/en/chat_templating): native control tokens and generation markers.
- [Model loading](https://huggingface.co/docs/transformers/en/main_classes/model): local-only/remote-code options and loading information.
- [vLLM generative scoring](https://docs.vllm.ai/en/latest/serving/online_serving/generative_scoring/): related next-token label scoring; not an integration shipped by this PR.

References inspected on 2026-09-24. They explain mechanisms; they are not evidence that
our selected checkpoint, prompt or deployment meets a quality or performance target.
