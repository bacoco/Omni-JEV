# OJ-001: artifact audit and reuse boundary

Audit date: 2026-09-24. Base: `653d9a6e72277ed9144a05a171bf147352b0274d`.
Related: [issue #1](https://github.com/bacoco/Omni-JEV/issues/1).
This change is independent of the unmerged typed-contract PR #14.

**Status: a completed static-inspection and audit-tooling slice, not completion of OJ-001.**
No pretrained model passed an inference smoke test in this execution environment.
Issue #1 must remain open until the remaining acceptance criteria have evidence.
The code below does not fabricate embeddings or substitute random weights for a baseline.

## Files and evidence levels

- `audits/OJ-001/manifest.json`: seven immutable GitHub code/document revisions, inspected-file Git blob hashes, five checkpoint candidates, sources and explicit blockers.
- `audits/OJ-001/environment.json`: observed local package versions, not a portable runtime lock.
- `audits/OJ-001/validation.json`: tests actually run and what remains untested.
- `scripts/omni_jev_audit.py`: dependency-free inventory validation, opt-in HF metadata resolution and local file-integrity verification.
- `scripts/omni_jev_smoke_colmodernvbert.py`: an opt-in native integration runner; its successful path is NOT claimed executed here.
- `requirements/oj001-colmodern.in`: exact code revision with upstream dependency resolution still pending. It is intentionally not named a lock file.

The inventory distinguishes a source declaration, inspected code, server-provided hashes,
locally verified artifact bytes, and a successful inference report. These are different
levels of evidence. `validate` only checks structure and invariants. Even
`--require-ready` requires declared runtime evidence, not cryptographic proof that an
experiment occurred; inspect the referenced report and verify its digest before admission.
A Git blob ID for a README does not pin a model checkpoint. LFS pointer blob IDs do not
hash the large weight payload: `verify-snapshot` uses its SHA-256 instead.

## Findings that affect implementation

### Ovis-Omni and Ovis-VL must remain distinct

The pinned [Omni README](https://github.com/ATH-MaaS/Ovis-Omni-Embedding/blob/c46bd3603cfdf539754694a0c7b9753819386e4b/README.md)
describes a Qwen2.5-Omni-based embedding model with text, image, video and audio input,
a shared Thinker path, removed generation heads and last-non-padding-token pooling.
Its reported embedding width is 2,048. None of these tensor properties was measured here.

The pinned [VL README](https://github.com/ATH-MaaS/Ovis-VL-Embedding/blob/48019b2d9342de28af715488e66f0c31ceb8aceb/README.md)
explicitly excludes audio, including the soundtrack of a video. The 2B visual model
must not satisfy a speech-related evidence requirement merely because its input is a video.
Use the native visual embedding path as a visual-only candidate, not an audio substitute.

Both inspected READMEs still announce weight release as forthcoming. Earlier conversation
and the user reported actual VL weights under `model/`. These observations may reflect
stale documentation. In this pass, the HF file inventories were not accessible and the
execution container could not resolve `huggingface.co`. **We therefore do not conclude
that the weights are absent, were removed, or never existed.** Neither a weight revision,
a subfolder nor a weight digest has been invented. Re-resolve the official HF repositories
before implementing their loaders. A differently owned mirror is not silently substituted.

### Reuse colpali_engine instead of rebuilding ColModernVBERT

The inspected [model implementation](https://github.com/illuin-tech/colpali/blob/97487f8871ff4d5d2284411fe61bdcd2cfe99894/colpali_engine/models/modernvbert/colmodernvbert/modeling_colmodernvbert.py)
uses native Transformers `ModernVBertModel`, a trainable 128-dimensional projection,
L2 normalization and attention masking. Optional image-only masking is off by default;
do not describe all returned vectors as image patches.

The [processor](https://github.com/illuin-tech/colpali/blob/97487f8871ff4d5d2284411fe61bdcd2cfe99894/colpali_engine/models/modernvbert/colmodernvbert/processing_colmodernvbert.py)
provides `process_texts`, `process_images` and `score`, with native Idefics3 processing
and left text padding. Preserve its actual image-splitting configuration before comparing
latency or token budgets. The integration runner records it rather than assuming defaults.

The pinned [pyproject](https://github.com/illuin-tech/colpali/blob/97487f8871ff4d5d2284411fe61bdcd2cfe99894/pyproject.toml)
requires Transformers `>=5.3.0,<6.0.0`. This differs from the older
[model-card instructions](https://huggingface.co/ModernVBERT/colmodernvbert-merged)
telling readers to use the `vbert` branch. Do not mix an old recipe with current code.
The HF tree surfaced a historical full revision hint in the manifest; the hint is not
promoted to a verified weight lock or claimed to be the latest revision.

### Reuse GLiFormer/GLiClass interfaces, not interchangeable weights

[GLiFormer vision.py](https://github.com/Knowledgator/GLiFormer/blob/b5c0a0fd2aacff64736fbfa0bac0bdcc032d5eff/gliformer/encoders/vision.py)
provides vision tokens, masks, spatial shape and prefix-token metadata. Its spatial contract
is a contiguous prefix followed by a dense grid. An interleaved Ovis state is not that
contract. The adapter must preserve modality/time metadata instead of inventing a square
patch grid. `VisionBiEncoder` delegates text-label encoding to a separate media interface.
No pretrained GLiFormer vision-head checkpoint is admitted by this audit.

[GLiClass config.py](https://github.com/Knowledgator/GLiClass/blob/40baa67cdea577449bc3f6a251646377b2cb0a6c/gliclass/config.py)
exposes dynamic markers, encoder variants, projections and scorers. An unknown encoder
configuration can invoke `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
Audit the nested encoder revision and that execution path before reuse. Config inspection
alone does not certify every encoder-decoder implementation.

### Reuse Laya option semantics and Decider as a separate control

[Laya common.py](https://github.com/NandhaKishorM/laya/blob/23a17522aa4942da6cce53a995a275760320b691/laya/common.py)
constructs a state/question/options sequence and scores option markers. Sharing tokenized
state IDs is not the same as reusing encoded state memory. Its current code separates
max-probability answer confidence from normalized-entropy concentration. Preserve that
distinction; do not import claims of automatic calibration. The HF root is the English
checkpoint according to its [model card](https://huggingface.co/convaiinnovations/laya).
A multilingual variant needs its own pinned checkpoint and evaluation.

[Decider's visual path](https://github.com/Mapika/decider/blob/b44b4c9880a67291206499b86aac89004850134a/decider/vision/model.py)
uses native image-text processing, then selects language-head rows for lettered answers.
It therefore cannot simply receive an Ovis embedding checkpoint with its language head
removed. Keep it as a separate candidate control. Its HF weights and exact load path were
not verified in this pass. Do not execute the unrelated game demonstration in that file.

## Reproducible audit commands

The metadata/integrity tools use Python 3.11+ standard-library modules only. No ML package,
network access, API token or model download is required for these tests:

```bash
python -m unittest discover -s tests -p 'test_omni_jev_audit.py' -v
python scripts/omni_jev_audit.py validate audits/OJ-001/manifest.json
python scripts/omni_jev_audit.py environment
```

The following deliberately exits 2 while all native model tests remain blocked:

```bash
python scripts/omni_jev_audit.py validate audits/OJ-001/manifest.json --require-ready
```

On a network-enabled machine, explicitly resolve one model. The first request discovers
its revision; the second requests that immutable revision and checks repository identity.
Only metadata is fetched, without credentials or remote-code execution. Output files are
not overwritten. A 401, 404 or network error is recorded as a resolution failure, not proof
of non-release.

```bash
python scripts/omni_jev_audit.py resolve-hf ModernVBERT/colmodernvbert-merged \
  --output /tmp/colmodern.snapshot.json
# Or supply a full --revision obtained from a trusted artifact source.
```

Review the resulting inventory, license, configuration, tokenizer and processor. Materialize
the complete snapshot at the lock's exact revision into a local directory. Do not replace
the SHA with `main`, execute custom code while inspecting it, or copy model weights into
this Git repository. Verify the materialized files:

```bash
python scripts/omni_jev_audit.py verify-snapshot /tmp/colmodern.snapshot.json /tmp/colmodern-model
```

This command supports content SHA-256 for LFS and Git blob SHA-1 for ordinary files,
rejects incomplete metadata and path traversal, and refuses symlinks escaping the directory.
Use a materialized local snapshot, not HF cache symlinks pointing outside it. Verification
covers inventoried files only and is not a code-safety certificate.

## Native smoke runner: prepared, not validated on a checkpoint

In a separate environment, review and check out the pinned `colpali_engine` revision from
the manifest. Resolve its dependencies, capture the full installed environment, and prepare
the verified local model snapshot. The optional requirements input is not a tested full lock.

```bash
python scripts/omni_jev_smoke_colmodernvbert.py \
  --lock /tmp/colmodern.snapshot.json \
  --snapshot /tmp/colmodern-model \
  --engine-source /tmp/colpali-checkout \
  --allow-reviewed-code \
  --report /tmp/colmodern-native-smoke.json
```

The runner checks the exact clean code revision, verifies model files, disables Hub access,
uses native classes with `trust_remote_code=False` and safetensors only, and rejects non-exact
checkpoint loads. It then checks text/image output shapes, finite values, masks, unit token
norms and native MaxSim parity using a procedural image and two text strings. It records
actual processor settings, device, precision and shapes. No label-accuracy assertion or
Noul/Choice/Score behavior is implied. No audio/video support is asserted by this runner.
Its successful ML execution path remains an integration test to perform, not a tested adapter.

## License and readiness gates

Code license observations: MIT for colpali_engine; Apache-2.0 headers for GLiFormer,
GLiClass, Laya and Decider; Apache-2.0 declarations in the Ovis READMEs. Evidence paths
and file hashes are in the manifest. This is an inventory, not a legal clearance opinion.
Weight licenses, tokenizer/base-model obligations, redistribution terms and notices still
need review at each selected artifact revision. No upstream source or model weights are
vendored by this PR, and no implementation license is assigned to Omni-JEV.

Before completing issue #1:

1. Resolve and verify at least one complete model snapshot, including native configs and
   any nested dependencies; produce a resolved environment lock.
2. Run the native text/image smoke test and record actual tensor/processor properties.
3. Audit Ovis loaders and their text, image, audio, video and joint paths with permitted
   fixtures, including audio-on/off controls; mark untested paths explicitly.
4. Attach reports and hashes, then update the capability manifest. Only then use it to
   unblock the native adapter in issue #4. No upstream retrieval score substitutes for this.
