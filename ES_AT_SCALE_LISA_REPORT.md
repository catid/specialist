# ES-at-Scale Audit and Qwen3.6 Layer-Location Experiments

Date: 2026-07-13
Target: Qwen3.6-35B-A3B, 40-layer hybrid MoE
Hardware: 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (97,887 MiB each)

## Outcome

The local SGLang-resident ES design is a better foundation than the upstream
implementation: it uses antithetic sampling, deterministic per-unit noise, and
persistent FP32 masters rather than repeatedly adding and subtracting noise in
BF16. The audit nevertheless found correctness bugs in rank shaping, resume,
targeted replay, checkpoint export, replica verification, evaluation, and data
splitting. Those high-risk paths have been repaired or made fail-loud.

The strongest low-risk test of the proposed front/back-ensemble prior is a
location-restricted ES sweep over complete four-layer motifs. On Qwen3.6 the
motif is `[linear attention, linear attention, linear attention, full attention]`.
Selecting dense front+back units, including the GDN input projections, updates
about 286 million parameters and needs roughly 1.07 GiB of FP32 masters per
replica. This is about two orders of magnitude less master movement than the
original supported full-model target.

Actual motif insertion is feasible on these GPUs. Three 44-layer checkpoints
were built by inserting a copied four-layer motif at the front, middle, or back,
with the copied motif's attention output and routed/shared-expert down
projections scaled by ε=0.05. Each adds 6.27 GiB of BF16 weights and roughly
10% nominal layer compute.

## Paper and upstream artifacts

- Manually cleaned paper transcription: `references/papers/2509.24372.md`
- Original PDF: `references/papers/2509.24372.pdf`
- Rendered figures: `references/papers/2509.24372-figures/`
- Requested upstream clone: `es-at-scale/`
- Paper-source clone: `es-fine-tuning-paper/`
- Both clones resolve to commit `574a9d134da1ffce2a8bb812019899e5c96b588a`.
- Consolidated experiment summary: `experiments/es_location/summary.json`
- Generation-20/30 FP32 exports and digest sidecars:
  `experiments/es_location/insert_back_seed1101_gen{20,30}_fp32_masters.*`

## P0 fixes implemented

### Reward shaping

The old centered-rank code assigned distinct utilities to tied fitness values.
Because all positive members preceded all negative members, an all-tie sparse
reward population produced a systematic nonzero update. Ties now receive their
average rank, and an all-tie population maps exactly to zero. Constant-population
z-score shaping also maps to zero.

### Reproducible resume and targeted replay

- Batch and perturbation RNGs are derived from `(global seed, generation)`.
- Generation numbers must be contiguous.
- Schema-v2 journal rows record the target regex and layer plan, shaping,
  hyperparameters, RNG identity, batch indices, data/eval SHA-256 identities,
  member fitness, and replica state.
- Resume rejects a changed dataset, target, seed, or recipe before touching a
  server.
- Legacy journals fail safe unless their full-model or targeted scope is
  explicitly asserted.
- Replay honors each generation's target regex.

### Replica and quantization safety

- Every target registration emits a deterministic unit manifest and hash.
- Separate SHA-256 digests cover authoritative FP32 masters and serving casts.
- Training compares all four replicas after commits.
- Zero-match selectors, later target expansion, unsupported shapes, and FP8 or
  integer targets fail loudly.
- The FP8 guard is necessary: writing into a float8 tensor without recomputing
  its block scales silently corrupts the effective weight.

### Exact checkpoint semantics

- Journal-to-HF replay now retains one FP32 master through all generations,
  honors per-row targets, and casts once at export.
- Saved master artifacts remain FP32 rather than being prematurely cast to BF16.
- Master overlays validate every name and shape, build in a temporary directory,
  then atomically rename the verified output.

## Dataset and evaluation findings

The original evaluation is development-only. It contains fact-level leakage:
for example, held-out eval-v2 asks when Akechi Denki died (`2005`), while the
training set contains the same entity/relation/answer under a paraphrase from a
different URL. The default verified set also contained 39 normalized questions
that exactly overlapped eval-v2.

The new deterministic leakage gate checks every rendered/paraphrased question
against both eval sets. It combines question similarity with entity/context and
answer identity, drops conflicting answers and duplicate examples, removes
obvious commerce/relative-time trivia, and adds stable fact IDs plus document
provenance.

| Dataset | Input | Leak-free output | Main removals |
|---|---:|---:|---|
| `train_qa_verified` | 3,510 | 3,113 unique facts | 40 exact questions; 112 near questions; 140 entity-answer facts; 56 low-value/time-sensitive; 20 normalized QA duplicates; rendered duplicates/conflicts |
| `train_qa_v3_clean` | 57,792 | **Invalidated**; 27,756 projected before manual review | Correct parsing collapses 28,896 paired raw/chat renderings, removes 984 entity-answer leaks, 154 near-question leaks, 804 rendered duplicates, 86 low-value/time-sensitive items, and conflicts |

The first leakage-gate artifact contained 3,133 provenance rows; the ES loader
collapsed those to 3,113 normalized question/answer pairs. The v2 builder now
performs this fact-level deduplication itself and assigns source-independent
fact IDs. The completed experiments used the byte-identified v1 artifact and
record `items: 3113`; new runs default to the v2 artifact.

The first large v2 build is invalid: a broad instruction regex ran before the
ChatML grammar and leaked role/`<think>` delimiters into 28,017 metadata rows.
It also allowed 270 evaluation collisions to evade the leakage gate. The large
artifact is excluded from version control and must not be used for training.
Structural, fail-loud parsing now projects 27,756 unique raw facts; those facts
are undergoing source-document-level manual keep/edit/drop review before a
replacement dataset is published.

The old evaluator also used each candidate as its own judge. Among byte-identical
candidate answers, pairwise judges disagreed on as many as 54 of 340 aligned
items (15.9%); other runs showed 51/392 and 50/384 disagreements. Older reruns
regenerated some candidates, so mismatched candidates were excluded rather than
counted in the denominator. New-schema result files record answer/judge
checkpoint names, raw verdict, prompt protocol, item ID, and eval hash; frozen
candidates can be reused across judges. Wilson intervals, paired bootstrap
intervals, McNemar tests, and multi-judge agreement utilities are included.

## Training-recipe improvements

- Accuracy reward is Unicode NFKC/casefold aware; Japanese-only answers no
  longer receive invariant zero reward.
- Normalized exact correctness receives 70% of reward and token-F1 supplies
  bounded dense credit.
- Training walks a deterministic shuffled fact stream without replacement
  instead of repeatedly sampling a small fraction of the corpus.
- Evaluation-fact normalization is cached during leakage gating so repeated
  source-grounded rebuilds do not redo Unicode and token preprocessing.
- Each antithetic pair runs consecutively on the same replica.
- Probe sets are sharded over all four GPUs.
- The post-perturb warmup decodes one token instead of 64 throwaway tokens.
- LoRA SFT uses the checkpoint chat template, masks prompt tokens, avoids a
  duplicate EOS, supports 1,024-token examples, step checkpoints, validation,
  and resume.

Every server log also warns that SGLang has no RTX PRO 6000 Blackwell fused-MoE
Triton configuration for `E=256, N=512`, including the `_down` variant. Building
and benchmarking those device-specific kernels is the clearest remaining
systems-speed opportunity before long sweeps.

## What transfers from es-at-scale

Useful lessons retained:

- persistent inference engines and in-place parameter mutation;
- seed-only noise reconstruction;
- deterministic greedy decoding with common prompt batches;
- explicit cache invalidation after weight changes;
- decomposed, layer-wise updates to bound peak memory;
- dynamic work dispatch and fully sharded evaluation as throughput goals.

Upstream behavior not copied:

- BF16 perturb/add/subtract restoration, which drifts the base;
- BF16 update accumulation, which loses sub-ULP updates;
- correlated same-prefix noise for equal-shaped tensors;
- tensor-parallel workers racing to write incomplete checkpoint shards;
- serial reward-pool submission and timeout paths that can reuse stale values;
- unconditional CUDA synchronization and allocator purges after every mutation.

## Interpreting the parallel-ensemble prior

For a residual stack $h_{l+1}=h_l+f_l(h_l)$ whose branch updates are small,
expanding each branch around the same input gives, to first order,

$$
h_L \approx h_0 + \sum_l f_l(h_0).
$$

That is the precise regime in which serial residual blocks behave like a
parallel ensemble whose outputs are summed. The terms omitted by this
approximation are cross-layer Jacobian interactions, such as
$J_{f_j}f_i(h_0)$, which are second order in small residual updates but need not
remain small everywhere. Layer normalization, attention/state recurrence, MoE
routing, and feature refinement can all make those interactions important.
Thus the hypothesis is plausible and testable, but serial and parallel models
are not generally identical.

This motivates two controlled interventions rather than arbitrary single-layer
duplication: restrict ES masters to complete motifs near the input/output edges,
and insert a whole motif with a damped residual contribution. Preserving the
`[linear, linear, linear, full-attention]` cadence avoids confounding the test
with a changed hybrid-attention schedule. A learnable zero-initialized gate is
the clean next version because it begins function-preserving and directly
measures whether the copied branch earns nonzero contribution.

## Layer-location design

The base model has 40 layers and ten aligned motifs. Full-attention layers are
3, 7, 11, ..., 39. The controlled first sweep is:

| Plan | Layers | Dense units |
|---|---|---:|
| Front | 0–3 | 35 |
| Back | 36–39 | 35 |
| Front + back | 0–3 and 36–39 | 70 |
| Matched middle control | 16–23 | 70 |

"Dense" includes full-attention projections or all five GDN input/output
projections, the shared expert, and router; the 256 routed experts remain
frozen. Unlocking routed experts at front+back jumps master memory from about
1.07 GiB to roughly 25 GiB, so it is a second-stage ablation only.

This is LISA-like selection, but ES already runs under `no_grad`: there is no
activation tape, gradient, or optimizer state to remove. The savings are FP32
master storage, CPU→GPU reconstruction, random-noise work, checkpoint size, and
the ability to keep the frozen bulk FP8. The safe mixed-precision route is a BF16
source checkpoint with online FP8 while explicitly ignoring selected target
modules; a prequantized FP8 checkpoint cannot simply reinterpret selected FP8
tensors as BF16.

## Equal-budget layer-location screens

The probe reward is $0.7\times$ normalized exact match plus $0.3\times$ token
F1; it is not pure F1. "Held" below means the contaminated 169-item
development-held split, not an untouched final test set.

| Model / target | Layers | Units / parameters | Commits | Train probe | Development-held probe | Held delta |
|---|---:|---:|---:|---:|---:|---:|
| Base, front | 0–3 | 35 / 142,999,552 | 3 | .0310 → .0264 | .0289 → .0303 | +.0014 |
| Base, back | 36–39 | 35 / 142,999,552 | 3 | .0305 → .0302 | .0297 → .0291 | −.0006 |
| Base, front + back | 0–3, 36–39 | 70 / 285,999,104 | 3 | .0310 → .0297 | .0300 → .0299 | −.0001 |
| Base, matched middle | 16–23 | 70 / 285,999,104 | 3 | .0302 → .0293 | .0297 → .0287 | −.0010 |
| 44-layer, front insertion | 4–7 | 35 / 142,999,552 | 3 | .0250 → .0254 | .0297 → .0293 | −.0004 |
| 44-layer, middle insertion | 20–23 | 35 / 142,999,552 | 3 | .0257 → .0256 | .0296 → .0297 | +.0001 |
| 44-layer, back insertion | 40–43 | 35 / 142,999,552 | 3 | .0190 → .0204 | .0287 → .0287 | .0000 |

These screens used seed 1101, eight antithetic pairs, $\sigma=.01$, learning
rate $.04$, rank-4 noise, rank shaping, and 32 QA items per population member.
Only base-front moved development-held reward by more than .001; the other six
were flat or negative. With one seed, no untreated continuation control, a
contaminated development probe, and visible probe variability, this does not
identify a winning layer location.

The back-inserted checkpoint was then continued to 30 commits as a systems and
stability test. A fresh post-commit probe moved from .0190/.0287 to
.0196/.0289 (train/development-held), still essentially flat. Exact FP32
snapshots were preserved after commits 20 and 30; each contains 35 tensors and
142,999,552 parameters (572,002,960 bytes), with matching master and serving
digests on all four replicas.

## GPU utilization verification

Eight pairs assigned two antithetic pairs to each of four single-GPU replicas
per generation. Probe shards were also balanced: 59 train-split items per GPU
and 43/42/42/42 development-held items. Every commit recorded a successful
four-replica FP32-master and serving-weight digest check.

A 12-second, approximately 1 Hz trace during the inserted-back extension
captured 48 device samples. Mean utilization on GPUs 0–3 was 93.6%, 87.1%,
95.3%, and 91.8%; all four reached 100%, 43/48 samples were at least 80%, and
87,702–87,706 MiB remained resident on every card. One isolated sample showed
GPU 1 at 0% during request turnover. Thus all GPUs received balanced work and
remained highly utilized, without claiming that every device was at 100% at
every instant. `gpu_utilization_guard.py` now makes this a fail-loud check for
future runs.

## Motif-insertion pre-screen

Each inserted checkpoint has 44 layers. The destination-to-source map and
checksums are recorded in its `layer_source_map.json`.

| Variant | Common-judge raw-prompt accuracy | Common-judge chat-prompt accuracy | Chat delta vs base | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| Base 40-layer | 12.10% | 12.59% | — | — | — |
| Front insertion | 11.85% | 13.09% | +0.49 pt | [−0.74, +1.73] pt | 0.69 |
| Middle insertion | 11.85% | 12.10% | −0.49 pt | [−1.98, +0.74] pt | 0.73 |
| Back insertion | 11.85% | 12.35% | −0.25 pt | [−1.73, +1.23] pt | 1.00 |

These are statistically indistinguishable. The useful result is that a damped
whole-motif insertion is operationally viable and does not catastrophically
damage the base. It is not evidence that front insertion is better.

These are binary LLM-judge results on the contaminated 405-item development
evaluation. "Raw" and "chat" describe answer prompting, not unjudged
exact-match accuracy.

## Recommended experiment order

1. Preserve the completed one-seed screens as exploratory baselines.
2. Repeat front, back, and matched-middle targets across at least three seeds,
   alongside an equally trained 40-layer continuation control.
3. Compare 44-layer insertions only at equal training budgets; do not select a
   location from the single-seed back continuation.
4. Unlock routed experts only after a location effect replicates.
5. Sweep insertion damping (ε=0, 0.01, 0.05, 0.1) and test a learnable
   zero-initialized residual gate, which requires scalar-parameter perturbation
   support.
6. Build a separately sourced, expert-reviewed final test set before making a
   domain-quality claim.

## Validation

- 47 root regression tests and 6 SGLang ES-state tests pass.
- Modified Python files compile; all top-level shell scripts pass `bash -n`;
  the SGLang diff passes `git diff --check`.
- The paper transcription has 136 unique anchors, 50 resolving internal links,
  and 13 existing/valid figure images, with balanced code and math delimiters.
- QA parsers are structural and fail closed on role-token leakage or metadata
  disagreement; the malformed large artifact is explicitly invalidated.
- Both FP32 exports pass SHA-256, tensor-count, dtype, and parameter-count
  verification against their sidecar manifests.
