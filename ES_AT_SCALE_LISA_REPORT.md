# ES-at-Scale Audit and Qwen3.6 Layer-Location Experiments

Date: 2026-07-14
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

A new deterministic-inference screen corrects an important flaw in the earlier
location runs. All four replicas and three repeated 405-item base probes now
produce identical answer hashes. Under that controlled runtime, the accuracy
reward is much sparser than the old runs suggested: front+back was positive on
the development-held split in all three six-commit seeds (mean +.0021), while
the matched middle control was positive in one of three (mean -.0005). This is
provisional evidence for the edge-location prior, not a final result: each
front+back run had only two informative antithetic pairs out of 48, four of six
generations had no update signal, and the evaluation remains contaminated.

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
- Curated-data smoke-screen summaries:
  `experiments/es_curated/{summary,summary_final}.json`
- Deterministic probe and raw-shaping screen:
  `experiments/es_determinism/probe_v1.json` and
  `experiments/es_deterministic/summary_raw_v1.json`
- Generation-20/30 FP32 exports and digest sidecars:
  `experiments/es_location/insert_back_seed1101_gen{20,30}_fp32_masters.*`

## P0 fixes implemented

### Reward shaping

The old centered-rank code assigned distinct utilities to tied fitness values.
Because all positive members preceded all negative members, an all-tie sparse
reward population produced a systematic nonzero update. Ties now receive their
average rank, and an all-tie population maps exactly to zero. Constant-population
z-score shaping also maps to zero. A deterministic follow-up exposed a second
problem: with only one informative antithetic pair, centered ranks converted a
raw reward difference of .00075 into a commit coefficient of .2667. The probe
then collapsed to zero. Raw reward shaping is now the default, while rank and
z-score modes remain explicit ablations. Journals and summaries report
informative-pair counts, zero-signal generations, and maximum coefficients.

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
| `train_qa_v3_clean` | 57,792 | **Invalidated**; 27,756 structural projection / 27,729 alias-aware review candidates | Correct parsing collapses 28,896 paired raw/chat renderings; the strengthened gate also catches 27 unique spacing/transliteration aliases before manual review |
| Manual pilot | 156 candidates across 9 documents | 81 evidence-carrying facts | 104 candidate IDs consolidated into 26 edits; 52 IDs dropped; 55 missing facts added by reviewers |
| Curated v1 | 3,113 base + 81 manual + 29 directory + 38 resource-fact rows | 3,258 unique facts | Three additional distinctive-answer aliases removed from the base during the final merge |

The first leakage-gate artifact contained 3,133 provenance rows; the ES loader
collapsed those to 3,113 normalized question/answer pairs. The v2 builder now
performs this fact-level deduplication itself and assigns source-independent
fact IDs. The completed experiments used the byte-identified v1 artifact and
record `items: 3113`; `es_train_acc.py` retains v2 as its legacy reproducibility
default, while `sft_lora.py` now defaults to curated v1.

The first large v2 build is invalid: a broad instruction regex ran before the
ChatML grammar and leaked role/`<think>` delimiters into 28,017 metadata rows.
It also allowed 270 evaluation collisions to evade the leakage gate. The large
artifact is excluded from version control and must not be used for training.
Structural, fail-loud parsing now projects 27,756 unique raw facts; those facts
are undergoing source-document-level manual keep/edit/drop review before a
replacement dataset is published.

An alias-aware follow-up gate catches compact transliteration and title variants
such as `Takatekote`/`Takate Kote`, `ipponnawa`/`Ippon nawa`, and punctuation or
spacing variants embedded in short lists. A manual audit of all 27 newly removed
unique candidates found direct evaluation collisions; the stricter future
candidate pool is therefore 27,729 facts.

The first three manual batches deliberately kept each reviewer to one complete
document at a time. They exposed additional semantic leakage that lexical
thresholds missed (including Somerville bowline, agura shibari,
`Ipponnawa`/`Ippon nawa`, Demon Tie/`tengu shibari`, and bend/half-hitch
function aliases), unsupported definition and authorship questions, duplicated
paraphrases, and safety claims that needed explicit source attribution. The
resulting 81-row pilot is suitable for inspecting the target quality schema,
but is too small to replace the 3,113-fact training set.

The published curated v1 artifact combines the 3,110 surviving base facts with
those 81 manual facts, 29 owner-curated resource-directory answers, and 38
independently reviewed resource facts. Its manifest preserves all 23 supplied
resource URLs verbatim. Live collection was relevance-bounded rather than a
site mirror: 165 useful public documents were retained, while robots/content
signals, terms, access challenges, paid material, and thin client-rendered pages
were recorded as exclusions. Resource recommendations are not treated as
independent safety endorsements; vendor and manufacturer claims remain
attributed, including Shibari Supply's explicit no-guarantee warning for bamboo.

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
- Deterministic serving is an explicit launch mode; an answer-hash probe checks
  the server flags, cross-replica canaries, and repeated complete evaluations.
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

The table above is the original seed-1101 screen. The four base-model targets
were subsequently repeated with seeds 2202, 3303, and 4404 using the same eight
antithetic pairs, $\sigma=.01$, learning rate $.04$, rank-4 noise, rank shaping,
and 32 QA items per population member. Three zero-learning-rate runs exercised
the same perturbation, fitness, probe, and commit paths while committing a zero
update; they measure short-run probe drift, not the effect of an untreated
learned continuation.

| Base-model plan | Runs | Positive / negative | Mean held delta | Sample SD | Mean minus zero-LR drift |
|---|---:|---:|---:|---:|---:|
| Front | 4 | 3 / 1 | +.000825 | .000862 | +.000292 |
| Matched middle | 4 | 3 / 1 | +.000425 | .000964 | −.000108 |
| Back | 4 | 3 / 1 | +.000250 | .000661 | −.000283 |
| Front + back | 4 | 2 / 2 | +.000250 | .000592 | −.000283 |
| Zero-LR drift control | 3 | 3 / 0 | +.000533 | .000208 | — |

Front has the largest raw mean, but its advantage over the zero-update drift
baseline is only .000292—about one third of its across-seed sample standard
deviation. All location effects are the same order as measurement drift, and
the probe is contaminated development data. These experiments therefore do
not support selecting front, back, middle, or front+back at this budget. They
also do not falsify a front/back ensemble prior; they show that this particular
three-commit screen is too weak to resolve it.

### Curated-data smoke screens

A one-seed screen on an intermediate 3,255-row curated build produced held
deltas of +.0003 (front), .0000 (back), +.0011 (matched middle), and −.0001
(front+back); its matched zero-update run drifted +.0002. Because the final
manual audit changed the dataset hash, these runs are recorded separately and
are not pooled with the base-data location sweep.

The apparent middle signal was then checked on the final 3,258-row artifact
(SHA-256 `4b6f6140ad2256f6c14b2fb182f1266490cb535ba963855ef93e0a1e70f54770`)
for six commits with seed 6606. The learned run moved development-held reward
from .0288 to .0296 (+.0008); an otherwise matched zero-learning-rate run moved
from .0300 to the same .0296 (−.0004). The two fresh base probes differed by
.0012 before any learned update, and both runs ended at the same score. This is
direct evidence that probe variability is large enough to explain the apparent
gain, not evidence for a middle-layer winner.

Those screens did not enable SGLang's deterministic-inference mode, so their
probe drift and dense-looking member-fitness variation cannot be cleanly
separated from runtime nondeterminism. They remain useful systems baselines,
but are superseded for layer-location inference by the deterministic screen
below.

### Deterministic final-data screen

SGLang 0.5.15 was relaunched with deterministic inference, PyTorch sampling,
radix-cache disablement, and FlashInfer attention. The four replicas produced
the same 32-answer canary hash. Three complete base probes produced the same
405-answer SHA-256 and exactly the same train/development-held rewards:
.0295713212/.0304041190. A two-trial probe after the seed-9909 front+back run
was likewise answer-identical at .0312422530/.0314957051.

The comparison used the final 3,258-row curated artifact, three paired seeds,
six commits, 8 antithetic pairs, 32 facts per member, sigma .01, learning rate
.04, rank-4 noise, 70 dense units, and raw reward shaping for both targets.

| Target | Held deltas by seed (7707, 8808, 9909) | Positive | Mean delta | Sample SD | Informative pairs | Zero-signal generations |
|---|---|---:|---:|---:|---:|---:|
| Front + back | +.0006, +.0046, +.0011 | 3/3 | +.0021 | .00218 | 6/144 pairs | 12/18 |
| Matched middle | -.0005, +.0003, -.0013 | 1/3 | -.0005 | .00080 | 15/144 pairs | 8/18 |

“Informative” means that the positive and negative member of an antithetic
pair received different reward. Each target has 8 pairs × 6 generations × 3
seeds = 144 pair comparisons.

The front+back-minus-middle difference in mean held delta is +.0026. This is
the first controlled directional support for the front/back prior in this
repository, but the sparse fitness makes effect attribution fragile. In
particular, every front+back seed learned from only two informative pairs, its
train-probe changes were inconsistent, and the development-held set is not an
independent final test.

The interrupted rank-shaped middle diagnostic is intentionally retained. Its
second generation had 15 zero member rewards and one reward of .00075; rank
shaping assigned that pair a .2667 commit coefficient. After three commits,
both probes were zero. Raw shaping keeps coefficients at the reward's actual
scale (the six completed runs had maxima from .000323 to .000768) and avoids
this particular amplification, but it does not solve reward sparsity.

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

The final curated-data learned/control pair also used all four replicas for
every population and probe shard. Across their utilization traces, per-GPU
mean utilization was 78.2–85.2% including request-turnover samples; every GPU
reached 100%, and 87,434–87,888 MiB remained resident. All 12 commits recorded
matching FP32-master and serving digests across all four replicas.

The six deterministic raw-shaping runs recorded 246 samples per GPU. Mean
utilization on GPUs 0-3 was 86.3%, 81.7%, 84.3%, and 83.7%, including launch
and request-turnover samples; every GPU reached 100%. All 36 commits verified
matching FP32-master and serving digests across all four replicas. The
answer-hash probes additionally establish that work sharding did not introduce
cross-replica answer differences.

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

1. Replace the mostly binary batch reward with a denser correctness-anchored
   signal, such as a small correct-answer log-probability component, without
   reverting to the old pure-likelihood objective that encouraged format blur.
2. Build a separately sourced, expert-reviewed final test set. Keep the current
   evaluation for development only, then repeat front+back versus middle at a
   longer budget with deterministic serving and a genuinely trained control.
3. Treat the three-seed front+back result as the leading hypothesis, not a
   selected winner; require replication under the denser reward and clean test.
4. Compare 44-layer insertions only at equal training budgets; do not select a
   location from the single-seed back continuation.
5. Unlock routed experts only after a location effect replicates.
6. Sweep insertion damping (ε=0, 0.01, 0.05, 0.1) and test a learnable
   zero-initialized residual gate, which requires scalar-parameter perturbation
   support.
7. Evaluate the owner-curated resource/factual QA tranche separately from the
   original 3,113-item training set so dataset changes are not confused with a
   layer-location effect.

## Validation

- 104 root regression tests and 6 SGLang ES-state tests pass.
- Modified Python files compile; all top-level shell scripts pass `bash -n`;
  the SGLang diff passes `git diff --check`.
- The paper transcription has 136 unique anchors, 50 resolving internal links,
  and 13 existing/valid figure images, with balanced code and math delimiters.
- QA parsers are structural and fail closed on role-token leakage or metadata
  disagreement; the malformed large artifact is explicitly invalidated.
- Both FP32 exports pass SHA-256, tensor-count, dtype, and parameter-count
  verification against their sidecar manifests.
