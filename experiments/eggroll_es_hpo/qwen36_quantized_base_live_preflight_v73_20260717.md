# Qwen3.6 BF16 versus serialized-FP8 LoRA preflight V73

Date: 2026-07-17 UTC

Status: **valid data-free diagnostic; scored/training promotion remains
fail-closed.**  V76 now proves the clean environment, routed-expert ownership,
and zero-forbidden-fallback-log gates on four GPUs.  Exact reference-repeat
identity plus semantic and source-disjoint OOD non-inferiority remain
unsatisfied, so no scored evaluation or training is authorized.

## Counterbalanced result

Two waves placed BF16 on GPUs 0/2 and serialized FP8 on GPUs 1/3, then swapped
the assignments.  Every physical GPU therefore ran both arms.  Each actor used
Qwen3.6-35B-A3B, the same two rank-32 LoRA artifacts, 68 synthetic prompts,
two warmups, eight alternating adapter calls, greedy fixed-seed decoding, eager
execution, one resident GPU LoRA slot, and `gpu_memory_utilization=0.82`.

| Metric | BF16 median [range] | Serialized FP8 median [range] |
|---|---:|---:|
| Timed generation phase | 45.196 s [44.824, 45.592] | 49.393 s [48.420, 50.494] |
| Per-GPU FP8/BF16 runtime ratio | reference | 1.095 [1.080, 1.113] |
| Reported model-load memory | 68.24 GiB | 36.93 GiB |
| Available KV-cache memory | 6.87 GiB | 38.14 GiB |
| KV-cache tokens | 139,264 | 775,372 |
| Steady/peak NVML memory | 83,820 MiB | 83,878 MiB |
| Plateau GPU utilization | 53.07% | 44.31% |
| Plateau memory activity | 31.52% | 25.03% |
| Plateau power | 195.12 W | 171.86 W |
| Reference repeat changed rows | 1 [1, 2] | 5 [4, 6] |
| Candidate repeat changed rows | 0 [0, 0] | 0 [0, 0] |
| Reference/candidate separating rows | 6.5 [6, 7] | 13.5 [13, 15] |

FP8 saved 31.31 GiB in live model residency and exposed 31.27 GiB more KV
memory at the same allocation fraction.  The 5.57x KV-token increase matches
the preregistered expectation that frozen-weight quantization does not shrink
the BF16 KV bytes per token; it frees capacity for more tokens instead.

The nearly identical 84-GiB NVML plateaus are not evidence that FP8 failed to
save VRAM.  vLLM deliberately filled the freed space with KV blocks because
both arms requested the same 0.82 memory fraction.  A production layout that
only needs BF16-like KV concurrency should lower the FP8 fraction to about
0.50, converting the weight saving into roughly 31 GiB of actual device
headroom.  That single-variable V74 diagnostic is separate from this report.

## Throughput and output behavior

FP8 was 9.5% slower by the median paired ratio, and every physical GPU measured
an 8.0-11.3% regression.  Its lower GPU and memory-controller activity suggests
kernel underutilization, not a saturated-bandwidth limit.  The logs confirm
that no matching block-FP8 MoE table was found, so vLLM used its default config.
This makes hardware- and shape-specific FP8/LoRA kernel tuning a credible
follow-up; the previously rejected V29 selected table is not reusable.

## Right-sized FP8 follow-up

V74 changed only FP8 `gpu_memory_utilization` from 0.82 to 0.50 and repeated the
same workload concurrently on all four GPUs.  The live-resolved fraction was
0.50 in every actor.

| Metric | FP8 at 0.82 | FP8 at 0.50 | BF16 at 0.82 |
|---|---:|---:|---:|
| Median timed runtime | 49.393 s | 48.958 s | 45.196 s |
| Steady/peak NVML memory | 83,878 MiB | 52,738 MiB | 83,820 MiB |
| Available KV memory | 38.14 GiB | 7.73 GiB | 6.87 GiB |
| KV-cache tokens | 775,372 | 157,286 | 139,264 |
| Maximum 2,048-token concurrency | 378.60x | 76.80x | 68.00x |

Right-sizing therefore freed 31,140 MiB (30.41 GiB) per GPU relative to the
same FP8 model at 0.82, and 31,082 MiB (30.35 GiB) relative to BF16, while still
retaining 12.9% more KV tokens than BF16.  Minimum unused physical memory rose
to 45,149 MiB per GPU.  Runtime improved by 0.9% relative to FP8 at 0.82, which
is small enough to treat as no throughput cost rather than a speed claim.

All four right-sized actors loaded, switched adapters, changed candidate
outputs, repeated candidate outputs exactly, and cleaned up.  Reference repeats
still changed 4-5 rows and the same environment/fallback/routed-expert gates
remain open.  V74 is strong capacity evidence, not precision promotion.

V75 then made the kernel environment explicit: a fresh empty FP8 tuning
directory, `enable_flashinfer_autotune=False`, and
`VLLM_USE_DEEP_GEMM=0`.  Its four runtimes had a 49.542 s median, so the
environment cleanup did not materially change throughput.  It removed the
FlashInfer default-tactic warning and proved the empty-folder start, but vLLM
0.25 still emitted its DeepGemm-to-CUTLASS fallback warning in every actor.
Source inspection found an ordering bug: `Fp8Config.use_deep_gemm` is still
`None` when the warning condition runs, and that branch does not consult the
documented environment switch before setting the field false.  This is tracked
as `specialist-nen.28`; V75 correctly remains failed on the literal zero-warning
gate rather than suppressing the message.

## V76 routed-method and fallback attestation

V76 fixed the DeepGemm-disable ordering only inside the trusted local probe
process, before vLLM's config post-initialization.  It did not edit installed
vLLM or suppress warnings.  The worker callback was serialized only between
the local controller and its colocated worker, then every returned structure
was validated and self-hashed.  Four concurrent actors completed the full
synthetic adapter-switch workload:

| Metric | V76 result |
|---|---:|
| Timed generation median [range] | 48.574 s [48.377, 48.773] |
| Physical FP8 routed weight owners | 40 per actor |
| Expected LoRA wrapper FP8 references | 80 per actor |
| Routed owner / runtime wrapper | `RoutedExperts` / `FusedMoEModularMethod` |
| Underlying quantization method | `Fp8MoEMethod`, block `[128, 128]` |
| Routed backend / implementation | `TRITON` / `TritonExperts` |
| Routed W13 and W2 dtype | `torch.float8_e4m3fn` |
| Candidate repeat changed rows | 0 on all actors |
| Reference repeat changed rows | 1-4 of 68 |
| Reference/candidate separating rows | 12-14 of 68 |
| Forbidden fallback log matches | 0 across four logs |
| Sampled peak / peak utilization | 50,858 MiB / 100% on every GPU |

All actors agreed on routed-owner name hash
`2b645a7e4c4488c548549ecd7326411fb7347bf569c78cd8640f9824b2178b55`
and wrapper-reference hash
`aaa98d3eac3e160f1ca905a366175ed97d31f27da873fb641d5b1920fdf7eb95`.
The monitor captured 81 rows per GPU, useful activity on every GPU, and final
4 MiB idle residency.

This attestation also corrected an important assumption in V75: disabling
DeepGemm does not prove that routed MoE uses CUTLASS.  The actual routed backend
is Triton because the probe explicitly requests `moe_backend=triton`.  The V75
field that claimed a CUTLASS path was intent-derived rather than runtime-derived;
`specialist-nen.29` now blocks kernel tuning and production-layout selection
until V77 is regenerated against the attested Triton baseline.

V76 remains non-promotable.  The process-local ordering workaround is not an
upstream integration fix, reference token hashes are still nondeterministic,
and no semantic or OOD dataset was opened.  The result closes instrumentation
uncertainty without weakening the quality gate.

## V78 FP8 per-token-head KV-cache diagnostic

Three independent four-GPU V78 replications changed only V76's resolved KV-cache
dtype from BF16 (`auto`) to `fp8_per_token_head`.  vLLM computed dynamic scales
per token and head, changed the language attention backend from FlashAttention
to `TRITON_ATTN`, retained the hybrid Mamba cache on its resolved FP32 SSM path,
and kept the same 0.50 device-memory budget.

| Metric | V76 BF16 KV, three replicates | V78 FP8 KV, three replicates |
|---|---:|---:|
| Available KV memory | 7.77 GiB | 7.77 GiB |
| KV-cache tokens | 157,696 | 198,656 |
| 2,048-token concurrency | 77x | 97x |
| Timed runtime median | 48.584 s | 49.102 s |
| Candidate repeat changed rows | 0 | 0 on all 8 actors |
| Reference repeat changed rows | 1-4 | 5-10 |
| Reference/candidate separating rows | 12-14 | 21-26 |
| Forbidden fallback matches | 0 | 0 |

The FP8 KV arm raises token capacity by 25.97% at the same allocation and its
combined runtime median is 1.07% above the V76 control.  The capacity gain
is much smaller than 2x because Qwen3.6 is a hybrid architecture and its Mamba
SSM state remains FP32.  A capacity-matched follow-up should convert the gain
into roughly 1.6 GiB of device headroom instead of extra tokens.

This arm is not promoted on capacity alone.  First-call FP8-KV output hashes
differ from V76 on 5-7 of 68 reference rows and exactly 21 of 68 candidate rows;
within-arm reference variation also increases.  Those hashes are not a quality
metric, but they make the preregistered semantic and source-disjoint OOD gates
mandatory.  V78 remains data-free and nonpromoting.

V78b then changed only the hybrid Mamba SSM cache from its resolved FP32 dtype
to BF16.  Three four-GPU replicates increased capacity from 198,656 to
310,886 tokens (151.8x 2,048-token concurrency), a further 56.49% increase and
97.14% above the V76 BF16-KV control.  The 12-actor median runtime was 48.987 s,
0.23% below V78 and 0.83% above V76; the slow first replicate was not repeated.
Candidate repeats remained exact, all GPUs reached
100% sampled utilization, no forbidden fallback occurred, and cleanup was
idle.

This is a high-capacity but high-risk arm.  Qwen's model config explicitly
specifies FP32 Mamba SSM state, and vLLM warns that the user-supplied BF16 value
overrides it.  Reference repeats changed 4-11 of 68 rows; first candidate calls
differed from V76 on 15 rows and from V78 on 20.  V78b therefore cannot be a
default memory optimization without strict semantic and source-disjoint OOD
non-inferiority.  Its value at this stage is identifying FP32 hybrid state as
the dominant remaining cache-capacity cost.

V78c isolates that hybrid-state change while retaining V76's BF16 attention
KV and FlashAttention backend.  Its first four-GPU replicate provides 218,843
tokens (106.86x concurrency), 38.78% above V76, at a 49.083-s median runtime
(+1.03%).  Candidate repeats remain exact, first calls differ from V76 on only
2-3 reference and 14 candidate rows, and no forbidden fallback occurs.  This
is a better preliminary capacity/perturbation tradeoff than quantizing both
attention KV and SSM state, but it still overrides Qwen's explicit FP32 SSM
recommendation and therefore remains conditional on replication plus strict
semantic/OOD non-inferiority.

## Text-only live residency audit

The serialized-FP8 checkpoint contains 0.832 GiB of visual weights and 0.795
GiB of MTP weights in addition to 33.256 GiB of language weights.  That looked
like a possible 1.627-GiB text-only optimization, but V76 worker introspection
shows vLLM already performs it.  Every one of four live actors exposed exactly
813 named parameters, all under the language component and all on `cuda:0`
inside its isolated physical-GPU namespace:

- 35,712,084,096 logical parameter bytes per actor;
- 303 BF16, 270 FP32, and 240 FP8 parameter objects;
- no visual or MTP named parameter;
- common parameter-name SHA-256
  `a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90`.

Therefore a filtered or rebuilt text-only checkpoint would add artifact and
loader risk without reducing live VRAM.  The correct production choice is to
retain the sealed checkpoint and vLLM's existing text-only/non-speculative
loading path, with live residency attestation guarding against future drift.

Both precision arms loaded, generated, switched adapters, repeated the
candidate exactly, and shut down cleanly.  FP8 resolved `quantization=fp8` with
the sealed block `[128, 128]` checkpoint.  The adapter affected outputs more
often under FP8 (13-15 rows versus 6-7), and first FP8 candidate calls differed
from BF16 on exactly 17 of 68 rows in both waves.  First reference calls differed
on 0-4 rows depending on the actor pairing.  Those token hashes cannot establish
semantic equivalence or quality.

## Why the original V73 full gate failed

The V67 preflight correctly requires more than successful inference:

- Exact deterministic reference restoration failed in every actor.  BF16
  reference repeats changed 1-2 rows and FP8 changed 4-6.  This is a token-level
  inference nondeterminism problem; adapter files themselves remained immutable.
- The FP8 logs explicitly auto-disabled DeepGemm on Blackwell/Qwen3.5 because
  its E8M0 scale path would degrade accuracy, then reported a CUTLASS fallback.
  The preregistration requires zero fallback messages.  A retry must explicitly
  set `VLLM_USE_DEEP_GEMM=0` so CUTLASS is selected intentionally.
- FlashInfer warmup reported an empty autotune cache and default tactics in both
  arms.  A strict retry must explicitly disable that autotuner or seal a valid
  cache rather than accepting a warning.
- The inherited probe pointed at the BF16 tuning directory.  No FP8 entry was
  loaded, but the precision contract requires an explicitly empty/default FP8
  folder, not merely the absence of a matching filename in another arm's
  folder.
- The compact probe did not attest that all 40 routed-expert layers use the
  expected FP8 method, nor enumerate unexpected unquantized prefixes.  V76 now
  satisfies this instrumentation gate, but not the remaining quality gates.

These are fail-closed preflight findings, not reasons to discard FP8.  The
capacity result is strong; promotion still requires a clean environment-bound
worker attestation, exact or preregistered semantic repeat handling, tuned
throughput, and paired dev/OOD non-inferiority.

No training dataset, dev/OOD set, terminal holdout, model update, checkpoint,
or promotion path was opened.  Only synthetic prompts and token-hash receipts
were persisted.
