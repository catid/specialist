# Structured EGGROLL-ES perturbation oracle and LoRA comparison: CPU evidence

Date: 2026-07-17 UTC
Bead: `specialist-0j5.9`
Status: CPU correctness complete; runtime dependencies pending; no launch authorized

## Outcome

The CPU correctness oracle and a six-arm comparison contract are implemented and
sealed. The causal comparison is restricted to the same 4,528,128-element LoRA
surface: absolute-index IID noise versus structured outer-product noise at ranks
1, 4, 8, and 16. A 35,951,822,704-element dense full-weight IID arm is explicitly
an unmatched systems/reward anchor and cannot be used for a causal quality claim.

The artifact does not authorize GPU execution, dev/OOD evaluation, protected
holdout access, live-run reads, candidate commits, or promotion. No dataset,
evaluation payload, live run artifact, checkpoint payload, or GPU was opened by
this work.

## Sealed identities

- Preregistration: `experiments/eggroll_es_hpo/preregistrations/structured_es_lora_comparison_v1.json`
  - file SHA-256: `1daa8372cef13613736534b6539eceea112616e32abbf0fa7ec30b457b3aeb4b`
  - content SHA-256: `de119d12099c381c299ff6de7484d882e805661c7223205943204d19e2e0b405`
- Oracle: `structured_es_oracle_v1.py`
  - file SHA-256: `8fca35f89744f292ef0d9327f547196dd26f93268336f3fad4812a065f35f740`
- Builder: `build_structured_es_preregistration_v1.py`
  - file SHA-256: `b14ecf26357a2c8d0a56aaaede8dc1535f6f3be0aded7a208968b0ff7274b856`
- Oracle tests: `test_structured_es_oracle_v1.py`
  - file SHA-256: `d6a890266ec95127d932c4d8df4fa46076e5cb1c73e16547aab0e056fdcfd913`
- Builder tests: `test_build_structured_es_preregistration_v1.py`
  - file SHA-256: `cc76dea58ead30a7afbb090127cd0db319c3805dd0370f3235d7eea326f5efcc`

The contract reuses the completed FP32 optimizer/global-sigma artifact
(`428d1de...` file, `e8c646b5...` content) and the accepted V66d telemetry
identities (`12a5e854...` report file, `87d1eca1...` report content,
`a31d9c4c...` GPU telemetry, and `aa10617c...` actor log). The builder reads the
sealed V66d evidence document, not the live result artifacts.

## Correctness semantics

- RNG is a domain-separated SHA-256 key plus SplitMix64 counters and Box-Muller,
  rounded once to FP32. A value depends only on direction seed, full tensor key,
  method/factor domain, and absolute element/factor ordinal. Mutable generator
  state and local-chunk indexing are prohibited.
- IID reconstruction supports all positive-rank tensor shapes, including the
  Qwen vector and 3-D expert tensors. Structured outer products are intentionally
  restricted to matrices.
- Structured noise is `epsilon = (U @ V.T) / sqrt(k)`, with FP32 products and
  FP32 additions in ascending component order. Every entry has mean zero,
  variance one, expected squared Frobenius norm `rows*columns`, fourth moment
  `3 + 6/k`, and excess kurtosis `6/k`.
- Finite-rank structured noise is claimed only as an isotropic first-order
  central-difference estimator. It is not claimed to equal a Gaussian-smoothed
  score estimator at finite sigma.
- Candidate and weighted-update generators emit bounded chunks and declare zero
  dense whole-surface noise/candidate allocation. The mirrored estimator applies
  exactly `1 / (2*N*sigma)` once.
- Exact restoration always writes the complete canonical surface. Noise
  subtraction is prohibited. A gap, overlap, duplicate, incomplete restoration,
  or master/runtime identity mismatch terminally poisons the actor, which then
  rejects further work.

## CPU verification

Command:

```text
.venv/bin/pytest -q test_structured_es_oracle_v1.py \
  test_build_structured_es_preregistration_v1.py \
  test_fp32_es_optimizer_ablation_v1.py \
  test_build_fp32_es_optimizer_sigma_preregistration_v1.py
```

Result: `114 passed in 3.63s`.

The new 67 tests pin absolute RNG vectors and full-stream hashes; reconstruct all
five LoRA perturbation methods exactly across 1, 4, and 7 shards and multiple
chunk sizes; validate arbitrary-rank dense IID tensors; compare streamed
antithetic candidates and weighted updates with one-chunk CPU oracles; verify
rank scaling and scratch accounting; exercise partial-write full restoration and
terminal poison behavior; and reject re-signed plan/receipt mutations covering
authorization, source identity, local-index RNG, scaling, allocation, bytes,
scratch, work, seed, update error, rollback, GPU attribution, access, and compute
ceiling violations.

The checked-in artifact also passed its deterministic builder check:

```text
.venv/bin/python build_structured_es_preregistration_v1.py --check
```

## Memory and bandwidth findings

The LoRA FP32 master is 18,112,512 bytes. The derived BF16 inference/runtime view
is 9,842,688 bytes per signed candidate. With 16 signed candidates, every LoRA
arm must write 157,483,008 bytes (150.1875 MiB) merely to install candidate
parameters. Structured factors do not reduce those runtime writes without a
fused perturbation path.

| Method | Random/factor values per direction | Fraction of IID draws | Candidate scratch ceiling | Weighted-update scratch ceiling | Entry fourth moment |
|---|---:|---:|---:|---:|---:|
| IID | 4,528,128 | 100.00% | 131,072 B | 196,608 B | 3.0 |
| Rank 1 | 143,744 | 3.17% | 163,968 B | 229,504 B | 9.0 |
| Rank 4 | 574,976 | 12.70% | 262,656 B | 328,192 B | 4.5 |
| Rank 8 | 1,149,952 | 25.40% | 394,240 B | 459,776 B | 3.75 |
| Rank 16 | 2,299,904 | 50.79% | 657,408 B | 722,944 B | 3.375 |

Thus rank 1 removes 96.8% of normal draws, but has the heaviest tails and a
small factor-cache scratch increase. Rank 16 is closer to Gaussian and still
halves random draws. These are theory/accounting results, not measured GPU
speedups or quality results.

The dense anchor is 7,939.67 times the LoRA optimization surface. Its four FP32
masters total 575,229,163,264 bytes (535.72 GiB), its largest single-tensor FP32
noise buffer is 2,147,483,648 bytes, and its two-buffer weighted-update ceiling
is 4,294,967,296 bytes. Sixteen BF16 candidate installs write
1,150,458,326,528 bytes (1.046 TiB), 7,305.29 times the LoRA runtime-view writes.
This makes the dense arm a capacity/bandwidth anchor, not a currently launchable
recipe.

## Empirical blockers and next gates

No training or GPU speed/quality result exists yet. Runtime execution remains
ineligible until all three blockers are resolved:

1. Implement a production CUDA/vLLM absolute-index generator and direct chunk
   writer that reproduces the pinned oracle, never allocates a whole-surface
   noise/candidate, and passes the two-ULP final-update gate.
2. Add direct per-phase H2D/D2H/PCIe byte and elapsed-time receipts. NVML samples
   alone cannot establish whether factor generation or candidate installation is
   the bottleneck.
3. Complete a dense full-weight capacity preflight. The 535.72 GiB four-master
   requirement and 1.046 TiB/update candidate-write floor make failure likely
   without host/offload/sharding changes; a failed preflight must skip the dense
   arm without changing matched-LoRA work.

After those gates, the preregistered systems rung is one update at seed 1701,
eight mirrored directions, 16 signed candidates, 64 train units per candidate,
and 1,024 equal rollouts per arm. Only the five matched LoRA arms advance to the
quality rung: three seeds, sigma `[0.0006, 0.0003]`, 2,048 rollouts per arm/seed,
fixed SGD/global sigma, and the existing 0.0005 update-norm budget. Protected
holdout access remains prohibited.
