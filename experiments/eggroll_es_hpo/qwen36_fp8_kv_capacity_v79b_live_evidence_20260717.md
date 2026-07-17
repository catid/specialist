# Qwen3.6 FP8 attention-KV capacity-matched V79B live evidence

## Outcome

V79B passes its **data-free runtime, capacity, backend, activity, VRAM, log,
PID-attribution, and corrected external-cleanup gates**.  It does not authorize
scored evaluation, training, checkpointing, or a production-layout promotion.
Reference restores remain token-hash-nonexact, and neither the source-disjoint
semantic gate nor the one-shot protected OOD gate has run.

The accepted run is
`v79b_fp8_kv_capacity_0485_r5_sealed_cleanup`.  Its eleven-file bundle SHA-256
is `17a4f99164cd17f8e55ddcd879460c99e356e34d8e90e017abbbd7597085dde7`.
The deterministic postrun receipt is
`qwen36_fp8_kv_capacity_v79b_postrun_20260717.json`, with canonical content
SHA-256
`dc2c3f47f28bd74bec0ddb385652c6263d38328f637f31c6769b1e48277ed46a`.

## Evidence classes

| Evidence | Permitted use | Cleanup status |
|---|---|---|
| V79 `r1_fix`, `r2`, `r3` | Same-model performance replicates | The old monitor observed its own 602 MiB NVML context and did not record the corrected external cleanup fields; these are not V79B acceptance runs. |
| V79B `r5_sealed_cleanup` | Full preregistered data-free runtime acceptance | Three consecutive external `nvidia-smi` batches satisfy all four GPUs at no more than 4 MiB, exactly 0% utilization, dead actor roots, empty compute PID sets, and no foreign PID. |
| V79 `r1` | None | Incomplete failed first attempt. |
| V79 `r4_cleanup` | Diagnostic history only | It preceded the immutable V79B cleanup preregistration. |

The analyzer rejects substituting an older run for r5.  It also seals each
postrun directory inventory so a receipt, log, telemetry row, or PID-map change
invalidates the analysis.

## Immutable contract and exact runtime

The V79B preregistration has content SHA-256
`7669c2f720f2a0d17e976de42cc5b7c08fba60a3251175a62eddf05de2dc1b5d`
and file SHA-256
`8e1940db5134bb77ef9959d10b4eec5d43fab4e8653d62733b42939b5fd7300f`.
The accepted run binds these exact sources:

- `probe_vllm_fp8_kv_capacity_v79.py`:
  `6b72de1bd7d7878ba4183bae618108f8cd1cf997e33c7447ee2459700e15ff45`
- `monitor_qwen36_fp8_kv_capacity_v79.py`:
  `6035eb32f90815ed2a2d8734d9e9072123b8ecc70d74449a2710e94b673ed3df`
- `launch_qwen36_fp8_kv_capacity_v79.sh`:
  `4ca93e3a171787bb56613bf3648365ae96a355e28ec95f094b12d1982b6772df`

All four actor receipts self-hash and independently resolve:

- serialized block-FP8 model weights and 40 physical routed expert owners;
- `TRITON`/`TritonExperts` routed MoE execution, with
  `FusedMoEModularMethod` wrapping `Fp8MoEMethod`;
- `[128, 128]` FP8 weight blocks and FP8 E4M3 W13/W2 tensors;
- `TRITON_ATTN` attention;
- `fp8_per_token_head` attention KV, dynamic per-token-head scales, no skipped
  layers, and FP32 Mamba SSM state;
- only the language named-parameter component: 813 parameters and
  35,712,084,096 logical bytes;
- eager FCFS execution, 68 sequences, one GPU LoRA slot, two CPU LoRA slots,
  and `gpu_memory_utilization=0.485`.

Each actor log contains exactly one required TRITON routed-MoE line, one
TRITON_ATTN line, one live capacity/concurrency set, and one disabled
FlashInfer-autotune line.  There are zero DeepGemm auto-disable/E8M0, traceback,
OOM, or broad fallback fragments.  Default MoE and LoRA kernel warnings remain
visible: one default-MoE warning, one default-LoRA warning, and six missing-LoRA
configuration warnings per actor.  They were not hidden to pass the gate.

## Capacity, performance, and VRAM

| Arm | KV tokens per actor | Complete 2,048-token contexts | Three/four-replicate actor median | Peak VRAM evidence |
|---|---:|---:|---:|---:|
| V76 BF16 attention KV | 157,696 | 77 | 48.584 s across 12 actors | 50,858 MiB external total in paired r7 |
| V78 FP8 attention KV at 0.500 | 198,656 | 97 | 49.102 s across 12 actors | 50,856 MiB external total in paired r3 |
| V79/V79B FP8 attention KV at 0.485 | 162,304 | 79 floor (79.25× reported) | 48.984 s across 16 actors | 49,994 MiB external total; 49,384 MiB attributed actor process in accepted r5 |

V79B exceeds the 161,792-token live gate by 512 tokens and V76 by 4,608
tokens, or 2.92%.  It removes 36,352 of V78's excess tokens and retains 81.70%
of V78 capacity.  This is the intended capacity match: two complete contexts
remain above V76 without reserving V78's additional 18.3% capacity.

The three old-monitor V79 timing replicates have run medians 48.891, 48.893,
and 49.350 seconds.  The fully accepted r5 median is 49.100 seconds.  Across all
16 same-arm actors, V79/V79B is 0.82% slower than the three-replicate V76
control and 0.24% faster than the three-replicate V78 arm.  Accepted r5 is
0.53% faster than paired V78 r3 and 1.21% slower than paired V76 r7.  It passes
both the absolute 50.841-second and relative 1.03× V78-r3 preregistered gates.

The external V79B peak is 862 MiB below V78 r3 and 864 MiB below V76 r7,
leaving 47,893 MiB physical headroom per GPU.  The process-attributed V79B
actor peak is 1,472 MiB below V78 r3's external peak, matching the intended
budget reduction much more closely.  These are deliberately separate claims:
the V79B in-process NVML observer holds about 598 MiB after actor exit, so the
49,994-MiB external total is the fail-closed gate, while the 49,384-MiB
attributed peak is a diagnostic estimate rather than an interchangeable total.

## Four-GPU activity, traffic, and cleanup

The accepted telemetry contains 528 rows forming 132 exact four-GPU batches.
Sampling intervals range from 0.666 to 0.729 seconds.  Every physical GPU has a
unique UUID, root PID, and engine PID; no row contains a foreign compute PID.

| GPU | Positive GPU-util samples | Positive HBM-util samples | Peak GPU util | Peak HBM util | Peak power | Physical headroom |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 89 | 78 | 100% | 73% | 283.590 W | 47,893 MiB |
| 1 | 87 | 81 | 100% | 66% | 277.979 W | 47,893 MiB |
| 2 | 89 | 76 | 100% | 74% | 283.662 W | 47,893 MiB |
| 3 | 89 | 78 | 100% | 76% | 278.666 W | 47,893 MiB |

The sampled PCIe rates integrate to left-rectangle estimates of 173,112,780,581
RX bytes and 17,645,958,415 TX bytes summed over the GPUs and observed window.
They are estimates, not exact traffic counts.  No HBM bytes/s is reported:
NVML memory-utilization percentages cannot be converted into a byte rate, so a
CUPTI or equivalent counter pass is still required for a true memory-bandwidth
roofline.

Batches 129, 130, and 131 are the required consecutive external cleanup
observations.  All have dead roots, empty compute and foreign PID sets, exactly
0% external GPU utilization, and 4 MiB external memory on GPUs 0 through 3.

## Hash drift and unresolved correctness gate

The matrix compares the same GPU, measured call index, and 68 hash-only output
rows.  Each pair therefore compares 2,176 rows.  A differing commitment is not
automatically a semantic error, but it cannot be silently called exact.

| Pair | Differing rows | Agreement | Differing rows by call (summed over GPUs) |
|---|---:|---:|---|
| V76 r7 vs V78 r3 | 422 | 80.61% | 23, 84, 84, 20, 23, 84, 84, 20 |
| V76 r7 vs V79B r5 | 430 | 80.24% | 29, 84, 84, 18, 20, 84, 84, 27 |
| V78 r3 vs V79B r5 | 72 | 96.69% | 22, 0, 0, 16, 15, 0, 0, 19 |

The V78-to-V79B candidate calls agree on every row.  All 72 differences occur
on reference calls, consistent with the known reference-repeat nondeterminism
rather than a candidate-state discontinuity.  Within V79B, candidate repeats
are exact on all four GPUs; reference repeats change 6, 8, 11, and 8 rows, and
the candidate differs from the reference on 22, 25, 22, and 22 rows.

This is enough to accept the capacity and execution mechanism, but not enough
to select it for scored work.  Next in dependency order is the preregistered
source-disjoint semantic check.  Only after selection is frozen may the
one-shot protected OOD gate open.  Exact reference restore or an explicitly
approved semantic resolution remains mandatory before promotion.

## Deterministic validation

The postrun analyzer imports no GPU or dataset stack.  It validates exact
source and preregistration hashes, every run inventory, actor self-hashes,
hash-only output summaries, backend/capacity logs, PID/GPU binding, complete
telemetry batches, PCIe support, useful activity, absence of foreign PIDs, and
the corrected external cleanup sequence.

```text
es-at-scale/.venv/bin/python -m pytest -q \
  test_analyze_qwen36_fp8_kv_capacity_v79b.py
12 passed in 0.53s

python3 -m py_compile analyze_qwen36_fp8_kv_capacity_v79b.py
python3 analyze_qwen36_fp8_kv_capacity_v79b.py --check
status=data_free_runtime_gates_passed_semantic_ood_and_exact_restore_pending
```
