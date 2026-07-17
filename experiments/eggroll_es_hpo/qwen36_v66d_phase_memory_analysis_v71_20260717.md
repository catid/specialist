# Qwen3.6 V66d phase/VRAM/memory-bandwidth analysis V71

Date: 2026-07-17 UTC

Status: accepted CPU-only analysis of a successful sealed V66d diagnostic run.
This evidence is sufficient to unblock downstream recipe experiments that need
proof of four-GPU residency, phase attribution, and safe capacity. It does not
close `specialist-0j5.14` under its literal acceptance criteria because the run
has no checkpoint phase, allocator/component telemetry, host-memory telemetry,
or exact transfer/HBM timing.

## Result

The analyzer accepted all 644 NVML rows as 161 contiguous, strictly ordered
four-GPU batches over the exact 18-phase epoch sequence (`setup` plus epochs
1-17). It accepted all 16 self-hashed actor CUDA-event receipts: one for every
signed candidate, four actors in each of four waves, with exact PID/GPU
bindings and 64 request outputs, samples, and generated tokens per candidate.
No foreign compute process was observed. The calibration generated 1,024
optimization rollouts/tokens, performed a nonzero update candidate, committed
nothing, exactly restored all four actors, opened no protected/dev/OOD/holdout
data, and returned every GPU to idle cleanup.

The run charged `492.405638704` GPU-seconds (`123.101409676` resident seconds
per GPU) and took `156.60223997599678` wall seconds including load and cleanup.
The analyzer's deterministic content SHA-256 is
`a4503e18cb6185ee872ad571d24a51fd8a7ac5154e1fd69793fba629842166e6`.

## VRAM capacity

All four GPUs reported the same capacity and memory envelope:

| Metric | Each physical GPU |
|---|---:|
| NVML total | 97,887 MiB |
| Setup/install resident allocation | 83,212 MiB |
| Peak device allocation | 84,138 MiB (82.17 GiB) |
| Peak expected-process allocation | 83,528 MiB (81.57 GiB) |
| Candidate/update increase over setup | 926 MiB (0.90 GiB) |
| Minimum device headroom | 13,749 MiB (13.43 GiB; 14.05%) |

The current single-slot mirrored recipe therefore has comfortable measured
capacity on this hardware. The earlier replicated V66 adapter-switch profile
provides the component context NVML cannot recover here: vLLM reported 68.24
GiB for BF16 model loading, eager KV capacity of 6.87 GiB, typical graph KV
capacity of 6.46 GiB, and a typical 0.62 GiB graph pool. The serialized V434
adapter is only 17.3 MiB. Base weights remain the main VRAM floor; the V66d
candidate/update scratch increment is under 1 GiB. The prior two-slot rejection
still applies: preallocating a second vLLM LoRA slot raised model-load
allocation by 3.54 GiB and reduced KV capacity while slowing the workload.

## Phase duration, activity, PCIe, and energy

`Observed span` is first-to-last sample within the phase. `Upper window` is the
first acknowledged sample of the phase to the first acknowledged sample of the
next phase. Because V66d waits for that first four-row sample before launching
phase work, the latter bounds controller work plus the next sampling-barrier
latency. The final abort phase has no following sample barrier and therefore
only an observed-span lower observation. PCIe values are left-rectangle
integrals over adjacent same-phase NVML samples, summed over four GPUs. They are
sampling estimates—not exact byte counts and not lower bounds. NVML memory
utilization is an activity duty cycle, not achieved HBM bandwidth.

| Phase/category | Instances | Observed span (s) | Upper window (s) | Mean GPU util. | Sampled RX (MiB) | Sampled TX (MiB) | Energy (J) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Activate LoRA slot | 1 | 1.259 | 1.676 | 0.19% | 3.06 | 5.85 | 383 |
| Install/certify/reference | 1 | 8.815 | 9.233 | 0.30% | 114.39 | 5,259.21 | 2,671 |
| Candidate materialization | 4 | 18.457 | 20.132 | 0.17% | 392.40 | 2,534.62 | 5,656 |
| Generation | 4 | 2.095 | 3.801 | 11.47% | 49.97 | 19.12 | 635 |
| Exact wave restore | 4 | 14.695 | 16.372 | 2.63% | 179.98 | 1,479.59 | 5,911 |
| Update prepare | 1 | 1.252 | 1.670 | 0.00% | 2.99 | 5.96 | 389 |
| Update execute/all-reduce | 1 | 4.202 | 4.622 | 5.84% | 8,075.28 | 1,496.05 | 1,322 |
| Update abort/final audits | 1 | 9.204 | unavailable | 0.07% | 167.29 | 3,599.56 | 2,875 |

The four actor-side generation critical paths were 801.466, 424.407, 420.728,
and 411.533 ms by wave, totaling 2.058 s. Every CUDA event fit both its worker
monotonic receipt and its phase handshake window. Peaks sampled under adjacent
restore labels (for example 97% in wave 1) are consistent with NVML's averaging
window crossing a short generation boundary; actor events, not the lagged
label, are the authoritative work attribution.

Across all phases, the sampled integrals were 9,421,838,461 RX bytes (8.77 GiB)
and 15,099,460,768 TX bytes (14.06 GiB). The update-execute phase alone accounts
for 8,075 MiB sampled RX, far above the logical CPU update return, which is
consistent with collective communication traversing the PCIe-visible path.
That is an inference from phase alignment, not a link-level NCCL byte receipt.

## Conservative implementation-derived transfer ledger

The independent byte ledger binds the exact V66d/V66/V41A code hashes, the 70
FP32 canonical tensors (4,528,128 elements; 18,112,512 bytes), the 82 packed
BF16 runtime views (4,921,344 elements; 9,842,688 bytes), and the 23 audited
base tensors (142,999,552 BF16 elements; 285,999,104 bytes). A successful run
proves 60 runtime materialization/audit calls across four actors:

- 12 during install, post-install certificate, and reference capture;
- 16 candidate materializations;
- 16 exact wave restores;
- 4 update-execute materializations; and
- 12 update-abort rollback, unconditional restore, and final-certificate calls.

Each runtime call writes every BF16 view H2D, reads it D2H for `torch.equal`,
reads it D2H again for SHA-256, and then hashes all relevant frozen base weights
through CPU bytes. Candidate formation additionally copies the FP32 master H2D
and the candidate back D2H; update execution returns one reduced FP32 adapter
per actor. These successful-path lower bounds exclude allocator/protocol
overhead, vLLM activation, NCCL/NVLink/peer traffic, model load, KV allocation,
generation, cleanup, checkpoint traffic, and failure-only repairs.

| Logical component | Direction | Lower-bound bytes | GiB | Share of bidirectional lower bound |
|---|---|---:|---:|---:|
| Frozen base exact-hash readback | D2H | 17,159,946,240 | 15.981 | 87.62% |
| Runtime equality + SHA readbacks | D2H | 1,181,122,560 | 1.100 | 6.03% |
| Candidate FP32 return | D2H | 289,800,192 | 0.270 | 1.48% |
| Reduced update FP32 return | D2H | 72,450,048 | 0.067 | 0.37% |
| Runtime BF16 materialization | H2D | 590,561,280 | 0.550 | 3.02% |
| Candidate FP32 master upload | H2D | 289,800,192 | 0.270 | 1.48% |
| **Total H2D** | H2D | **880,361,472** | **0.820** | **4.50%** |
| **Total D2H** | D2H | **18,703,319,040** | **17.419** | **95.50%** |
| **Bidirectional total** | both | **19,583,680,512** | **18.239** | **100%** |

The sampled 14.06 GiB TX estimate being lower than the 17.42 GiB D2H logical
minimum is not a contradiction: samples omit phase boundaries and short
bursts. Conversely, sampled RX exceeds the 0.82 GiB enumerated H2D minimum
because the conservative ledger intentionally excludes collective, load,
activation, allocator, and protocol traffic.

## Top bottlenecks and ordered actions

1. **Repeated frozen-base audits dominate proven host bandwidth.** The current
   call graph forces at least 15.98 GiB D2H solely to re-hash 23 unchanged base
   tensors, 87.62% of the logical transfer minimum. Install and final abort
   also show 5.26 GiB and 3.60 GiB sampled TX. Retain pointer/storage/version
   invariants at every transition, but move full content audits to sealed trust
   boundaries or compute a fused on-device digest. Unknown RPC completion must
   still trigger exact restore or terminal poison, and the final boundary must
   remain exact.

2. **Update collective traffic is the largest sampled RX burst.** The 4.622 s
   update window carries about 7.89 GiB sampled RX and 1.46 GiB TX despite only
   69.1 MiB of logically required FP32 CPU return across actors. Before changing
   math, capture NCCL algorithm/link/channel receipts and an in-process paired
   profile. Then test topology-aware sharding or bounded overlap; retain native
   tensor boundaries because the V68 paired flat-buffer benchmark was a tie and
   a flat shadow costs 545.5 MiB per rank.

3. **Per-view materialize/restore/audit orchestration dominates loop time.**
   Candidate materialization and restores consume at most about 36.5 s across
   four waves, versus 2.058 s of actor CUDA generation critical path. Batch the
   82 view writes/readbacks, use pinned bounded buffers and streams where a host
   copy remains necessary, cache immutable CPU master identities, and test a
   seeded in-place GPU candidate path. Preserve a bounded single resident slot,
   exact candidate/restore identities, and one final full audit.

For VRAM, quantizing the 68.24 GiB frozen base has much more leverage than
shrinking the 17.3 MiB serialized adapter. The current 13.43 GiB minimum
headroom is safe for the sealed single-slot recipe, so capacity work should
focus on frozen-base precision and right-sized KV/graph pools rather than
adding resident LoRA slots.

## Remaining acceptance blockers

The Bead remains `IN_PROGRESS` for these exact gaps:

1. No checkpoint was created (`checkpoint_count=0`), so checkpoint publication
   memory, copy bytes, latency, atomicity, and failure behavior are unmeasured.
2. No host RSS/USS, pinned allocator, NUMA, or per-actor CPU-master telemetry
   separates four FP32 masters, clones, hashes, and update buffers.
3. NVML total/process memory does not split base model, KV cache, LoRA slots,
   CUDA graph pools, candidate/noise scratch, or NCCL workspaces. A CUDA
   allocator snapshot/receipt is still needed for those component peaks.
4. PCIe throughput sampling is not CUPTI/Nsight transfer timing or an HBM
   roofline. NCCL peer/link bytes, achieved HBM GB/s, stalls, and overlap remain
   unknown.
5. Exact audit time is combined with install/materialize/restore/abort labels,
   and reward/judge work is combined with generation resolution. Dedicated
   worker subphase events are required to attribute their time directly.

These do not block the already sealed downstream experiments from using the
measured 13.43 GiB capacity margin and V66d's accepted four-GPU activity gate.

## Reproduction and adversarial validation

Files added by this analysis:

- `analyze_v66d_phase_memory_v71.py`
- `test_analyze_v66d_phase_memory_v71.py`
- `preregistrations/v66d_phase_memory_analysis_v71.json`

The post-run analysis contract content SHA-256 is
`e8c2ecc2fafc2ded90ff7cde57f028f83720725a38692865bf1d01f8f26427de`.
The analyzer rejects missing/duplicate/reordered rows, incomplete batches,
phase/epoch drift, foreign PIDs, missing residency, invalid process memory,
schema extras, naive timestamps, missing/hash-drifted receipts, CUDA events
outside worker or phase evidence, output-cardinality drift, actor/GPU binding
drift, duplicate assignments, adapter-metadata drift, and completion-proof
drift. Partial PCIe support is reported as unavailable rather than fabricated.

Validation:

- `.venv/bin/pytest -q test_analyze_v66d_phase_memory_v71.py`: 19 passed.
- `python3 -m py_compile analyze_v66d_phase_memory_v71.py test_analyze_v66d_phase_memory_v71.py`: passed.
- `python3 analyze_v66d_phase_memory_v71.py --check-preregistration`: passed.
- `git diff --check` on the V71 files: passed.

Bound V66d artifact SHA-256 values:

- report: `12a5e854856d28bd8439cf3ed004664317086f8d117ae08e78b59f857f6102bb`
- population: `9d172d15f82a54c697b8b860ff3131733d59006f1e4b790b5b9b87ded679e9d4`
- update: `f958f90b26c5b2afa4a81b03a0ab91c12d9684c2ce236bbb658d674e7a5eeffd`
- NVML telemetry: `a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475`
- actor receipts: `aa10617c347b7ce5449165580dd4eaa98bb5131cfde5fcf9cda1134b380390e0`

No GPU was initialized by the analyzer or tests, and no dataset, dev, OOD,
holdout, or protected content was read.
