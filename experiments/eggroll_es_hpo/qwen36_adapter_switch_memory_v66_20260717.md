# Qwen3.6 V66 adapter-switch memory and utilization evidence

Date: 2026-07-17 UTC

Status: diagnostic evidence only. The compiled/CUDA-graph arm is **not**
promoted, and this report does not establish token identity, semantic quality,
holdout equivalence, OOD equivalence, or end-to-end ES-training speed.

## Result

Across three four-GPU repetitions per arm (12 independent single-GPU actors per
arm), the compiled/CUDA-graph bundle reduced the median timed generation phase
from 46.090 s to 23.550 s: a 48.9% runtime reduction, or 1.957x the eager
throughput for this fixed synthetic workload. Its median resident plateau and
peak were both 81,944 MiB (80.02 GiB), versus 83,820 MiB (81.86 GiB) for eager,
a reduction of 1,876 MiB (1.83 GiB, 2.24%). Plateau GPU utilization increased
from a median 52.43% to 91.89%.

This is not an isolated CUDA-graph comparison. With vLLM 0.25.0,
`enforce_eager=False` also enabled the vLLM compilation path and changed kernel
configuration. The result therefore belongs to the complete
**compiled/CUDA-graph bundle**, not to graph capture alone.

The bundle also changed generated token hashes. In particular, its candidate
first call differed from the eager anchor on 8 of 68 rows, and the number of
rows separating the reference and candidate adapters fell from a median 7 to a
median 4. These differences may be numerically small greedy-decoding boundary
effects, but only token hashes were retained, so their semantic or reward impact
cannot be evaluated from these artifacts. Performance is promising; quality
equivalence is unproven.

## Scope and fixed workload

The sealed eager baseline is
`preregistrations/qwen36_memory_bandwidth_baseline_v66.json`. The six analyzed
run directories are:

- E1: `runs/v66_four_gpu_adapter_switch_memory_baseline`
- E2: `runs/v66_four_gpu_adapter_switch_memory_eager_r2`
- E3: `runs/v66_four_gpu_adapter_switch_memory_eager_r3`
- G1: `runs/v66_four_gpu_adapter_switch_memory_graph`
- G2: `runs/v66_four_gpu_adapter_switch_memory_graph_r2`
- G3: `runs/v66_four_gpu_adapter_switch_memory_graph_r3`

Each actor used one physical GPU, Qwen3.6-35B-A3B in BF16, tensor parallelism
1, vLLM 0.25.0, the Triton MoE backend, `max_model_len=2048`,
`max_num_seqs=68`, `gpu_memory_utilization=0.82`, one GPU-resident LoRA slot,
and two CPU LoRA slots. Four actors ran concurrently on physical GPUs 0-3. Each
actor processed two warm-up calls followed by the fixed eight-call sequence
`reference, candidate, candidate, reference, reference, candidate, candidate,
reference`; every call generated 64 tokens for each of 68 synthetic prompts.
No training dataset, holdout, protected OOD set, or terminal set was opened, and
no model or adapter update was performed.

The two adapter weight files were only about 17.3 MiB apiece. The checkpoint was
66.97 GiB on disk, and every engine log reported 68.24 GiB of device memory for
model loading. Thus these measurements primarily reflect base-model, runtime,
and KV-cache residency rather than LoRA parameter size.

## Measurement definitions

- **Runtime** is the receipt field
  `wall_runtime_seconds_excluding_model_load_and_cleanup`. The timer starts
  after engine construction and ends after all ten generation calls, before
  shutdown. It therefore excludes model loading, compilation, CUDA-graph
  capture, and cleanup.
- Device telemetry came from each run's `nvidia_smi_samples.log`. Samples were
  approximately 0.34-0.49 s apart; the nominal 250 ms request was not achieved
  because command and collection latency are included.
- **Resident samples** are device samples with more than 1 GiB allocated.
- **Steady memory** is the modal allocated-memory value among resident samples
  for that actor. This identifies the dominant post-initialization plateau and
  avoids treating staged model-load allocations as steady state.
- **Peak memory** is the maximum allocated memory among resident samples.
- **GPU util.** and **memory activity** are means over samples at that actor's
  steady-memory plateau. NVIDIA's `utilization.memory` is an activity-duty
  metric, not achieved GB/s and not a roofline measurement. It must not be read
  as a percentage of peak HBM bandwidth.
- **Ref Δ** and **Cand Δ** are the receipt counts of rows whose token hash
  changed across repeated calls of the same adapter. **R/C Δ** is the row
  count separating the first reference and first candidate calls.

## Per-run, per-GPU evidence

| Arm | Run | GPU | Runtime (s) | Steady (MiB) | Peak (MiB) | GPU util. (%) | Memory activity (%) | Ref Δ | Cand Δ | R/C Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | E1 | 0 | 45.871 | 83,820 | 83,820 | 51.7 | 26.6 | 1 | 0 | 7 |
| eager | E1 | 1 | 46.593 | 83,820 | 83,820 | 52.3 | 29.5 | 1 | 0 | 7 |
| eager | E1 | 2 | 45.855 | 83,820 | 83,820 | 52.7 | 32.1 | 1 | 0 | 7 |
| eager | E1 | 3 | 46.082 | 83,820 | 83,820 | 51.5 | 33.3 | 1 | 0 | 6 |
| eager | E2 | 0 | 46.692 | 83,820 | 84,377 | 52.0 | 27.5 | 1 | 0 | 7 |
| eager | E2 | 1 | 45.648 | 83,820 | 83,820 | 53.5 | 29.7 | 2 | 0 | 8 |
| eager | E2 | 2 | 46.789 | 83,820 | 83,820 | 52.6 | 32.1 | 1 | 0 | 7 |
| eager | E2 | 3 | 46.098 | 83,820 | 83,820 | 53.4 | 34.2 | 0 | 0 | 7 |
| eager | E3 | 0 | 47.458 | 83,820 | 83,820 | 52.9 | 27.8 | 2 | 0 | 6 |
| eager | E3 | 1 | 46.554 | 83,902 | 83,902 | 50.3 | 26.7 | 1 | 0 | 7 |
| eager | E3 | 2 | 45.723 | 83,820 | 83,820 | 52.6 | 32.1 | 2 | 0 | 7 |
| eager | E3 | 3 | 45.488 | 83,820 | 83,820 | 52.0 | 33.4 | 1 | 0 | 7 |
| compiled/graph | G1 | 0 | 23.566 | 81,404 | 81,404 | 91.9 | 57.9 | 1 | 0 | 4 |
| compiled/graph | G1 | 1 | 22.897 | 81,944 | 81,944 | 91.0 | 62.9 | 0 | 0 | 4 |
| compiled/graph | G1 | 2 | 23.660 | 81,944 | 81,944 | 89.7 | 65.5 | 1 | 0 | 4 |
| compiled/graph | G1 | 3 | 24.711 | 81,948 | 81,948 | 91.9 | 69.8 | 1 | 0 | 4 |
| compiled/graph | G2 | 0 | 23.605 | 81,944 | 81,944 | 92.9 | 58.6 | 0 | 0 | 4 |
| compiled/graph | G2 | 1 | 22.852 | 81,944 | 81,944 | 92.5 | 62.4 | 1 | 0 | 3 |
| compiled/graph | G2 | 2 | 23.691 | 81,944 | 81,944 | 92.7 | 66.8 | 1 | 0 | 3 |
| compiled/graph | G2 | 3 | 23.573 | 81,944 | 81,944 | 92.1 | 69.8 | 1 | 0 | 4 |
| compiled/graph | G3 | 0 | 23.506 | 81,944 | 81,944 | 90.9 | 57.8 | 0 | 0 | 4 |
| compiled/graph | G3 | 1 | 22.801 | 81,944 | 81,944 | 92.0 | 60.5 | 0 | 0 | 4 |
| compiled/graph | G3 | 2 | 23.535 | 81,944 | 81,944 | 90.6 | 65.5 | 1 | 0 | 3 |
| compiled/graph | G3 | 3 | 23.401 | 81,944 | 81,944 | 90.2 | 68.8 | 1 | 0 | 3 |

## Aggregate evidence

Aggregates below are medians and full ranges across the 12 actor receipts in
each arm. Utilization and power first reduce each actor's plateau samples to a
mean, then aggregate those 12 actor means.

| Metric | Eager median [range] | Compiled/graph median [range] | Observed bundle effect |
|---|---:|---:|---:|
| Timed generation runtime (s) | 46.090 [45.488, 47.458] | 23.550 [22.801, 24.711] | -48.9%; 1.957x throughput |
| Steady allocated memory (MiB) | 83,820 [83,820, 83,902] | 81,944 [81,404, 81,948] | -1,876 MiB (-2.24%) |
| Peak allocated memory (MiB) | 83,820 [83,820, 84,377] | 81,944 [81,404, 81,948] | -1,876 MiB (-2.24%) |
| Plateau GPU utilization (%) | 52.43 [50.35, 53.53] | 91.89 [89.65, 92.88] | +39.46 percentage points |
| Plateau memory activity (%) | 30.88 [26.57, 34.18] | 64.15 [57.80, 69.81] | +33.28 percentage points |
| Plateau power (W) | 198.12 [182.33, 204.73] | 294.09 [292.24, 299.49] | +95.97 W |
| Reference within-state changed rows | 1 [0, 2] | 1 [0, 1] | Identity not stable in either arm |
| Candidate within-state changed rows | 0 [0, 0] | 0 [0, 0] | Stable within each receipt |
| Reference/candidate differing rows | 7 [6, 8] | 4 [3, 4] | Adapter effect is arm-dependent |

The per-run median runtimes were 45.976, 46.395, and 46.139 s for E1-E3,
versus 23.613, 23.589, and 23.453 s for G1-G3. Thus the runtime result is
replicated rather than being driven by one actor or one run. The single eager
84,377 MiB peak was transient; the corresponding steady plateau was 83,820
MiB.

## Output identity and quality caveats

The workload used greedy decoding (`temperature=0`) with fixed seeds, yet exact
token hashes were not fully deterministic:

- Eleven of 12 eager receipts and eight of 12 compiled/graph receipts changed
  at least one reference row across repeated calls. No receipt changed a
  candidate row across its repeated candidate calls.
- Across the 12 first reference calls, eager produced three distinct aggregate
  row hashes and compiled/graph produced two. Each arm produced one internally
  consistent first-candidate aggregate hash.
- Relative to eager E1/GPU0, every compiled/graph first-candidate call differed
  on 8 of 68 rows. Compiled/graph first-reference calls differed on 1-2 rows.
- The same two adapter artifacts separated 6-8 rows in eager but only 3-4 rows
  in compiled/graph.

The engine configuration explains why this is not a pure capture ablation.
Eager logged compilation mode `NONE` and CUDA-graph mode `NONE`; the other arm
logged vLLM compilation mode 3 and `FULL_AND_PIECEWISE` CUDA graphs, along with
different custom-op and fusion settings. Numerical route or kernel differences
can move greedy decoding across token boundaries. Because decoded text and
rewards were intentionally not persisted, the present evidence cannot say
whether those row changes are harmless, beneficial, or damaging.

Promotion therefore requires an isolated compile-versus-capture experiment and
the preregistered clean holdout/OOD quality gates. Exact cross-arm token identity
would be the strongest fail-closed gate; if exact identity is unattainable, the
minimum acceptable substitute is replicated semantic/reward non-inferiority
with the same prompts, sampling payload, judging payload, and compute budget.

## Credible bottlenecks and ordered optimization implications

1. **Eager execution is dispatch/launch limited on this decode workload.** Its
   steady GPU utilization was only about 52%, while the compiled/graph bundle
   reached about 92% and nearly doubled throughput. Memory activity increased
   at the same time, so the eager arm was not saturating either compute or HBM
   activity. Persistent compiled/graph execution is the strongest measured
   speed candidate, subject to the numeric-quality gate above.

2. **Base-weight residency is the dominant VRAM floor.** The engine reports
   68.24 GiB for model loading, about 83% of eager's 81.86 GiB plateau. The two
   LoRA files together are only about 34.6 MiB. Quantizing the frozen base is
   therefore the highest-leverage capacity and likely weight-traffic ablation;
   shrinking LoRA state alone cannot materially change the observed footprint.
   This probe did not measure achieved HBM GB/s, so a bandwidth-speed claim for
   quantization remains a hypothesis to test.

3. **KV-cache policy consumes several GiB and differs between arms.** Every
   eager actor reported 6.87 GiB available for 139,264 KV tokens (68.0x
   2,048-token concurrency). The compiled/graph median was 6.46 GiB, 131,072
   tokens, and 64.0x; G1/GPU0 was the lone lower outlier at 5.91 GiB, 120,012
   tokens, and 58.6x. Every graph actor estimated 0.55 GiB of graph memory. Eleven
   then reported a 0.62 GiB actual graph pool, while G1/GPU0 reported 0.07 GiB
   and also had the 540 MiB lower resident plateau. Right-sizing
   `max_model_len`, `max_num_seqs`, and `gpu_memory_utilization` to real ES wave
   shapes, and explaining this first-run graph/KV allocation outlier, are
   credible ways to recover VRAM. They require a prompt length/concurrency sweep
   because under-sizing KV can lower throughput or reject workloads.

4. **Alternating two adapters through one GPU LoRA slot may create avoidable
   host/device traffic.** The probe sets `max_loras=1`, `max_cpu_loras=2` while
   alternating adapters. Resident dual-slot execution could remove swaps, but
   this log has no PCIe byte counters and cannot prove that swaps are currently
   expensive. Compare one versus two resident slots with phase-tagged PCIe RX/TX
   counters and identical token hashes before adopting it.

5. **Engine setup must be amortized.** Model loading took 11.02-12.37 s per
   actor. The compiled/graph logs additionally report 7-9 s of graph capture,
   excluded from the timed runtime. Actual graph-pool memory was 0.62 GiB for 11
   actors and 0.07 GiB for the G1/GPU0 outlier. Long-lived engines and resident
   candidate slots are necessary for the measured steady-state win to survive
   at short ES horizons.

6. **The present telemetry is not a bandwidth roofline.** It lacks HBM bytes,
   PCIe bytes, per-phase CPU wait, adapter-install timing, perturbation/update
   timing, checkpoint timing, and judge timing. A phase-tagged NVML/CUPTI follow-up
   should separate load, graph capture, adapter install/restore, prompt prefill,
   decode, reward/judge, ES update, and cleanup. Until then, the memory-activity
   percentages identify underutilization but not the byte source responsible.

## Two-resident-slot follow-up

A later single-variable diagnostic changed only `max_loras` from 1 to 2, while
retaining `max_cpu_loras=2`, both adapter artifacts, prompts, call order, seeds,
and all other engine settings. One four-GPU eager run and two four-GPU graph
runs were completed. This follow-up rejects the initial resident-slot
optimization for the current vLLM/Qwen configuration:

- The eager median timed runtime increased from 46.090 s with one GPU slot to
  80.096 s with two slots (+73.8%).
- The graph median increased from 23.550 s to 31.965 s across the eight
  two-slot actors (+35.7%). The two run medians were 32.249 s and 31.838 s.
- vLLM reported that model loading occupied 71.78 GiB with two slots rather
  than 68.24 GiB with one: an additional 3.54 GiB even though each serialized
  adapter weighs only about 17.3 MiB. At fixed `gpu_memory_utilization=0.82`,
  eager available KV cache fell from 6.87 GiB/139,264 tokens to 3.32
  GiB/67,174 tokens. Graph available KV fell from a typical 6.46
  GiB/131,072 tokens to 2.91 GiB/58,982 tokens.
- Repeated-output stability did not improve. The second graph replication had
  one actor change three of 68 candidate rows across nominally identical
  candidate calls, and its reference/candidate separation fell from seven rows
  to four. The other seven graph actors separated seven rows.

The most likely interpretation is that vLLM's configured LoRA capacity
preallocates much larger runtime workspaces than the adapter file size suggests,
reducing KV headroom and making the LoRA kernels slower at `max_loras=2`.
Avoiding adapter transfers did not offset that cost. Direct PCIe counters are
still needed to quantify the transfers themselves, but they cannot reverse the
observed latency, capacity, and repeatability rejection. The production search
should retain one GPU LoRA slot and investigate bounded in-place state writes or
a custom compact candidate buffer rather than increasing `max_loras`.

## Static sequence-wave follow-up

A second data-free diagnostic retained the 68 submitted prompts but reduced
vLLM's concurrent `max_num_seqs` from 68 to either 48 or 32. This reduces graph
capture sizes and makes vLLM schedule the fixed request batch in internal waves.
Two four-GPU runs were collected for the 32-sequence condition; the 48-sequence
condition has one four-GPU run and is descriptive only.

| Sequence cap | Actors | Median timed runtime | High-utilization memory plateau | Difference from cap 68 |
|---:|---:|---:|---:|---:|
| 68 | 12 | 23.550 s | 81,944 MiB | reference |
| 48 | 4 | 28.989 s | 81,826 MiB | +23.1% runtime; -118 MiB |
| 32 | 8 | 31.972 s | 80,835 MiB | +35.8% runtime; -1,109 MiB |

The table deliberately uses the high-utilization generation plateau. An early
live snapshot around 71-73 GiB occurred during compilation/warm-up and is not a
valid generation-memory result. Phase-aligned parsing corrects that misleading
snapshot.

The 32-sequence cap reduced the actual graph pool from the usual 0.62 GiB to
0.36 GiB and increased available KV capacity from a typical 6.46 GiB/131,072
tokens to 6.66 GiB/135,168 tokens. The 48-sequence condition used a 0.47 GiB
graph pool and exposed 6.51 GiB/132,300 KV tokens. Thus most of the already
small observed memory gain is graph/runtime workspace, not less model weight or
a lower KV allocation target. Halving sequence concurrency is not an attractive
default for only 1.35% lower generation VRAM.

Decode identity again changed with batching. The cap-48 actors separated the
two adapters on 12 of 68 rows, and cap-32 separated them on 8-9 rows, versus
3-4 rows for cap 68. Reference repeats changed 0-1, 1-3, and 0-1 rows in the
48, 32, and 68 conditions respectively; candidate repeats stayed stable in
these runs. No semantic-quality conclusion can be recovered from token hashes.
The static-wave settings are therefore rejected as production defaults, while
length-bucketing remains worth testing on the real prompt distribution because
it can avoid padding/state waste without imposing one low global concurrency
cap.

## Compilation-without-CUDA-graphs isolation

V72 separated vLLM compilation from CUDA-graph replay by resolving
`CompilationMode.VLLM_COMPILE` together with `CUDAGraphMode.NONE` in every live
engine.  It retained the same model, adapters, 68 prompts, seeds, call order,
eager request execution, and one-slot LoRA configuration.  One four-GPU
diagnostic produced these actor runtimes: 46.294, 45.497, 49.505, and 46.279 s.
The median was 46.286 s, effectively the 46.090 s eager baseline rather than
the 23.550 s compiled/graph result.  Compilation by itself therefore did not
produce the observed throughput gain; CUDA-graph replay, or an interaction
requiring it, accounts for nearly all of the measured speedup.

The compile-only memory plateau was 83,694 MiB on all four GPUs, only 126 MiB
below eager and 1,750 MiB above compiled/graph.  Median plateau GPU utilization
was 47.05% and memory activity was 30.48%, again matching the underutilized
eager regime rather than the graph regime.

Numerical behavior did change without CUDA graphs.  Reference/candidate
separation was 3-4 rows and candidate repeats were stable within every actor,
matching the compiled/graph counts rather than eager's 6-8 rows.  Relative to
the eager E1/GPU0 anchor, the first candidate call changed 8-9 of 68 rows.
Three compile-only actors exactly matched the compiled/graph G1/GPU0 candidate
hashes; the fourth differed on three rows.  Reference calls differed from the
eager anchor on 1-2 rows.  This single diagnostic therefore suggests that the
compiled kernels cause most of the token-hash shift while CUDA graphs supply
the speed, but it is not enough to assign every numerical difference to one
component.

The useful follow-up is the converse isolation: compilation disabled with a
supported full-decode graph mode, with resolved configuration attestation.  If
vLLM silently disables graphs when compilation is `NONE`, that arm is
infeasible rather than evidence against CUDA graphs.  No compiled or graph arm
is promoted before the registered semantic/reward and OOD gates.

Finally, this is an inference-only adapter-switch feasibility probe. It does not
measure perturbation generation, EGGROLL transforms, parameter installation,
restore, aggregation, optimizer state, checkpointing, reward computation, or
failure recovery. Those phases can reverse the end-to-end ranking and must be
included before making a trainer-level speed or VRAM claim.
