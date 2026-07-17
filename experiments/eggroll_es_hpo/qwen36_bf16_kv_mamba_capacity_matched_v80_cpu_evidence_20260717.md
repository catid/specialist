# Qwen3.6 BF16 hybrid-cache capacity matching V80 CPU evidence

## Outcome

V80 is implemented and CPU-preregistered, but it has not been launched. It
reconstructs the reviewed V78c r1 FP8-weight runtime with auto/BF16 attention
KV and an explicitly overridden BF16 Mamba SSM cache, then changes only
`gpu_memory_utilization`, from `0.500` to `0.479`. The live actor fails during
engine construction unless group-aware capacity is at least 161,792 tokens,
or 79 complete 2,048-token contexts. A projection cannot satisfy that gate.

The preregistration is
`experiments/eggroll_es_hpo/preregistrations/qwen36_bf16_kv_mamba_capacity_matched_v80.json`.
Its canonical content SHA-256 is
`7527ed6fe0154a79ecc0de46b00af4601b0e3deaac184f2af094fba15740149a`;
its file SHA-256 is
`be7de6c8ac1d59feef0ec0d0e289be85612644efbe72fdb3847fa34d5dd50aad`.

## Evidence ancestry

V80 binds only the current reviewed ancestry:

- V76 r7 is the FP8-weight, auto/BF16-attention-KV control at utilization
  0.500. All four actors report 7.77 GiB, 157,696 cache tokens, and 77.00
  complete contexts. Median data-free runtime is 48.5125 seconds. Its sealed
  nine-file bundle is
  `46cf5ab3e6d3688de25cfdcf101710a129fdba309a5f11a9404d17344848e5e6`.
- V78c r1 changes only the Mamba SSM cache from FP32 to BF16. All four actors
  report 7.77 GiB, 218,843 tokens, and 106.86 contexts at utilization 0.500.
  Median runtime is 49.0834 seconds. Its sealed nine-file bundle is
  `a9d82f71bb6beecc420be135737f2048fb770bee5d97309f9e90da4e31ef833f`.
- Every V78c actor resolves attention KV `auto`, Mamba cache `auto`, Mamba SSM
  cache `bfloat16`, FP8 weights, eager execution, FLASH_ATTN attention, and
  TRITON routed FP8 MoE. There was no DeepGemm auto-disable/E8M0, traceback,
  or OOM log fragment.
- Every actor binds the same current named-parameter residency: 813 language
  parameters and 35,712,084,096 logical bytes. No separate vision or MTP
  named-parameter component was visible. V80 requires the exact same identity.
- The model configuration recommends an FP32 Mamba SSM state. V78c logs the
  explicit BF16 override warning exactly once per actor. V80 requires that
  warning rather than suppressing it, and semantic/OOD promotion remains
  blocked pending strict review of this material numerical change.

The V78c probe SHA-256 is
`761857944064a0b21ff528971d3f497e4e67865679fa51a30d385cab65835dcb`.
Unreviewed later runs are not launch ancestry.

## Why utilization 0.479

Each sealed GPU has 97,887 MiB. Lowering utilization from 0.500 to 0.479
removes 2,055.627 MiB from vLLM's requested memory budget. Scaling from the
V78c observation of 7.77 GiB and 218,843 tokens projects 5.76255 GiB and
162,303.0 raw tokens. Conservatively retaining only complete 2,048-token
contexts gives 79 contexts or 161,792 tokens, which is two contexts and 4,096
tokens above the V76 floor.

The next lower 0.001-grid point, 0.478, removes 2,153.514 MiB and projects
159,610.6 raw tokens. That is only 77 complete contexts, or 157,696 tokens,
and fails the required margin. Therefore 0.479 is the minimum preregistered
grid point eligible for a live test. This interpolation selects the arm; it is
not a live capacity result.

V78c's legacy NVML samples peaked at 50,808 MiB on each GPU and ended with two
zero-utilization batches at no more than 5 MiB. Subtracting the released budget
projects a V80 peak of 48,752.373 MiB, but that value is deliberately not an
acceptance gate. V80 must measure its actual peak, stay no higher than the
V78c peak, and retain at least 47,079 MiB of physical headroom per actor.

## Wrapper and telemetry contract

The additive V80 wrapper reconstructs the V78c runtime directly around the
sealed V73 data-free engine call. It does not import the V79 runtime wrapper or
rewrite any historical V76/V78c certificate. The actor independently attests:

- serialized block-FP8 weights, eager FCFS execution, TRITON routed MoE,
  FLASH_ATTN attention, one GPU LoRA slot, `max_num_seqs=68`, and the unchanged
  adapter workload, prompt-free call plan, and seeds;
- utilization 0.479, attention cache `auto` (effective BF16), Mamba cache
  `auto`, Mamba SSM cache `bfloat16`, disabled FlashInfer autotuning, and at
  least 161,792 live group-aware cache tokens with concurrency at least 79;
- the exact routed-expert and named-parameter residency identities;
- ten timed generation calls (two warmups and eight measured), generated
  tokens per second, and median, nearest-rank p95, and maximum call latency;
- only token counts and SHA-256 output commitments, never prompts, generated
  text, or token IDs.

The launcher starts one actor on each physical GPU concurrently and starts the
reviewed NVML monitor at 2 Hz. The monitor SHA-256 is
`6035eb32f90815ed2a2d8734d9e9072123b8ecc70d74449a2710e94b673ed3df`.
It binds actor-root ancestry, retains already attributed engine PIDs through
reparenting, and reports GPU/HBM utilization, VRAM, power, and PCIe RX/TX for
each physical device. Missing PCIe counters fail closed. Post-run integration
must label PCIe byte totals as sampled left-rectangle estimates; HBM bytes/s
must not be inferred from `utilization.memory`. The in-process pynvml observer
adds a reproducible 598 MiB to its own raw memory view on this host, so the
cleanup-only 4 MiB/0% gate uses an external unitless `nvidia-smi` snapshot
while retaining the raw pynvml value in the evidence. This distinction was
sealed before live V80 execution.

## Acceptance and promotion gates

All four actors must self-hash, bind distinct root PIDs to physical GPUs 0-3,
show positive GPU and memory utilization, have no foreign compute PID, and
agree between live engine capacity fields and logs. Median data-free runtime
must be at most 50.5559 seconds (1.03 times V78c r1). Actual VRAM, throughput,
tail latency, PCIe, power, GPU utilization, and memory utilization are all
reported and gated.

Required log fragments include TRITON routed MoE, FLASH_ATTN, the exact FP32
model-config/BF16-user-override warning, cache GiB, cache tokens, concurrency,
and disabled FlashInfer autotuning. DeepGemm auto-disable/E8M0, traceback, and
OOM fragments are forbidden. Cleanup requires engine shutdown, process-group
destruction, no remaining actor/descendant compute PID, and three consecutive
post-exit batches at zero GPU utilization and at most 4 MiB used. Failure to
reach that idle state within 60 seconds fails closed.

The data-free output check requires candidate-state separation, exact
candidate repeats, and a paired V78c/V80 token-hash agreement/drift matrix
that does not conceal reference-repeat nondeterminism. Only after every
data-free gate passes may a paired source-disjoint semantic check run (mean
delta at least -0.001 and paired 95% lower bound at least -0.002, with no new
safety failure). Protected OOD is one-shot after selection freeze (paired 95%
lower bound at least -0.005, worst-stratum point delta at least -0.01, and no
new safety failure), with no retuning after access.

The explicit BF16 Mamba SSM override and any unresolved reference-restore
drift require explicit review. Scored evaluation, training, checkpointing, and
production promotion remain false by default.

## Exact live command

This sealed command was not executed by the CPU work:

```bash
RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/v80_bf16_kv_mamba_capacity_0479_r1 bash /home/catid/specialist/launch_qwen36_bf16_kv_mamba_capacity_v80.sh
```

The launcher SHA-256 is
`3a335c0e44a9b8130adec8c0e11f52d2246264aa0a47f5b173cbdb35093fc7ad`.

## CPU validation

The validation initializes no GPU, opens no dataset or protected split,
performs no training/model update, writes no checkpoint, promotes no
configuration, and modifies no site package.

```text
es-at-scale/.venv/bin/python -m pytest -q \
  test_probe_vllm_quantized_adapter_switch_v73.py \
  test_probe_vllm_fp8_rightsized_v74.py \
  test_probe_vllm_fp8_attested_v76.py \
  test_probe_vllm_bf16_kv_mamba_bf16_v78c.py \
  test_probe_vllm_bf16_kv_mamba_capacity_v80.py \
  test_build_qwen36_bf16_kv_mamba_capacity_preregistration_v80.py \
  test_monitor_qwen36_fp8_kv_capacity_v79.py
36 passed in 1.02s

python3 build_qwen36_bf16_kv_mamba_capacity_preregistration_v80.py --check
status=cpu_preregistered_live_launch_not_performed

python3 -m py_compile \
  probe_vllm_bf16_kv_mamba_capacity_v80.py \
  build_qwen36_bf16_kv_mamba_capacity_preregistration_v80.py \
  monitor_qwen36_fp8_kv_capacity_v79.py

bash -n launch_qwen36_bf16_kv_mamba_capacity_v80.sh
```
