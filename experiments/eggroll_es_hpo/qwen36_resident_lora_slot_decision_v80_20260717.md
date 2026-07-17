# Qwen3.6 resident LoRA slot decision V80

Date: 2026-07-17 UTC  
Bead: `specialist-0j5.16`  
Disposition: **close as a negative result; reject `max_loras=2`**

## Outcome

The tested two-GPU-slot layout is not a viable optimization for the current
Qwen3.6-35B-A3B/vLLM 0.25 path. It does not duplicate the 66.97 GiB base
checkpoint, but it expands vLLM's preallocated LoRA tensors, including the
large fused-MoE tensors. Live model/LoRA allocation increased from 68.24 to
71.78 GiB (+3.54 GiB), available KV capacity fell by about 52-55%, and the
complete switched-generation workload became substantially slower.

Retain the already supported layout:

```text
max_loras=1
max_cpu_loras=2
base model instances per actor=1
prefix caching=false
```

With only the two reference/candidate adapters in use, vLLM's registered CPU
LRU holds both after their first load. A switch evicts the prior active GPU
slot and copies the selected registered adapter into that one slot. This avoids
both a second GPU base and repeated disk adapter loads. It does **not** eliminate
the CPU-to-GPU adapter copy, whose isolated latency and PCIe bytes were not
measured by V66.

## Bound evidence

The machine decision binds every file in 14 immutable V66 run directories:

- Primary comparator: three one-slot eager runs, three one-slot graph runs, one
  two-slot eager run, and two two-slot graph runs (36 actors).
- Stability supplement: five later one-slot graph runs (20 actors). These are
  reported separately and are not post-selected into the original runtime
  comparator.
- Total: 56 independent GPU actors, with GPUs 0, 1, 2, and 3 represented in
  every run. Every receipt reports engine shutdown, zero dataset rows opened,
  no adapter update/HPO, and no protected holdout/OOD access.

Every arm used the same two adapter hashes, 68 synthetic prompts, BF16 base,
TP=1, `gpu_memory_utilization=0.82`, `max_num_seqs=68`, `max_model_len=2048`,
and prefix caching disabled. The only intended resident-slot change was
`max_loras: 1 -> 2`; `max_cpu_loras` stayed 2.

## Runtime and capacity

| Arm | Runs / actors | Median timed workload | Recorded-call token-rate lower bound | Model/LoRA load | Available KV | KV tokens | Device resident mode |
|---|---:|---:|---:|---:|---:|---:|---:|
| one slot, eager | 3 / 12 | 46.090 s | 755.4 tok/s/actor | 68.24 GiB | 6.87 GiB | 139,264 | 83,820 MiB |
| two slots, eager | 1 / 4 | 80.096 s | 434.7 tok/s/actor | 71.78 GiB | 3.32 GiB | 67,174 | 83,254 MiB |
| one slot, graph | 3 / 12 | 23.550 s | 1,478.4 tok/s/actor | 68.24 GiB | 6.46 GiB median | 131,072 median | 81,944 MiB |
| two slots, graph | 2 / 8 | 31.965 s | 1,089.2 tok/s/actor | 71.78 GiB | 2.91 GiB | 58,982 | 81,186 MiB |

Relative to one slot, two slots increased median complete-workload runtime by
73.78% in eager mode and 35.73% in graph mode. The conservative recorded-call
token rate fell 42.46% and 26.33%, respectively.

The timer is important: it starts before two warmup generations and spans two
first adapter loads, eight recorded generation calls, and six adjacent adapter
state transitions. It excludes base-engine construction and cleanup. Therefore
these are complete switched-workload results, not isolated switch-latency
measurements. The rate is explicitly a lower bound because only the eight
recorded calls' 34,816 output tokens are counted while both warmups remain in
the denominator.

The lower two-slot NVML plateaus do not contradict the +3.54 GiB allocation.
The engine target stayed fixed at 0.82 of physical VRAM, so the extra LoRA
allocation displaced KV instead of raising total steady residency. Compared
with one slot, eager KV tokens fell 51.76% and graph KV tokens fell 55.00%.
The machine artifact preserves both legacy process-memory and device-memory
telemetry schemas and forbids subtracting values across schemas.

## Output behavior and cache isolation

Only token counts and token-ID SHA-256 values were persisted; decoded text,
rewards, and token IDs were intentionally absent. Consequently, no semantic or
quality-equivalence conclusion is possible.

- One-slot eager: candidate repeats were identical in 12/12 actors; reference
  repeats changed 1-2 of 68 rows in 11/12 actors.
- Primary one-slot graph: candidate repeats were identical in 12/12 actors;
  reference repeats changed one row in 8/12 actors.
- Supplemental one-slot graph: candidate repeats were identical in all 20
  actors; reference repeats changed one row in 13/20 actors.
- Two-slot eager: candidate repeats were identical in 4/4 actors; reference
  repeats changed one row in 2/4 actors.
- Two-slot graph: one of eight actors changed 3/68 candidate rows across
  nominally identical candidate calls. Two actors changed one reference row.

Prefix caching was disabled in all 56 actors, removing that cache-reuse path.
There was no direct KV/mapping instrumentation, direct GPU LoRA tensor digest,
or exact restore digest. The nonzero repeat changes mean exact output restore
was not established. This is a rejection signal; it is not evidence that a
different custom buffer would be safe.

## Installed vLLM source audit

Read-only inspection of the exact installed vLLM 0.25 source established:

1. `max_cpu_loras` is the registered-adapter LRU capacity and `max_loras` is
   the active GPU-slot LRU capacity.
2. If an adapter is already registered, the worker touches it instead of
   calling `_load_adapter`; activation then evicts the oldest active slot when
   needed and calls each wrapper's `set_lora`.
3. Linear LoRA A/B tensors have a leading `max_loras` dimension. Fused-MoE
   w13/w2 LoRA tensors also have leading `max_loras` and expert dimensions,
   explaining why a nominally 17.3 MiB serialized adapter can require far more
   slot capacity at runtime.
4. Wrappers retain one `base_layer` reference. `max_loras=2` adds LoRA capacity;
   it does not instantiate a second base model.
5. `load_inplace` is not a lazy optimization here: it explicitly re-enters
   `_load_adapter`, removes the old registered adapter, and replaces it.

The source audit supports the allocation class but not an exact byte-level
causal breakdown. V66 did not collect CUDA allocator categories, so it would be
overclaiming to assign every byte of the 3.54 GiB delta to one tensor family.

## What is and is not decided

Decided:

- Reject two resident GPU LoRA slots for this model/runtime.
- Keep one GPU slot and a two-entry CPU adapter cache.
- Do not use `load_inplace` as a supposed no-reload path.
- Close `specialist-0j5.16` as a tested negative result without claiming that
  its positive acceptance criteria were met.

Not measured and not imputed:

- isolated adapter activation or restore latency;
- per-switch PCIe RX/TX bytes or HBM bytes;
- direct GPU LoRA tensor restore identity;
- direct stale-KV/mapping instrumentation;
- decoded semantic quality, reward, or OOD equivalence.

No supported compact alternate GPU buffer exists in the inspected path. Such a
buffer would be new implementation work, not a safe configuration change. The
forward challenger preregistration requires exact tensor restore, explicit
cache isolation, synchronized switch timers, phase-aligned PCIe traffic,
allocator categories, at least three counterbalanced four-GPU runs per arm,
and source-disjoint semantic/OOD non-inferiority. Missing metrics fail closed.

## Reproducibility and validation

- Decision:
  `experiments/eggroll_es_hpo/decisions/qwen36_resident_lora_slot_decision_v80.json`
  - content SHA-256: `4c255e41d930e2338f5c3a9d02a9cf99671be3b0b7bbdfa4690a7d39aea86b6d`
  - file SHA-256: `93a1386cfde06c65509ea8a43b20bbd50838d6f3e248f7e8bd97d81bf9bf7c8d`
- Future challenger preregistration:
  `experiments/eggroll_es_hpo/preregistrations/qwen36_resident_lora_slot_challenger_v80.json`
  - content SHA-256: `1723fe1be88cdca1c73decd38bfc2a1614a78332ae7670b834396f70816cc4e7`
  - file SHA-256: `5c7c68ae24a5c0c2cb9e62526fb5f40bc38e582665cf3bb1e1e2ceb3c39e5f73`
- Builder SHA-256:
  `24ca3e36e14187509ccfb2cbad6f39c1f99d0f73603a49cc96bc8c992163ee4a`
- Tests: 20 passed with `.venv/bin/pytest -q
  test_build_qwen36_resident_lora_slot_decision_v80.py`.
- Deterministic check: `python3
  build_qwen36_resident_lora_slot_decision_v80.py --check` passed.

This analysis was CPU-only. It did not initialize a GPU, access a dataset or
protected split, mutate V66 artifacts, patch site-packages, or alter
`es-at-scale`.
