# Qwen3.6 LoRA rank/target-surface Pareto CPU evidence (V1)

Status: **CPU preregistered and quarantined; live launch and promotion are not authorized.**

This artifact defines seven LoRA parameter and active-packed-payload budgets around the exact current adapter control. The builder reads sealed metadata and safetensors headers only and opens no dataset row, tensor payload, or GPU.

A broad upstream regression-test invocation outside the builder did open the V1 protected source. This is recorded as a real nonzero access event in `specialist-0j5.30` and the content-addressed incident record. The entire V1 source (all 59 rows, including all 18 legacy heldout candidates) is quarantined. No protected row text escaped the test process or entered these artifacts, but V1 may not be reset to zero or used for terminal evaluation. A fresh untouched V2 contract and rebind are mandatory before any live HPO.

## Exact geometry and byte budgets

| Arm | Rank | Modules | Logical params | Active packed elems | Packing duplicates | FP32 master bytes | Active BF16 bytes | BF16 padding in shared-r64 diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_current_r32_control | 32 | 35 | 4,528,128 | 4,921,344 | 393,216 | 18,112,512 | 9,842,688 | 9,842,688 |
| full_current_r8 | 8 | 35 | 1,132,032 | 1,230,336 | 98,304 | 4,528,128 | 2,460,672 | 17,224,704 |
| full_current_r16 | 16 | 35 | 2,264,064 | 2,460,672 | 196,608 | 9,056,256 | 4,921,344 | 14,764,032 |
| full_current_r64 | 64 | 35 | 9,056,256 | 9,842,688 | 786,432 | 36,225,024 | 19,685,376 | 0 |
| shared_expert_only_r32 | 32 | 12 | 983,040 | 983,040 | 0 | 3,932,160 | 1,966,080 | 1,966,080 |
| attention_gdn_only_r32 | 32 | 19 | 3,250,176 | 3,643,392 | 393,216 | 13,000,704 | 7,286,784 | 7,286,784 |
| frozen_router_dense_r32 | 32 | 31 | 4,233,216 | 4,626,432 | 393,216 | 16,932,864 | 9,252,864 | 9,252,864 |

The deployable measurements use a dedicated engine whose `max_lora_rank` equals the arm rank, so selected-wrapper rank padding is exactly zero. The final column is a diagnostic showing how much selected-wrapper padding a shared rank-64 engine would add; it is excluded from Pareto selection. Unselected wrappers and allocator reservations are deliberately not inferred from payload arithmetic and require live receipts.

The rank-only cohort fixes layers 20-23 and all supported families at ranks 8/16/32/64. The family-only cohort fixes layers 20-23 and rank 32 while testing shared-expert, attention/GDN, dense-without-router, and the full control surfaces. Cross-cohort comparisons cannot be called rank effects. No arm changes layer topology.

Rank 64 is a function-preserving zero expansion of the current rank-32 factors. Ranks 8 and 16 use deterministic modulewise truncated SVD, so their compression loss is part of the estimand and must be reported before ES. If the V70 topology task selects anything other than layers 20-23, this entire ladder must be preregistered again on the winning inventory.

## Control-arm transfer and checkpoint ledger

- Canonical FP32 master: 18,112,512 bytes.
- Active packed BF16 payload: 9,842,688 bytes, including 786,432 bytes of vLLM packing duplicates.
- Each audited materialization projects 9,842,688 bytes H2D plus 9,842,688 bytes D2H for the packed runtime view.
- The current candidate path additionally projects 18,112,512 FP32-master bytes H2D and 18,112,512 bytes D2H per signed candidate.
- One 32-candidate update plus canonical restore projects 324,808,704 packed H2D bytes and 324,808,704 audit D2H bytes. Candidate-master H2D and owned-candidate D2H are 579,600,384 bytes each.
- Logical checkpoint payloads including the uint64 step are 18,112,520 bytes (SGD), 36,225,032 bytes (momentum), and 54,337,544 bytes (AdamW). Safetensors header/manifest and whole-file bytes must be measured.

Every live actor must report exact module identities, PEFT/runtime shapes and slices, rank/alpha/dtype, updated parameter counts, packed elements, duplicates, padding, install/audit traffic, checkpoint bytes, allocator reservations, peak VRAM, safe headroom, throughput, cleanup, and positive finite update norms by selected family/layer.

## Compute, validation, and selection

Each arm uses seeds [1701, 1702, 1703], exactly 2,048 optimization rollouts, 171 generated evaluation rollouts, and a charged ceiling of 14,400 GPU-seconds. Both rollout/CRN identity and charged GPU-seconds within 2% are required; idle padding and adaptive extra work are prohibited.

Against the common frozen pre-ES rank-32 reference, source-disjoint gates require all three seeds: dev reward 95% LCB >= -0.01 with point delta > 0.0; OOD-QA reward LCB >= -0.02 and exact-count delta >= -1; OOD prose token-logprob LCB >= -0.02; safety/hallucination component LCBs >= 0.0; and zero new hard events. The trained rank-32 full arm is the paired Pareto control, not the frozen baseline. These V1-bound thresholds are design evidence only until they are rebound unchanged to a fresh V2 boundary; protected data remains terminal and unavailable during HPO.

Only arms passing every compute, audit, quality, OOD, and safety gate may enter a nondominated Pareto set over measured quality, bytes, VRAM, transfer, throughput, SNR, and GPU-seconds. Scalarization and post-hoc weights are prohibited; the current full control remains in the report even if dominated.

## Outstanding dependencies

- `live_packed_allocation_and_transfer_ledgers_complete`: false
- `pareto_frontier_frozen`: false
- `protected_terminal_evaluation_complete`: false
- `specialist_0j5_14_phase_profile_complete`: false
- `specialist_0j5_30_fresh_evaluation_v2_complete`: false
- `specialist_0j5_6_live_topology_selection_complete`: false
- `task_must_remain_in_progress`: true
- `three_seed_quality_ood_and_reward_snr_complete`: false
- `twenty_one_arm_seed_runs_complete`: false
- `v69_live_family_evidence_complete`: false

No empirical quality, VRAM, bandwidth, throughput, SNR, or Pareto winner is claimed by this CPU preregistration.

Canonical content SHA-256: `9150d217025b29474a83a090a997dc6a8bec5fe85049c95a265c883f6bad5fec`
