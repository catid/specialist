# Qwen3.6 front/tail LoRA-ES topology V70: CPU evidence

Status: CPU protocol and adversarial acceptance are complete. `specialist-0j5.6`
must remain in progress until `specialist-0j5.14`, the preregistered GPU runs,
and the downstream one-shot protected terminal evaluation are complete.

No GPU API, dataset content, raw model output, protected item, or live-run file
was opened by this work. Model inspection was limited to configuration, index,
and safetensors headers.

## Frozen architecture facts

- The text model has 40 layers in ten exact four-layer motifs:
  `[linear_attention, linear_attention, linear_attention, full_attention]`.
- The audited active LoRA surface on layers 20--23 has 35 modules, 70 FP32
  PEFT tensors, and 4,528,128 logical parameters.
- Across all 40 layers there are 350 compatible 2D target modules. Header
  signatures are identical at each corresponding motif position.
- Shape compatibility is not evidence of semantic mergeability. Only complete
  aligned motifs are admitted as checkpoint duplication/insertion candidates;
  arbitrary individual layers are not described as mergeable units.

## Parameter-matched arms

| Arm | Layers | Rank / alpha | Modules | Logical parameters |
| --- | --- | ---: | ---: | ---: |
| Early contiguous | 0--3 | 32 / 64 | 35 | 4,528,128 |
| Late contiguous | 36--39 | 32 / 64 | 35 | 4,528,128 |
| Symmetric early + late | 0--3, 36--39 | 16 / 32 | 70 | 4,528,128 |
| Current middle/late control | 20--23 | 32 / 64 | 35 | 4,528,128 |
| Distributed motif control | 0--3, 12--15, 24--27, 36--39 | 8 / 16 | 140 | 4,528,128 |

Every arm has the same target-family budget: attention/GDN 3,250,176,
shared-expert projections 983,040, and routers 294,912 parameters. Every run
has 16 directions, 32 signed candidates, one update, 72,450,048 perturbation
scalar draws, 2,048 optimization rollouts, 83 primary evaluation rollouts,
and 64 layer-sensitivity rollouts.

The audited vLLM packed mapping stores 4,921,344 BF16 runtime elements for
each arm. Its 393,216 elements of A-view packing duplication are counted
explicitly and are not rank padding. Runtime `max_lora_rank` must equal the
arm's logical rank; any additional padding fails the contract.

## Non-compensable gates

The analyzer requires all 15 counterbalanced arm/seed runs, non-overlapping
four-GPU residency receipts, exact target and runtime mappings, active updates
on every selected layer and family, equal router budget/scaling, bounded router
drift and expert utilization, three-seed train/dev/OOD trust, exact rollout and
compute accounting, per-layer masked-reward sensitivity, at least 4 GiB safe
VRAM headroom, and internally consistent throughput and memory-traffic
receipts. Aggregate quality cannot override an integrity failure.

A candidate also needs paired mean reward improvement, improvement on at least
two seeds, throughput of at least 95% of the current control on every seed,
no more than 2 GiB additional reserved VRAM, and at most 1.05x memory traffic.
Layerwise sensitivity is diagnostic only and cannot authorize an unregistered
post-hoc range.

## Duplication/insertion boundary

The prior V23 front, middle, and back epsilon-0.05 checkpoints have exact
checkpoint shapes and audited vLLM mappings, but all three train-only
hypotheses closed negative. V70 preserves those closures. Reconsidering a new
strength requires a fresh complete-motif checkpoint, mapping audit, and
preregistration; this topology result cannot reopen insertion automatically.

## Acceptance

- Scoped topology tests: 26 passed.
- Topology + MoE targeting + multiobjective trust + evaluation contract: 82
  passed.
- Module SHA-256:
  `9e2ac60f51940cfcb2c3b5f625d859d2eb8477c2f06e1bd03385d81eb0b0f8d7`.
- Test SHA-256:
  `16c4f06453a8f5bb80f137b59494457b9b27287d908b31beaf9c68cadd7084f9`.
- Preregistration file SHA-256:
  `4da6b721a3d62653ab227522822b4f18c84e0a025530faf3193d186167b91dbb`.
- Preregistration canonical content SHA-256:
  `69b9e07b96b060d8c6b3a2854fe5cf79e39f4706b77f11460e48dd931743e88a`.

## Exact blockers

1. `specialist-0j5.14` must finish its phase-separated VRAM and bandwidth
   profile so the live measurements can be interpreted against the same
   recipe.
2. All five arms need the sealed three-seed, all-four-GPU execution and
   aggregate evidence.
3. Only after HPO selection may the downstream one-shot protected terminal
   evaluation run; it cannot adapt the selected topology.
