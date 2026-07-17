# FP32 ES optimizer + module-normalized sigma V1: CPU evidence

Date: 2026-07-17

Bead: `specialist-0j5.8`

Status: the numerical implementation, actual adapter inventory, nonadaptive
grid, compute/update-norm contract, memory/bandwidth accounting, checkpoint
transaction, evaluation firewall, preregistration, and adversarial CPU tests
are complete. The Bead remains in progress: mirrored ES has substantive
nonzero/restore evidence but still needs an accepted all-four-GPU activity
receipt, its roofline dependency still needs optimizer-phase transfer/bandwidth
receipts, and the preregistered 24-run grid has not executed.

## Numerical design

The sealed source is the exact 70-tensor, 35-logical-module, 4,528,128-element
LoRA FP32 master used by the mirrored-ES worker. Its tensor inventory identity
is `eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6`.
The source master has global RMS `0.009133813367694695` and L2 norm
`19.436205436381517`.

The global arm assigns every element the sealed rung sigma. The module arm
groups each LoRA A/B pair and computes:

```text
raw_m   = max(sqrt(sum(theta_m^2) / elements_m), 2^-24)
norm    = sqrt(total_elements / sum(elements_m * raw_m^2))
sigma_m = base_sigma * raw_m * norm
```

Thus both modes have exactly the same expected perturbation L2:
`sum(elements_m * sigma_m^2) = total_elements * base_sigma^2`. The scale is
based on FP32 RMS, not raw module L2, so module size is not counted twice.
On this checkpoint the first rung's module sigmas range from
`0.0003772318167248829` to `0.0008371383126489394`, around the global
`0.0006`; the second rung is exactly half. The largest module's expected
perturbation-energy share falls from the global shape-only ceiling of 7.24% to
2.89% in the RMS-normalized arm.

Candidates use unscaled unit Gaussian noise and apply `sigma_m` exactly once.
The raw mirrored estimator applies its inverse exactly once:

```text
g_m = sum_i((Rplus_i - Rminus_i) * epsilon_i,m)
      / (2 * directions * sigma_m)
```

Only three coefficient modes are registered:

- raw pair difference, the smoothed-objective gradient estimator;
- sign of the complete pair difference; and
- average rank of absolute complete-pair differences with the sign restored.

The latter two are explicitly direction heuristics, not unbiased gradients.
They are antisymmetric under swapping plus/minus. Ranking signed candidates
independently, unpaired centering, and any mode that can break pair
antisymmetry are prohibited.

The optimizers operate on the canonical CPU FP32 master and FP32 slots:

- SGD uses the raw FP32 estimator direction;
- momentum uses `velocity_t = 0.9 * velocity_(t-1) + gradient_t`;
- projected AdamW uses FP32 first/second moments, betas `0.9/0.999`, epsilon
  `1e-8`, exact bias correction at the new committed step, and a
  `-0.01 * master` decay direction.

Every nonzero optimizer/decay direction is projected after optimizer math to
the same target:

```text
target_update_l2 = 0.0005 * max(master_l2, 2^-24 * sqrt(elements))
```

The observed FP32 candidate-minus-master norm must match within 50 ppm. This
tolerance is applied after FP32 rounding; a real-surface all-ones direction
landed within 18.1 ppm without an extra correction pass. Zero pair-difference
variance or a zero optimizer direction skips without advancing moments,
checkpoint identity, sigma rung, RNG, or panel cursor. Any nonfinite value
aborts and restores the exact prior checkpoint without a replacement seed.

## Nonadaptive equal-work grid

There are six primary arms—three optimizers crossed with global and
module-RMS sigma—and two global-SGD shaping diagnostics for pair-sign and
signed-absolute-rank coefficients. No result-dependent arm addition or budget
reallocation is permitted.

Each arm runs registered seeds `1701`, `1702`, and `1703`. Each seed has two
fixed sigma rungs (`0.0006`, then `0.0003`), eight fresh shared directions per
rung, 16 signed candidates, and 64 identical train conflict units per
candidate. Therefore every arm/seed uses exactly:

- 16 directions;
- 32 signed candidates;
- 2,048 optimization rollouts; and
- the same 4,528,128-element parameter surface and relative update budget.

The complete grid is 24 receipts and 49,152 optimization rollouts. Each
receipt has a 14,400 charged-GPU-second ceiling and must attribute useful
activity to physical GPUs 0–3. GPU seconds and optimizer runtime are reported
outcomes; failed work is charged and may not be reassigned.

After a run's final checkpoint is sealed, and only then, it performs identical
registered evaluation requests: 83 dev generation plus 83 dev teacher-forced
requests, 24 OOD-QA generation plus 24 OOD-QA teacher-forced requests, and 16
OOD-prose teacher-forced requests. OOD is a noninferiority gate, not a point
score to optimize. Protected holdout access remains false.

## Optimizer memory and bandwidth expectations

Optimizer slots are host-resident FP32 and replicated exactly across all four
actors. Persistent optimizer VRAM is zero in every arm; BF16 optimizer state
is a contract failure.

| Optimizer | FP32 slot bytes/replica | Logical checkpoint bytes/replica | Four-replica slot bytes | Algorithmic minimum host traffic/update/replica |
| --- | ---: | ---: | ---: | ---: |
| SGD | 0 | 18,112,520 | 0 | 126,787,584 |
| Momentum | 18,112,512 | 36,225,032 | 72,450,048 | 181,125,120 |
| AdamW | 36,225,024 | 54,337,544 | 144,900,096 | 271,687,680 |

The checkpoint figures include the FP32 master and an eight-byte logical step
counter. Each transaction additionally reserves one 18,112,512-byte rollback
master and one equally sized candidate master per replica. Streaming one
maximum-size FP32 noise tensor plus accumulator has a 2,097,152-byte GPU
scratch ceiling; no complete optimizer vector is allowed to become persistent
GPU state.

Per update and replica, the reduced FP32 gradient accounts for 18,112,512 D2H
bytes and a committed derived BF16 runtime view accounts for 9,842,688 H2D
bytes. The fixed 16-candidate population plus exact restores accounts for
314,966,016 H2D bytes across all replicas per update, independent of optimizer
or sigma mode. Runtime receipts must separately measure actual allocator/copy
traffic, phase time, achieved host bandwidth, PCIe time, checkpoint I/O, and
allocated/reserved VRAM; measured traffic may exceed but cannot be silently
reported below the sealed algorithmic minimum.

## Checkpoint and access firewall

All 24 initial checkpoint identities are preregistered from the exact master,
empty optimizer state, arm, seed, next direction-set identity, train-panel
cursor, and sigma schedule. Every update records:

- previous/resume/rollback checkpoint identity;
- pre/rollback/post optimizer-state identity and committed step;
- candidate and four-replica committed checkpoint identity;
- direction-set and train-panel identities; and
- target/observed update norm and all-four-GPU activity.

A resume must form one contiguous master, optimizer, RNG/direction, panel, and
sigma-rung chain. Candidate commit requires exact four-replica consensus;
otherwise master and optimizer state roll back together. Monotonic receipts
must prove the last update preceded the training seal, which preceded any dev
or OOD open. Protected access or a model update after evaluation is forbidden.

## CPU acceptance

CUDA-hidden numerical, builder, adversarial, and parent-contract suite:

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  test_fp32_es_optimizer_ablation_v1.py \
  test_build_fp32_es_optimizer_sigma_preregistration_v1.py \
  test_recipe_evaluation_contract_v1.py
```

Result: `57 passed in 7.80s`.

Adjacent mirrored-ES integration suite in the environment that supplies vLLM:

```text
CUDA_VISIBLE_DEVICES='' es-at-scale/.venv/bin/python -m pytest -q \
  test_eggroll_es_mirrored_v66.py \
  test_run_lora_es_mirrored_calibration_v66.py \
  test_recipe_evaluation_contract_v1.py
```

Result: `36 passed in 7.95s`.

Adversarial coverage includes Adam bias-correction drift, applying/inverting
sigma twice, raw-L2 module-size domination, broadcasted perturbation shapes,
silent BF16 slots/receipts, unequal update norm, unequal directions/rollouts,
altered parameter surface, incomplete/duplicated grid receipts, checkpoint or
optimizer resume mismatch, replica/rollback drift, early dev/OOD access,
protected access, zero variance, zero direction, and nonfinite rewards/state.

## Artifact identities and remaining blockers

- Preregistration:
  `experiments/eggroll_es_hpo/preregistrations/fp32_es_optimizer_module_sigma_ablation_v1.json`
- File SHA-256:
  `428d1de245a5cd5ad3cb976aa5312f6eda0874efb895d298bb5731a05f326924`
- Canonical content SHA-256:
  `e8c646b5929de49805421035bb56f2eca2ed2010f7d1fce6893f5b095303dbc9`
- Parent evaluation-contract content SHA-256:
  `2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad`
- Mirrored-ES dependency content SHA-256:
  `f706b63befbd9da93cdda6ad9e612bf8fccfeda395e573ae59ff3515f24e8eef`

No GPU, live run directory, base-model tensor, dev/OOD content, or protected
content was opened. The builder loaded only the sealed 18 MB source LoRA
adapter on CPU to calculate exact FP32 module RMS/shape tables; train and
evaluation inputs were bound by pre-existing identities only.

Launch remains blocked until:

1. `specialist-0j5.2` records a V66d all-four-GPU activity receipt for the
   already substantive nonzero mirrored calibration and exact final restore;
2. `specialist-0j5.14` records optimizer-phase VRAM, host bandwidth, transfer,
   and checkpoint evidence; and
3. the 24 preregistered arm/seed runs report train SNR/stability/runtime,
   validation, OOD noninferiority, and complete checkpoint/GPU receipts.
