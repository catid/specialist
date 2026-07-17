# Reward-shaping ablation V1: CPU evidence

Date: 2026-07-17

Bead: `specialist-0j5.3`

Status: the fail-closed CPU transform, adversarial tests, and experiment
preregistration are complete. The Bead remains in progress because its
acceptance criteria require the registered multi-seed Qwen3.6 result, dev/OOD
deltas, and a method-selection receipt. None of those outcomes is inferred
from synthetic data.

## Current-recipe audit

- The legacy `es-at-scale` z-score helper applies
  `(reward - mean) / (std + 1e-8)` using caller-supplied population
  statistics. It does not itself enforce prompt locality or reject nonfinite
  input.
- V43G centered ranks already reject nonfinite values and assign exact ties
  their average rank. V1 preserves those semantics while moving the
  statistical boundary inside each prompt/repeat/common-random-number group.
- V66 compiles one coefficient per exact antithetic direction as
  `Rplus - Rminus`. For a complete mirrored population this is algebraically
  identical to applying signed raw rewards to `+epsilon/-epsilon` and summing.
  Consequently, raw rewards and direct pair differences are one gradient
  estimator, not two independent training arms. Direct pair reduction remains
  the canonical implementation of the raw arm.

## Implemented boundary

`reward_shaping_ablation_v1.py` accepts exactly one train population per call.
Mixing dataset role, training seed, generation, population identity, or global
evaluation contract is an error. Within that population, z-score mean/variance
and centered ranks are computed independently for each opaque prompt identity,
repeat, and evaluation seed. Each such group must contain exactly one reward
for both signs of all eight registered directions.

The transform rejects extra semantic fields, missing or duplicate candidates,
wrong direction identities, booleans masquerading as numbers, and NaN or
infinite rewards before producing any statistic. Exact ties receive identical
midranks; a zero-spread group produces exactly zero z-score/rank utilities.
Input record order cannot change the result.

The four recorded views are:

1. raw signed rewards;
2. within-prompt centered z-scores with population variance and the existing
   `1e-8` denominator guard;
3. within-prompt exact-tie centered midranks; and
4. direct antithetic pair differences.

All views consume the same reward tensor. The preregistration allocates 2,048
optimization rollouts per method and seed: 64 prompts by 16 signed candidates
by two frozen repeats before an arm-specific update. It prohibits a duplicate
raw-versus-pair GPU run. The three unique parameter updates are positively
rescaled to the exact FP32 LoRA-delta L2 of the current z-score reference, so
arbitrary coefficient units do not confound the direction comparison.

## Adversarial evidence

A deterministic single-reward contamination of `1e12` changed only its own
prompt group for every method. In the sealed synthetic diagnostic:

| Method | Coefficient L2 change | Unchanged prompt groups |
| --- | ---: | ---: |
| raw rewards | 333,333,333,333.11414 | 2/3 |
| direct pair difference | 333,333,333,333.11414 | 2/3 |
| prompt-local z-score | 1.51654 | 2/3 |
| prompt-local centered rank | 0.355556 | 2/3 |

These are mechanics/sensitivity diagnostics, not model-quality evidence and
not a basis for method selection.

Focused CPU command (CUDA hidden):

```text
CUDA_VISIBLE_DEVICES='' es-at-scale/.venv/bin/python -m pytest -q \
  test_reward_shaping_ablation_v1.py \
  test_build_reward_shaping_ablation_preregistration_v1.py \
  test_eggroll_es_mirrored_v66.py \
  test_lora_es_robust_consensus_v43g.py
```

Result: `45 passed in 2.62s`.

The 26 new focused tests cover exact ties, zero spread, nonfinite and finite-
overflow input,
complete pair coverage, order invariance, every forbidden boundary mixture,
single-prompt outlier isolation, output-seal tampering, registered-seed
stability input validation, and exact compatibility with the V66 central
difference reduction.

## Sealed artifacts

- Preregistration:
  `experiments/eggroll_es_hpo/preregistrations/reward_shaping_ablation_v1.json`
- File SHA-256:
  `06a24c50c44f684534d1e189dd145c8705c5f3867053982590f7f262d86a2615`
- Canonical content SHA-256:
  `00024ec2deacb463fddedd0883849f86808b8a3acdee3e2b86809c0719f87ccf`
- Evaluation-contract content SHA-256:
  `2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad`
- Sampling-contract content SHA-256:
  `18cab815193d05de6e7416b17c1ffeae334a6a613f3899faa459cc719144e97f`
- V66 protocol file SHA-256:
  `06b8cfd775051e1a20d30f969442cdcdc1b2b56ad10c1291d287f031a47594ad`

## Remaining registered comparison

After the V66 calibration dependency is accepted, run all three unique methods
for seeds 1701, 1702, and 1703. Report split-half and cross-seed FP32 update
cosines, dev equal-conflict-unit reward deltas and paired intervals,
leave-one-prompt/outlier influence, every global OOD noninferiority delta,
charged GPU seconds, peak VRAM, and useful activity on physical GPUs 0-3.

Choose only among methods passing every OOD and integrity gate, using pooled
dev improvement and the preregistered tie breakers. Persist that selection
before the one terminal protected-holdout access. A protected result may not
change the selected method or recipe. No GPU, model, train/dev/OOD semantics,
protected holdout, live run directory, or checkpoint was accessed for this CPU
evidence.
