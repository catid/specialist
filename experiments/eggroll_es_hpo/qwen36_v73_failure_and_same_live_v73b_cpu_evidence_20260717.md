# Qwen3.6 V73 failure diagnosis and additive V73B gate

Date: 2026-07-17 UTC

Status: V73B sealed CPU-only; no V73B model, Ray, GPU, protected-dev,
OOD, or holdout access was performed while preparing this contract.

## V73 finding

The V73 live integration run reached all 16 Qwen3.6-35B-A3B candidate
evaluations across physical GPUs 0-3 and then failed closed while writing the
population artifact.  Its exact error was:

```
v73 population differs from accepted V66d: ['signed_rewards', 'signed_reward_sha256']
```

This comparison enumerates every population acceptance field.  Therefore the
plan, evaluation contract, all candidate FP32 and runtime identities, all
restore identities, input receipt, installation consensus/count, work
cardinality, drain/restore flags, and protected-data flags were exact; only the
new floating answer-logprob measurements and their derived digest differed.
Cleanup succeeded, all four compute-process lists were empty, no checkpoint or
promotion occurred, and no protected data was opened.

The rejected reward vector cannot be recovered from the failed run.  The
population writer rejected before persistence, while raw prompts, answers, and
outputs are intentionally not persisted.  Consequently that run provides no
prospective basis for an absolute/relative drift threshold.  Selecting a
tolerance after observing a later vector would be post-hoc and is explicitly
forbidden by V73B.

Bound evidence:

- `failure_v73.json`: file SHA-256
  `b2e75dc55a84c2dab494bd28d06e0161cd6cd199cb36e70ada86a220acb8cba8`
- `actor_cuda_work_receipts_v73.jsonl`: file SHA-256
  `227e38fd3fd8e6e54d33e302c21d9b77b00fcde1ead04a2df36d19a0524a7860`
  (16 unique work IDs, 64 outputs/candidate, GPUs 0-3)

## V73B correction

V73B does not weaken a numerical identity gate with a tolerance.  It replaces
the invalid *cross-run reward measurement* identity with a stricter
*same-input implementation* identity:

1. All non-reward population fields remain exact to immutable accepted V66d.
2. Every live reward must be finite and appear exactly once for the exact V66d
   pair, sign, direction index, direction seed, and evaluation contract.
3. V71 exact candidate audits and four distinct rank-local population
   acceptance tokens must complete before update math.
4. The canonical V66 compiler and an independently implemented one-pass pair
   compiler receive the identical live reward object.  Their entire result
   mappings, including pair values, coefficients, scales, seeds, and digest,
   must be exactly equal.  No tolerance is used.
5. The V72 distributed executor must consume that exact coefficient digest,
   produce one non-master candidate and runtime identity shared exactly by all
   four actors, retain exact V71 audits, and return to the original master on
   all four actors through the existing fail-closed abort path.
6. Static prepared-shard, no-commit, protected-data, audit-traffic, host-state,
   telemetry, and cleanup semantics remain exact.  Historical reward-derived
   update identities are recorded diagnostically, not used to approve a new
   live measurement.

The V73 adapter remains immutable.  V73B is an additive wrapper with distinct
artifacts and preregistration, and restores every temporarily rebound V73
global after exit.

## Sealed identities

- preregistration file SHA-256:
  `9c5ce43c36e08e038ee33e86380ab6c287ae1b4bcba4c80def623daeb00f7ed9`
- preregistration content SHA-256:
  `50b51ee2d71dc85d024c4a63cc57183d2b9c2d925c18924a719dacb1fb61dc94`
- builder file SHA-256:
  `b72dc4a134853484e556f09bc9e5eed22cccc9c5fb7bdb2aff5ec2e73b57669f`
- runner file SHA-256:
  `8e6b00adcfcdb81c81a9bb78032e993114203660af526026350a298a9a4320db`

## CPU validation

The focused V73/V73B suite passed 24 tests.  The adjacent V66, V66b, V66c,
V66d, V71 audit/worker, V72 ownership/worker, V73, and V73B suite passed 88
tests.  Covered failures include duplicate, missing, non-finite, and
metadata-mutated rewards; compiler disagreement; one-actor output identity
divergence; acceptance ordering; private in-memory control rebinding; exact
abort behavior inherited from V73; builder/source binding; dry-run purity; and
restoration of the patched integration surface.

## Exact next command

Run only when all four GPUs are exclusively available:

```bash
/home/catid/specialist/es-at-scale/.venv/bin/python \
  /home/catid/specialist/run_lora_es_v71_v72_same_live_calibration_v73b.py \
  --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/lora_es_v71_v72_same_live_calibration_v73b.json \
  --preregistration-sha256 9c5ce43c36e08e038ee33e86380ab6c287ae1b4bcba4c80def623daeb00f7ed9 \
  --preregistration-content-sha256 50b51ee2d71dc85d024c4a63cc57183d2b9c2d925c18924a719dacb1fb61dc94 \
  --execute
```

Success requires the same-live compiler receipt, exact four-actor candidate and
runtime consensus, exact abort, V71/V72 traffic and ownership receipts, useful
GPU attribution for every actor, host telemetry, and final four-GPU idle
cleanup.  The run remains train-only and cannot commit or promote a model.
