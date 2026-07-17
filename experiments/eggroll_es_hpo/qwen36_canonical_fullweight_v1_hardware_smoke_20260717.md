# Canonical full-weight EGGROLL-ES V1 hardware smoke

Date: 2026-07-17

## Outcome

The canonical FP32 full-weight state path completed a four-engine, four-GPU
Qwen3.6-35B-A3B hardware smoke. All four replicas installed the same canonical
master, participated in a disjoint-seed ES update, converged on identical
post-update master and runtime hashes, materialized distinct candidates, and
restored exactly to the post-update state.

This closes the numerical-state defect in the legacy full-weight path:

- Noise now advances once across the complete sorted parameter manifest instead
  of restarting the same random stream for every parameter.
- Candidate restore rematerializes from the immutable FP32 master instead of
  subtracting BF16 noise.
- ES updates accumulate in FP32, retaining changes below one BF16 ULP.
- The canonical FP32 master, rather than the quantized runtime projection, is
  saved in resumable checkpoints.
- Failed or uncertain partial updates poison the worker rather than allowing a
  potentially divergent state to continue.

## Resident model inventory

The live vLLM model exposed the following canonical inventory on every engine:

| Field | Value |
|---|---:|
| Canonical parameters | 573 |
| Elements | 34,660,610,688 |
| Runtime bytes | 69,321,223,296 (64.5604 GiB) |
| FP32 master bytes per engine | 138,642,442,752 (129.1208 GiB) |
| Manifest SHA-256 | `c94ff73cf6605bdab1b7750a64596e486c753536477b7f4ff8336feed13f906f` |

The first hardware attempt discovered that Qwen/vLLM publishes alternate names
for some shared parameter storage, including an MLP gate alias. Installation
failed before mutation. The manifest was then changed to retain one canonical
parameter name, seal alternate names as aliases, reject incompatible overlapping
storage, and mutate each storage exactly once. The second hardware run passed.

## Initial state certificate

All four engines agreed on both hashes, and the read-only runtime audit confirmed
that every live tensor exactly matched the native-dtype projection of its FP32
master.

| State | SHA-256 |
|---|---|
| FP32 master | `b439fb3d3966f77f4b0cdc7576786415631b6274966e933d4ede67b1425a9ebf` |
| Runtime/live projection | `c4699bccb05d223de7aebb9b05b7df8312d5b19b7f4f376e84e255841804adbb` |

Initialization took 173.596 seconds. The initial live-runtime/projection audit
took 85.679 seconds.

## Distributed update

The fixed update used seeds `12001..12004`, coefficients
`[1.0, -1.0, 0.5, -0.5]`, alpha `2.5e-5`, and population size four. The applied
scale was `6.25e-6`. Each of the four engines generated exactly one seed stream;
the seed shards covered indices `0..3` once without overlap. Every parameter was
then reduced in FP32 and applied to all four masters.

| Measurement | Value |
|---|---:|
| FP32 master update L2 | 1.8397773361484746 |
| FP32 master maximum absolute delta | 0.000066112530475948 |
| Nonzero update elements | 34,660,610,329 |
| Runtime elements changed | 7,618,866,396 (21.9813391766%) |
| Coefficient SHA-256 | `cb9ff3ccb95b9389a2c847f7322e0fcec72d4cca811831358252469c5749a641` |
| Post-update master SHA-256 | `1e52b315c01821d73ea401c06266ad3cb5fc7084af8583b46d3c5b17fbf93b51` |
| Post-update runtime SHA-256 | `4831b47d3115291bfb9801b0e9708e8556cc6ba44b9908fbd53d57ab3c760108` |

The communicator receipt certified a four-replica hash consensus. During the
update, all GPUs held the resident model at 82,494 MiB and rose to 89,248 MiB;
simultaneous utilization samples were 18%, 22%, 14%, and 14%. This confirms that
the update was distributed across all four GPUs rather than prepared only on
engine zero.

## Candidate and restore

Seeds `13001..13004` produced four distinct candidate hashes. Each engine then
restored by directly projecting the FP32 master; no native-dtype algebraic
subtract was used. The concurrent candidate/restore wave took 56.903 seconds.

Every final hardware certificate passed with the post-update master and runtime
hashes above. The final live-runtime/projection audit took 86.988 seconds. After
actor teardown, all four GPUs returned to 4 MiB resident memory and zero
utilization.

The subsequent adversarial review strengthened the certificate to recompute and
compare the complete FP32-master identity as well. The hardware update had
already computed and reached four-replica consensus on that complete master
hash; the new read-only audit check was separately exercised with an injected
sub-BF16 mutation.

## Regression coverage

The production venv test command was:

```bash
es-at-scale/.venv/bin/python -m pytest -q \
  test_eggroll_es_specialist*.py \
  test_eggroll_es_worker_fullweight_v1.py
```

Result after the final adversarial review fixes: **148 passed in 5.81 seconds**.

Focused regressions cover deterministic full-manifest noise replay, non-reused
successive parameter streams, exact restore for both perturbation signs,
mid-materialization repair, idempotent restore, read-only drift detection,
uneven four-rank seed coverage, concurrent four-rank update consensus, retained
sub-BF16 residuals, canonical checkpoint round trips, alias-safe manifests,
partial-update poisoning, and TP greater than one rejection.

The final review additionally injected a sub-BF16 FP32-master mutation, changed
canonical checkpoint contents while retaining its claimed hash, changed the
checkpoint noise schedule, and forced post-update identity hashing to fail. The
certificate and checkpoint loader reject the first three without repairing or
partially loading them; detected canonical/runtime drift and the post-update
failure terminally poison the worker and block all subsequent reuse.

## Deliberate limits and follow-up

- V1 supports tensor parallelism one only. A TP-sharded canonical manifest and
  checkpoint protocol must be designed before enabling TP greater than one.
- The generator schedule is deterministic for the sealed parameter manifest and
  traversal. It is not invariant to arbitrary future rechunking or repartitioning;
  an absolute-index counter-based generator would be the appropriate V2 design.
- Full cryptographic runtime audits are deliberately read-only but are expensive:
  the live-runtime/projection portion took about 86 seconds at this model size.
  The final implementation also hashes the 129.1-GiB FP32 master and has not
  separately re-timed that additional pass at full scale. These checks belong at
  state boundaries and in hardware qualification, not after every candidate.
- A partial update is terminally poisoned instead of rolled back. Keeping a second
  516.5-GiB four-replica FP32 bank solely for rollback would be disproportionate;
  recovery is actor recreation from the last canonical checkpoint.
- This run qualifies state correctness and meaningful update magnitude. It does
  not make a validation-quality or OOD-quality claim about a trained checkpoint.
