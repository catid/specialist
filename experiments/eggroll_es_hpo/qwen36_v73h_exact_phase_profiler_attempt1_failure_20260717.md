# Qwen3.6 V73H exact-phase profiler attempt 1

Status: failed closed during real Ray actor initialization on 2026-07-17. This
attempt is immutable and must not be deleted, edited, or reused.

## Result

V73H proved that the Ray job runtime environment carried the exact actor flag
and guard hash into all four production actors, but the actors did not execute
the V73H `sitecustomize` before resolving the worker extension. Each actor
reached vLLM argument parsing, model-architecture resolution, and engine
configuration, then the extension rejected the absent process-start guard:

`RuntimeError: V73H process-start sitecustomize guard is absent or invalid`

The production-only difference from the passing CPU Ray probe is now
identified. `train_eggroll_es_specialist.load_trainer` reconstructs
`PYTHONPATH` as repository root, `eggroll_es_compat`, upstream, and only then
the inherited V73H guard path. Python therefore imports
`eggroll_es_compat/sitecustomize.py` and shadows the later V73H
`sitecustomize.py`. The CPU probe kept the V73H guard directory first and did
not reproduce this trainer rewrite.

An additive successor must reassert the guard directory as the first
`PYTHONPATH` entry before `ray.init`, then run the exact compatibility
sitecustomize behavior only after the actor guard is installed. Its real-Ray
test must reproduce both the production trainer rewrite and the actor-specific
runtime environment. V73H paths may not be reused.

## Authority and resource accounting

- Protected dev, OOD, holdout, prompt, answer, or semantic content opened: no.
- Successful protected open, resolve, metadata, or enumeration operations: 0.
- Production Ray actors launched: four.
- Model weights loaded: no; failure occurred while resolving the worker
  extension before model loading.
- Actor CUDA work receipts: 0.
- Checkpoint, snapshot, model update, or promotion performed: no.
- Promotion-charged GPU seconds: 0.
- Reserved wall GPU-seconds: 164.856030452; this is not accepted as measured
  model residency or useful GPU work.
- Final compute-process lists were empty on all four GPUs before the next
  systems-only keepalive launched.

## Sealed source and artifact identities

- V73H sealed source commit: `8e89e6d12c9b0501f70c1da9b4ff67b3ffad45b1`
- Preregistration file/content SHA-256:
  `86ea93c94c973e64a84ece2677f2b58fa40c1dbe5bd1e2123b49738bba08cbc6` /
  `7cc60f2976203ceb32f395295450f074fc730e72be7e1d8e61ac39aa646adb04`
- Profile attempt file/content SHA-256:
  `9ae72d1c7692fec50fc664483e7361cb075bbde412ccbabbadf37f177a2e0546` /
  `802a7a5d928cb89baeb66a83ef531eff342b2a826a217225484957e4bcd5b655`
- Profile failure file/content SHA-256:
  `a045aee62ff65b4f15145eaced61dad30ed01f0014c04335d6b6e5ef14821c93` /
  `69a32077c034d722c3caf8badb1c0429858cb6eaf59cb8286e1ac532914e49c2`
- Nsight report SHA-256:
  `a6eb4cf9606cbe1582d5582c5b1d549656da26bf178c085d4d6365bb04945c48`
- Nsight SQLite export SHA-256:
  `cb4fe9eadb18ac0858b7905cf8cfb90f5bcf63b20d2a3af714994adc43d5c3b3`
- Run attempt file/content SHA-256:
  `c4d83c6ff5b8e9bf1ad224b55f097886909e68d7d4c3fbe1337d43cc8eaa374f` /
  `efa9f8a9e1d0ddc21b4e02caf8972a284c21c90fa71934986077213168811547`
- Run failure file/content SHA-256:
  `69e956f987ab28c05daff62e8c0193329bf44c09b8a681e5fac98c5f3ee3d86f` /
  `643a747bba9b64896ce7ca57e1dd0b3f6a022a92e153443a6ee9413f282ff177`
- Partial phase receipt file/content SHA-256:
  `fc9521a99d82a75025926730f08e4ded9f9789efbaeb7c4d7d4868823bae62cb` /
  `a79b5df6be4edfbbc08404e841723c498f1b92711b4d6dd95a4ed21c80b88b53`
