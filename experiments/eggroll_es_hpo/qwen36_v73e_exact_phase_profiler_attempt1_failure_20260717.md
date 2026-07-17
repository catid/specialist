# Qwen3.6 V73E exact-phase profiler attempt 1

Status: failed closed during controller setup on 2026-07-17. This attempt is
immutable and must not be deleted, edited, or reused.

## Result

The target stopped while `load_trainer()` validated the external worker class.
Importing `eggroll_es_worker_lora_v73e` raised:

`RuntimeError: V73E controller worker validation guard state changed`

The controller sitecustomize guard had been installed before target imports,
but the historical trainer import closure had already imported the V72/V71
worker parents by the time `validate_worker_state_mode()` imported the V73E
worker contract. The V73E worker's controller branch incorrectly required
those already-guarded parent modules to be absent. That absence requirement is
appropriate for a newly spawned actor, but not for controller-side class
validation after the controller guard has covered the complete import graph.

The successor must preserve the actor-side pre-parent requirement while making
the controller branch prove the preinstalled sitecustomize mechanism, matching
controller PID, non-actor environment, and zero successful protected access.
It must use fresh run and profile paths; this attempt is not rerunnable.

## Authority and resource accounting

- Protected dev, OOD, holdout, prompt, answer, or semantic content opened: no.
- Model load reached: no.
- Ray actor bootstrap reached: no.
- Actor CUDA receipts: 0.
- Observed phases: setup only.
- Checkpoint, snapshot, model update, or promotion performed: no.
- Promotion-charged GPU seconds: 0.
- Final compute-process lists were empty on all four GPUs.

## Sealed artifact identities

- Profile attempt: `9f14cb7bd419d527a0ba4a2201cb63665b6de4ffa5607f20458d5a1ec67a522f`
- Profile failure: `43541b858d6cb60088f46bb065e8057db2768bc4979fd02e7064a7f7000d5d58`
- Nsight report: `51f1c511e9dc7d308fdc973cbbb84ee9c5484201c46765a89dd18c4e2bd0b743`
- Nsight SQLite export: `6dd132da295c24a2a6ab0ac994dc5aa07e791e03810aabe1df02ca38be0ca478`
- Run attempt: `dbcb55bfa29b3ced1b8d404a61a14904610253770b5dbf6382dccf7fc9a8182f`
- Run failure: `9381ce465efa44c3d0d1686311efd91efb8ca612dde8e9ed54f0da11e5f00783`
- Partial phase receipt: `73bbdba0f98c80474e697bab0eb1cdfb22f5df154624f7876a5dc5673ece0586`
