# S6 V11c post-engine/pre-journal launch failure

V11c was launched once on 2026-07-14 with the committed complete-runtime-API
implementation at commit `7ab2daa79299f6800db31ff3bafa5a358cbd4ef3` and the fresh experiment name
`snapshot794_layer_v11c_middle_late_resident_sign_exact_v10_complete_api_d43d44_a43a44_basis20260714`.

The launch passed the V11c API preflight and created four Ray/vLLM actors, one
on each of GPUs 0--3. All four actors loaded Qwen3.6-35B-A3B successfully at
approximately 64.69 GiB per GPU. PID 3546921 then logged the start of PyNCCL,
and at 18:03:10 UTC driver cleanup killed all four actors and called
`ray.shutdown()`.

That single-rank PyNCCL log does **not** localize the failure to rendezvous:
successful V10 has the same rank-zero-only logging pattern and begins inference
roughly 22 seconds later, matching V11c's quiet interval. The durable boundary
is therefore only after all four engine/model loads and before initial journal
creation. The available evidence cannot distinguish inter-engine group
initialization, the remaining trainer-constructor/eval-cache work,
`configure_anchor`, or an external coordinator-side interruption.

The exact foreground exception was not durably captured: the unified terminal
chunk was truncated and the Ray logs contain no Python traceback. The durable
evidence therefore does not justify a narrower root-cause claim.

The failure preceded `execute_line_search`, coefficient construction, any
perturbation, and journal creation. It may or may not have entered
`configure_anchor`. The run directory contains only empty
`checkpoints/` and `eval-output/` directories. No parameter update was prepared
or applied, and validation, OOD, and sealed-heldout content were not scored or
opened.

## Frozen implementation identities

- driver SHA-256: `5bd650f727e3e32a0be530316c82043b012a623428fab47498f4d80f5ac48e76`
- trainer SHA-256: `c663b62f9d7990a2c59d8b46ad6258209b590bb29aa48946755a7d263a3d0799`
- worker SHA-256: `d75951483058de340185fc81f6ed050deeac1551107c0357349a1f311cdb2c22`
- Ray driver log SHA-256: `593409aacdc6e3e09ef5295d008308889d0aa8f74cf9f95e8a44d02f7c9425c6`

The four worker stdout SHA-256 values, ordered by PID 3546904, 3546921,
3546925, and 3546932, are:

- `dc5a944c6b2229ec630acfa7c141508da35a35ba7900997d065888164c425ec7`
- `9f9cbd717bf6d4dc5a9040e52dbf620eeef62e888e3da9270437353b6e6f2370`
- `5c205aec126d4c723bfabc479599237fa25ccde3c14436585f1f4e8f08eda997`
- `974ac79295477c3a4240117606436f52dd34b607de4b6ef40c2a64e56c7cf1cb`

The corresponding stderr SHA-256 values are:

- `7ba8cd98e6fcdfed5d0dee951881627923d521ac277d2d0efcf6433156fe1457`
- `e3050c279ec883b30dc463f2268c4e2dbb9f44abaa0a185f3e6e755975d560d8`
- `fff9a78eeda6200ec7d670450de9080467a05f6419fd0d477604dd46d9ccf1cc`
- `8d84ae80f7feabb5d295558f9131c6f2ed5fd972576ac7f6768d1a48da45096a`

V11c and its failed run directory are immutable evidence. Any retry must use a
new version, a new experiment name, and durable foreground logging. Later
forensics must retain the broad post-engine/pre-journal boundary unless new
durable evidence narrows it.
