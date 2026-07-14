# S6 V11d durable anchor-configuration failure

V11d was launched once on 2026-07-14 from committed source at
`25b9b2c9818f8d0ce6624f70aa39c11a7ec2666e`. It used the immutable experiment
name
`snapshot794_layer_v11d_middle_late_resident_sign_exact_v10_durable_launch_d43d44_a43a44_basis20260714`
and exact alpha-zero recipe identity
`3e91bc82ef50f528cbae4925931e5f2aee5b9c63e470c489dd11814b5df9f6a8`.

All four Qwen3.6-35B-A3B actors loaded, one per GPU, and the four-rank
inter-engine NCCL communicator completed initialization. The driver then raised
`ValueError: v5 requires every frozen anchor document` in
`DocumentLCBAnchoredMixinV5.configure_anchor`, before journal creation or any
perturbation.

The failing V5 check runs only after its complete `super().configure_anchor`
chain returns. That chain had already installed the frozen layer plan by RPC,
saved the exact reference on every engine, and inspected and validated the
distributed state. The run therefore passed rendezvous and initial cross-engine
RPCs. It failed in a coordinator-side effective-command invariant before any
anchor or domain scoring, coefficient construction, or update preparation.

The durable launch-attempt artifact is:

`experiments/eggroll_es_hpo/runs/.snapshot794_layer_v11d_middle_late_resident_sign_exact_v10_durable_launch_d43d44_a43a44_basis20260714.launch_attempt.json`

- file SHA-256: `5e75d421982e6eaa4f979692073c3018c37a5054636db611bb7f9a03cc2325b3`
- validated content SHA-256: `4f21152b42bd91b507327936fdfd16365022c22a057f48e3b546f1875ce6d1f3`
- exact argv SHA-256: `a3ec2d5cfb637e54e20947fc2d23b7216e5e452c7712474e1fa6eb6ad61dfcdd`
- committed driver SHA-256: `76862beb9957965eaf43f6629a6b5fa640e2ad8930a26eb3a51ce5c45741b675`
- status/phase: `failed` / `inside_v11c_driver_main`
- run directory exists; it contains only empty `checkpoints/` and
  `eval-output/` directories
- no journal exists
- no model update was applied
- sealed data was not opened or scored

## Root cause

This is a command-line normalization/default-divergence bug, not missing anchor
data. `validate_frozen_execution_cli_v5` parses absent anchor options with its
own frozen defaults of 128 items per step and cosine 0.8. It records those
values in the recipe. The delegated base driver parses the same absent options
again with independent defaults of 2 items per step and cosine 0.1. V11d's
exact real argv omitted both options, so its dry run and recipe validation
reported the intended frozen values while the real trainer received the base
defaults. The item-count invariant failed first; the cosine invariant would
have failed next.

Any retry must use a new version and experiment name, bind this immutable V11d
failure, explicitly forward the frozen anchor values, and test the effective
delegated parser values rather than only the outer recipe dictionary. V11d and
its run directory are immutable single-attempt evidence and must not be
retried, deleted, or rewritten.
