# S6 V11g inherited-policy forwarding retry

V11g is a fresh retry of V11f under experiment name
`snapshot794_layer_v11g_middle_late_resident_sign_exact_v10_policy_forwarded_d43d44_a43a44_basis20260714`.
It does not modify or reuse V11f evidence or its run directory.

## Bound failure

V11g binds the V11f failure document committed at
`190ddb7922cf834b81101c5927a78fb447295831` (SHA-256
`0e5f659ade028d77ab6fa7f82ad2ac89093de6d1aa045a56dd8bb106870ebb47`),
the committed V11f source at `d448c8a980bc6326a92b9938aaaeafe6114b0bab`
(driver SHA-256
`e4d50b4bac41c4c294e999dba61e8ced9aa95e5136f048c3db75350b97f999ff`),
and the exact V11f attempt (`3b97db50578af3d30364988e16ffdb50e1d54dd81c9084fd072e9c8160d9cd39`),
journal (`62cea23dbdba65f11d14f98d38d249dcbda3b3984739035990cf10db9e192eb4`),
plan (`2909b74c6d2bcec1b7db5186b2c4ab151511e34195af3a89b8468f8709c03af2`),
and identity audit
(`168ace19f541bac3a19a29757b3cfe198b19207c0852585c761b69f1cfc706a4`).

V11f used the correct frozen perturbation basis, completed population scoring,
and constructed a coefficient plan. The only completion mismatch was the
absence of these three inherited policy fields:

- `document_lcb_anchor_required: true`
- `ood_validation_heldout_as_objective: false`
- `optimization_data: train_and_anchor_only`

No expected policy had another missing or changed value, and no unexpected
policy was present.

## Sole correction

V11g retains V11f's exact RNG(seed=43)-to-frozen-basis wrapper unchanged. A
second scoped wrapper surrounds the inner V4 execute call. After unchanged V4
returns, it requires the policy to equal the exact inherited policy minus only
the three fields above, rejects conflicts, injects exactly those immutable
values, reseals the in-memory journal, and returns to unchanged V11c completion
handling. It does not weaken V11c validation or modify data, seeds, scoring,
coefficients, alpha, or update behavior. Both scoped functions restore their
original delegates in `finally`.

## Evidence and launch contract

The dry artifact exposes the full 27-field CLI audit, full V11f seed-forwarding
audit, exact before/after policy audit, committed V11g source, and pinned V11c
driver/trainer/worker bundle. Real execution requires the exact frozen CLI, a
fresh run name, committed matching source, an `O_EXCL` attempt claim, and a
second run-directory check after the claim. Escaping failures record their full
traceback and are re-raised.

Completion is not recorded until the journal has the frozen seed vector and
exact V11/V11b/V11c policy union, passes unchanged V11c validation, and is
bound by path, file hash, validated content hash, schema, seed hash, and policy
hash. Alpha remains exactly zero. Baseline validation/OOD scoring is recorded
separately from sealed evaluation data and model updates. No GPU launch or
sealed-data read is part of V11g implementation or testing.
