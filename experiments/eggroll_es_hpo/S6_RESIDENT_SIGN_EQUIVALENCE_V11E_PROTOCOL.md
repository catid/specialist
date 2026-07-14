# S6 V11e effective-CLI forwarding exact retry

V11e is a fresh, intended-recipe-identical retry of V11d with one explicit
effective-CLI forwarding correction. It binds the committed
V11d failure document at commit
`a6e4e6623d92aa35c999db29e819bfe0de6b09df` (file SHA-256
`f3ff548151b9767f1a4cbfa0968d6ae69cffe2bb71d1c569b622dad534cd7353`)
and the exact durable V11d launch-attempt artifact (file SHA-256
`5e75d421982e6eaa4f979692073c3018c37a5054636db611bb7f9a03cc2325b3`,
content SHA-256
`4f21152b42bd91b507327936fdfd16365022c22a057f48e3b546f1875ce6d1f3`).
That artifact is preserved byte-for-byte in the tracked, root-level
`S6_RESIDENT_SIGN_EQUIVALENCE_V11D_FAILURE_EVIDENCE_V11E.json`; V11e never
depends on the ignored `runs/` copy being present.
V11d source commit `25b9b2c9818f8d0ce6624f70aa39c11a7ec2666e` and driver SHA-256
`76862beb9957965eaf43f6629a6b5fa640e2ad8930a26eb3a51ce5c45741b675`
are part of the failure binding. V11d and its run directory remain immutable.

## Localized failure and sole runtime correction

V11d completed engine loading, cross-engine setup, frozen-plan installation,
exact-reference capture, and distributed-state inspection. It then failed in
the coordinator's V5 anchor invariant with
`ValueError: v5 requires every frozen anchor document`. No domain or anchor
scoring, perturbation, coefficient construction, or update occurred, and no
journal was created.

The outer frozen V5 parser supplied defaults of 128 anchor items and minimum
cosine 0.8 when those switches were absent. The delegated driver-v1 parser
independently supplied defaults of 2 and 0.1. A complete projection of all 27
driver-v1 runtime fields found exactly those two mismatches; the other 25
matched.

V11e changes only the new immutable experiment name and explicitly forwards:

- `--anchor-items-per-step 128`
- `--anchor-max-input-tokens 512`
- `--min-anchor-cosine 0.8`
- `--ood-prose-max-input-tokens 1024`
- the canonical frozen anchor JSONL, anchor report, and OOD-prose paths.

Before Ray can start, V11e parses the actual delegated driver-v1 argv and
requires all 27 effective fields to equal both the outer frozen projection and
the preregistered canonical projection. The dry-run artifact exposes the full
outer/effective projections, field count, empty mismatch list, base-argv hash,
and content hash. Removing the two formerly omitted switches must reproduce
exactly `anchor_items_per_step` and `min_anchor_cosine` as mismatch fields and
must fail the V11e audit.

Evidence therefore records
`frozen_recipe_or_data_changed_from_v11d=false`,
`effective_cli_forwarding_corrected=true`, and an exact two-field correction:
`anchor_items_per_step` changes at the delegated parser boundary from the
accidental V11d value 2 to intended value 128, and `min_anchor_cosine` changes
from accidental 0.1 to intended 0.8. It does not make the inaccurate claim
that V11d and V11e had identical effective delegated runtime values.

Everything else remains V11d/V11c: Qwen3.6-35B-A3B, layers 20--23, sigma
`0.0003`, the exact 32-direction basis, D43/D44, A43/A44, antithetic signs,
exact restoration, four TP=1 engines, and alpha exactly zero. Model, data,
prompts, sampling, directions, coefficient rules, and update policy do not
change. Validation and OOD are not selection surfaces; sealed data remains
forbidden.

Immediately before delegation, V11e rehashes the actual V11c runtime bundle
and requires the V11d-pinned files exactly: driver
`5bd650f727e3e32a0be530316c82043b012a623428fab47498f4d80f5ac48e76`,
trainer
`c663b62f9d7990a2c59d8b46ad6258209b590bb29aa48946755a7d263a3d0799`,
and worker
`d75951483058de340185fc81f6ed050deeac1551107c0357349a1f311cdb2c22`.
The completed evidence binds their paths, individual hashes, and canonical
bundle hash, so unchanged algorithm/data claims fail closed on source drift.

## Fresh-run and offline evidence contract

The exact real and dry-run CLI tuples are closed allowlists. A real launch is
forbidden unless the V11e driver is committed at current HEAD and its working
bytes match that blob. An `O_EXCL` sibling launch-attempt file reserves the
fresh experiment before delegating to V11c; an existing claim or run directory
is terminal. Any escaping `BaseException` is atomically recorded with its full
traceback and re-raised. V11e never retries inside the same name.

A completed attempt has an exact key set, recursively rejects V11e-owned
heldout-bearing keys or values, and binds the exact V11d failure, effective
27-field CLI audit, recipe, source, diagnostic environment, and completed
journal. Completed validation freshly reparses the exact frozen CLI and
requires the stored effective audit—including its strict key set—to equal the
new audit exactly. It also freshly rehashes and compares the runtime bundle.
The journal binding includes its exact frozen path, file SHA-256,
validated content SHA-256, and schema. Inherited journal false sentinels are
validated by the exact inherited schema rather than by substring matching.
The reporter requires the exact sibling attempt path and cryptographically
recomputes the attempt-to-journal binding before reporting exact V10
equivalence.

No GPU launch is part of the V11e implementation or test step.
