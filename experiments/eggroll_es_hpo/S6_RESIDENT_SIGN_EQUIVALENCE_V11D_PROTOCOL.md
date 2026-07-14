# S6 V11d durable-launch exact retry

V11d is an evidence-bound, algorithm-identical retry of V11c's broad
post-engine/pre-journal failure. The cause remains unknown. V11d changes no
model, layer, data, prompt, sampling, seed, perturbation, sign, restore,
coefficient, or alpha-zero setting. It uses a fresh immutable experiment name
and binds the committed V11c post-engine/pre-journal failure evidence.

Before delegating to the exact V11c driver, V11d writes a sibling launch-attempt
artifact outside the still-absent run directory. This preserves driver-v1's
fresh-run guard. The artifact pins the V11c implementation and recipe, records
the diagnostic environment, and is atomically rewritten with the complete
exception type, message, and traceback if any `BaseException` escapes. It also
records whether the run directory and V11c journal exist at the failure
boundary. The exception is always re-raised; V11d does not retry inside a run.

Ray log deduplication is disabled and NCCL initialization/network diagnostics
are enabled before Ray starts. These are broad launch diagnostics, not a claim
that V11c failed in NCCL, and they do not alter the ES algorithm.
The V11c target remains exactly alpha zero, so no model update is authorized.
Validation/OOD are not selection surfaces, and sealed-heldout content remains
forbidden.

The launch-attempt artifact is a sibling named
`.snapshot794_layer_v11d_middle_late_resident_sign_exact_v10_durable_launch_d43d44_a43a44_basis20260714.launch_attempt.json`
under the frozen output directory. Its existence, or existence of the V11d run
directory, forbids another V11d attempt.

Offline validation and reporting must scope the same frozen V11d experiment
name and recipe hash while invoking V11c's exact-equivalence validator. The
report additionally requires the content-addressed completed launch-attempt
artifact; validating a V11d journal as ordinary V11c without that scope is
forbidden.

A real launch additionally fails before Ray starts unless the V11d driver is
tracked at the current Git HEAD and its working-tree bytes exactly match that
commit. The launch artifact records this commit/blob/file binding. Offline
validation re-reads the recorded Git blob and requires both the current driver
and recorded driver hash to match it, and it requires the exact diagnostic
environment rather than merely recording unverified provenance fields.

The real and dry-run command lines are exact frozen tuples. The sibling launch
artifact is claimed with an exclusive create before delegation, and completed
evidence must bind the exact run path plus both file and validated-content
hashes of its journal. V11d-specific launch evidence is recursively checked for
sealed-data references; the inherited journal instead uses its schema-aware
validator because that frozen schema intentionally contains explicit
`...heldout...: false` policy sentinels.
