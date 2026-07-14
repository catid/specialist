# S6 V12 consensus candidate, preseal, and release protocol

V12 is the first nonzero candidate protocol admitted by the V10/V11 response
diagnostics. It is not a new direction search. It deterministically collapses
the duplicate A43/A44 cells and constructs
`c* = sqrt(32) * unit(unit(c_D43) + unit(c_D44))`.

The frozen result is:

- coefficient SHA-256
  `1a85502564020048f634b0a8ced3952343f135f67ffd1faad1aaa697aebea8a8`;
- cosine to each unique domain cell `0.938139575931748`;
- cosine to the centered common anchor response `0.8527515739920155`.

## Required V11g evidence

The original V11 journal is a known failed artifact and is not admissible V12
evidence. No real V12 stage may launch until V11g has a canonical completed
durable launch attempt and its corrected journal passes exact V10 equivalence.
`build_eggroll_es_v11_evidence_v12.py` binds the committed V11g launch source,
V11f failure ancestry, exact effective CLI, seed and policy forwarding audits,
and the launch-attempt journal binding. It then performs one deep
V9/V10/V11c/V11g ancestry replay and emits a compact self-hashed artifact with
exclusive creation. It copies no validation, OOD, or heldout result content.

The V12 driver requires hard-coded file and content hashes for that artifact;
it never substitutes CLI-provided ancestry claims. Fixture evidence is
accepted only together with `--v12-dry-run`. The one-time minter completed
successfully from canonical V11g evidence. The compact evidence identities are:

- file SHA-256
  `d68dafd50e229bd444b5ff0a666aabb508d3d021d44cd2060d7aace391fc6745`;
- content SHA-256
  `b6212a4bdafaf234f8445b11c18ef96e15526d450f89d366542c63fae2d15e8f`.

V12 extends the successful V11c trainer/worker facade. This is mandatory:
V11b separated raw-question manifests from templated-prompt manifests, and V11c
provided the complete substituted-anchor API. Inheriting the original V11
trainer would reproduce its deterministic manifest failure before a usable
coefficient estimate.

## Independent train-only screens

C45 and C46 contain 64 frozen training rows each. Construction first excludes
every D43/D44 row identity; C46 additionally excludes C45. Their Q/A manifest
hashes are respectively
`45944527c98eaf7446a89f85672f83ec6c42d28047288cddbc66f1d547b09490`
and
`be80a4973772d0457d769fcb12462192bc1153f7eec9e301f961825c67e95e74`.
No validation, OOD QA, OOD prose, or heldout path is accepted in the preseal or
confirmation parser.

The alpha grid is exactly `[0, 7.8125e-7, 1.5625e-6]`. Each candidate uses a
fixed paired 20,000-resample lower confidence bound. It is eligible only when
both C45 and C46 bounds are strictly positive and both A43/A44 prose-anchor
bounds are nonnegative. The smallest positive eligible alpha is selected.
Alpha zero is a baseline, never a fallback candidate.

## Three immutable stages

1. `preseal` uses only train and anchor data. If a candidate is eligible it
   writes `candidate_seal.json` after the completed screen report. The seal
   fixes alpha, coefficient, gate, V11g evidence, implementation, and report.
   Its key set and nested binding key sets are exact; extra surfaces fail
   closed even if an artifact is re-hashed.
2. `confirm` starts fresh engines at base weights, applies only the sealed
   alpha, and repeats the direct C45/C46 plus A43/A44 gate. A failure closes
   the candidate; no alternative alpha is attempted. The passing confirmation
   has an exact key set, a canonical eligible gate, an application bound to the
   sealed alpha and coefficient, and an exact candidate-seal binding.
3. `release` requires exact file/content hashes for both the immutable seal and
   passing fresh confirmation. Only then may validation, OOD QA, and OOD prose
   be opened. Validation mean must strictly improve; all strict QA counts and
   OOD prose must not degrade. Failure means reject without fallback.

The heldout split remains unopened. A later heldout command may be defined only
after the fixed candidate passes release.

## Hardware and restoration

Every real stage requires four TP=1 engines on GPU IDs `0,1,2,3`. Partial
population waves are forbidden. V11c's exact perturbation restoration is
revalidated before V12 replaces the projected coefficient vector. Every
nonzero update uses the V4 two-phase distributed commit, records all four
post-commit worker states, and requires unanimous exact final identities.

## Launch order

The launch dependencies are intentionally serial:

1. complete V11g and retain its immutable attempt and journal (complete);
2. run the V12 evidence minter once at its canonical output path (complete);
3. hard-bind the emitted file and content hashes in
   `run_eggroll_es_consensus_candidate_v12.py` (complete);
4. commit the reviewed V12 implementation bundle and run the full offline
   contract suite;
5. obtain the current implementation bundle SHA from a dry run and pass it to
   the real stage.

Only then may preseal launch. Preseal, confirmation, and release use distinct
experiment directories and refuse resume or overwrite. Confirmation depends
on a selected candidate seal; release additionally depends on a passing fresh
confirmation. No heldout command is defined by this protocol.
