# S6 V11c complete-anchor-facade retry

V11c is a minimal API-surface retry of V11b. The V11b algorithm and its
raw-versus-templated manifest correction remain unchanged. V11b failed before
engine creation because its substituted anchor module did not export
`load_anchor_prose`; it also omitted `coefficient_sha256`, which the inherited
line-search helper would have accessed later.

The exact committed V11b failure evidence is
`S6_RESIDENT_SIGN_EQUIVALENCE_V11B_FAILURE.md`, file SHA-256
`c517bfa02bcf1fb3a75bc063cf3d33c721c1d25f99a5c4f4964d520de53c1cfd`,
commit `013d042371b9e9b0d2a050b53072b392a86a2574`. V11c refuses a changed
artifact and uses a fresh run directory.

## Complete substituted-module API

Static and runtime audit of `run_eggroll_es_anchor_line_search.py` fixes these
four direct surfaces:

- `__file__`
- `coefficient_sha256`
- `load_anchor_prose`
- `load_trainer`

Before any engine can be created, V11c loads the exact 128-row train-only prose
anchor, resolves the frozen V11c trainer class, and deterministically exercises
the coefficient hashing facade. This preflight runs in dry-run mode and is
repeated immediately on the real path before the inherited main is called.
Failure is terminal and precedes Ray/vLLM engine creation.

Everything else remains V11b: Qwen3.6-35B-A3B, layers 20--23, sigma `0.0003`,
32 base directions, D43/D44 raw and templated identities, A43/A44 references,
resident order `D43 -> A43 -> A44 -> D44`, 64 perturb/restore cycles, four TP=1
engines on GPUs `0,1,2,3`, alpha zero, no update, and exact V10 equivalence.
Validation, OOD, and heldout results are not selection surfaces.
