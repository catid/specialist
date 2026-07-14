# S6 exact data-seed-44 replay v9b result

The Qwen3.6-35B-A3B middle-late alpha-zero control replay completed from the
exact v8 data-seed-44 inputs under a distinct no-overwrite experiment name.
Its final journal passes the complete v9b-to-v4 provenance and zero-update
audits.  The population rollout used exactly four independent one-GPU actors,
repeatedly reached 100% utilization on every GPU, and had no CUDA co-tenant.

The preregistered replay gate passes.  The coefficient cosine is
`0.9999999999999999`, standardized raw domain-score cosine is
`1.0000000000000002`, and standardized raw anchor-score cosine is `1.0`, all
above the `0.99` threshold.  The coefficient, unstandardized domain-score,
and unstandardized anchor-score SHA-256 identities also match the original v8
seed-44 run exactly.  Together with v9's exact seed-43 replay, this confirms
that v8's cross-seed instability comes from changed domain/reference sampling,
not execution nondeterminism.

Alpha zero remains selected.  No parameter update was applied, validation or
OOD outputs were not used for selection, and the sealed holdout remains
unopened.  The next protocol uses antithetic perturbations and crossed frozen
domain/anchor references to reduce and decompose estimator variance before
any nonzero-alpha or LISA-style architectural comparison.

`S6_DATA44_REPLAY_V9B_REPORT.json` is the machine-readable result.  Its file
SHA-256 is
`c608047faaafe2c081417328e794f43d2201bd789b017a61ecfed1dd10031248`
and its self-reported content SHA-256 is
`e69f2a6f43836ed8735bcccb2a190a841bd340fc4421c8830b8775300efb14e8`.
The raw replay journal remains in the ignored `runs/` workspace and is bound
by file, content, coefficient-plan, and perturbation-basis identities in the
report.
