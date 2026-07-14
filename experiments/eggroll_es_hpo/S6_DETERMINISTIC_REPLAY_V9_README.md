# S6 exact deterministic replay v9 result

The corrected Qwen3.6-35B-A3B middle-late alpha-zero replay completed from
the exact v8 data-seed-43 inputs.  Its journal passes the complete v9-to-v4
provenance and zero-update audits.  The population rollout used exactly four
independent one-GPU actors, repeatedly reached 100% utilization on all four
GPUs, and had no CUDA co-tenant.

The preregistered replay gate passes.  Coefficient, standardized raw domain
score, and standardized raw anchor score cosines are all exactly `1.0`, above
the `0.99` threshold.  The unstandardized coefficient, domain-score, and
anchor-score arrays also match their original v8 SHA-256 identities exactly.
This establishes that the fixed-input execution path is deterministic and
that v8's `0.427694` split-seed coefficient cosine reflects changed
domain/reference sampling rather than execution nondeterminism.

Alpha zero remains selected.  No parameter update was applied, validation or
OOD outputs were not used for selection, and the sealed holdout remains
unopened.  The next protocol reduces and decomposes estimator variance with
antithetic perturbations and crossed frozen domain/anchor references before
any nonzero-alpha or LISA-style architectural comparison.

`S6_DETERMINISTIC_REPLAY_V9_REPORT.json` is the machine-readable result.  Its
file SHA-256 is
`8dac33ad828a29021f107074a79aa536267349a6903f3f3b8d9a89146e7859a3` and
its self-reported content SHA-256 is
`7871c2d959e9b70f9d0912a1af5571767da155fd3ffc5a3588ab6e8cbdb7a50f`.
The raw replay journal remains in the ignored `runs/` workspace and is bound
by file, content, coefficient-plan, and perturbation-basis identities in the
report.
