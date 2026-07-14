# S6 exact v8 data44 replay control v9b

V9b repeats the completed v8 data44 alpha-zero run under the distinct,
no-overwrite experiment name
`snapshot794_layer_v9b_middle_late_exact_replay_data44_basis20260714_control1`.
It exists only to test reproducibility and cannot select on validation,
OOD-QA, OOD prose, or heldout results.

The exact v8 trainer, worker, snapshot builder, population basis, seed 44,
middle-late layer plan, sigma `0.0003`, population `32`, domain batch `64`,
128-document anchor, cosine floor `0.8`, and four TP=1 GPU engines are reused
unchanged. Target alphas are exactly `[0.0]`; nonzero updates are impossible.

Before execution, the exact failed v8 report and both strict journals are
fully revalidated. The embedded seed44 reference is fixed to journal file
SHA-256
`3f317ec14c47cd8fccea63dc09401e85416e8d4f0c2dfbb1d5e6d8f26385a30d`,
content SHA-256
`f00f8f4d34c9885ed9904ad59d895bcb787d0fe1292977d7be8bf93841d68aa4`,
coefficient SHA-256
`c63bf1e2081245a36b45704ac416675943bf2311b018f3489511d0ac231241a3`,
and robust-plan SHA-256
`66867658644c5ff406e72033eea61b5eb4aa007f67d5080e0da298d28f001c78`.

The reporter compares coefficients and standardized raw domain/anchor score
vectors at a preregistered cosine threshold of `0.99`. Passing additionally
requires exact coefficient, raw domain-score, and raw anchor-score hashes.
