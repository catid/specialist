# S6 guarded 794-row baseline

This package freezes the valid alpha-zero evaluation of the guarded S6
candidate. It used Qwen3.6-35B-A3B with seed 42 and evaluated the model before
population estimation: validation was `0.08381010452961674`, OOD QA was
`0.714128787878788`, and OOD prose was `-1.2632580042542214`. The strict OOD
gates passed, and the sealed 18-row heldout split was neither loaded nor
evaluated.

No training update was applied (`target_alphas=[0]`, `applied_alpha=0`, zero
plan applications). The coefficient plan produced after the baseline is
retained only as a diagnostic and is non-selectable. Its v1 path has two known
confounds: BF16 additive perturb/restore is not bit-exact, and combining domain
and anchor requests changes domain generation batching. These confounds occur
after the baseline and do not invalidate the baseline measurement.

## Frozen identity

- Guarded training JSONL: `f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776` (794 rows)
- Training Arrow: `6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6`
- Validation Arrow: `19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db` (41 rows)
- OOD-QA Arrow: `b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced` (24 rows)
- Sealed heldout Arrow: `df23a704d0f621bffd8b55fb4a0a296e06a79feaf79cfe1bd357d55bb4f07cf1` (18 rows; not loaded)
- Domain/OOD-QA/OOD-prose JSONL: `ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b`, `25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d`, `3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57`
- Dataset manifest: `a48fba83ba41f3495f1c397f05576e66852d6ec7ef44edba725ed38753510435`
- Eval rebuild report: `2b011982350bcedf6e6e247ecffa0caa990c66d380d3aa896e84372b9a8929dd`
- Run journal: `a5acf0ecf46770abe4af80d622cc5ce35ded49e578fe4e6f40b07970a368bcdf`
- Diagnostic plan: `d3b1983b1cfebcafeb61909edcf59e3d91cb18d2ba5879303e734f798da1af91`
- Raw OOD-prose baseline: `38276f2dedaaf5062ca8e58aacb9990c3a33bb1b44e3c7172798334da5bfd239`
- GPU trace: `a4d2e3f91648aaf4aaec5474186781c467bc17795c882338acad675f0567f3fe`

The compact machine-readable record is `baseline_summary.json` (SHA-256
`a42f0589a2734fab1c497465b42eaacae6bf52291e31827d17ad9cb57dbac9cb`).
