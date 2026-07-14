# S6 V11b resident-sign dual-manifest retry

V11b is a minimal retry of V11. V11 failed before perturbation because the
trainer compared already-templated prompts against hashes defined for raw
questions. The failed journal is immutable and bound by exact file and
canonical hashes. V11b does not edit, resume, or overwrite that run.

The retry keeps V11's model, layer plan, sigma, 32 directions, D43/D44 rows,
A43/A44 references, generation order, resident-sign schedule, alpha zero, and
exact V10-equivalence gate. It changes only manifest typing and the fresh run
name.

Raw question/answer identities remain:

- D43 `b864cfcc4ebcd987d8091f1067f631366c128d63d09fb7160a09561d10063a0f`
- D44 `3574ff126f727a262957f34ab83fbefce6754ae9e4be790f810f42656e692bc2`

They are checked by the crossed loader and snapshot. After the unchanged
specialist template is applied, trainer capture separately requires:

- D43 `54f53464e479fa9dd0c80263f0e424a3d225681c1d8f15554b171f6d5b40c637`
- D44 `44cc0ba38c7b2c685a2c44699be9f6dd6313c1391765e13c046812f06e280c23`

The two namespaces may never be substituted for each other. Regression tests
must change one surface at a time and require the corresponding validator to
fail.

The run uses four TP=1 engines on GPUs `0,1,2,3`, forbids partial waves, and
applies no update. It must exactly match V10 raw responses, crossed cells, and
projected plans. Validation, OOD, and heldout results are not selection
surfaces. A failed equivalence closes the retry.
