# S6 exact data43 deterministic replay v9

V9 is an alpha-zero reproducibility diagnostic. It cannot select on
validation, OOD-QA, OOD prose, or heldout results and cannot apply a nonzero
parameter update.

## Frozen predecessor evidence

The exact failed v8 report is required at
`experiments/eggroll_es_hpo/S6_SPLIT_SEED_POP32_V8_REPORT.json`, file SHA-256
`c1cbeb05ba25db1451d66c7d344f17cd86f535eb1a55d54584496029a0bca05d`
and content SHA-256
`ae5c861705268ceadc9254c0ee8ddd367f6a2971ba45f35d495426b077a8c438`.
The report must rebuild exactly from both strict v8 journals and retain its
formal failure: same-basis coefficient cosine `0.4276943787514416` below the
preregistered `0.5` threshold.

The report and both journals are rehashed and fully validated before v9 can
start or validate a completed replay. Their expected file, content,
coefficient, and robust-plan hashes are embedded in the v9 driver.

## Exact replay

V9 reuses the v8 trainer and worker unchanged so that a renamed distributed
extension cannot confound the replay. It repeats only the original v8 data43
run under the distinct experiment name
`snapshot794_layer_v9_middle_late_exact_replay_data43_basis20260714`.
The first launch completed GPU computation but exposed a redundant nested
offline-audit scope before release validation. Its immutable failed directory
is retained. The corrected no-overwrite replay name is
`snapshot794_layer_v9_middle_late_exact_replay_data43_basis20260714_retry1`;
the computational recipe is otherwise identical.

All computational inputs remain exact:

- Qwen3.6-35B-A3B with no checkpoint overlay;
- frozen S6 train/eval artifacts;
- middle-late layers 20--23 and the exact plan/model hashes;
- sigma `0.0003`, population `32`, batch and mini-batch `64`;
- global/data seed `43`, maximum generated tokens `32`;
- the exact v8 perturbation basis seed `20260714`, with basis SHA-256
  `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`;
- all 128 frozen anchor documents, 512-token cap, cosine floor `0.8`;
- four TP=1 engines on GPUs 0--3;
- target alphas exactly `[0.0]`.

The inherited evaluation pass remains a mechanical alpha-zero identity audit.
Its validation/OOD outputs cannot enter the v9 decision.

## Decision

The reporter compares the original v8 data43 journal with the v9 replay. It
reports coefficient cosine, standardized raw domain-score cosine, and
standardized raw anchor-score cosine. It also hashes the coefficient plan and
both unstandardized raw score arrays.

The preregistered cosine threshold is `0.99` for all three vectors. A
deterministic replay passes only if all three cosines meet that threshold and
the coefficient, raw domain-score, and raw anchor-score identities match
exactly. A mismatch is diagnostic evidence and must remain reportable; it does
not authorize a nonzero update, recipe tuning, or heldout access.
