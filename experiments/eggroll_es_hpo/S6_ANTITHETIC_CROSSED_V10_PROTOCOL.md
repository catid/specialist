# S6 antithetic crossed D x A diagnostic v10

V9 proved exact execution determinism: the original and replayed data43
coefficient, raw domain-score, and raw anchor-score vectors matched bit for
bit, with all cosines `1.0`. V10 therefore measures data/reference variation
without applying a parameter update or consulting validation, OOD, prose, or
heldout outcomes.

## Frozen estimator

- Model and plan: exact Qwen3.6-35B-A3B middle-late layers 20--23.
- Base perturbation basis: the exact v8/v9 32-seed basis, SHA-256
  `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`.
- Signs: each base seed is evaluated at both `+sigma` and `-sigma`, sigma
  `0.0003`, with exact restoration after every sign. Every wave contains four
  seeds so all four TP=1 engines remain occupied.
- Domains: the exact first 64 training rows under loader seeds 43 and 44,
  identified respectively by SHA-256
  `b864cfcc4ebcd987d8091f1067f631366c128d63d09fb7160a09561d10063a0f`
  and `3574ff126f727a262957f34ab83fbefce6754ae9e4be790f810f42656e692bc2`.
  Their ordered 128-row concatenation is
  `a1b77aed57313c0dec44195a35232818426668a771db2c5055bb6b28c304289a`.
- Anchors: all 128 frozen anchor documents are generated at seeds 43 and 44.
  Document-LCB bootstrap configuration remains unchanged and fixed.
- Central responses: `(score_plus - score_minus) / 2`, calculated before
  standardization and cosine-0.8 projection.
- Cells: `D43xA43`, `D43xA44`, `D44xA43`, and `D44xA44`.
- Runtime: one combined batch of 128, mini-batch 64, 32 base directions / 64
  unique signed directions, four GPUs, alpha exactly zero. The two domain
  minibatches are evaluated in separate parent calls, so those 64 signed
  directions are applied once to D43 and once to D44: exactly 128 actual
  perturb/restore cycles and 128 domain-signed scores. Anchors are evaluated
  only during the D43 residency, producing 64 signed responses for each of
  two generation seeds, or 128 anchor-signed responses total.

The positive A43 path remains a complete v5-compatible legacy plan so all
inherited dense-reward, exact-restore, population-boundary, partition, and
document-LCB audits continue to run. The other sign/reference results are
stored as a separately hashed, numerically replayed diagnostic artifact.

## Decision

The report exposes raw central domain and anchor cosines and all six pairwise
coefficient cosines across the four cells. The preregistered pass requires
minimum pairwise coefficient cosine at least `0.5` and median at least `0.7`.
No benchmark result participates. Passing permits design of a separate
consensus/nonzero protocol; failing keeps alpha zero and calls for additional
batch averaging or anchor pooling.
