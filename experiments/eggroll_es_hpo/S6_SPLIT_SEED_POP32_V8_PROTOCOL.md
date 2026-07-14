# S6 split-seed population-32 direction diagnostic v8

V8 is a coefficient-only diagnostic. It does not apply a parameter update or
select on validation, OOD-QA, OOD prose, or holdout results.

## Frozen design

- Model: `models/Qwen3.6-35B-A3B`, no checkpoint overlay.
- Data: the exact frozen S6 train/eval artifacts inherited from v7.
- Layer plan: `middle_late`, layers 20--23, exact v6/v7 plan hashes.
- Runtime selection: 35 source units, 23 runtime parameters, 142,999,552
  elements, and 285,999,104 bf16 bytes.
- ES: sigma 0.0003, population 32, batch 64, mini-batch 64.
- Anchor: 128 documents, 512 input tokens, minimum cosine 0.8.
- Engines: four TP=1 engines on GPUs 0,1,2,3.
- Stage: `stability`; target alphas: exactly `[0.0]`.
- Data/bootstrap seeds: 43 and 44.
- Perturbation basis seed: 20260714. Both runs must contain the same exact 32
  perturbation seeds, bound by SHA-256
  `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`.
- Preregistered same-basis coefficient-cosine threshold: at least 0.5.

The data/bootstrap seed continues to control training-batch order, anchor
selection, and document bootstrap sampling. The v8 execution wrapper replaces
only the population perturbation seeds immediately before the inherited exact
v4/v5 alpha-zero execution path.

## Required predecessor evidence

The exact completed v7 family report and all four v7 journals named by that
report must pass their completed validators. The v7 report is evidence, not an
optimization input.

## Interpretation

Because both runs use the same perturbation basis, coefficient cosine measures
the reproducibility of the held-in train-plus-anchor response shape when data
and bootstrap sampling change. Passing does not establish QA improvement, OOD
safety, or holdout performance. A nonzero-alpha experiment requires a separate
versioned protocol.
