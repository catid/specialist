# S6 layer-partitioned v4 conservative-anchor rescue protocol

## Status and motivation

The original frozen v4 family is closed with the alpha-zero baseline.  Both
seed-42 population-16 pilots reproduced the S6 baseline exactly and left OOD
QA unchanged at every alpha, but no nonzero state passed the paired-document
OOD-prose gate.  The front/back arm's validation-improving states had
materially negative prose deltas.  The middle control produced one state with
both a small validation increase and a positive mean prose delta, but its
paired-document lower confidence bound still allowed degradation.

This is a new, predeclared rescue family.  It changes only the strength and
coverage of the already-audited prose-anchor projection and the alpha bracket.
It does not use the sealed heldout split, change S6, tune against individual
validation examples, or reinterpret any result from the closed family.

## Frozen identities and execution order

The two layer plans, model, data, dense reward, runtime mappings, four-engine
topology, and all v4 identity/immutability audits remain exactly as specified
in `S6_LAYER_PARTITION_V4_PROTOCOL.md`:

- middle control first: `experiments/layer_plans/middle_matched_dense.json`,
  file SHA-256
  `f2b38054e3cdaf41619cce579d3ba2e030fa3cfa87fd42b50543f655ff5f6dc0`,
  semantic plan SHA-256
  `b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0`;
- front/back second: `experiments/layer_plans/front_back_dense.json`, file
  SHA-256
  `8e855cbd0d6130278e87b1af348e39dd0f683b8575d9abcb9260f3fe7b29d824`,
  semantic plan SHA-256
  `6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb`;
- model-config SHA-256
  `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`;
- frozen S6 train/validation/OOD-QA and 128-item anchor identities pinned by
  the v4 driver.

Both arms use seed 42, sigma 0.0003, population 16, domain batch 64,
evaluation mini-batch 64, generated-answer maximum 32 tokens, and four TP=1
Qwen3.6-35B-A3B engines on GPUs 0--3.  The middle arm runs first because it
was the only closed-family arm to show a simultaneous positive validation
and mean-prose response; the front/back run remains mandatory as the matched
location comparison.

## Single predeclared rescue change

Each population evaluates all 128 frozen prose-anchor documents rather than a
rotating subset of 32, and projects the domain direction to a cosine floor of
0.8 rather than 0.5.  This reduces anchor sampling variance and reduces the
largest permitted component orthogonal to the anchor direction from about
86.6% to 60%.  No other direction-estimation parameter changes.

The monotonic screening targets are fixed before either run:

`0, 0.00000078125, 0.0000015625, 0.000003125, 0.00000625`

The smaller bracket is required because the smallest nonzero alpha in the
closed family already crossed the strict prose boundary.  Sigma is not tuned
in this family: the prior direction moved validation, while lowering sigma
would reduce BF16 perturbation signal-to-noise and confound the anchor change.

## Gates and decision rule

Alpha zero must reproduce the frozen S6 validation, OOD-QA, and OOD-prose
baselines exactly in each arm.  A nonzero state is eligible only when:

- document-disjoint validation mean reward is strictly higher than that
  arm's alpha-zero value;
- OOD-QA exact count, nonzero count, and mean reward are all nondecreasing;
- the paired-document OOD-prose lower confidence bound proves no degradation;
- every v4 plan, mapping, selected/unselected partition, coefficient,
  reference, application, and final-state audit passes on all four engines.

After both arms finish, select among eligible states by largest validation
improvement, then smaller alpha, then lexicographically smaller plan name.  If
there is no eligible nonzero state, alpha zero wins and this rescue family
closes.  A path-dependent pilot state is never a final model: its chosen alpha
must be restored from alpha zero and directly replicated with fixed seeds
43--47 under the same 128-anchor/cosine-0.8 recipe before any heldout access.
No seed-specific alpha or recipe changes are allowed.

The sealed heldout split remains unopened until a replicated candidate and
all selection rules are fixed.  Concurrent manual dataset curation remains
isolated and cannot enter this byte-frozen S6 family.
