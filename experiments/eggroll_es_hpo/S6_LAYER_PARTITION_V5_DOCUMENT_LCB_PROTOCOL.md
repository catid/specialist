# S6 layer-partitioned document-LCB anchor v5 protocol

## Question and separation from v4

The closed v4 mean-anchor families established that front/back layer updates
can improve document-disjoint specialist validation, but the update can harm
some prose documents even when its mean training-anchor geometry is strongly
positive.  V4 reduced all 128 training-anchor documents to one token-weighted
scalar per perturbation, discarding the cross-document variation measured by
the strict OOD-prose gate.

V5 is a new, predeclared family.  It retains the frozen S6 data, both audited
layer plans, dense gold-answer objective, exact-reference restoration,
four-engine update protocol, evaluation gates, and sealed heldout policy.  Its
only optimization change is to replace the scalar mean training-anchor
fitness with an exact-reference-relative, paired-document bootstrap lower
confidence bound.  No OOD, validation, or heldout example is an optimization
input.

## Frozen model, data, plans, and topology

The identities remain those pinned by the v4 driver and protocols:

- Qwen3.6-35B-A3B model-config SHA-256
  `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`;
- middle control plan file SHA-256
  `f2b38054e3cdaf41619cce579d3ba2e030fa3cfa87fd42b50543f655ff5f6dc0`
  and semantic SHA-256
  `b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0`;
- front/back plan file SHA-256
  `8e855cbd0d6130278e87b1af348e39dd0f683b8575d9abcb9260f3fe7b29d824`
  and semantic SHA-256
  `6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb`;
- frozen S6 train, validation, OOD-QA, 128-document training anchor, and
  16-document OOD-prose identities enforced by the inherited v4 snapshot;
- four TP=1 engines on GPUs 0--3, with no partial population wave.

Sigma is fixed at 0.0003, generated-answer evaluation uses mini-batch 64 and
maximum 32 tokens, and all 128 frozen training-anchor documents are evaluated
for every population member.  The coefficient cone floor remains 0.8 so the
new objective is isolated from cone-strength HPO.

## Document-LCB objective

For each training-anchor document `j`, retain its ordered identity, selected-
token log-probability sum, and selected-token count at the exact alpha-zero
reference.  Retain the same numeric summary for perturbation `i`.  Raw prose
text is never written to the plan.

One common document-resampling plan is generated without mutating global RNG
and is reused for every population member.  For bootstrap resample `r`:

`delta[i,r] = perturbed_token_sum[i,r] / token_count[r]
              - reference_token_sum[r] / token_count[r]`

The anchor fitness for population member `i` is the linearly interpolated
2.5th percentile across 20,000 document resamples.  Higher is better.  The
fixed configuration is:

```json
{
  "schema": "eggroll-es-document-lcb-anchor-v1",
  "source_split": "anchor_prose",
  "reference": "exact_alpha_zero_selected_plan_weights",
  "document_unit": "document_id",
  "within_document": "sum_selected_token_logprob_and_scored_token_count",
  "document_identity_in_provenance": "sha256_utf8",
  "bootstrap_document_sampling": "uniform_with_replacement",
  "bootstrap_samples": 20000,
  "bootstrap_seed": 20260715,
  "bootstrap_prng": "python_random_mt19937_randrange_v1",
  "common_resamples_across_population": true,
  "percentile": 0.025,
  "percentile_interpolation": "linear",
  "fitness": "paired_document_bootstrap_lower_bound",
  "higher_is_better": true,
  "population_standardization": {
    "method": "population_zscore",
    "epsilon": 1e-8,
    "zero_spread": "return_all_zero_coefficients"
  }
}
```

The canonical configuration SHA-256 is
`da49dd210bf5375cc8c96220744695e5f772546fc55c997efa239053c6498cae`.

Document identities/counts must align exactly across reference and every
population member.  Input rows are canonically ordered by hashed document
identity; persisted rows in any other order, or duplicate, missing,
non-finite, or token-count-drifted summaries, fail closed.  Zero robust-score
spread produces a zero update.  The reference summaries, population matrix,
common resampling plan, robust scores, objective configuration, projection,
coefficients, and their canonical hashes are bound into the persisted v5
coefficient plan and independently recomputed before any update is accepted.

## Execution stages

1. Run one seed-42 mechanical smoke for the middle plan and one for the
   front/back plan, each with population 4, domain batch 8, all 128 anchor
   documents, cosine floor 0.8, and targets 0 then 7.8125e-7.  Smokes validate
   mechanics and provenance only; they cannot select a model.
2. Only if both smokes reproduce alpha zero exactly and every identity,
   robust-objective, partition, and update audit passes, run paired seed-42
   pilots for both plans with population 16, domain batch 64, and fixed targets
   `0, 7.8125e-7, 1.5625e-6, 3.125e-6, 6.25e-6`.
3. Among nonzero pilot states passing every strict gate, select largest
   validation improvement, then smaller alpha, then lexicographically smaller
   plan name.  If none qualifies, alpha zero wins and v5 closes.  The v4
   mean-anchor runs are controls, not adaptive v5 tuning observations.
4. A selected path-dependent state must be restored directly from alpha zero
   and replicated at the one fixed alpha for seeds 43--47 in both the selected
   arm and capacity-matched control.  No seed-specific recipe changes are
   allowed.

## Selection and release gates

A candidate is eligible only if validation mean reward strictly improves,
OOD-QA exact/nonzero/mean do not decrease, and the OOD-prose point delta and
paired-document lower confidence bound are both nonnegative.  Every v4
selected/unselected partition audit and every v5 robust numeric/hash
recomputation must pass on all four engines.  Any incomplete run, provenance
drift, non-finite statistic, or disagreement fails closed.

The heldout split remains unopened and unscored until a directly replicated
candidate, alpha, plan, objective, and decision rule are fixed.  Manual
dataset work may continue concurrently but cannot enter byte-frozen S6.
