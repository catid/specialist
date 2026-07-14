# S5 guarded multi-seed HPO protocol

This protocol selects on the 41-item document-disjoint `validation` split and
fails closed on the 24-item `ood_qa` split and 16-document `ood_prose` probe.
The sealed holdout is not an HPO input and must remain unopened until the
candidate, horizon, seeds, and stopping rule have all been fixed.

## What the completed probes establish

The provisional 908-row transfer of `sigma=0.001`, `alpha=0.00025`, and three
updates was not seed-stable. On the original 109-item provisional validation
cohort, seed 42 changed mean reward by `-0.007801` and seed 43 by `+0.009888`.
Seed 42 also changed OOD-QA mean reward by `-0.001190`, while seed 43 left it
unchanged. This setting is therefore not eligible for transfer.

After the evaluation audit reduced validation to 41 grounded items, the
one-seed 908-row probes produced the following exploratory results. All three
left OOD-QA mean reward and exact answers unchanged.

| Sigma | Alpha | Updates | Validation delta | Changed pairs | Paired 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0.00015 | 3 | +0.000102 | 1 win / 0 losses | [0.000000, +0.000305] |
| 0.0005 | 0.00025 | 3 | +0.003796 | 2 wins / 0 losses | [0.000000, +0.010412] |
| 0.0010 | 0.00025 | 3 | +0.002820 | 1 win / 0 losses | [0.000000, +0.008461] |

These effects depend on only one or two examples and are not replications.
More importantly, the same `sigma=0.0005`, `alpha=0.00025`, three-update
candidate on the final 784-row snapshot changed validation by `-0.000375`
(2 wins, 2 losses, 37 ties; 95% CI `[-0.010513, +0.008568]`). OOD-QA mean and
exact answers were unchanged, but OOD-prose mean token log-probability changed
by `-0.004004`; its source-document bootstrap interval was
`[-0.007470, -0.000239]`. The final-snapshot `sigma=0.001`,
`alpha=0.00025` probe improved validation by only `+0.000763` (2 wins, 1 loss,
38 ties; 95% CI `[-0.001955, +0.003690]`) and left OOD QA unchanged, while
OOD prose changed by `-0.004475` with interval
`[-0.007307, -0.001170]`. Both candidates pass the historical `0.02`
non-inferiority margin but fail the strict no-OOD-degradation rule below. The
baseline remains the current selected model.

## Frozen boundary

Every journal in one aggregate must pin the same identities:

- final S5 training JSONL: 784 rows,
  SHA-256 `62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507`;
- final S5 training Arrow: SHA-256
  `c4935458da6e887064ed181e9ec8ee490752cca2b6a1d33ecb7aa58c201c851f`;
- validation Arrow: SHA-256
  `19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db`;
- OOD-QA Arrow: SHA-256
  `b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced`;
- OOD-prose JSONL: SHA-256
  `3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57`;
- model snapshot identity, trainer SHA-256, candidate settings, and raw
  evaluation-output SHA-256 values.

A dataset, trainer, model, metric, or threshold change starts a new experiment
family; its results cannot be pooled with the old family.

## Predeclared search and gates

Use seeds `42,43,44,45,46`. Treat a full setting as
`(sigma, alpha, updates, population_size, batch_size, max_tokens)`; changing any
field makes a different candidate. A compact next grid is the three
already-probed conservative pairs above at three updates. Only after one pair
passes the multi-seed rule may a separately declared horizon comparison be run.

For every candidate and every seed:

1. Validation is the sole optimization objective: `mean_reward` must be
   recorded with exact, nonzero, wins/losses/ties, and a paired item bootstrap.
2. OOD QA is a strict guard: mean-reward delta must be at least zero and exact
   count delta must be at least zero. Both raw arrays must align by
   `(prompt, answer)` before comparison.
3. OOD prose is a strict source-document bootstrap guard: both the point delta
   and the lower endpoint of its paired 95% interval must be at least zero.
   Use 20,000 resamples, seed `20260714`, grouping by
   `normalized_source_url`, with the token-weighted corpus mean.
4. A missing, malformed, hash-mismatched, or failed guard makes the candidate
   ineligible. Do not replace a failed metric with a cached value from another
   seed or dataset snapshot.

Across the five seeds, a candidate is eligible only when every per-seed OOD
gate passes, mean and median validation deltas are positive, at least four of
five validation deltas are positive, and
`mean_delta - 0.5 * population_stddev(delta)` is positive. Rank eligible
candidates by that risk-adjusted score. If none is eligible, select the
baseline. This rule is fixed before seeing the remaining seeds.

The current aggregation command supports the mean/exact/prose fail-closed
checks. The strict thresholds are:

```bash
python3 aggregate_eggroll_es_hpo.py seed42.json seed43.json seed44.json \
  seed45.json seed46.json \
  --selection-split validation --guard-splits ood_qa \
  --max-guard-degradation 0 --max-guard-exact-loss 0 \
  --require-ood-prose-guard --min-positive-seed-fraction 0.8 \
  --risk-penalty 0.5 --output aggregate.json
```

Run generation must also use `--max-ood-prose-degradation 0`; otherwise a
`true` prose decision was made under a looser policy and is not admissible for
this strict aggregate.

## Seed-journal contract

`run_eggroll_es_hpo.py` already emits the necessary information. A compact
consumer must require the following shape (extra provenance fields are
encouraged):

```json
{
  "schema": "eggroll-es-hpo-v1",
  "seed": 42,
  "selection_split": "validation",
  "guard_splits": ["ood_qa"],
  "max_guard_degradation": 0.0,
  "max_guard_exact_loss": 0,
  "max_ood_prose_degradation": 0.0,
  "final_holdout_used_for_selection": false,
  "dataset": {
    "snapshot": {
      "train_arrow": {"sha256": "..."},
      "eval_arrows": {
        "validation": {"sha256": "..."},
        "ood_qa": {"sha256": "..."}
      },
      "ood_prose_jsonl": {"sha256": "..."}
    }
  },
  "model": {"metadata": {}, "weight_shards": []},
  "trainer_sha256": "...",
  "baseline": {
    "evaluation_scores": {"validation": 0.0, "ood_qa": 0.0},
    "evaluation_details": {
      "validation": {"rows": 41, "exact": 0, "nonzero": 0, "sha256": "..."},
      "ood_qa": {"rows": 24, "exact": 0, "nonzero": 0, "sha256": "..."}
    }
  },
  "results": [{
    "name": "sigma5e-4_alpha2.5e-4",
    "sigma": 0.0005,
    "alpha": 0.00025,
    "steps": 3,
    "evaluation_scores": {"validation": 0.0, "ood_qa": 0.0},
    "evaluation_details": {
      "validation": {"rows": 41, "exact": 0, "nonzero": 0, "sha256": "..."},
      "ood_qa": {"rows": 24, "exact": 0, "nonzero": 0, "sha256": "..."}
    },
    "ood_prose_guard_passed": false,
    "ood_prose_gate": {
      "delta": 0.0,
      "paired_document_bootstrap_95_ci": [0.0, 0.0],
      "max_degradation": 0.0,
      "passed": false
    },
    "run_summary": "..."
  }]
}
```

The aggregate journal must preserve every per-seed baseline, treatment,
split delta, exact-count decision, prose decision, eligibility reason, and the
fixed thresholds. Paths are convenience fields; SHA-256 values are the
identities. Holdout paths, scores, and decisions do not belong in either HPO
journal.
