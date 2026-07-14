# S6 four-arm edge-split document-LCB v6 protocol

## Frozen question and inheritance

V6 tests whether the front and back four-layer motifs behave differently from
capacity-matched middle motifs under the committed v5 document-LCB objective.
It changes only the frozen selected-parameter partition.  V5's train-only
20,000-resample document LCB, exact alpha-zero reference, cosine-0.8
projection, and numeric/hash replay remain unchanged.  V4's exact restoration,
selected/unselected byte identities, four TP=1 engine audits, two-phase update,
strict validation/OOD-QA/OOD-prose gates, and no-overwrite rule remain
unchanged.  Validation, OOD, and heldout data cannot enter optimization; the
heldout split remains unopened and unscored.

The unchanged legacy v4 audit runs inside a nonblocking, non-reentrant v6
scope that admits exactly the four v6 plan identities and the shared
35-unit/23-runtime-parameter/142,999,552-element capacity.  The scope verifies
the original v4 constants before entry and restores their exact objects in a
`finally` block on success or failure.  Its policy/configuration identity is
bound into every journal.

Every implementation entry is exact-keyed and independently rehashed from
disk during offline validation.  The bundle covers the complete inherited
v1--v5 driver/trainer/worker chain, reward and anchor helpers, v6 sources, the
offline auditor, model configuration, all four plan manifests, this protocol,
and the v6 contract tests.  Missing, extra, stale, or self-consistently forged
implementation identities make a journal ineligible.

The only predeclared comparisons are `front` versus `middle_early` and `back`
versus `middle_late`.  Cross-pair comparisons are descriptive, not selection
tests.

## Frozen arms

Every arm contains one complete [linear, linear, linear, full-attention]
motif, 35 checkpoint units, 23 packed runtime parameters, 142,999,552 BF16
elements, and 285,999,104 bytes.  The model-config SHA-256 is
`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.

| arm | layers | file | file SHA-256 | semantic SHA-256 | paired arm |
|---|---:|---|---|---|---|
| front | 0--3 | `experiments/layer_plans/front_dense.json` | `02e5ce4cc2e20cf6b0910578a3e7982569d323b3458593000f77c624a8db62bf` | `af9dcf4e5c932aeb192ee0d195e9c4fee9d4a510467d850d7cf26f6db4c2d823` | middle_early |
| middle_early | 16--19 | `experiments/layer_plans/middle_early_dense_v6.json` | `1496184e483071537cd95e10fd8cd051d7bd18c947df1b1e76d72f7d47bafab1` | `d72624f2ef55b49b40aa8e52910394f079827a2d848bacc1ee42abb82c47846d` | front |
| middle_late | 20--23 | `experiments/layer_plans/middle_late_dense_v6.json` | `d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df` | `03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9` | back |
| back | 36--39 | `experiments/layer_plans/back_dense.json` | `73bfc82ba057908c0071d3c5e190581fecf6147cc398f06a994231f31908187e` | `6da92a4db760676acda1bcbcaec4a925a6dd7b641c250a58a3fe4837d97ac93a` | middle_late |

`build_eggroll_es_edge_split_plans_v6.py --check-directory
experiments/layer_plans` must reproduce all four artifacts byte-for-byte.
Neither a custom path containing copied bytes nor any fifth semantic identity
is accepted by the controller or workers.

## Mechanical smokes

Run all four arms at seed 42, sigma 0.0003, population 4, domain batch 8,
evaluation batch 64, all 128 frozen training-anchor documents, cosine floor
0.8, and targets `0, 7.8125e-7`.  All four GPUs run as independent TP=1
engines and a partial population wave is forbidden.  Smokes establish
mechanics and provenance only; their metrics cannot select an arm.

The train/eval roots, model, absent checkpoint, GPU order, evaluation split
order, anchor and OOD-prose paths/caps, reward timeout 10, output root,
disabled external logging, and arm/stage/seed run-name structure are also
exact.  Each is persisted in the v6 recipe and replayed offline.

The front smoke command template is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 es-at-scale/.venv/bin/python \
  run_eggroll_es_anchor_line_search_v6.py \
  --v6-stage smoke \
  --layer-plan-json experiments/layer_plans/front_dense.json \
  --expected-layer-plan-file-sha256 02e5ce4cc2e20cf6b0910578a3e7982569d323b3458593000f77c624a8db62bf \
  --expected-layer-plan-sha256 af9dcf4e5c932aeb192ee0d195e9c4fee9d4a510467d850d7cf26f6db4c2d823 \
  --expected-model-config-sha256 93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99 \
  --model-name /home/catid/specialist/models/Qwen3.6-35B-A3B \
  --train-dataset /tmp/specialist-s6-candidate-guarded-ead1b21/dataset/train \
  --eval-dataset /tmp/specialist-s6-candidate-guarded-ead1b21/dataset/eval \
  --sigma 0.0003 --population-size 4 --batch-size 8 \
  --mini-batch-size 64 --max-tokens 32 --seed 42 \
  --n-vllm-engines 4 --n-gpu-per-vllm-engine 1 --use-gpus 0,1,2,3 \
  --eval-splits validation,ood_qa \
  --target-alphas 0,0.00000078125 \
  --anchor-prose-jsonl data/general_prose_anchor_v1.jsonl \
  --anchor-prose-report data/general_prose_anchor_v1.report.json \
  --anchor-items-per-step 128 --anchor-max-input-tokens 512 \
  --min-anchor-cosine 0.8 \
  --ood-prose-jsonl data/ood_prose_v3.jsonl \
  --ood-prose-max-input-tokens 1024 --reward-function-timeout 10 \
  --output-directory experiments/eggroll_es_hpo/runs \
  --experiment-name snapshot794_layer_v6_front_smoke_seed42 --logging none
```

Substitute only the frozen plan path, both plan hashes, and unique arm name for
the other three smokes.  Existing output directories cannot be resumed or
overwritten.

## Gated pilots

Only after every smoke is complete and passes the v6 offline audit, create a
strict JSON smoke-gate artifact listing exactly four entries with keys `arm`,
`journal`, and `content_sha256` under schema
`eggroll-es-edge-split-smoke-gate-v6`.  The pilot CLI independently reads and
fully audits each journal and binds all four coefficient, robust-plan, and
journal identities into its snapshot.

Then run each arm at seed 42, population 16, domain batch 64, and fixed targets
`0, 7.8125e-7, 1.5625e-6, 3.125e-6, 6.25e-6`, adding:

```text
--v6-stage pilot --v6-smoke-gate-json PATH
```

The pilot CLI rejects any other population, batch, seed, target grid, anchor
configuration, topology, model, data identity, or evaluation split.

## Eligibility

A nonzero pilot state is eligible only if specialist validation mean strictly
improves, OOD-QA exact/nonzero/mean do not decrease, and OOD-prose point delta
and paired-document lower confidence bound are nonnegative.  Every inherited
partition, application, identity, content-hash, and numeric replay must pass.
Within each predeclared pair, compare the best eligible state by validation
improvement, then smaller alpha, then lexicographically smaller arm.  If a pair
has no eligible nonzero state, alpha zero wins for that pair.  No heldout score
may be consulted before a directly replicated candidate and decision rule are
fixed.
