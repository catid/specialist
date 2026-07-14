# Deterministic layer-location screen

These artifacts are the first location comparison in this repository run with
SGLang deterministic inference enabled. They supersede the older
nondeterministic screens for interpreting layer-location effects, but they are
still development experiments rather than final-test results.

## Runtime and determinism

The four single-GPU Qwen3.6-35B-A3B replicas were launched with:

```bash
ES_DETERMINISTIC=1 ./serve_es.sh
```

This adds `--enable-deterministic-inference`. SGLang 0.5.15 reported PyTorch
sampling, FlashInfer attention, and radix-cache disablement on every replica.
The independent check was:

```bash
.venv/bin/python probe_es_determinism.py \
  --trials 3 \
  --output experiments/es_determinism/probe_v1.json
```

All four replicas returned the same hash for a 32-question canary. All three
sharded 405-item trials returned answer hash
`a692b4d8c8ec87217e32b466c3cf12e491fae47d7e474b96df5c7bb629da6ff2`
and exact train/development-held rewards
`.02957132118376376/.0304041190300207`.

The post-trained seed-9909 front+back check is
`../es_determinism/front_back_raw_seed9909_post.json`; its two complete probes
were also identical.

## Recipe

Every completed run used the byte-identical final curated dataset, 70 dense
units, six commits, eight antithetic pairs, 32 facts per member, sigma .01,
learning rate .04, rank-4 noise, and raw fitness shaping. For example:

```bash
.venv/bin/python es_train_acc.py \
  --data data/train_qa_curated_v1.jsonl \
  --layer-plan front_back \
  --unit-groups dense \
  --gens 6 --pairs 8 --qa-per-eval 32 --eval-every 3 \
  --sigma .01 --lr .04 --rank 4 --shaping raw --seed 7707 \
  --journal experiments/es_deterministic/front_back_raw_seed7707.jsonl
```

The paired `middle_matched` run changed only the layer plan, seed-specific
journal name, and freshly relaunched base replicas. Seeds were 7707, 8808, and
9909. The summary can be regenerated with:

```bash
.venv/bin/python summarize_es_experiments.py \
  experiments/es_deterministic/middle_raw_seed7707.jsonl \
  experiments/es_deterministic/front_back_raw_seed7707.jsonl \
  experiments/es_deterministic/middle_raw_seed8808.jsonl \
  experiments/es_deterministic/front_back_raw_seed8808.jsonl \
  experiments/es_deterministic/middle_raw_seed9909.jsonl \
  experiments/es_deterministic/front_back_raw_seed9909.jsonl \
  --aggregate-generations 6 \
  --output experiments/es_deterministic/summary_raw_v1.json
```

## Result and limits

Front+back development-held deltas were +.0006, +.0046, and +.0011 (3/3
positive; mean +.0021; sample SD .00218). Matched-middle deltas were -.0005,
+.0003, and -.0013 (1/3 positive; mean -.0005; sample SD .00080).

The reward was extremely sparse. Each front+back run had only two informative
antithetic pairs across 48, and four of six generations had no signal. Middle
runs had 4, 6, and 5 informative pairs. The evaluation is known-contaminated
development data, not an independent final set, and train-probe movement did
not consistently track held movement. The result supports prioritizing the
edge-layer hypothesis in the next experiment; it does not establish a winner.

`middle_seed7707.jsonl` is an intentionally retained, interrupted rank-shaped
diagnostic. One .00075 member reward among 15 zeros became a .2667 commit
coefficient and the probes collapsed to zero. It motivated making raw shaping
the default. The next recipe should add a denser correctness-anchored component
before spending on longer location sweeps.
