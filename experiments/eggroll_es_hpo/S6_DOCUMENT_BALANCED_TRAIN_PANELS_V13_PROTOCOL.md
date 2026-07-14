# S6 document-balanced crossed train panels v13

## Why this experiment exists

The strongest remaining instability is associated with the training-data
panel, not nondeterministic execution. With the perturbation basis held fixed,
data43 versus data44 produced coefficient cosine
`0.4276943787514416`. V10's raw central domain cosine was only
`0.34415939658856237`. Meanwhile the exact data43 and data44 replays each
reached coefficient cosine `1.0`. The next estimator experiment therefore
changes only train-panel construction; it does not reinterpret validation,
OOD, heldout, or benchmark evidence.

The frozen 794-row training source has only 310 source documents. One document
contributes as many as 27 rows, so ordinary row shuffling can make a page an
implicit high-weight expert in an ES direction score.

## Frozen sampling frame

`eggroll_es_train_panel_sampler_v13.py` requires the exact source JSONL SHA-256
`f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776`
and records the exact Arrow SHA-256
`6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6`.
It never opens an evaluation split.

Rows connected by either the same `document_sha256` or a conservative,
deterministic lexical-semantic cluster are collapsed into one conflict unit.
Only one deterministic representative can be sampled from a unit. This gives
the frozen frame 310 conflict units and guarantees that selected panels cannot
share a source document or discovered semantic cluster. The current curated
snapshot already removed semantic duplicates, so all 794 conservative
lexical-semantic clusters are singletons; the document constraint does the
material balancing work in this snapshot.

Every conflict unit is assigned to its dominant explicit stratum, with fixed
tie priority:

- safety/consent: 48 units;
- technique: 94 units;
- equipment/material: 39 units;
- resources/general: 129 units.

Keyword rules and their hash are stored in the generated manifest. Safety and
consent take priority over equipment, equipment over technique, and unmatched
content goes to resources/general.

## Panels, weights, and common random numbers

Five globally disjoint 56-row panels are frozen from keyed SHA-256
permutations. `optimization_0` through `optimization_2` are crossed
optimization panels. `train_screen_0` and `train_screen_1` are separate,
train-only screens. Each contains exactly 9 safety/consent, 16 technique,
6 equipment/material, and 25 resources/general units. This consumes 280 of
the 310 conflict units without reuse.

For a stratum with `N` units and quota `q`, every unit has exact per-panel
inclusion probability `q/N` because a panel is a fixed block of a uniformly
keyed permutation. The manifest records both this probability and the exact
Horvitz-Thompson unit weight `N/q`. The estimand is explicitly the equal-weight
document-semantic conflict-unit mean, not the original row-weighted mean.

For every optimization panel, every direction and both antithetic signs must
reuse its exact ordered row identity. No dataloader reshuffle, refill, or
direction-specific hard-example choice is allowed. The same direction set is
crossed with all three optimization panels. Direction aggregation and any
consensus rule must be preregistered before responses are observed.

The two screen panels may measure train-panel transfer of a frozen estimator
or candidate, but may not be folded back into the optimization response. A
future runtime adapter must validate each panel's ordered identity immediately
before generation and journal the three optimization responses separately.

## Why hard-example oversampling is disabled

The design caps any future hard allocation at 25 percent, but the frozen
manifest sets it to zero. There is no independently replicated, content-free,
train-only difficulty artifact bound to this frame. Labeling examples as hard
from validation, OOD, heldout, or from these same ES responses would leak the
selection target. It would also make adaptive inclusion probabilities unsafe
to claim as importance weights.

Enabling a nonzero hard tier requires an artifact bound to the frame hash,
out-of-fold or independent-train-panel difficulty estimates, a tier fixed
before candidate responses, and exact per-tier inclusion probabilities and
inverse weights. Until all four conditions hold, uniform stratified conflict-
unit sampling is the auditable experiment.

## Selection firewall and next run

Only the three frozen optimization panels may produce ES coefficients. Only
the two frozen train screens may compare coefficient transfer or train reward.
Validation, OOD, heldout, and benchmark outcomes are forbidden selection
surfaces. No model update is part of the sampler build.

The first launch-shaped use should retain Qwen3.6-35B-A3B, the existing frozen
layer/basis contract, antithetic signs, exact restoration, and four TP=1
engines. It should first run alpha zero and compare:

1. pairwise coefficient cosine across all three optimization panels;
2. sign agreement and robust aggregate coefficient magnitude;
3. transfer of the preregistered aggregate to both untouched train screens;
4. document-stratum weighted and unweighted train responses, both reported.

No GPU launch is authorized by or performed in this V13 sampler artifact.
