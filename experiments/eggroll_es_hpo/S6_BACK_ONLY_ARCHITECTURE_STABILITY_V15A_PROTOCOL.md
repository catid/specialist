# S6 V15A paired back-plan architecture stability

## Question and scope

V15A tests one train-only architecture hypothesis: with selected capacity held
exactly fixed, does moving the dense EGGROLL-ES partition from the retained
middle-late layers 20--23 to back layers 36--39 produce a materially more
stable train direction estimator?

This is an alpha-zero diagnostic. It cannot apply an update, open validation,
OOD, heldout, or benchmark surfaces, or select a releasable model. It does not
insert or duplicate layers. Front-plus-back is not an adaptive fallback; that
would be a separate hypothesis requiring a separate preregistration.

## Why the middle-late arm is rerun

The completed V13 aggregate is the absolute floor, but it used perturbation
basis `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`.
Comparing a new-basis back run only with that historical result would mix the
layer-location effect with the basis draw. V15A therefore runs both the
middle-late control and back candidate on one newly frozen basis. The back arm
must beat both the historical V13 floor and this contemporaneous control.

The arm order is middle-late then back. Both arms use the identical direction
seeds, panels, generation order, scoring objective, and numeric analysis. The
only intended difference is the selected dense-layer location.

## Frozen architecture and capacity

The control is plan
`03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9`
at layers 20--23. The candidate is plan
`6da92a4db760676acda1bcbcaec4a925a6dd7b641c250a58a3fe4837d97ac93a`
at layers 36--39. Each plan selects exactly 35 checkpoint units, 23 runtime
parameters, 142,999,552 elements, and 285,999,104 bytes. The model-config hash
is `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.

## Exact V13 estimator

V15A reuses the exact 794-row, 310-document V13 train source and five globally
disjoint 56-row panels. The three optimization panels and two untouched train
screens retain their committed order, Horvitz--Thompson document-semantic
conflict-unit weights, and common-random-number generation. Each direction and
sign generates the same 280 prompts per arm.

For each panel, the central response is `(plus-minus)/2`, followed by the V13
standardization with epsilon `1e-8`. The robust optimization aggregate remains
the coordinatewise median of the three independently standardized optimization
vectors. No train screen enters that aggregate.

Only V13-native endpoints are admissible:

- optimization-panel pairwise cosine and sign agreement (three pairs);
- robust aggregate-to-optimization cosine and sign agreement (three panels);
- robust aggregate-to-train-screen cosine and sign agreement (two screens).

The last family remains V13 train-screen transfer, not a disjoint crossfit.
The middle family remains robust aggregate-to-optimization, not full-frame to
optimization. V15A does not manufacture V14 aliases or change panel geometry.

## Fresh common basis and runtime

Basis seed `20260715` uses the exact earlier direction-seed algorithm:
`numpy.default_rng(seed).integers(0, 2**30, size=32, dtype=int64)`. Its frozen
basis hash is
`6c358060c5f9a0a7b00e953bd230b18f915950f0233f38321e0e048a67ea05e7`.
The same ordered seeds are used by both arms.

The future runtime must use Qwen3.6-35B-A3B, population 32, alpha zero, greedy
generation seed 43, plus then minus, and four TP=1 engines on GPU IDs 0--3.
Every signed wave contains exactly four directions and all four engines must
participate. Each arm has eight population waves and sixteen signed waves.
Partial waves are forbidden.

Every sign is restored to an exact reference in a `finally` path. Pre- and
post-population base probes, a per-arm boundary audit, dense result hashes, a
fresh exclusive attempt/run, committed source bundle, and report self-hashes
are mandatory. Response vectors and row content are not persisted.

## Historical evidence and numeric gate

The gate binds completed V13 aggregate evidence and the compact negative V14a
and V14b evidence. Those two sampler changes failed and retain V13; neither
authorizes a confirmation, evaluation, or update.

The historical V13 native baselines are:

| Endpoint | Median | Worst |
| --- | ---: | ---: |
| optimization pairwise cosine | 0.47411088498906484 | 0.3900621868364503 |
| optimization pairwise sign | 0.59375 | 0.5625 |
| aggregate-to-optimization cosine | 0.7608236805612648 | 0.7082628389768383 |
| aggregate-to-optimization sign | 0.8125 | 0.75 |
| train-screen cosine | 0.3936314430866483 | 0.314941371734614 |
| train-screen sign | 0.65625 | 0.53125 |

The promotion screen is a strict conjunction:

1. For every cosine family, both the back median and worst must be at least
   `0.05` above the corresponding historical V13 value.
2. For every cosine family, both the back median and worst must also be at
   least `0.05` above the fresh-basis middle-late control.
3. For every sign family, both back median and worst must be no lower than the
   corresponding historical V13 value and no lower than the contemporaneous
   control.
4. Every panel spread in both arms must be nonzero, both robust aggregates
   must have 32 nonzero coordinates and positive finite norm, and every
   integrity audit must pass.

Thus all twelve median/worst endpoint summaries have nonregression protection,
while all six cosine median/worst comparisons require a fixed material margin
against two controls. This is a conservative effect-size screen, not a claim
of a calibrated statistical test. The single hypothesis and conjunctive rules
prevent choosing a favorable endpoint after the run. A second fresh-basis
confirmation remains mandatory even if V15A passes.

Failure retains V13 middle-late and keeps every evaluation surface closed.
Passing authorizes only a separately preregistered alpha-zero back-plan
confirmation on another fresh 32-direction basis. Passing does not authorize
evaluation or a model update.

## Launch status

This commit preregisters the design only. The V15A trainer, runner, and runtime
tests do not yet exist, so no GPU launch is authorized by this artifact.
