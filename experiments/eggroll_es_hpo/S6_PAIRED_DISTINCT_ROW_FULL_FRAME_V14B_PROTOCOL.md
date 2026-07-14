# S6 V14b paired-distinct-row estimator preregistration

## Bound result and decision

V14a completed cleanly at alpha zero with exact restoration, four TP=1
engines, all eight response spreads nonzero, and no model update or evaluation
surface. It nevertheless failed its preregistered conjunctive V13-baseline
gate. Matched-56 pairwise cosine was `0.4427854137278768` / 
`0.26764423806475524` median/worst, and full-to-optimization cosine was
`0.7132916285087167` / `0.6803342639761624`. Full-to-optimization sign
agreement also missed its median threshold at `0.75`. Crossfit screen metrics
passed. The decision is therefore to retain V13, not run the V14a row-draw-1
confirmation, and keep model-update and evaluation surfaces closed.

The compact V14a negative evidence has file/content SHA-256
`9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9` /
`ee4ded3d974dfd0becaedb1007f96888e133db51e62130d2844ab9c25e2ccf2b`.
It contains only hashes, aggregate metrics, decisions, and integrity booleans;
it contains no response vectors, dense-result hashes, source rows, or
evaluation content.

## Single k=2 hypothesis

V14b tests one hypothesis: reduce within-document row-choice variance before
forming the equal-document estimator. The frozen snapshot has 310 documents:
139 have one row and 171 have multiple rows. Every singleton contributes its
sole row; every multirow document contributes exactly two distinct rows. This
produces 481 unique train prompts per direction and sign.

Rows are selected without replacement. For each document, sort all its rows by
`SHA256(master_seed, within-document-without-replacement, document_sha256,
row_sha256)` and take the first `min(2,row_count)`. Sort documents by
`SHA256(master_seed, full-frame-document-order, document_sha256)`, then emit
each selected row in selection-rank order. The master seed is
`specialist-s6-paired-distinct-row-v14b-20260714`.

The full-frame ordered document identity is
`3574c320bee17d7df2f31b2cc35d0ee018e903702c1dc1c3dda2778c77e05c0f`.
The exact 481-prompt identity/order is
`b3e7c0eb24a04377fc7727cb1972fefdca016dd112c0076068f2527677866ba9`.
The full-frame content identity is
`9d9fc31e928948cae12d7dc4b5ffedfd9def8482a4e6af8e82dc7dcfce7cb3d4`.

For each direction/sign, score all 481 prompts, average the one or two row
rewards within each document, and only then aggregate documents. The primary
response is the arithmetic mean of 310 document means. Row rewards must never
be pooled directly across documents.

## Matched allocations and crossfits

V14b reuses the exact five globally disjoint V14/V14a matched-56 document
allocations. Their prompt counts are 92, 81, 87, 88, and 86. The three
optimization responses are stratum-weighted equal-document Horvitz-Thompson
means of document means. The two train-screen responses use the same rule.

Each screen is compared with an independently standardized arithmetic mean of
document means from its exact disjoint 254-document complement. The complement
prompt counts are 393 and 395. Every matched and complement response is derived
from the same 481 generated outputs, with no extra generation. The machine
preregistration freezes every ordered document, prompt, panel, and complement
identity.

## Frozen runtime and integrity

The future runtime must use Qwen3.6-35B-A3B, the exact middle-late layers
20--23 control, the existing 32-direction basis
`29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`,
sigma `0.0003`, plus then minus, alpha zero, and four TP=1 engines on GPU IDs
0,1,2,3. Eight complete four-direction waves are required.

For each wave, all four engines perturb and generate the plus sign, then every
engine restores exactly in `finally`; all four then perturb and generate the
minus sign and restore exactly. A pre-population base probe must equal the
post-population probe. Exact-reference and population-boundary audits must
pass. Partial waves and update RPCs are forbidden.

The driver must use fresh exclusive attempt and run paths, bind a committed
implementation bundle, reject validation/OOD/heldout/benchmark tokens before
argument parsing, and durably record cleanup and failure. No per-document
reward vector may be persisted. The diagnostic may retain only the aggregate
signed vectors and dense-result hashes required for recomputation.

## V13-baseline conjunctive gate

The gate uses the completed V13b aggregate baseline, whose compact evidence
file/content SHA-256 is
`d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54` /
`06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3`.
All rules are conjunctive:

1. matched-56 optimization pairwise cosine median/worst must strictly exceed
   `0.47411088498906484` / `0.3900621868364503`;
2. matched-56 sign median/worst must be at least `0.59375` / `0.5625`;
3. full-to-matched-optimization cosine median/worst must be at least
   `0.7608236805612648` / `0.7082628389768383`;
4. corresponding sign median/worst must be at least `0.8125` / `0.75`;
5. complement-to-screen cosine median/worst must strictly exceed
   `0.3936314430866483` / `0.314941371734614`;
6. corresponding sign median/worst must be at least `0.65625` / `0.53125`;
7. all eight spreads must be nonzero, the full-frame coefficient must have 32
   finite nonzero coordinates, and every provenance/restoration audit must
   pass.

A failure retains V13 and keeps all evaluation and update surfaces closed. A
pass authorizes only a separately preregistered k=2 alpha-zero confirmation on
a fresh 32-direction basis. It does not authorize evaluation, architecture
HPO, layer insertion, or a model update.

## Launch firewall

This preregistration does not authorize a GPU launch. Before launch, add,
review, test, commit, and hash exactly the V14b trainer, fail-closed driver, and
focused runtime tests named in the machine preregistration. Until that adapter
exists, there is deliberately no valid real-run command.
