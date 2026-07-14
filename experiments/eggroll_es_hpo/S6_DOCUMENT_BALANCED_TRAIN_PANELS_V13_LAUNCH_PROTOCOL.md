# S6 V13 five-panel alpha-zero launch contract

This adapter is the first launch-shaped implementation of the committed V13
document-balanced panel design. It is a train-only diagnostic and makes no
promotion decision. It applies no model update.

## Frozen runtime

- Model: the local Qwen3.6-35B-A3B snapshot.
- Layer plan: the exact middle-late dense plan used by V10/V11.
- Perturbations: the exact 32-seed basis identified by seed `20260714` and
  SHA-256
  `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`.
- Signs: plus then minus for every direction, with exact restoration after
  each sign.
- Hardware: four TP=1 engines on GPU IDs 0,1,2,3. Every population wave has
  exactly four directions; partial waves are forbidden.
- Alpha: exactly zero. The V13 worker refuses every update preparation,
  execution, commit, and legacy update RPC.

The runtime inherits the repaired V11c trainer and worker surface, while its
entrypoint directly supplies the frozen V11/V11g perturbation seeds. This
avoids the earlier raw-versus-templated manifest and seed-forwarding failures.
V13 does not invoke the V11 line-search driver.

## Exact train panels

The runtime accepts only the committed V13 manifest and exact 794-row train
source. It independently validates the source JSONL hash, frozen Arrow hash,
manifest file hash, manifest content hash, all five ordered panel identities,
and the materialized prompt-answer identities before generation.

Every direction and both signs see one combined request in the fixed order
`optimization_0`, `optimization_1`, `optimization_2`, `train_screen_0`, then
`train_screen_1`. Each constituent panel retains its exact internal order.
Generation is greedy and uses one fixed seed, so the panel is a common random
number across directions and signs.

No command-line option or data path exists for any non-train scoring surface.
The driver rejects such tokens before parsing. Its trainer is constructed with
an empty evaluation-loader mapping.

## Preregistered response analysis

For each panel, direction, and sign, the runtime reports both:

1. the ordinary mean selected-answer token log probability across 56 units;
2. the Horvitz-Thompson equal-conflict-unit mean, using the exact stratum
   weights from the manifest, whose total is all 310 conflict units.

Stratum means and weighted stratum contributions are also retained. Only the
weighted response defines coefficient diagnostics. The central response is
`(plus - minus) / 2` and is independently standardized within each panel using
the existing epsilon `1e-8` convention.

The robust optimization aggregate is fixed before responses are observed:
take the coordinate-wise median of the three independently standardized
optimization-panel coefficient vectors. It is not renormalized. Report its L2
magnitude, all optimization-panel pairwise cosines and sign agreements, and
its cosine/sign agreement with each train screen's independently standardized
coefficient vector. The two screen vectors are expressly excluded from the
aggregate. Undefined zero-vector cosines are recorded as null rather than
silently promoted or rejected.

No numeric threshold is a V13 promotion rule. Any later sampling decision must
be specified separately after this immutable diagnostic is complete.

## Durability and launch gate

A real launch requires the exact current implementation-bundle hash and every
implementation file committed at the recorded Git HEAD. The driver exclusively
creates a sibling launch-attempt artifact before trainer construction, requires
a fresh run directory, records full failure telemetry, closes trainer resources
before writing the final report, and binds the completed report by file and
content hashes. Resume and overwrite are forbidden.

Dry-run mode validates the complete recipe, train Arrow, source, manifest,
panels, and implementation identities without constructing a trainer or
launching a GPU actor.
