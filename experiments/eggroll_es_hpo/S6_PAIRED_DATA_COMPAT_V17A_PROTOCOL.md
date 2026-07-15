# S6 V17A paired production/v283 train-only compatibility protocol

## Scope and attribution

V17A asks exactly one question: with model, layer plan, perturbation basis,
sigma, alpha, scoring, hardware, joint-unit membership, and request order
fixed, is the v283 train-only estimator at least as stable as the matched
production estimator? The only intended difference is the dataset version.

V17A is not optimizer HPO. Sigma is `0.0003`, alpha is zero, no model update
or checkpoint write is reachable, and no evaluation surface is accepted. A
pass authorizes only a separately committed V17B train-only preregistration.
It does not promote v283. Failure retains production and the V13 recipe.

V17B, if later preregistered, uses the full v283 frame and its independent
5x39 geometry (14 safety/consent, 8 technique, 3 equipment/material, and 14
resources/general). V17A's common-only paired subset is never substituted for
that final training estimand.

## Frozen inputs and paired joint frame

The complete v283 replacement candidate is frozen at commit `2bf505e`:

- 492-row artifact SHA-256
  `83d14d9d42740c836b49a8ec9e4237766e9d751c827c21d4d2c79500ee4bc3b9`;
- freeze manifest SHA-256
  `014f37177073d5a433b2da2b01298463cc87856f0278a60d66e53a0dce55bbfb`;
- active 784-row production SHA-256
  `62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507`.

`build_eggroll_es_joint_panels_v17a.py` constructs connected components over
the combined candidate and production rows using shared document SHA, every
normalized URL identifier, raw/raw-document/raw-successor lineage family, and
the pinned V13 lexical-semantic rule. The result has 276 joint components:
195 paired, 4 candidate-only, and 77 production-only. Candidate-side dominant
strata among paired units are 70 safety/consent, 41 technique, 13
equipment/material, and 71 resources/general.

Five globally disjoint paired panels contain 38 units each, with exact quotas
14/8/2/14. They consume 190 of 195 paired units and leave five reserves. The
first three panels are optimization panels; the final two are train-only
compatibility screens. Inclusion probability in a stratum and panel is
`quota / paired_stratum_units`; the exact Horvitz-Thompson unit weight is its
inverse. The estimand is the equal-weight joint-conflict-unit mean.

The corrected manifest file SHA-256 is
`bfd53bb2c2148381e0b5b9b24102a67e20ef65f7dabe96314e097ad800ea7ff1`
and its content SHA-256 is
`eaad58e01a429cae00f85af1d057dbde9e72a3a0c196cd90089ad0f1366ca194`.

## Fixed representatives and common random numbers

Direction-index row rotation is forbidden. It would confound an ES seed
coefficient with the row chosen inside a conflict unit.

Each paired unit binds one deterministic document SHA that occurs on both
sides and has a candidate row in the unit's assigned dominant stratum. It then
binds one candidate representative in that stratum and one production
representative from the same document. Those exact, ordered side batches are
reused for all 32 directions and both antithetic signs.

For each resident perturbation, both dataset versions are scored before exact
restoration. Signed waves alternate production-first and candidate-first to
balance order and thermal drift. The same engine, seed, sign, layer weights,
panel-unit order, and sampling contract therefore serve both versions.

## Full-answer dense reward

V17A retains V4's teacher-forced dense-gold objective. The specialist prompt
and complete answer are tokenized together. `prompt_logprobs=1` returns every
gold-token log probability; scoring averages all aligned answer-token
log-probabilities within an example, then aggregates examples with the frozen
unit/stratum weights. `max_tokens=1` is only a dummy generated-token trigger.
It is not an answer-length cap. Detokenization is disabled and EOS is not
scored.

The total prompt-plus-answer cap remains 1024 tokens. Frozen tokenizer audits
found zero boundary mismatches and zero examples above the cap. Production
combined-length p50/p90/p95/p99/max is 54/63/67/86/142; v283 is
67/91/106/131/144. V283 answer-length p50/p90/p95/p99/max is
18/42/53/81/86. V283 is more prefill-heavy but fully covered.

This likelihood objective is retained only for controlled attribution to the
existing ES estimator. V17A makes no claim that likelihood alone prevents
answer-format blur. Any later nonzero update requires separate direct-answer
and external guard stages; V17A cannot open them.

## Frozen model and estimator mechanics

- model: Qwen3.6-35B-A3B;
- layer plan: middle-late layers 20-23, file SHA-256
  `d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df`;
- sigma: `0.0003`;
- alpha: `0.0`;
- population: 32 directions from basis seed 20260714, basis SHA-256
  `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`;
- signs: plus then minus;
- hardware: four TP=1 engines on GPU IDs 0,1,2,3, with no partial wave.

Each version independently forms the V13 robust direction: standardize each
of the three optimization-panel central-response vectors, then take their
coordinatewise median. The screens never enter either direction.

## Preregistered compatibility gate

Six within-version metric families are frozen:

1. optimization-panel pairwise cosine;
2. optimization-panel pairwise sign agreement;
3. robust aggregate to optimization-panel cosine;
4. robust aggregate to optimization-panel sign agreement;
5. robust aggregate to train-screen cosine;
6. robust aggregate to train-screen sign agreement.

Both the median and worst value of every family are endpoints, for 12 total.
All panel response spreads must be nonzero. Candidate must be no lower than
matched production on every observed endpoint; there is no tolerance.

A deterministic 20,000-replicate paired bootstrap (seed 20260719) resamples
joint conflict units within each panel stratum, using the same resample on
both versions and preserving each panel's 14/8/2/14 stratum counts. Every
replicate recomputes Horvitz-Thompson panel scores, all 32 coefficients, and
then every nonlinear median/worst endpoint. The five reserve units remain
unused. Per-unit scores and bootstrap replicates are never persisted. For
every endpoint, the one-sided familywise lower bound of
candidate minus production uses Bonferroni alpha `0.05/12` and must be at
least zero. All 12 observed comparisons and all 12 bounds are conjunctive.

Cross-dataset coefficient cosine is recorded only as a diagnostic. Improved
data may legitimately change the direction. Base mean reward is not compared
because the gold answers differ.

## Power and interpretation

The design uses 190/195 (97.44%) of the paired frame. A 38-unit panel has an
i.i.d. standard-error inflation of `sqrt(56/38) = 1.21395...` relative to a
V13 56-unit panel, but pairing the exact joint units, perturbations, and order
reduces variance of the dataset-version delta. The zero-margin, familywise
gate is deliberately conservative. Equipment has only two units per panel,
so no stratum-specific superiority claim is authorized.

Expected wall time is 15-30 minutes with all four GPUs, or approximately 1-2
aggregate GPU-hours. Longer v283 prompts increase prefill cost. This estimate
does not authorize launch.

## Fail-closed runtime requirements

The preregistration commit does not implement or authorize a runtime. A later
runtime commit must bind the preregistration file/content hashes and committed
source bundle, claim fresh O_EXCL attempt/run paths, reject every non-train
surface and every nonzero alpha/update/checkpoint entrypoint, and prove:

- exact fixed ordered batch identity before every generation;
- both versions completed on the same resident perturbation before restore;
- complete alternating arm-order balance;
- four-engine coverage for all 16 signed waves;
- exact reference restoration after every sign;
- identical pre/post base probes for both versions;
- passed population-boundary and unselected-origin audits;
- exact prompt-token IDs, answer boundaries, and prompt-logprob coverage;
- compact aggregate persistence only, with no response vector or row content.

The compact summary is also fail-closed: every persisted field whose name
contains `sha256` must be exactly 64 lowercase hexadecimal characters. The
cross-dataset diagnostic must contain exactly `used_for_gate` and
`content_sha256`; no raw value, vector, metadata, or extra key may be hidden in
that non-gating object.

Any mismatch, partial result, failed bootstrap endpoint, or cleanup failure
retains production and V13. No fallback sigma, alpha, panel geometry, dataset,
or endpoint may be attempted.
