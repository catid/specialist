# V35A lagged-replay calibration protocol

V35A is a train-only calibration stage. It is not a replay comparison, model
update, checkpoint, dataset promotion, or evaluation authorization.

The exact V13 production optimization panels are scored once with the frozen
unperturbed Qwen3.6-35B-A3B model. All four TP=1 engines generate the identical
168-request union, and token/log-probability results must be exactly equal
across engines. Source/container bytes may be read solely to bind and project
the exact optimization indices. Screen rows are never materialized as
requests, interpreted, generated, scored, ranked, selected, or used.

Within each optimization panel and stratum, rows are ranked from lowest to
highest mean gold-answer token log-probability, with row SHA-256 as the frozen
tie-breaker. The lowest 50% form a provisional manual-review pool. Reviewers
receive a deterministic shuffle without scores, ranks, or future HPO results.
A row is eligible only when its question is useful and clear, its answer is
source-supported and direct, it contains no protocol/control leakage, and it
does not omit material safety context. Review cannot rewrite or substitute a
row inside this experiment.

After audit, the lowest-ranked eligible 25% per panel/stratum form the hard
tier. Insufficient eligible rows fail the calibration and authorize nothing.
Runtime artifacts contain hashes, ranks, and eligibility enums only—never row
content, raw scores, tokens, log-probabilities, or outputs.

Only a later separately committed preregistration may compare 0%, 10%, and 20%
lagged replay on a fresh perturbation basis. That comparison must preserve V13
stratum mass, use identical requests and perturbations across fractions, keep
the train screens untouched until the tier is frozen, and use fixed-sequence
familywise noninferiority gates before any update or clean evaluation.
