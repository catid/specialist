# S6 guarded candidate

This immutable candidate incorporates the manually reviewed pending data work
without changing the canonical S5 files while S5 experiments were in flight.
Relative to S5 it adds 38 source-backed safety, care, consent, and practical
QAs; applies 27 merit drops; excludes one otherwise useful Rope365 addition
whose document is assigned to validation; and applies four reviewed edits.
The net result is 794 unique rows (S5: 784).

The unguarded 795-row projection collided with the frozen validation document
`web://rope365.com/storing-rope`. The tracked S6 disjointness guard removes the
one colliding training row. A deterministic eval-v3 rebuild then passed every
identity check and reproduced the frozen domain, OOD-QA, and OOD-prose bytes
exactly. No heldout question or answer was manually opened, and the heldout
split was not scored.

## Frozen identity

- Training JSONL: `f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776`
- Training report: `f0e36cefe360cabd82527b054ed433c43fd0817c0d75166a3e841b3581e40fc2`
- Eval-v3 rebuild report: `2b011982350bcedf6e6e247ecffa0caa990c66d380d3aa896e84372b9a8929dd`
- Arrow manifest: `a48fba83ba41f3495f1c397f05576e66852d6ec7ef44edba725ed38753510435`
- Training Arrow: `6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6`
- Validation Arrow: `19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db`
- OOD-QA Arrow: `b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced`
- Sealed heldout Arrow identity: `df23a704d0f621bffd8b55fb4a0a296e06a79feaf79cfe1bd357d55bb4f07cf1`

The guarded JSONL was rebuilt twice byte-identically. The frozen eval content
hashes remain `ab9a391e...` (domain), `25a48b94...` (OOD QA), and
`3299457c...` (OOD prose). The first S6 alpha-zero evaluation reproduced the
same model baseline as S5: validation `0.08381010452961674`, OOD QA
`0.714128787878788`, and OOD prose `-1.2632580042542214`. It occurred before
population evaluation and is valid; the subsequent v1 coefficient diagnostic
is non-selectable because the v1 perturbation path is not bit-exact.

## Status

This is a segregated experiment snapshot, not yet a replacement of
`data/train_qa_curated_v1.jsonl`. It is safe for a new experiment family, but
must not be mixed with S5 treatments in one A/B comparison. Selection still
requires strict validation improvement, nondegradation on OOD QA, and a
nonnegative OOD-prose point delta and paired-bootstrap lower bound.
