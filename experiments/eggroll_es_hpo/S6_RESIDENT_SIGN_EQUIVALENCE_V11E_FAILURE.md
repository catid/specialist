# S6 resident-sign equivalence V11e failure

## Outcome

V11e passed the corrected 27-field downstream-CLI audit, started exactly four
TP=1 Qwen3.6-35B-A3B engines, reproduced the frozen baseline validation and OOD
scores, and completed the expensive resident population generation.  It then
failed closed while constructing the coefficient plan with:

`RuntimeError: v11 resident-sign artifact identity changed`

No coefficient plan escaped, no model update was applied, and sealed data was
not opened or scored.  The failed journal has one baseline state and reports
`coefficient_plan: null` while still in
`estimating_fixed_coefficient_plan`.

## Exact cause

The persisted V11e anchor plan contains the driver-v1 seed-43 population
schedule:

`[542587408, 700401000, 430899501, 47003395, ...]`

Its canonical seed-list SHA-256 is
`78c046a76d8f31123ec42d189cf134b4424949f720173c7873417924c3401a89`.
The frozen V10/V11 perturbation basis instead starts:

`[140002291, 1028842752, 480373990, 1037026679, ...]`

with canonical seed-list SHA-256
`07fa4900cd10fd17b678355389adcfa4f5ac7ec356be46088466cd32b032e6e1`.

`run_eggroll_es_anchor_equivalence_v11.py:439-448` validates the inherited
seed-43 schedule and replaces it with the frozen V11 basis before delegating.
The V11b and V11c retry adapters instead delegate directly to driver v4;
`run_eggroll_es_anchor_equivalence_v11c.py:389-392` therefore bypasses that
substitution.  `_build_resident_artifact_v11` correctly records the effective
runtime seeds, and `train_eggroll_es_specialist_anchor_v11.py:153` correctly
rejects those seeds because they are not `PERTURBATION_SEEDS_V11`.

This was a retry-adapter forwarding defect, not stochastic model behavior and
not evidence against resident-sign equivalence or document-first sampling.

## Durable evidence

- Launch attempt file SHA-256:
  `4297a4dd0ca47aa61c0359b6aa06a281ff389821f79d42d92949d945d5e39ac3`
- Launch attempt content SHA-256:
  `314f3c1bd4cbe4d9614022e5b6570f963f88d80a4871d1a53b6d4b5523aa81a4`
- Failed journal file SHA-256:
  `106168485a7cdc82726845f2f97384717a51daede75271ab9855006d01f716a6`
- Persisted anchor-plan file SHA-256:
  `50afc13272e5f5cdd4aca51a37aa499c37bdef59ce6c3b4d2062798b5ad3ba62`
- Alpha-zero identity-audit file SHA-256:
  `37ad7696840f00472c6b03a734b7ab084b9f5881a92175084fcae49d0f9c9ad9`

## Required retry

Any retry must use a new experiment name and bind the exact V11e attempt,
journal, anchor-plan, and failure document.  It must minimally restore the V11
seed-forwarding wrapper before any engine work and test both sides of the
boundary: the inherited RNG schedule must be the exact expected seed-43 list,
and the delegated driver-v4 call must receive the frozen V11 perturbation
basis.  All other recipe, data, alpha-zero, four-engine, and sealed-holdout
guards remain unchanged.
