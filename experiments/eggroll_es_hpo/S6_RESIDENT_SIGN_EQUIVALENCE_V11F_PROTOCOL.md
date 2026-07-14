# S6 V11f frozen-seed forwarding retry

V11f is a fresh retry of V11e under experiment name
`snapshot794_layer_v11f_middle_late_resident_sign_exact_v10_seed_forwarded_d43d44_a43a44_basis20260714`.
It does not modify or reuse the V11e run directory.

## Bound failure and sole correction

V11f binds the committed V11e failure document at commit
`c21255925aaa4a8367a092c153a386cf20e26209` and SHA-256
`c7c9afe977c4ada9c19c2e3eea1bf8e4d9039f1b5f720f0bc38507a1b6403e55`,
the V11e attempt (`4297a4dd0ca47aa61c0359b6aa06a281ff389821f79d42d92949d945d5e39ac3`),
failed journal (`106168485a7cdc82726845f2f97384717a51daede75271ab9855006d01f716a6`),
persisted plan (`50afc13272e5f5cdd4aca51a37aa499c37bdef59ce6c3b4d2062798b5ad3ba62`),
and alpha-zero identity audit
(`37ad7696840f00472c6b03a734b7ab084b9f5881a92175084fcae49d0f9c9ad9`).

V11e received the exact inherited RNG(seed=43) vector, canonical SHA-256
`78c046a76d8f31123ec42d189cf134b4424949f720173c7873417924c3401a89`,
but V11b/V11c bypassed V11's seed substitution. V11f requires that exact
incoming vector, substitutes the frozen V10/V11 basis with canonical SHA-256
`07fa4900cd10fd17b678355389adcfa4f5ac7ec356be46088466cd32b032e6e1`,
and delegates all other arguments to the unchanged V11c execute wrapper. The
patch is scoped and restores V11c's original function in a `finally` block.
Any other incoming vector is terminal.

Preflight and durable evidence also freshly rehash the unchanged V11c driver,
trainer, and worker against the exact V11e-pinned implementation bundle.

The intended recipe and data remain unchanged, but the effective perturbation
directions are corrected relative to the accidental V11e execution. The
validator is not weakened, old scores are not relabelled, and V11e population
results are not reused.

## Evidence and safety

The dry artifact contains both the 27-field effective CLI audit and the full
32-entry incoming/forwarded seed audit. A real launch requires exact frozen
CLI bytes, committed V11f source at HEAD, a fresh run directory, and an
`O_EXCL` sibling launch-attempt file. Escaping failures record a full traceback
and are re-raised. Alpha remains exactly zero.

V11e did score its baseline validation and OOD surfaces before the coefficient
plan failure; V11f evidence records that fact separately from the facts that
no coefficient plan escaped, no model update occurred, and sealed evaluation
data was not opened or scored. Baseline validation/OOD is not described as
sealed data. No GPU launch or sealed-data read is part of V11f implementation
or testing.
