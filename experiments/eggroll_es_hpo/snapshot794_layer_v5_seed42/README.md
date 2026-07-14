# S6 document-LCB layer-partition v5 seed-42 results

This package closes the frozen v5 middle-versus-front/back experiment. It
records only aggregate metrics, immutable identities, journal hashes, and GPU
trace hashes. No per-example evaluation output was inspected or copied here.

Both population-16 pilots reproduced the alpha-zero baselines exactly:
validation `0.08381010452961674`, OOD QA `0.714128787878788`, and OOD prose
`-1.2632580042542214`. OOD QA was unchanged at every tested alpha.

No nonzero middle state improved validation. Front/back improved validation by
only `0.00010180275715800102` at alpha `0.00000625`, while OOD prose declined
by `-0.0018140599552813885` with paired 95% CI
`[-0.004557531586730917, 0.0014154743361985847]`. Therefore no nonzero state
passed every strict gate, alpha zero won mechanically, direct confirmation was
not run, and heldout remained unopened and unscored.

The machine-readable record is [summary.json](summary.json). The governing
protocol is [S6_LAYER_PARTITION_V5_DOCUMENT_LCB_PROTOCOL.md](../S6_LAYER_PARTITION_V5_DOCUMENT_LCB_PROTOCOL.md),
SHA-256 `acdad6bbe4d1f993582002c2114cdc6dd3c6b11966ea89753a773b7a36aade7b`.

