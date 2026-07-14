# S6 corrected anchored-v2 rescue pilot

This is the predeclared full-model rescue family after the items16/cosine-0.25
pilot produced no selectable nonzero state. It used the same frozen S6 train,
validation, OOD-QA, and OOD-prose artifacts; population 16; 32 prose-anchor
items; minimum anchor cosine 0.5; sigma 0.0003; and seed 42. Exact CPU-reference
restoration and the four-engine alpha-zero identity audit remained mandatory.

All three nonzero states failed. Validation deltas at alphas 1.5625e-6,
3.125e-6, and 6.25e-6 were respectively -0.0230000, -0.0243902, and
-0.00360976. OOD-QA remained exactly flat, but OOD-prose point deltas were
-0.00168943, -0.00183529, and -0.00340836; every lower confidence bound was
negative. The largest alpha's paired-document 95% interval was entirely below
zero: [-0.00620986, -0.000315319]. Baseline therefore remains selected and no
checkpoint was saved.

These are path-dependent monotonic BF16 pilot states, not direct-alpha
confirmations. The fail-closed aggregation in `summary.json` marks selection
as forbidden. The source journal is retained under the ignored-run path via an
explicit tracked exception, and the GPU trace is retained for utilization
audit. The sealed heldout split was never loaded or scored.

Source identities:

- journal SHA-256: `5343e42cb8bb00ac1bb8921593624011dfe2d43576b329dcaebe065e9e4259b5`
- GPU trace SHA-256: `c1179d78edc95aabf0893de0ecdaa67cfa9b0efc9b83d9081d05efdfd5d90e5e`
- fail-closed summary SHA-256: `cf3fed60ba395c692187a458c0e8b22b0cee03bd8919ac0352e192c3cf3c8cb3`
