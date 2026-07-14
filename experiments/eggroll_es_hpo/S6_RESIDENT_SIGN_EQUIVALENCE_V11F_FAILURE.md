# S6 resident-sign equivalence V11f failure

## Outcome

V11f passed the exact 27-field CLI audit and the full frozen-seed forwarding
audit. It ran the intended V10/V11 perturbation basis (canonical seed-list
SHA-256 `07fa4900cd10fd17b678355389adcfa4f5ac7ec356be46088466cd32b032e6e1`),
completed population scoring, and constructed a coefficient plan. It then
failed closed during unchanged V11c completion validation with:

`RuntimeError: v11c journal identity, policy, or completion changed`

The coefficient plan exists, alpha remained exactly zero, no model update was
applied, baseline validation and OOD were scored, and sealed evaluation data
was not opened or scored.

## Exact cause

The reconstructed V11c journal policy is missing exactly three inherited V11
policy keys:

- `document_lcb_anchor_required: true`
- `ood_validation_heldout_as_objective: false`
- `optimization_data: train_and_anchor_only`

There are no conflicting values for those keys and no other missing, changed,
or unexpected policy keys relative to the exact expected V11c policy union.
The V11c outer wrapper adds V11, V11b, and V11c policy keys, but the inner V4
execution path returns without the three V5 document-LCB policy keys. This is
a retry-adapter policy-forwarding defect, not a model, data, seed, coefficient,
or resident-sign failure. The exact completion validator is correct to reject
the incomplete journal.

## Durable evidence

- V11f committed source: `d448c8a980bc6326a92b9938aaaeafe6114b0bab`
- V11f driver SHA-256:
  `e4d50b4bac41c4c294e999dba61e8ced9aa95e5136f048c3db75350b97f999ff`
- Launch attempt file SHA-256:
  `3b97db50578af3d30364988e16ffdb50e1d54dd81c9084fd072e9c8160d9cd39`
- Launch attempt canonical self SHA-256:
  `1111325aee79004764083e70ff32bb28f1f1f3353b8194a2ef5d99e639c77c36`
- Failed journal file SHA-256:
  `62cea23dbdba65f11d14f98d38d249dcbda3b3984739035990cf10db9e192eb4`
- Persisted anchor-plan file SHA-256:
  `2909b74c6d2bcec1b7db5186b2c4ab151511e34195af3a89b8468f8709c03af2`
- Alpha-zero identity-audit file SHA-256:
  `168ace19f541bac3a19a29757b3cfe198b19207c0852585c761b69f1cfc706a4`

## Required retry

Any retry must use a fresh experiment name and bind this committed document,
the exact V11f attempt, journal, plan, identity audit, and committed source.
It must retain V11f's seed forwarding unchanged. The sole correction is a
scoped inner V4 wrapper that requires exactly the three keys above to be
absent without conflicts, injects the immutable inherited values, and returns
to unchanged V11c completion handling. Both the inner V4 and outer V11c
patches must be restored in `finally`. Existing V11f evidence is immutable.
