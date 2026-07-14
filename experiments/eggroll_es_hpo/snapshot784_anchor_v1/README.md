# Final-784 prose-anchored ES family

This family adds a train-only general-prose anchor to the ES coefficient
estimate. Its zero-step baseline is exactly equivalent to the frozen legacy
baseline on all 41 validation and 24 OOD-QA items and on the 16-document OOD
prose score. The baseline therefore remains selected: no completed nonzero
anchored state passes the strict no-degradation policy.

## Validity warning: nonzero v1 results are diagnostic only

**Every nonzero anchored-v1 result below is non-selectable, irrespective of
its measured validation or OOD score.** A post-run implementation audit found
two confounds that prevent causal comparison with the unperturbed baseline:

1. Population exploration perturbs and restores model parameters with two
   separate in-place native-precision additions. In BF16, adding noise and
   then adding its negation need not restore the original bits, so successive
   population members and the eventual update can inherit rounding residue.
   The frozen upstream worker implements the forward add at
   `es-at-scale/es_at_scale/utils/worker_extension.py:139-155` and the inverse
   add at `:48-54` (SHA-256
   `b5e99e4f050f6882529eae98d3b1cd9cb54894362fefcb2b7b802ce3f75a933a`).
2. The anchor adapter submits domain and anchor requests together in one
   generation call. Adding the anchor requests therefore changes scheduler
   batching for the domain requests, confounding the domain reward estimate
   with the anchor batch size. The frozen implementation is
   `train_eggroll_es_specialist_anchor.py:250-313`, especially the combined
   prompt construction at `:291-303` (SHA-256
   `e429e2150859bb308f45efa6b6147edff5bd4207e45b902bffb03f58c827ce2a`).

The alpha-zero baselines remain valid: they are evaluated and recorded before
population estimation begins at `run_eggroll_es_anchor_line_search.py:350-383`
(frozen driver SHA-256
`0705184534949034e169f0d22d28b27008636719dfbc1590d38f876b0cb662cb`).
Their exact agreement with the legacy baseline is therefore unaffected. The
nonzero rows remain useful only as diagnostics motivating a corrected
experiment family; they must not support model or hyperparameter selection.

| Run | Steps | Anchors | Cosine floor | Validation | OOD QA | OOD prose delta (95% CI) | Selectable |
|---|---:|---:|---:|---:|---:|---:|:---:|
| selected baseline | 0 | — | — | 0.08381010 | 0.71412879 | 0 `[0, 0]` | yes |
| anchored cos 0.1 | 1 | 2×512 | 0.1 | 0.05952166 | 0.71537879 | -0.002135 `[-0.005261, 0.001216]` | no—diagnostic |
| anchored cos 0.25 | 1 | 2×512 | 0.25 | 0.05683560 | 0.71412879 | -0.004126 `[-0.006012, -0.002049]` | no—diagnostic |
| resident alpha 0.0000125 | 1 | 8×512 | 0.1 | 0.08446864 | 0.71412879 | -0.001937 `[-0.004157, 0.000462]` | no—diagnostic |
| resident alpha 0.000025 | 1 | 8×512 | 0.1 | 0.08391191 | 0.71412879 | -0.002040 `[-0.003767, -0.000008]` | no—diagnostic |
| resident alpha 0.00005 | 1 | 8×512 | 0.1 | 0.08293902 | 0.71412879 | -0.003119 `[-0.005678, -0.000407]` | no—diagnostic |
| resident-16 alpha 0.00000625 | 1 | 16×512 | 0.25 | 0.08164472 | 0.71412879 | -0.003084 `[-0.005676, -0.000319]` | no—diagnostic |
| resident-16 alpha 0.0000125 | 1 | 16×512 | 0.25 | 0.08488752 | 0.71412879 | -0.001982 `[-0.004303, 0.000488]` | no—diagnostic |
| resident-16 alpha 0.000025 | 1 | 16×512 | 0.25 | 0.08464361 | 0.71537879 | -0.002044 `[-0.004442, 0.000501]` | no—diagnostic |
| resident-32 alpha 0.000003125 | 1 | 32×512 | 0.5 | 0.08202962 | 0.71412879 | -0.002823 `[-0.005379, -0.000270]` | no—diagnostic |
| resident-32 alpha 0.00000625 | 1 | 32×512 | 0.5 | 0.05952166 | 0.71412879 | -0.003029 `[-0.005273, -0.000555]` | no—diagnostic |
| resident-32 alpha 0.0000125 | 1 | 32×512 | 0.5 | 0.08381010 | 0.71412879 | -0.000474 `[-0.002461, 0.001585]` | no—diagnostic |
| resident-p16-s43 alpha 0.000003125 | 1 | 32×512 | 0.5 | 0.08544425 | 0.71412879 | -0.001288 `[-0.003705, 0.001232]` | no—diagnostic |
| resident-p16-s43 alpha 0.00000625 | 1 | 32×512 | 0.5 | 0.08281435 | 0.71412879 | -0.000925 `[-0.002189, 0.000398]` | no—diagnostic |
| resident-p16-s43 alpha 0.0000125 | 1 | 32×512 | 0.5 | 0.08206171 | 0.71412879 | -0.001931 `[-0.004226, 0.000420]` | no—diagnostic |

The following paragraphs describe the recorded diagnostics, not trustworthy
treatment effects; corrected experiments are required before using any trend.

The 0.1 run projected the domain/anchor coefficient cosine from -0.18894 to
0.1 (`update_norm_ratio=0.98694`). Raising the floor to 0.25 increased the
projection but worsened both validation and prose. Both runs used the same
seed plan and the same two selected anchor documents.

The resident search broadened the estimate to eight documents and reused one
fixed population across four alpha states. Its measured cosine was +0.22247,
so the domain direction was accepted without projection. Alpha 0.0000125
improved validation by 0.79%, but its prose point estimate was negative and
therefore ineligible.

The 16-document estimate required projection from cosine +0.05574 to 0.25.
Its best validation state, alpha 0.0000125, improved validation by 1.29%, but
again had a negative prose point estimate. The 32-document estimate pointed
strongly against the domain direction (cosine -0.43434) and required a large
projection to 0.5. None of its nonzero states improved validation; the largest
alpha reproduced the baseline validation answers exactly but still degraded
the prose point estimate. Within the confounded diagnostics, increasing anchor
count or the cosine floor did not rescue the direction.

Doubling the population to 16 and changing to seed 43 made the unprojected
cosine positive (+0.30564), but it still required projection to the 0.5 floor.
Its smallest alpha improved validation by 1.95% (paired CI `[0, 0.004244]`),
yet its prose delta remained negative. The failure therefore persists under a
larger population and a different seed rather than being specific to the
eight-sample estimate in the recorded diagnostics; this needs a corrected
replication.

Eligibility requires validation improvement, nondegradation in OOD-QA mean,
exact, and nonzero counts, a nonnegative OOD-prose point delta, and a
nonnegative lower bound of its paired document-bootstrap 95% CI. Every
treatment is compared independently with the zero-step baseline. No checkpoint
was saved, and the sealed heldout split was neither loaded nor evaluated.

## Frozen identity

- Anchor trainer: `e429e2150859bb308f45efa6b6147edff5bd4207e45b902bffb03f58c827ce2a`
- Base trainer: `bbffbf16747ec514c67e48daab696560eb3309f5a3edf0a700257969cad35c23`
- Coefficient projection: `a5e71ae4eebf36907e9a8da2d91e61fe46d0c581b5ef286a0668124feb02af52`
- Anchor builder: `5df8f4441f2f719c676560be857b2f90e5cb918a31d4e197040c47157e4b36f8`
- Anchor JSONL: `a693e23c48e558e9b72c30b0ae31f0b3e580a665371846978ad4d3eca7ef5f7d` (128 documents)
- Anchor report: `913ff2cb786ac50ffe86770291b6173a14220afce3682dfea67359c45cf6e9f5`
- Final-784 manifest: `7d56567a2116ad11814d8bc9b62e6b9341593dc3e3f0854511c151449b76b056`
- OOD prose JSONL: `3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57`

Compact summary hashes are:

- `baseline_summary.json`: `35aa9f1bc89aeb416f3ede7cd42279e462fd53344721194b39a19529fbed3cb5`
- `cos01_summary.json`: `49354627656cb96673a2b56b44881f22bb9f659a368d543a62a4a3d377fec428`
- `cos025_summary.json`: `4dc1e27c1a3e838f0a84278c895640a9514f0c102c957243fddd85b092c45186`
- `resident_items8_summary.json`: `8b2d3737d501516f91b48a96f17996b5c919bf470a99560e700c0e04f09ae1c3`
- `resident_items16_summary.json`: `f8258177f435776b6ab4035d7cad62ef1b6cc0481ec601d002d623088071e1d4`
- `resident_items32_summary.json`: `81277c067fcb20a40ff9573dd6ff632fc342be8d9c6d2dca02d66d362d566f3d`
- `resident_pop16_items32_seed43_summary.json`: `6e52bbfa4731a0103ab84407f497b89fd68f860f21cda03df15e16d1fbb1dd7d`

Each compact summary pins its raw run, prose output, projection plan,
comparison files, and GPU-utilization trace.
