# Final-784 guarded ES snapshot

The selected result is the zero-step Qwen-3.6-35B-A3B baseline. Several ES
updates improved validation, including a one-step +29.16% point, but every
treatment degraded the independent OOD prose point estimate and/or its paired
95% confidence interval. Under the strict no-degradation policy, none is
eligible for selection.

## Selection policy

Every treatment is compared independently with the same zero-step baseline.
A treatment is eligible only if validation improves, OOD QA does not decline,
the OOD prose mean-token-logprob delta is nonnegative, and the lower bound of
its 20,000-sample paired document-bootstrap 95% CI is nonnegative. The final
strict runs set the trainer margin to exactly `0.0`. The earlier three-step
runs used a legacy `0.02` point-degradation margin; they are reported for
completeness but are ineligible under the final policy.

| Run | Steps | Sigma | Alpha | Validation | Change | OOD QA | OOD prose delta (95% CI) | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| selected baseline | 0 | — | — | 0.08381010 | — | 0.71412879 | 0.000000 `[0, 0]` | yes |
| legacy sigma 0.0005 | 3 | 0.0005 | 0.00025 | 0.08343466 | -0.45% | 0.71412879 | -0.004004 `[-0.007470, -0.000239]` | no |
| legacy sigma 0.001 | 3 | 0.001 | 0.00025 | 0.08457317 | +0.91% | 0.71412879 | -0.004475 `[-0.007307, -0.001170]` | no |
| legacy sigma 0.0003 | 3 | 0.0003 | 0.00015 | 0.08520035 | +1.66% | 0.71412879 | -0.003527 `[-0.006124, -0.000398]` | no |
| strict sigma 0.0003 | 1 | 0.0003 | 0.00015 | 0.08381010 | 0.00% | 0.71412879 | -0.002248 `[-0.004431, 0.000247]` | no |
| strict sigma 0.001 | 1 | 0.001 | 0.00025 | 0.06105401 | -27.15% | 0.71412879 | -0.003043 `[-0.005867, 0.000031]` | no |
| strict sigma 0.0005 | 1 | 0.0005 | 0.00005 | 0.08112718 | -3.20% | 0.71537879 | -0.000532 `[-0.003190, 0.002329]` | no |
| strict sigma 0.0003 | 1 | 0.0003 | 0.00005 | 0.10824913 | +29.16% | 0.71412879 | -0.001922 `[-0.004062, 0.000218]` | no |
| strict alpha 0.000025 | 1 | 0.0003 | 0.000025 | 0.08391191 | +0.12% | 0.71537879 | -0.002591 `[-0.004101, -0.001004]` | no |
| strict sigma 0.0001 | 1 | 0.0001 | 0.00005 | 0.08488752 | +1.29% | 0.71412879 | -0.001751 `[-0.003768, 0.000714]` | no |
| strict sigma 0.00005 | 1 | 0.00005 | 0.00005 | 0.08381010 | 0.00% | 0.71412879 | -0.000862 `[-0.003242, 0.001639]` | no |
| strict sigma 0.000025 | 1 | 0.000025 | 0.00005 | 0.08464361 | +0.99% | 0.71412879 | -0.002643 `[-0.004843, -0.000099]` | no |
| negative-alpha sign control | 1 | 0.0003 | -0.00005 | 0.06020035 | -28.17% | 0.71412879 | -0.003356 `[-0.005686, -0.000920]` | no |
| seed-43 exploratory direction | 1 | 0.0003 | 0.00005 | 0.08333424 | -0.57% | 0.71412879 | -0.000517 `[-0.002409, 0.001446]` | no |
| seed-44 exploratory direction | 1 | 0.0003 | 0.00005 | 0.08407143 | +0.31% | 0.71412879 | -0.002734 `[-0.005337, 0.000003]` | no |

No treatment checkpoint was emitted. The 18-row heldout split remains sealed:
it was not inspected, scored, or included in any command in this campaign.

## Frozen inputs and integrity

- Trainer SHA-256: `bbffbf16747ec514c67e48daab696560eb3309f5a3edf0a700257969cad35c23`
- Dataset manifest SHA-256: `7d56567a2116ad11814d8bc9b62e6b9341593dc3e3f0854511c151449b76b056`
- Train JSONL SHA-256: `62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507` (784 rows)
- Train Arrow SHA-256: `c4935458da6e887064ed181e9ec8ee490752cca2b6a1d33ecb7aa58c201c851f`
- Validation Arrow SHA-256: `19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db` (41 rows)
- OOD QA Arrow SHA-256: `b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced` (24 rows)
- OOD prose JSONL SHA-256: `3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57` (16 documents, 10,926 scored tokens)

Each compact summary records the raw run-summary, raw prose result, and GPU
trace hashes. Its paired validation and OOD-QA comparison files preserve all
aligned item-level rewards and bootstrap inputs.

| Compact summary | SHA-256 |
|---|---|
| `baseline_summary.json` | `6a5480d0f331b836d527b770a8c60fe7d0060ac5a349c2c2164959583b1916fb` |
| `sigma0005_summary.json` | `14f3642076267f9b9133d41d09f37ed838108ebf173763a4665273a185390b7f` |
| `sigma001_summary.json` | `0bf47d8b8a13fb93ffa55dc10186f5b57b9631fcdfaac1c6c046a2e86fe7a9e7` |
| `sigma0003_summary.json` | `e1decff15039238f275fb19a19e94907c6d6b4fcc7816a8c7bb09b0e2bcb3849` |
| `strict_sigma0003_steps1_summary.json` | `7852e4e944f17976508faf497a73572a88d97c5693cdc8f61185dba5a9816c89` |
| `strict_sigma001_steps1_summary.json` | `6144366a1c810148bc03392ab6545648162aaf2384d18024e425fa049cd48985` |
| `strict_sigma0005_alpha00005_steps1_summary.json` | `81d9aa83d44d47ff2c9b9df3b5ba7196a83b6afdbffbafc01443d714aefd3862` |
| `strict_sigma0003_alpha00005_steps1_summary.json` | `73f25170088e686cbe1daa831590e73f7c88b209a3d0d1a8761d28283a13ad27` |
| `strict_sigma0003_alpha000025_steps1_summary.json` | `1e295d7f5f6135cf758d6b43169b300fdf8c44f43a5c0b2e7975a95c18ebeb34` |
| `strict_sigma0001_alpha00005_steps1_summary.json` | `1c39ba47260364e0686a57e789f5e928f2ad453ec582c22c037f4d187c21307d` |
| `strict_sigma00005_alpha00005_steps1_summary.json` | `4da4da57a0af0b47b3b53de77ce2b8b4b4cf9fb2a3555d8ead347cbd715635b6` |
| `strict_sigma000025_alpha00005_steps1_summary.json` | `30e4a908356fecbe79aa8619797d2b900e40bb60cf9783fc804d336708c1b21a` |
| `strict_negative_alpha_sigma0003_alpha_minus00005_steps1_summary.json` | `683a5975f58414b8e08f4a588717f5d73069ef3f2cb1f52884b1b99ebf740987` |
| `strict_fresh_direction_sigma0003_alpha00005_steps1_seed43_summary.json` | `7ad1c93fe6e171abd26bcaaa7a8652478f2d1b0329ee3ed498b92579fa39f5f1` |
| `strict_fresh_direction_sigma0003_alpha00005_steps1_seed44_summary.json` | `e3f34a4ce5bddb1c8e32709224f821df3b8631d31b91f3fbb0a67ad8818483e4` |
