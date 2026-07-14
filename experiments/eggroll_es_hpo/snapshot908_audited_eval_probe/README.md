# S5 audited-evaluation HPO probes

These probes use the same frozen 908-row training snapshot and the final
manually audited `eval_qa_v3.jsonl` artifact with Qwen3.6-35B-A3B. All three
candidates used seed 42, population 8, batch and mini-batch size 64, 32
generated tokens, and three exact ES updates. The baseline is a zero-step
evaluation of the same model and artifacts. Only the 41-example validation
split and 24-example OOD QA split were evaluated; the 18-example heldout split
has not been opened and no checkpoint was saved.

| Sigma | Alpha | Validation | Delta vs baseline | Changed validation items | OOD QA delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0005 | 0.00025 | 0.0876061134 | +0.0037960089 (+4.53%) | 2 wins, 0 losses | 0 |
| 0.0010 | 0.00025 | 0.0866305036 | +0.0028203991 (+3.37%) | 1 win, 0 losses | 0 |
| 0.0003 | 0.00015 | 0.0839119073 | +0.0001018028 (+0.12%) | 1 win, 0 losses | 0 |

The baseline validation score is 0.0838101045. The paired bootstrap intervals
for the three rows above are [0, 0.0104124169], [0, 0.0084611973], and
[0, 0.0003054083], respectively. Their lower percentiles equal zero, so none
is independently conclusive. All candidates preserve 2 exact matches and 13
nonzero rewards. Every OOD QA prediction is byte-for-byte unchanged from the
baseline evaluation: score 0.7141287879, 16 exact matches, and 23 nonzero
rewards.

The best point result is therefore `sigma=0.0005`, `alpha=0.00025`, but this
is validation-selected, single-seed evidence rather than a final recipe. The
training artifact was still provisional at the time of these runs even though
the evaluation artifact had completed its audit. A clean-seed replication and
a frozen prose-distribution gate should precede any one-time heldout decision;
the 24-item OOD QA set alone cannot establish broad non-degradation.

The pre-run and post-run gates fixed training Arrow SHA-256
`05ad1e523032026d59bf2da953b5c15bfd8f6ea738067685b4eb947b012b86b2`,
validation Arrow
`19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db`,
OOD QA Arrow
`b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced`,
manifest
`6eafb6b4119b64534d0baf5736af1ea4162a60df0d84e3ca047ba0b84424ffc2`,
and trainer
`e095d90eef32c3ae1e9109ceeff94994bfae974b46a35666c4dd6b306b8d3196`.

The machine-readable [`result.json`](result.json) links every compact summary,
self-contained paired comparison, and one-second GPU trace by hash. The three
treatment traces contain 83, 82, and 84 samples in table order. During their
parallel population phases all four GPUs reached 100% utilization together;
startup, centralized weight updates, and single-engine QA evaluation account
for the lower whole-trace averages and expected idle intervals.
