# S5 provisional-snapshot transfer check

This package tests whether the six-step S4 setting transferred to a newer
908-row training snapshot before the domain evaluation set finished its final
manual audit. Both treatments used Qwen3.6-35B-A3B, `sigma=0.001`,
`alpha=0.00025`, population 8, batch and mini-batch size 64, and 32 generated
tokens. The retained zero-step baseline scored 0.0744407192 on 109 provisional
domain-validation examples and 0.7141287879 on the 24-example OOD QA set.

| Seed | Validation | Delta | Exact change | Nonzero change | OOD QA delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.0666393946 | -0.0078013246 (-10.48%) | -1 | -2 | -0.0011904762 |
| 43 | 0.0843289741 | +0.0098882549 (+13.28%) | +1 | +1 | 0 |

Seed 42's validation interval was
[-0.0235624273, +0.0006563515] and its OOD interval was
[-0.0035714286, 0]. Seed 43's validation interval was
[+0.0000289026, +0.0289217398], while every OOD example tied the baseline.
The conflicting validation signs make this transfer strongly seed-sensitive;
it is not evidence for adopting the recipe. The domain artifact in this
package was later superseded by the final audited `eval_qa_v3.jsonl`, so these
scores are preserved only as a provenance-complete preliminary result.

The held-out split was never evaluated and no checkpoint was saved. OOD QA
contains only 24 examples, so unchanged scores do not establish broad
distributional robustness.

All dataset, trainer, comparison, summary, and telemetry hashes are recorded
in [`result.json`](result.json). The seed-42, baseline, and seed-43 one-second
GPU traces contain 122, 45, and 121 samples, respectively. The treatment runs
used all four devices concurrently during ES evaluation; lower whole-trace
averages include startup, centralized updates, and serial task evaluation.
