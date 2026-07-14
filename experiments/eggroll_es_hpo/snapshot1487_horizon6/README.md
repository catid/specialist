# S4 six-step horizon probe

The fresh S4 grid selected `sigma=0.001`, `alpha=0.00025` at three steps. This
probe extended only that setting to six steps on the same frozen dataset and
validation split; the holdout remained unopened.

Validation reward improved from 0.1031286700 at three steps to 0.1044079066 at
six. Relative to baseline 0.0976728523, the six-step delta is +0.0067350542
(+6.90%). It won 15 items, lost 8, and tied 213; exact matches changed 14 to 15
and nonzero rewards 93 to 95. The paired bootstrap 95% interval is
[0.0001860413, 0.0168399353]. Because the setting and horizon were selected on
this validation split—and validation has known document overlap with
training—the interval is descriptive rather than a correction for HPO bias.

| Artifact | SHA-256 |
| --- | --- |
| HPO journal | `5d46b9207ffe672366fe366be85dde85ea26f9b10572ce9e0f5aca724d663b97` |
| Self-contained paired comparison | `d082d2425beca0ae22da8174747db5c48fc992ecc24fc13a5b817093171c29a4` |
| GPU trace | `8e325783e421916856ba6240e667894d298300ddb1b74633774fed56e065626e` |

The 181-sample trace is
[`../../gpu_utilization_eggroll_es_s4_horizon6.jsonl`](../../gpu_utilization_eggroll_es_s4_horizon6.jsonl).
All four GPUs were at least 20% active together for 40 samples and exactly 100%
for 23. Whole-trace utilization means, including startup/evaluation/teardown,
were 46.10%, 23.92%, 23.85%, and 24.21%.
