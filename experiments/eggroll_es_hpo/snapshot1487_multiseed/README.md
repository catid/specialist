# S4 selected-treatment multiseed validation aggregate

This aggregate compares the selected six-step S4 treatment with the same
retained baseline over training seeds 42–46 on the frozen 1,487-row snapshot.
All runs used Qwen3.6-35B-A3B, `sigma=0.001`, `alpha=0.00025`, population 8,
batch and mini-batch size 64, and 32 generated tokens. Only the 236-item
`train` validation split is aggregated; no holdout or OOD score enters these
statistics.

| Seed | Treatment | Delta vs baseline | Exact change | Nonzero change | Paired 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.1044079066 | +0.0067350542 | +1 | +2 | [+0.000186, +0.016840] |
| 43 | 0.0934198433 | -0.0042530091 | -1 | -1 | [-0.014460, +0.002012] |
| 44 | 0.0979182133 | +0.0002453609 | 0 | 0 | [-0.012389, +0.012578] |
| 45 | 0.0970984390 | -0.0005744133 | 0 | -3 | [-0.013321, +0.011738] |
| 46 | 0.0970728458 | -0.0006000065 | 0 | -2 | [-0.013316, +0.011650] |

The baseline is 0.0976728523. Across all five seeds, mean delta is
+0.0003105973, median delta is -0.0005744133, and sample standard deviation is
0.0039911417 (population standard deviation 0.0035697857). Two of five seeds
have positive point deltas, but only seed 42 has a paired interval wholly
above zero. Aggregate exact-match change averages zero; aggregate nonzero
change averages -0.8 examples per seed.

Seed 42 was itself used to select the setting and six-step horizon on this
validation split, so the five-seed mean is biased upward. The four clean
post-selection replicates have mean delta -0.0012955170, median delta
-0.0005872099, and only one positive point delta. The evidence therefore does
not support treating seed 42's gain as a robust recipe improvement. The
validation split also has known document overlap with training, and the metric
is normalized answer-token overlap with exact-match credit rather than binary
accuracy.

The machine-readable [`aggregate.json`](aggregate.json) has SHA-256
`4954ed5ddeea888a8adf374fa50c0a365006e31b534b90b9e62e1ed658163633`
and records all per-seed summary, comparison, raw-evaluation, GPU-trace,
trainer, and dataset hashes. The source comparison hashes are:

| Seed | Comparison SHA-256 | Summary SHA-256 |
| ---: | --- | --- |
| 42 | `1c3ffb3b12829875e6a5a34e355628445092d611a637326fd43748169f024209` | `96749661a47cc04719a238259f6d56d029e22655f577b394fba89912d9df96ae` |
| 43 | `207e1ca39826c14b847bc8829145262fceaa6f8d033c08fb9ae83fb843b2a488` | `5d00e358c2289df4efe4de61e8d9057e46237c29e7f7df05c23588f158cda32d` |
| 44 | `e0854926f9f197723702f02ddb51d2ed5ff6ec33e7311ca5b85f46633296d17d` | `0778f0410d51de58bae915d76051de551df15e609ac601a7ba1f551a008e12b6` |
| 45 | `7e6405d85b31623eab7c7dde16b246f5fff560c6bd56829740c1b602dd0490e4` | `7d47d595700343fd30545e995f48eba8fa1f7d78fa406c676f81b551dcacd8b4` |
| 46 | `2ba61608075a2fa40e137713a3b22655faa4b3193fb9389a720d6f1feaa1b4a2` | `57b1cfa60b34de535849ce0f335bcdeab1414ea6c752d88980c9546335ad4fef` |
