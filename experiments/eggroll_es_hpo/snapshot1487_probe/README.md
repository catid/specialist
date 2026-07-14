# S4 frozen-snapshot transfer probe

This is the first foreground refresh after the stricter Kinbaku audit and ten
additional manually reviewed resource facts were promoted. Both A/B arms read
the same immutable 1,487-row Arrow snapshot. The HPO driver verified the train
Arrow, validation Arrow, dataset manifest, model identity, and trainer source
before and after each arm.

| Artifact | SHA-256 |
| --- | --- |
| Curated source JSONL | `d371114ca4e2dacf7dfb97adb2f669ee8b3f44455354a15015fd2b56ac982d5f` |
| Train Arrow | `ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c` |
| Validation Arrow | `3b8b980f4be5060ade3671a53dde03a975382c63ab1e44f0133a19cd741d06cb` |
| Dataset manifest | `f78314347895a67a1696e9e936c2710de4b7874833b6e4f28378ff4f960e8df2` |
| HPO journal | `74b18b7669399082d56450bcaf4a17625869dff750436326966332074f257d7c` |
| Paired comparison | `cacbabf7ca5017f9dc43de69835c66a368aadc0b32b6eb00eaeeed632464270d` |

The inherited S3 setting (`sigma=0.002`, `alpha=0.001`, three updates, seed
42) scored 0.09469050375328315 versus the unchanged 0.09767285234556067
baseline, a delta of -0.0029823485922775062. It had 15 wins, 16 losses, and
205 ties; exact matches changed 14 to 13. The paired bootstrap 95% interval was
[-0.013486769280468167, 0.003905317420240139]. The baseline was therefore
retained. This one-treatment transfer probe is not a new S4 hyperparameter
grid, and the holdout was not opened.

The GPU trace is
[`../../gpu_utilization_eggroll_es_s4_probe.jsonl`](../../gpu_utilization_eggroll_es_s4_probe.jsonl),
SHA-256 `c279105ed8d45688a012cb5a56f3cba06dfe2a357547a8f78dfeaaf91108500f`.
Across all 175 samples (including an initial virtualenv-path failure and
startup/teardown idle time), all four GPUs were at least 20% busy together in
21 samples and all four were exactly 100% in 11. Peak memory was 96,639 MiB on
GPU 0 and 82,494 MiB on GPUs 1–3. As in the upstream recipe, rollout phases use
all four replicas while evaluation and updates include GPU-0-only phases.
