# S6 layer-partition v4 seed-42 results

This package closes the original S6 v4 front/back-versus-middle experiment family. It records only aggregate line-search metrics, frozen identities, run metadata, and journal/GPU-trace hashes. Per-example evaluation output was not inspected or copied here.

## Outcome

No nonzero candidate passed the strict gates, so no candidate advanced to direct confirmation and the heldout remained unopened and unscored.

Both population-16 pilots reproduced the same alpha-zero baseline:

- validation mean reward `0.08381010452961674` (`2` exact, `13` nonzero, `41` rows);
- OOD QA mean reward `0.714128787878788` (`16` exact, `23` nonzero, `24` rows);
- OOD prose mean token log-probability `-1.2632580042542214` (`16` documents, `10,926` scored tokens).

OOD QA was exactly unchanged at every tested alpha. The front/back pilot improved validation most at alpha `0.00000625` (delta `+0.0007317073170731714`), but its prose delta was `-0.003325770186902499` with paired 95% CI `[-0.005465899153831232, -0.0008649941150006951]`. The middle pilot's only validation improvement was at the same alpha (delta `+0.00004878048780487809`), but its prose delta CI `[-0.0006753152389219585, 0.0031093444001601706]` crossed degradation. The zero-degradation policy rejected both, as well as every other nonzero state.

## Reproducibility scope

The machine-readable record is [`summary.json`](summary.json). It pins:

- the S6 train, validation, OOD QA, prose-anchor, reward, model-config, layer-plan, and protocol identities;
- smoke and population-16 recipes and aggregate states;
- the four complete alpha-line-search journal hashes;
- the four GPU telemetry hashes, each of which records GPUs `0`, `1`, `2`, and `3`.

The governing protocol is [`../S6_LAYER_PARTITION_V4_PROTOCOL.md`](../S6_LAYER_PARTITION_V4_PROTOCOL.md), SHA-256 `3b6f4683ea4ce91078df1e5d47c4d65d787925c1eb98720dbfa4144055160aa1`.

This package deliberately excludes the later rescue family. It is a closed negative result for the original seed-42 v4 recipes, not evidence against other anchor strength, projection, layer placement, or optimization choices.
