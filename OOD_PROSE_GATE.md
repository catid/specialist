# OOD prose log-probability gate

`train_eggroll_es_specialist.py` can opt into a frozen, teacher-forced prose
gate during exact-step EGGROLL-ES runs. It scores both the initial and final
weights before the live vLLM engines are closed:

```bash
es-at-scale/.venv/bin/python train_eggroll_es_specialist.py \
  --exact-train-steps 6 \
  --ood-prose-jsonl data/ood_prose_v3.jsonl \
  --max-ood-prose-degradation 0.0 \
  ...
```

The gate uses vLLM 0.25's `prompt_logprobs=1` path. Each document is tokenized
with the model tokenizer and `add_special_tokens=False`, then sent as explicit
token IDs. The first token is excluded because a decoder-only model has no
left context with which to score it. No document is silently truncated: the
default 1,024-token cap covers every frozen v3 item and an over-length item is
an error.

Documents are sharded round-robin over all live engines and results are put
back into source-file order. The corpus metric is

```text
sum(selected-token log probabilities) / number of scored tokens
```

so long documents and short documents contribute in proportion to the number
of evaluated tokens. Higher is better. The gate passes when
`final - baseline >= -max_ood_prose_degradation`.

`run_summary.json` records the input file SHA-256, scoring configuration,
aligned per-item metrics for both weight states, token-weighted means, delta,
tolerance, and pass/fail decision. Separate
`eval-output/ood_prose_baseline.json` and
`eval-output/ood_prose_final.json` files contain the same per-item metrics.
Text and token-ID hashes, token counts, and item IDs must match across the two
states or the comparison fails.

This path is deliberately opt-in and currently requires
`--exact-train-steps`; ordinary upstream-compatible training is unchanged.
