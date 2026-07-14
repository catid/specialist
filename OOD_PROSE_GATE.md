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

## Paired document-bootstrap non-inferiority decision

The eval-v3 protocol defines uncertainty at the independent source-document
level. Each current `ood_prose_v3.jsonl` item is one document, so item and
document resampling are presently identical. The implementation must still
group by `normalized_source_url` before resampling; this preserves the correct
unit if one source document is split into multiple items later.

For each of 20,000 replicates using Python's local `random.Random(20260714)`,
sample the same number of documents as the frozen cohort with replacement.
For the paired sample, recompute

```text
(final sampled logprob sum / final sampled token count)
  - (baseline sampled logprob sum / baseline sampled token count)
```

Do not sample tokens independently, bootstrap baseline and final separately,
or average document means without token weights. The 95% interval is the
linearly interpolated 2.5th and 97.5th percentiles, matching the repository's
existing paired-evaluation reports. The OOD prose gate passes only when the
interval's lower bound is at least `-0.02`; a point estimate alone cannot pass
the gate. The summary must pin the resampling unit, document count, sample
count, seed, interval, and threshold.

The CPU-only test matrix for this decision is:

1. Identical baseline/final rows produce point delta and interval `[0, 0]`.
2. Repeating the calculation with seed 20260714 produces byte-identical
   output, while the implementation does not touch process-global RNG state.
3. A fixture with unequal document token counts distinguishes the required
   token-weighted replicate from an unweighted mean of document means.
4. Multiple items sharing a normalized source URL are sampled as one document,
   not as independent observations.
5. Reordered, missing, changed-text, changed-tokenization, or changed-count
   final items fail alignment before resampling.
6. A lower bound exactly equal to `-0.02` passes; a lower bound below it fails,
   even if the point delta is above `-0.02`.
7. Zero/negative bootstrap sample counts, non-finite values, and a missing
   document identity fail closed.
8. `run_summary.json` records the fixed seed, 20,000 samples, paired-document
   unit, interval, and CI-based pass/fail result.

A compact reference fixture makes the weighting test unambiguous. Document A
has one scored token, baseline sum `-1.0`, and final sum `-0.5`; document B has
three scored tokens, baseline sum `-3.0`, and final sum `-3.6`. The required
full-cohort point delta is `-0.025`, not the `+0.15` obtained by averaging the
two document deltas equally. With two-document resamples, the only possible
replicate deltas are `+0.5` (A/A), `-0.025` (A/B or B/A), and `-0.2` (B/B).
This fixture therefore detects both an incorrect unweighted metric and an
incorrect token-level bootstrap without requiring model inference.

This path is deliberately opt-in and currently requires
`--exact-train-steps`; ordinary upstream-compatible training is unchanged.
