# Qwen3.6 exact speculative-decode V84 CPU evidence

- Contract: `04225231b9be3b868f1c06a2a64cb7c7a8b7bd1468b11f179f9155415db545da`
- Installed vLLM 0.25.0 explicitly supports `Qwen3_5MoeMTP` and a model-free CPU n-gram proposer.
- The checkpoint declares 1 MTP layer; baseline live actors load zero MTP named parameters.
- Enabling MTP may add up to 1,689,281,536 BF16 or 853,668,480 serialized-FP8 logical bytes before allocator/cache effects.
- V73D is a one-token workload, so it cannot measure multi-step speculative acceptance; the future systems screen is an additive synthetic 32-token contract.
- No GPU/model/data/protected access or training/promotion occurred. Live work remains unauthorized.
