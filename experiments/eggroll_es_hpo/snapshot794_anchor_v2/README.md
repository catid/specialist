# S6 corrected anchored-v2 pilot

This is the first nonzero S6 pilot without the two anchored-v1 confounds. Each
of the four engines restored every BF16 perturbation from a committed CPU
reference, domain and anchor requests used separate generation calls, and a
mandatory alpha-zero audit compared all 573 parameters plus pre/post train-only
output hashes before any update was allowed.

The audit passed on every engine with the same 69,321,223,296-byte weight state
(`39178d6f...5008`). Validation and OOD-QA baselines exactly reproduced the
frozen S6 baseline. Alpha `0.0000125` improved validation by
`0.0006585365853658404` (about 0.79%), but its OOD-prose delta was
`-0.0021104592187286553`; all four nonzero states had a negative prose point
delta and therefore failed selection. No checkpoint was saved, and the sealed
heldout split was not loaded or scored. Baseline remains the selected state.

These are monotonic BF16 pilot states, not direct-alpha confirmations: later
alphas were reached through incremental updates, so their exact weights are
path-dependent. The pilot may guide ranges but cannot substitute for fresh
`0 -> alpha` replication.

The GPU trace also isolates a recipe bottleneck. All GPUs reached 100% and were
simultaneously active during population/evaluation, but five samples show GPU 0
at 90%+ while the other three were idle because the inherited update computes
all seed noise on engine 0 before broadcasting. Anchored v3 must shard seed
work across all engines and FP32-all-reduce the update before more selection
runs.

The machine-readable record is `summary.json`; the frozen GPU trace is
`../../gpu_utilization_eggroll_es_s6_anchor_v2_exactrestore_items16_cos025_seed42.jsonl`.
