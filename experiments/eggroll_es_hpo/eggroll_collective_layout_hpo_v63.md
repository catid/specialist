# Four-GPU EGGROLL-ES collective-layout HPO (V63)

## Scope

This data-free microbenchmark measures the coefficient-update collective used
by the middle/late LoRA EGGROLL-ES recipe. It allocates synthetic FP32 tensors,
performs no model load or parameter update, opens no dataset or protected
split, and writes no prompt or generation content. All variants use the same
142,999,552 elements (571,998,208 bytes, or 545.5 MiB, per rank), four RTX PRO
6000 GPUs, NCCL, and five warmup iterations. The initial sweep used 60-second
measurement intervals; diagnostic replications used 10-, 15-, 20-, or
30-second intervals to expose order and runtime-state sensitivity.

The benchmark implementation SHA-256 is
`e1196567065baa6d89baeb64b180d3eed5c2df3101d9dce4bff65e257d3edcf7`.
The real 23-tensor size vector is bound by SHA-256
`7871fe2d020537cebb72053b8c3866b7b1d4296fee284833ac0b18c8c6a5c240`.
The single-flat-buffer vector is bound by SHA-256
`5f30ab8c1edf67a6081c2010f3cab43f6d01ac6597d81061ed647049a8d7ee61`.

## Results and replication warning

Reported throughput is the mean across ranks; rank-to-rank values agreed to at
least five decimal places. Independent utilization checks showed all four GPUs
participating at 100% during representative measurements. The first sweep
looked decisive, but later counter-ordered replications exposed a large
run-level throughput regime. The initial point estimates therefore must not be
treated as a stable layout comparison.

| Tensor layout | Collectives/iteration | Observed algorithm GiB/s | Replications |
|---|---:|---:|---:|
| Real 23 tensors, native boundaries | 23 | 6.24075–11.37704 | 10 |
| One flat buffer, one collective | 1 | 8.12296–11.52071 | 6 |
| One flat buffer, 25.2M buckets | 6 | 6.13414–6.22811 | 2 |
| One flat buffer, 8.39M buckets | 18 | 6.26450–11.28688 | 2 |
| Real tensors, 8.39M cap | 31 | 6.31537–11.12465 | 2 |
| Real tensors, forced Ring/Simple | 23 | 6.17519 | 1 |

The native layout alone varied by about 82% from its slowest to fastest run.
No leaked `NCCL_*` or `CUDA_VISIBLE_DEVICES` setting explained the switch;
clock, PCIe-link, and utilization spot checks were also normal. NCCL debug
identified NCCL 2.28.9, a PCIe/PHB topology, no IB plugin, unavailable NVLS,
and default Ring/Simple for medium/large messages with Ring/LL for 524,288-byte
tensors. Forcing Ring/Simple did not improve the slow regime.

Within the later high-throughput sequence, a flat single collective reached
11.52071 GiB/s versus 11.37704 GiB/s for native boundaries, while 8.39M flat
buckets reached 11.28688 versus 11.12465 GiB/s for capped real tensors. Those
small apparent flat-layout gains are order-confounded and much smaller than
the unexplained run-level swing. A subsequent 30-second native-then-flat pair
reversed even that local ranking (11.36866 versus 11.15858 GiB/s).

## Decision

Make no recipe change from this benchmark. Retain the existing real parameter
boundaries and NCCL defaults because no candidate has a replicated,
counterbalanced advantage after accounting for the throughput regime. Do not
promote flattening, bucket caps, or forced Ring/Simple yet.

A follow-up harness should randomize/counterbalance layout order within each
replicate, record clocks/link state and NCCL environment before every arm, and
require a paired improvement across both observed regimes. These measurements
do not authorize HPO population updates, model training, or protected-data
access.

## V67 high-regime diagnostic

A later bounded diagnostic added two 30-second native-boundary runs and two
30-second single-flat-buffer runs in the high-throughput regime. Native reached
11.2744 and 11.1705 algorithm GiB/s; flat reached 11.3426 and 11.3738 GiB/s.
The apparent flat advantage of 0.6-1.8% remains much smaller than the historical
run-regime swing and does not change the decision above.

A third native-then-flat pair made the confound explicit within one adjacent
pair: native reached 11.2296 GiB/s, then flat fell to 9.8467 GiB/s. This is not
credible evidence that flattening causes a 12.3% regression; it is evidence
that the unexplained runtime regime can change between neighboring processes
and dominate the layout effect. Process-to-process layout comparisons should
stop here. Any future decision requires both layouts inside the same long-lived
four-rank process, randomized/counterbalanced across many short blocks.

Matched 8,388,608-element caps produced 11.1524 GiB/s for 31 native
collectives and 11.2417 GiB/s for 18 flat collectives. At a 1,048,576-element
cap, 141 native collectives reached 10.8737 GiB/s while 137 flat collectives
reached 10.6940 GiB/s. The layout ordering reverses at the small cap, and both
small-cap arms trail the unbucketed high-regime results. This reinforces two
conclusions: avoid aggressive micro-bucketing, and do not pay VRAM for a flat
shadow buffer merely to chase an unreplicated low-single-digit collective gain.
All four ranks agreed within each run and all four GPUs were independently
observed at 100% utilization.

## V68 same-process paired decision

V68 keeps both layouts resident in one four-rank process and alternates 12
fixed-64-iteration AB/BA pairs. Two independent processes produced 24 paired
comparisons. The first process favored native boundaries in 11/12 pairs with a
1.00352 median native/flat speed ratio; the second favored flat in 9/12 pairs
with a 0.99771 median ratio. Each run contained one regime-transition outlier
(ratios 0.611 and 1.350), confirming that the state switch can occur even inside
a long-lived process.

Across all 24 pairs, the median native/flat speed ratio was 1.00303. Excluding
only the two obvious regime-transition ratios outside `[0.9, 1.1]`, the 22
stable pairs had median 1.00303 and geometric mean 1.00207; 13 favored native
and nine favored flat. This is effectively a tie, with a slight descriptive
edge to the real parameter boundaries. A flat shadow also requires another
571,998,208 bytes (545.5 MiB) per rank. The production decision is therefore
now firmer: retain native boundaries, do not allocate a flat shadow, and avoid
1M-element micro-buckets. V68 measures the update collective plus the real
layout's extra fill/launch operations, so this decision applies directly to the
current coefficient-update loop.
