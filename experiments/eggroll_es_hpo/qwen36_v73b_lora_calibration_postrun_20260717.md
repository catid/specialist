# Qwen3.6 V73B LoRA live-gate postrun analysis

Date: 2026-07-17 UTC

Verdict: **accepted** for the V71 exact-audit plus V72 LoRA host-ownership
gate.  The run is train-only evidence: no model update was committed, no
checkpoint or promotion occurred, and no protected dev/OOD/holdout data was
opened.

## Exact acceptance

- All self-hashes passed for the launch journal and six JSON run artifacts.
- All 16 actor CUDA receipts and all 412 host-process samples passed their
  per-row hashes.  The 16 work assignments/cardinalities exactly match V66d,
  with four candidates on each of four physical GPUs over four waves.
- The live reward matrix was finite, complete, and assignment-exact.  Its
  digest is `0122298c844c59665b46b9e09b4b9249fb065e49e0c7629ef9d2ad9aacea76c2`.
- The canonical V66 compiler and independent one-pass compiler produced the
  same complete result mapping from that same live vector.  Coefficient digest:
  `005182fc01f44066ce9728cbefcaca905b08c79cb1d59d39532bc9d154c3bc14`.
- All four actors agreed on candidate `f3bcdb9de5d9b815a1dc5f1f8678d02b476755521847649971a0ed9c796c068a`
  and runtime `819563416319fbf66590cd6603b9eb21305c9c0880aaf9d1323f452ecd53e3ef`.
- All 16 population restores and all four update aborts returned exactly to
  the original master/runtime; final cleanup removed all four placement groups
  and left all four compute-process lists empty.

Historical reward bits remained diagnostic, not an acceptance threshold:
maximum absolute drift was 0.00128856169189, and 5/8 pair
preference signs matched V66d.  No post-hoc tolerance was applied.

## Traffic and memory

| Measurement | V66d | V73B | Observed change |
|---|---:|---:|---:|
| Base + LoRA D2H bytes | 18,341,068,800 | 4,219,404,288 | -76.9948% |
| Peak GPU memory / GPU | 84,138 MiB | 84,140 MiB | +2 MiB |
| Wall runtime | 156.602 s | 127.274 s | -18.73% |

Worker counters exactly matched the preregistered values on every actor:
4,770,594,816 bytes
aggregate, with zero repeated master-validation host-copy bytes.  Including
explicitly accounted transfers outside those worker counters gives
6,566,641,664 bytes.

V72 ownership followed one/two/one/one banks per actor: 18,112,512 bytes after
install, 36,225,024 bytes with the executed candidate retained, 18,112,512
bytes after abort, and 18,112,512 bytes at final quiescence.  Host telemetry
sampled all actors on NUMA node 0; maximum sampled RSS was
5,317,107,712 bytes, maximum reported HWM was
6,314,668,032 bytes, major-fault delta was zero, and
fault counters were monotonic.  Pinned bytes remained
0; this run does not validate a future pinned
staging arm.

## Timing scopes

| Comparable sampled epoch window | V66d | V73B | Reduction |
|---|---:|---:|---:|
| Canonical install | 9.233 s | 5.038 s | 45.4% |
| Four materialization waves | 20.132 s | 8.791 s | 56.3% |
| Four generation waves | 3.801 s | 3.790 s | 0.3% |
| Four restore waves | 16.372 s | 9.255 s | 43.5% |
| Update execute | 4.622 s | 2.944 s | 36.3% |
| Final abort sampled span | 9.204 s | 4.207 s | 54.3% |

The table uses the same monitor-derived epoch-window definition for both runs.
It is not interchangeable with per-worker RPC duration.  V73B's worker RPC
medians were 1.595
seconds for candidate materialization and
1.777 seconds for restore;
four actors overlap, so these durations are not summed as serial wall time.
The final-abort row is a first-to-last sampled span because there is no next
phase epoch, and V73B update-prepare includes a V71 acceptance boundary absent
from V66d.

Generation itself was not faster.  The actor CUDA critical-path sum was
2.058 seconds in V66d and
2.141 seconds in V73B
(+4.01%).  The observed overall
gain therefore aligns with state/audit-window reductions, not faster model
inference.  This is one accepted run per implementation, so timing differences
are descriptive rather than a causal replicated benchmark.

## Task disposition

- Close `specialist-0j5.29`: every live acceptance condition passed.
- Close `specialist-0j5.21`: the LoRA exact-audit traffic reduction, exact
  final boundary, abort, and cleanup gates passed.
- Keep `specialist-0j5.14` open: CUPTI/equivalent HBM bandwidth, CUDA allocator
  decomposition, and live checkpoint-phase evidence remain absent.
- Keep `specialist-0j5.19` open: this accepts the LoRA one/two/one ownership
  sub-arm only.  It does not establish dense full-weight shared/mmap ownership
  or a live checkpoint path.

Canonical machine analysis content SHA-256:
`21689f75ecaaf583aedde50ad293ce3a9b5644009d62c2bc4624637280a651e7`.
