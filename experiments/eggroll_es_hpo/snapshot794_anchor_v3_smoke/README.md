# S6 distributed anchored-v3 guarded smoke

This diagnostic exercised the four-engine sharded FP32 update protocol on the
frozen S6 artifacts with Qwen3.6-35B-A3B.  It used population 8, 16 prose-anchor
items, minimum anchor cosine 0.25, sigma 0.0003, seed 42, and a single monotonic
BF16 step from alpha 0 to 6.25e-6.  All four TP=1 engines passed the allocation
preflight, independent manifest reconstruction, seed-shard coverage, final
weight-hash consensus, two-phase commit, and post-commit state consensus.

The alpha-zero state reproduced validation 0.08381010452961674, OOD-QA
0.714128787878788, and OOD-prose -1.2632580042542214.  The nonzero state left
validation and OOD-QA exactly flat but reduced OOD-prose to
-1.2657495474262415, a delta of -0.00249154317202005 with paired-document 95%
interval [-0.004954134740518329, 0.0003068238055387737].  It therefore failed
the strict nondegradation gate and is not selectable.

The offline fail-closed aggregator separately and correctly rejected this
journal because its snapshot omitted the inherited v2 `corrected_driver` and
`exact_worker` implementation identities.  The completed producer artifact is
retained byte-for-byte and was not retroactively edited or resealed.  The
producer was fixed in commit `e355015`; future v3-derived snapshots bind the
complete implementation chain.  This smoke is consequently mechanical and
diagnostic evidence only, not an aggregatable selection result.

The utilization trace contains 185 two-second samples.  Every GPU reached 100%
utilization, all four were simultaneously at 100% in eight samples, and peak
memory was 87,458, 86,254, 86,622, and 86,778 MiB on GPUs 0 through 3.  The
sealed heldout split was never loaded or scored.

Source identities:

- journal SHA-256: `bb9e3d5c78bc5f458733f96e2bc39d8cc03acdf5fba8a2ee7f847536799279cc`
- GPU trace SHA-256: `b320a2ec1ac8b0831191237a81bcbeff1ebede34433336239a96d47a80e37e94`
- journal self-content SHA-256: `f98b2969b0f9d2339094ac10c7e88019c439a2bc6ab9ed2e7804d265a6f831c8`
