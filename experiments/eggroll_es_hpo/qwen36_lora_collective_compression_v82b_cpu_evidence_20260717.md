# Qwen3.6 LoRA collective compression correction V82B

## Decision

Do not launch or implement a live compression arm from V82.  The exact V72
production update reduces 70 canonical PEFT FP32 tensors totaling 4,528,128
elements.  V82 modeled a different 23-tensor, 142,999,552-element selected
base-weight surface, overestimating the update surface by
31.58x.  Commit `7bfb666c5afa63d199b83de2e8f670f7e7857999`
remains immutable wrong-scope evidence; none of its live or promotion
conclusions are carried forward.

The corrected nominal ring saving is only
12.955 MiB per
actor per update (51.820
MiB across four actors).  A transactional BF16 error-feedback design adds
35.047 MiB peak GPU state and
at least 51.820
MiB local HBM traffic per actor, excluding NCCL internals.  That HBM increment
is 4.0x the
nominal ring-byte saving.  Dtype compression also leaves all 70 collective
launches intact.  No isolated canonical PyNccl collective time, link bytes, or
HBM bytes have been measured, so materiality is not established.

## Exact source-bound surface

- Initialization manifest: `a2fa79e6ac06f75743d3fee8f5c0b1aabe6bb83b52b05910ed6460438e2640a2`
- Manifest content: `5f885b415302c4e748e19f4d535f1e57ff87f785370f653b345cbdfafda3224b`
- Surface identity: `6c4c219f92fba3d7d01e08f439b7b1f21a1d07bc9893cdd18f860994668e0fb8`
- Ordered-key identity: `ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280`
- Ordered-shape identity: `e12f7199343477db3927bda67bf5f364030a47216be8aa2b30fc3b71c261da2b`
- V41A worker: `cc40337eba30fe0748996c22dcbf3914b8c12249f6e2e47d6128aadee575494c`
- V72 production worker: `547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2`
- Tensor/module/element count: 70 / 35 / 4,528,128

The full ordered 70-row key/shape manifest is embedded in the machine-readable
preregistration.  The V72 method `execute_sharded_adapter_update_v41a` loops
`self._v41_master.items()`, creates one same-shape FP32 accumulator, calls
`self.inter_pg.all_reduce` in place, and copies each reduced delta to CPU FP32.

## Corrected byte formulas

For `E = 4,528,128`, world size `N = 4`, FP32 width 4, and BF16 width 2:

| Quantity | Formula | Bytes | MiB |
|---|---|---:|---:|
| FP32 payload / actor | `4E` | 18,112,512 | 17.273 |
| BF16 payload / actor | `2E` | 9,056,256 | 8.637 |
| FP32 nominal ring / actor | `4E * 2(N-1)/N = 6E` | 27,168,768 | 25.910 |
| BF16 nominal ring / actor | `2E * 2(N-1)/N = 3E` | 13,584,384 | 12.955 |
| Nominal ring saving / actor | `3E` | 13,584,384 | 12.955 |
| Two FP32 residual banks | `8E` | 36,225,024 | 34.547 |
| Maximum BF16 staging | `2 * 262,144` | 524,288 | 0.500 |
| Incremental transaction peak | `8E + max_stage` | 36,749,312 | 35.047 |
| Fused prepare HBM lower bound | `read update 4E + read residual 4E + write residual 4E + write BF16 2E = 14E` | 63,393,792 | 60.457 |
| Incremental local HBM vs FP32 | `14E + BF16 D2H read 2E - FP32 D2H read 4E = 12E` | 54,337,536 | 51.820 |

Ring bytes are a projection, not a measurement of the canonical communicator;
NCCL-internal HBM traffic is deliberately excluded.

## V75 correction and next gate

V75 remains immutable and was already provisional.  Its safe-default,
conditional-FP8, and retained-choice `native_23_tensor` collective fields are
now explicitly superseded.  They are not promotable until V75 is rebuilt with
the canonical 70-tensor PEFT layout.  V82B does not rejudge V75's unrelated
precision, cache, scheduling, or residency fields.

Only an unchanged-FP32 V72 profile may reopen the question.  Across the three
registered seeds it must measure exact canonical PyNccl collective time, link
bytes, HBM bytes, attribution, and cleanup; the collective must be at least 5%
of update execution, have at least a 1% perfect-half-payload speedup upper
bound, and rank in the top three in at least two replicates.  V82B authorizes
neither that GPU profile nor a compression arm by itself.

Dataset/protected/model/GPU access by this builder: none.  Semantic/OOD gates
were not run, and promotion is false.

- V82B content SHA-256: `3efcc3a59652a6dbef73a5e0a963e4a86628992ef371556314c73d61245983f4`
