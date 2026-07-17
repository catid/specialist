# Qwen3.6 V80B three-run aggregate evidence

## Outcome

All three sealed V80 bundles and all 12 actor, candidate, cleanup, fallback,
and backend receipt sets validate.  Each run independently passes every V80
data-free parent gate except the immutable literal requirement that
`torch_process_group_destroyed` be true.  That historical result remains a
failure and is not rewritten.

V83 binds the exact shutdown source and the same 12 receipts: the legacy field
actually recorded `dist.is_initialized()` before shutdown.  All values are
false, so TP1 never initialized a process group and no destroy call was
required.  Every run independently ends with at least three external batches
showing dead actor roots, no compute or foreign PID, 0% GPU utilization, and at
most 4 MiB.  The additive cleanup interpretation therefore passes.  Semantic
and protected OOD evaluation remain unopened, so promotion remains false.

| Run | sealed bundle SHA-256 | four-actor median | worst external peak | nonlegacy gates | literal parent PG | additive V83 cleanup |
|---|---|---:|---:|---:|---:|---:|
| v80_bf16_kv_mamba_capacity_0479_r1 | 73adc7ebe416d6065808cf918415d077989b1a064b47ef5132422b32da118e47 | 48.409 s | 49,366 MiB | PASS | FAIL | PASS |
| v80_bf16_kv_mamba_capacity_0479_r2 | 3ac1156ac629d8c27d70756c08fa98652c74351549e5fea09711473900b584bc | 48.642 s | 49,366 MiB | PASS | FAIL | PASS |
| v80_bf16_kv_mamba_capacity_0479_r3 | cd5107654df8367284a5fc720c50c3007ce16513ae77b55e7c5ccc52af3c3d5f | 48.613 s | 49,366 MiB | PASS | FAIL | PASS |

## Timing, capacity, and VRAM

| Arm | actors | cache tokens / actor | full 2048 contexts | actor median | external peak |
|---|---:|---:|---:|---:|---:|
| V80B | 12 | 162,669 | 79 | 48.577 s | 49,366 MiB |
| V76 | sealed control | 157,696 | 77 | 48.584 s | 50,858 MiB |
| V78 | sealed control | 198,656 | 97 | 49.102 s | 50,856 MiB |
| V79B | sealed control | 162,304 | 79 | 49.100 s | 49,994 MiB |

Across 12 V80B actors, the median is
48.577 seconds and the range is
48.008-49.447
seconds.  Relative to the sealed control aggregates, V80B is
0.01% faster than V76,
1.07% faster than V78, and
1.07% faster than the cleanup-accepted V79B run.

V80B provides +4,973 tokens versus V76,
-35,987 versus V78, and +365 versus V79B.  Its worst
external peak is 49,366 MiB, for descriptive
savings of +1,492 MiB versus V76, +1,490 MiB versus
V78, and +628 MiB versus V79B.  V80B and V79B totals include an
in-process NVML observer, while V76/V78 use an older external monitor, so exact
cross-generation MiB differences are descriptive rather than causal.

PCIe RX/TX values in the machine report are left-rectangle estimates from
sampled rates.  HBM bandwidth is intentionally absent: NVML memory-utilization
percentages do not measure bytes per second.

## Evidence boundaries

- V80B preregistration: `8515a4680175a68233f2ff408bd0439daebf1a8a4c94a842c45e08ac9ec6b976`
- V83 additive interpretation: `45957ac5f53004456862596baacc09d95f0179995436e745eea7dd970e1a91de`
- V79B aggregate control: `dc2c3f47f28bd74bec0ddb385652c6263d38328f637f31c6769b1e48277ed46a`
- Aggregate analysis: `b38f450a1ae8d74e1e368998c43d74ff5b8d1c5d31726867beedf0f4336083f0`
- Dataset, prompts, generated text, token IDs, protected data, model, and GPU
  access by this analyzer: none
- Semantic gate run: false
- Protected one-shot OOD gate run: false
- Promotion authorized: false
