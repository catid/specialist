# Qwen3.6 BF16 hybrid-cache V80 r1 live evidence

## Outcome

V80 r1 passes every preregistered data-free capacity, runtime, backend,
residency, log, four-GPU activity, PCIe-observation, VRAM, output, engine
shutdown, and external cleanup check except one literal clause.  The parent
requires `torch_process_group_destroyed_per_actor=true`, while all four actor
receipts contain `false`.  The probe sets this field to true only if a Torch
process group was initialized before `destroy_process_group()` was called;
TP1 may initialize no such group.  That is a plausible semantic explanation,
but it was not preregistered, so this report does not reinterpret the gate
after seeing r1.  The literal parent data-free contract therefore **fails**.

The external cleanup evidence is independently strong: trailing batches
[125, 126, 127] show dead actor roots, no
compute or foreign PIDs, exactly 0% GPU utilization, and at most 4 MiB from
external `nvidia-smi` on all four GPUs.  Semantic and protected OOD evaluation
remain unopened, reference restoration remains non-exact, and promotion is
not authorized.

## Sealed evidence

- V80 r1 bundle: `73adc7ebe416d6065808cf918415d077989b1a064b47ef5132422b32da118e47`
- V80 parent content: `7527ed6fe0154a79ecc0de46b00af4601b0e3deaac184f2af094fba15740149a`
- V80B prospective confirmation content: `8515a4680175a68233f2ff408bd0439daebf1a8a4c94a842c45e08ac9ec6b976`
- Executable source bundle: `2c7927964ea3cfca1880d4cf3919609509e9d7655f2deec1646faef4b7bef0cd`
- Analysis content: `f59870ef504fcab5f80dd424c38f7e3741886add683a1059c270e8eb257d729f`

No dataset, prompt text, generated text, token IDs, protected evaluation data,
model, or GPU API was opened by this analyzer.

## Capacity, speed, and memory

| Arm | Attention KV / Mamba SSM | Utilization | Tokens / actor | Full 2048-token contexts | Four-actor median |
|---|---|---:|---:|---:|---:|
| V80 r1 | auto→BF16 / BF16 | 0.479 | 162,669 | 79 | 48.409 s |
| V79B r5 | FP8 per-token-head / FP32 | 0.485 | 162,304 | 79 | 49.100 s |
| V78c r1 | auto→BF16 / BF16 | 0.500 | 218,843 | 106 | 49.083 s |

V80 exceeds its 161,792-token minimum by
877 tokens and V76 by
4,973.  It has
365 more live cache tokens than V79B.  The
r1 median is 1.37% faster than
V78c r1 and 1.41% faster than
V79B r5, but a single replicate is not a promotion-quality timing estimate.

V80 external peak MiB by GPU is [49366, 49366, 49366, 49366]; attributed actor-process peaks
are [48756, 48756, 48756, 48756].  Its worst external peak is
49,366
MiB, within the 50,808 MiB gate.  New-monitor totals include the in-process
NVML observer, so the external total is used for the fail-closed gate and the
attributed value is reported only as a diagnostic.  Sampled PCIe RX/TX totals
are 177,041,298,305 /
17,716,005,185 bytes using the
preregistered left-rectangle approximation.  HBM bytes/s are not inferred
from NVML memory-utilization percentages.

## Parent data-free gate matrix

| Gate | Result |
|---|---|
| cardinality identity hash only and no data access | PASS |
| capacity live field log and minimum | PASS |
| hybrid cache precision backend residency and warning | PASS |
| required and forbidden logs | PASS |
| runtime and memory | PASS |
| four gpu activity and pcie telemetry | PASS |
| candidate output and paired hash matrix | PASS |
| engine shutdown completed per actor | PASS |
| external three batch post exit cleanup | PASS |
| torch process group destroyed per actor literal true | FAIL |

Candidate repeats are exact on all four actors.  Reference repeat changed-row
counts are [6, 5, 2, 6]; the
known nondeterminism remains visible.

## Paired token-hash drift

| Pair | Differing / compared rows | Agreement | Differences by call |
|---|---:|---:|---|
| V78c r1 vs V80 r1 | 46 / 2176 | 97.89% | [14, 0, 0, 13, 8, 0, 0, 11] |
| V79B r5 vs V80 r1 | 429 / 2176 | 80.28% | [31, 84, 84, 19, 18, 84, 84, 25] |

These are token-hash commitment comparisons, not semantic quality scores.

## Prospective V80B confirmations

V80B was sealed only after disclosing that r1 had been observed.  It copies
the parent runtime and all thresholds unchanged, admits exactly r2 and r3,
forbids replacement/exclusion after failure, and cannot authorize promotion.
The exact commands are:

```bash
RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/v80_bf16_kv_mamba_capacity_0479_r2 bash /home/catid/specialist/launch_qwen36_bf16_kv_mamba_capacity_v80.sh
RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/v80_bf16_kv_mamba_capacity_0479_r3 bash /home/catid/specialist/launch_qwen36_bf16_kv_mamba_capacity_v80.sh
```

Run them sequentially because each command uses all four GPUs.  Analyze both
independently; do not tune thresholds or reinterpret the literal process-group
clause from r1, r2, or r3.
