# Qwen3.6 TP1 process-group cleanup amendment V83

The immutable V80 parent verdict remains recorded as failed: it literally
required `torch_process_group_destroyed=true`, while all actors persisted
`false`.  V83 does not rewrite that result.  It binds the exact legacy source
and records that the field actually meant `dist.is_initialized()` **before**
shutdown.  The helper calls `destroy_process_group()` only when that value is
true.  Therefore the four TP1 actors in each run had no process group to
destroy.

| Run | legacy value | consecutive external idle batches | max memory | max util |
|---|---:|---:|---:|---:|
| v80_bf16_kv_mamba_capacity_0479_r1 | 4/4 false | 3 | 4 MiB | 0% |
| v80_bf16_kv_mamba_capacity_0479_r2 | 4/4 false | 3 | 4 MiB | 0% |
| v80_bf16_kv_mamba_capacity_0479_r3 | 4/4 false | 3 | 4 MiB | 0% |

Every accepted cleanup batch also has dead actor roots, no descendant compute
PID, and no foreign compute PID.  The additive cleanup interpretation passes;
all semantic, OOD, and promotion gates remain pending.

Future receipts must separately persist initialization-before, destroy
attempt, and initialization-after.  Never-initialized TP1 is accepted only as
`false/false/false`; an initialized group is accepted only as
`true/true/false`.  External dead-process and idle-GPU cleanup remains
mandatory in both cases.

- Legacy source SHA-256: `115774a63f54480fa4796f24f5b47a82fda1c2a761db4cf3ae0b6b83e85165d6`
- V83 content SHA-256: `45957ac5f53004456862596baacc09d95f0179995436e745eea7dd970e1a91de`
- Dataset/protected/model/GPU access by builder: none
