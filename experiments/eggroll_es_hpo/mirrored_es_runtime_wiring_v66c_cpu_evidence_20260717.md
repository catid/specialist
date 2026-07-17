# Mirrored LoRA-ES V66c runtime-wiring evidence

Date: 2026-07-17 UTC

## Outcome before the reserved V66c retry

- The mirrored-ES protocol remains eight antithetic directions, sixteen signed
  evaluations, four complete four-actor waves, common tokenized prompts/decode
  settings/judge state within every pair, and an uncommitted update followed by
  exact canonical-master restoration.
- The shared trainer now has two explicit state modes:
  `canonical_full_weight_v1` and `external_worker`. Unknown or mismatched modes
  fail during CPU-side worker-contract resolution, before engine allocation.
- The canonical LoRA factory selects `external_worker`. It does not invoke or
  allocate `install_full_weight_master_v1` state.
- V66c makes the live LoRA transition explicit on all four actors:
  `add_lora` -> read-only adapter-1/slot-0 certificate -> canonical FP32 state
  install -> canonical master/runtime certificate. Partial activation stops
  before a slot proof or canonical-state write.

## Fail-closed receipts retained

The original V66 attempt reached trainer construction and stopped before LoRA
installation because the then-new shared constructor unconditionally requested
the dense full-weight endpoint.

- Attempt file SHA-256:
  `b01e8be7ce15fb8b931d1c58ae4139db0b4ac2e6a1825023108817cad580c955`
- Failure file SHA-256:
  `5de0eeff1ab4b49c0b1d304385db3a679cee7bb7fd8836e3e88e2864a36bd475`
- Charged GPU-seconds: `245.997554524`
- Final GPU process lists empty: true
- Protected dev/OOD/holdout opened: false
- Checkpoint, candidate commit, or promotion performed: false

V66b proved the explicit external-LoRA mode fix: all four V66 actors constructed
and execution reached `install_adapter_state_v41a` without any dense-master RPC.
It then stopped because the runner had not registered the staged adapter in the
sole vLLM slot.

- Attempt file SHA-256:
  `750a604c53de34becb37394b61773bb90cccdf91d8b7f8b74abafee66821d805`
- Failure file SHA-256:
  `03f5aa98dbd3ad75d7a561fedabea6f585f460c95186bfe1e9946cedf42ec57b`
- Charged GPU-seconds: `201.59608662`
- Final GPU process lists empty: true
- Protected dev/OOD/holdout opened: false
- Checkpoint, candidate commit, or promotion performed: false

Both failed one-shot namespaces are immutable inputs to the V66c
preregistration. Neither is reused.

## CPU validation

The final adjacent command covered the V41A adapter master, V43I fused anchor,
V51 transitions, V52 nested population, V65A/V65B high-repetition calibration,
V66 mirrored protocol/worker/runtime, V66b/V66c retry contracts, and canonical
full-weight state tests.

Result: `204 passed in 19.80s`.

Focused activation tests additionally prove:

- full-weight mode invokes `install_full_weight_master_v1` once per actor;
- external-LoRA mode invokes it zero times;
- the selected V66 worker resolves the canonical install, active-slot,
  mirrored-materialize, and mirrored-restore endpoints;
- all four `add_lora` results must be exactly true;
- four read-only certificates must identify adapter 1 in sole slot 0 before
  canonical state installation;
- a partial activation result stops before any later RPC.

## Sealed V66c launch identity

- Preregistration:
  `experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66c.json`
- File SHA-256:
  `c5dbe872839f1f9de4af1d4337bd92694f9536aa882a50114d05b30de5acfd5e`
- Content SHA-256:
  `5876731d9b1cb5f79e34fcd8a3a2a1e2a36048471e40730b4344235673887802`
- Dry run: passed without writes, Ray/model loading, GPU access, or protected
  data access.
- Attempt and run paths: fresh after an operator-requested preflight interrupt;
  that interrupt occurred during base-model file hashing, before path claim,
  Ray startup, or GPU work.

The live run remains gated on an explicit GPU handoff followed by a fresh
no-foreign-process and idle-memory check.
