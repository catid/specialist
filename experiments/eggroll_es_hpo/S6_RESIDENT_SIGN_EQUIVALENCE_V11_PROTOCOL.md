# S6 resident-sign exact-equivalence diagnostic v11

V11 changes execution scheduling only. It must reproduce the completed V10 raw
responses and projected plans exactly before it can replace V10 in any later
experiment. It applies no parameter update and consults no validation, OOD, or
heldout result for selection.

## Frozen execution

- Exact V10 model, middle-late layers 20--23, sigma `0.0003`, 32 perturbation
  seeds, D43/D44 manifests, A43/A44 reference generations, and alpha zero.
- Each four-engine wave is perturbed once per sign. While that sign remains
  resident, every engine evaluates `D43 -> A43 -> A44 -> D44`; restoration is
  unconditional in the enclosing `finally` and all scoring follows restore.
- The inherited first parent minibatch call performs the resident evaluation.
  The second call must match D44 exactly, contain no anchors, issue no engine
  RPC or generation, and consume its cached plus metrics exactly once.
- Counts: 32 base directions, 64 unique signed directions, 16 all-engine sign
  residencies/restores, 64 engine-direction perturb/restore cycles, 128 domain
  signed scores, and 128 anchor signed responses.

## Equivalence gate

The exact completed V10 report and journal are frozen evidence. V11 must match
V10 bit-for-bit for base/sign/domain/reference identities, both domain sign
maps, both anchor sign maps, all central vectors, robust result bindings and
payloads, all four cells, and the inherited coefficient/domain/anchor/robust
plan fields. Any mismatch fails the journal; cosine or tolerance fallback is
forbidden. Only an exact pass permits a later speed or training protocol.

