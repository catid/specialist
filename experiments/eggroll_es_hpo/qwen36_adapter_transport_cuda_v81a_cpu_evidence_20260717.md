# Qwen3.6 V81A pinned transport: CPU/source evidence

Status: CUDA integration and prospective paired systems contract sealed; live
execution remains blocked.

## Result

V81A is an additive subclass of the accepted V72 worker. It leaves V71, V72,
V73B, vLLM, and site-packages unchanged. The only changed operation is the
canonical-to-runtime publication step:

- CPU FP32 remains the sole canonical, perturbation/update, optimizer, and
  checkpoint authority: 70 tensors, 4,528,128 elements, 18,112,512 bytes per
  actor.
- Each actor lazily allocates one stable pinned CPU BF16 bank: 4,921,344
  elements and 9,842,688 bytes.
- The bank is partitioned into the exact 82 production runtime views. Each
  segment is copied directly into the existing sole resident vLLM adapter slot
  with `non_blocking=True` on one dedicated copy stream.
- One generation-scoped CUDA event is recorded after all 82 copies. The
  consumer stream waits on that exact event and the host synchronizes it before
  the inherited V71 exact readback, activation, or generation can proceed.
- Publication H2D remains 9,842,688 bytes and 82 calls per transition. The
  challenger has zero temporary *publication-staging* device bytes and zero
  publication D2D payload. V71's shared fused exact-audit buffer is unchanged
  and is not misreported as an arm-specific saving.
- The pinned bank uses non-owning tensor weak references to prove legal
  same-source reuse without retaining a hidden FP32 candidate bank. V72's
  one/two/one host ownership remains authoritative.

The source fails closed on a pageable bank, insufficient memlock capacity,
partial view/byte coverage, missing or foreign events, an event that becomes
pending after publication, active-generation overlap, and cross-candidate bank
reuse. An interrupted copy enters unknown device state; only a synchronized
exact-master rewrite can recover it, otherwise the inherited actor poison path
wins. Final cleanup requires the exact master, quiescence, a complete event,
and logical release of the bank and stream.

## Memory and bandwidth scope

The current shell reports a 118,224,613,376-byte soft and hard memlock limit.
One actor requires 9,842,688 bytes and all four actor banks sum to 39,370,752
bytes. This is capacity context only: every live actor must still prove the
actual tensor reports `is_pinned()`, with VmPin/VmLck recorded only as
non-authoritative process context.

This arm does not reduce persistent runtime VRAM or H2D bytes: BF16 is already
the lowest supported installed vLLM LoRA execution width. Its prospective
benefit is removing the per-view host-to-temporary-device-to-runtime path and
the associated 9,842,688-byte D2D payload per transition. FP32 collective
coalescing belongs to `specialist-0j5.36` and is not folded into V81A.

## Prospective comparison

The paired contract freezes four exactly counterbalanced orders (AB, BA, AB,
BA), with four physical GPUs in every arm. It requires transition time, H2D
bytes/calls, publication D2D trace, peak allocated/reserved VRAM, PCIe RX/TX,
HBM activity with explicit units/exactness, actor PID/useful-CUDA attribution,
exact V71/V72/V73B lifecycle receipts, actual pinning, and final Ray/GPU idle.

The runner is systems-only and intrinsically nonpromotable. It recursively
rejects the quarantined V1 evaluation surfaces. Historical reward floats are
not reused as a cross-run bitwise oracle; each arm must independently prove
same-live dual-compiler integrity, exact audits, update abort, and final exact
restore, while deterministic population-plan, candidate-projection, and final
master/runtime identities are paired exactly.

Execution is blocked until an additive gate binds completed
`specialist-0j5.32` V73C phase evidence. The current CLI has no executor and
hard-fails `--execute`. Even a successful systems report can only authorize
preregistration of a separate quality successor bound to the sealed V2
evaluation boundary; it cannot promote a worker, recipe, checkpoint, or HPO
choice directly.

## Verification

- 25 original V81 CPU/source tests retained.
- 15 V71 tests retained.
- 8 V72 tests retained.
- 14 V81A worker integration/oracle tests pass.
- 11 prospective runner/gate/analyzer tests pass.
- 7 V81A preregistration tests pass.
- Combined adjacent result: 80 passed.
- Builder `--check`, `py_compile`, and `git diff --check` pass.
- No GPU, model, Ray, training, dataset/protected content, checkpoint, or
  promotion action was performed.

## Identities

- Preregistration file SHA-256:
  `f30006c1961509f25e3e4706f77a356aff226d6792466bfe64a8df7b8778d834`
- Preregistration content SHA-256:
  `a1a6202845087d0855c0ac027cec1af3058efa9c8f978fb0c132f37e7b32aef2`
- CUDA worker SHA-256:
  `17b20a974b2e21cf144b376080af0cffc2751cf0c52dbc538e615e5e2ae88879`
- Prospective runner SHA-256:
  `1a5feb969659cd3926d95a90c4c2f003444005e58d76039018d7992c5908f69e`
