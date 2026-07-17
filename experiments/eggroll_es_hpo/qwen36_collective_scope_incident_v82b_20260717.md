# Incident: collective update surface scope contamination

## Summary

V82 commit `7bfb666c5afa63d199b83de2e8f670f7e7857999` used the 23-tensor selected base-weight surface
(142,999,552 elements) as though it were the production
LoRA update.  The canonical V72 path actually reduces 70 FP32 PEFT master
tensors (4,528,128 elements).  V82 is retained
unchanged as wrong-scope evidence; this additive V82B artifact supersedes its
collective byte, VRAM, HBM, benchmark, live-arm, and promotion conclusions.

The V82 prospective benchmark also used `torch.distributed.all_reduce`, while
production calls `self.inter_pg.all_reduce`.  Its bound ProcessGroupNCCL dtype
inspection therefore does not prove BF16 support or performance through the
canonical PyNccl-style communicator.

## Downstream impact

V75 artifact `5dd23d1effbecec2068d8e21d7f8bf9e5afab85a9e8a58d38a913e835c0e0ed5` copied V68's
`native_23_tensor` choice into both layouts and its retained-choice list.  V75
is not modified, and its noncollective decisions are not assessed here.  Its
collective-layout field is superseded and cannot be promoted until a rebuilt
decision binds the canonical 70-tensor PEFT manifest.

## Containment

- `specialist-nen.31` records this incident and blocks `specialist-0j5.28`.
- No V82 compression code or receipt may authorize a GPU run or promotion.
- Exact FP32 V72 remains the only authorized update path.
- A future compression implementation is not registered unless a separately
  authorized, data-free, unchanged-FP32 canonical profile first passes the
  prospective V82B materiality thresholds.
- No dataset, protected evaluation content, model, or GPU was opened for this
  correction.

V82B content SHA-256: `3efcc3a59652a6dbef73a5e0a963e4a86628992ef371556314c73d61245983f4`
