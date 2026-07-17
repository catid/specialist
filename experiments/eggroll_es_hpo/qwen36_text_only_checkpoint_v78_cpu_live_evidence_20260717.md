# Qwen3.6 text-only checkpoint V78 CPU and live evidence

## Outcome

Do not build a text-only derivative checkpoint or add a loader filter under
the current runtime.  The serialized-FP8 checkpoint contains 1,746,810,976
bytes of visual and MTP tensors, but all four live V76 R6 actors report only
the language component in `named_parameters()`: 813 parameters and
35,712,084,096 logical bytes per actor.  Every actor reports zero visual and
zero MTP named parameters with the same parameter-name manifest
`a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90`.

This closes the proposed steady-state VRAM and model-weight memory-bandwidth
optimization as a negative result.  The checkpoint still has a possible
storage/startup-I/O opportunity, but that alone does not justify a derivative
artifact and its maintenance surface without a separate startup-bottleneck
measurement.

The immutable preregistration is
`experiments/eggroll_es_hpo/preregistrations/qwen36_text_only_checkpoint_v78.json`.
Its canonical content SHA-256 is
`7164f1ba8f581f5e6253ee89aead3058366526fbee24913501c2aa9aec6ec121`;
its file SHA-256 is
`6666f6e485d4ae12796d1731af34131a717e1d76497cc49a028281628204ec4e`.

## Checkpoint-byte inventory

The builder parses raw safetensor headers and binds every omitted key, dtype,
shape, source file, and byte count.  It does not read tensor payloads or import
torch/vLLM.

| Checkpoint | All logical bytes | Language | Visual | MTP | Visual + MTP |
|---|---:|---:|---:|---:|---:|
| BF16 | 71,903,645,408 | 69,321,221,376 | 893,142,496 | 1,689,281,536 | 2,582,424,032 (3.59%) |
| Serialized FP8 | 37,454,789,472 | 35,707,978,496 | 893,142,496 | 853,668,480 | 1,746,810,976 (4.66%) |

The BF16 inventory has 693 language, 333 visual, and 19 MTP tensors.  Its
omitted tensors are mixed with language tensors in shards 1, 2, 25, and 26,
so no whole-file ignore rule can remove them.  The serialized-FP8 inventory
has 62,303 language, 333 visual, and 1,560 MTP tensors.  Its MTP tensors are
isolated in `mtp.safetensors`, but all visual tensors share
`outside.safetensors` with three language tensors.  A complete omission would
therefore require rewriting a derivative artifact, not merely ignoring files.

Both checkpoints have byte-identical tokenizer, chat-template, vocabulary,
merge, and generation files.  No other tensor prefix exists beyond
`model.language_model.*`, `lm_head.*`, `model.visual.*`, and `mtp.*`.

## Installed vLLM behavior

Ten installed vLLM 0.25 source files are hash-bound.  Under the existing
text-only settings, image and video limits are both zero and speculative
configuration is absent.

- Qwen3.5 constructs the vision tower inside `_mark_tower_model`.  With all
  tower modalities limited to zero, vLLM constructs it on the meta device and
  replaces the registered child with `StageMissingLayer`.
- `StageMissingLayer` keeps the original meta module through `__dict__` rather
  than registering it as a persistent child.  `AutoWeightsLoader` skips the
  stage-missing module.
- The target Qwen3.5 loader explicitly skips `mtp.*`.  The separate
  `Qwen3_5MoeMTP` draft architecture is selected only when speculative
  configuration is non-null.
- The default lazy safetensors iterator calls `get_tensor` before model-level
  StageMissing/MTP skipping.  Thus the full artifact can still incur
  checkpoint traversal and payload-read work at startup even though those
  tensors do not become persistent model parameters.

Checkpoint bytes are therefore not evidence of live residency.  The source
trace predicts zero incremental persistent VRAM saving, and the live audit
confirms that prediction for the production serialized-FP8 path.

## Four-actor live audit

The exact nine-file V76 R6 bundle is
`experiments/eggroll_es_hpo/runs/v76_fp8_attested_050_r6_residency`, with
canonical inventory hash
`142fea7a45b62ec87d1d60c35f8819e017b79ac3a4004aa1fdb3e4882d775795`.
All four receipt self-hashes validate.  No actor opened source-dataset or
protected rows or performed an adapter/model update.

Each actor reports the same live inventory:

- component set: `language` only;
- parameter count: 813;
- logical parameter bytes: 35,712,084,096 (33.26 GiB);
- dtype counts: 303 BF16, 270 FP32, and 240 FP8 parameters;
- device count: 813 on the actor-local `cuda:0`;
- visual named parameters: 0;
- MTP named parameters: 0.

All four logs bind `limit_mm_per_prompt={'image': 0, 'video': 0}` and
`speculative_config=None`.  The audit concerns persistent named parameters;
total engine VRAM also includes runtime allocations and is not mislabeled as
checkpoint-component residency.

## Supported path if evidence changes

No artifact was created and no loader, checkpoint, site package, or
`es-at-scale` source was modified.  If a future audit finds live visual/MTP
CUDA parameters, or startup I/O is independently shown to be material, the
only documented candidate is a standard Hugging Face safetensors derivative:
retain exact language tensors, copy each retained payload bit-for-bit,
regenerate the index, and keep the config/tokenizer files byte-identical.
Custom loaders, ignore-pattern tricks, architecture rewrites, and site-package
patches are rejected.

The dormant validator remains fail-closed: any reopened experiment must prove
the complete retained/omitted key manifests, per-tensor payload identity,
exact live named-parameter, tokenizer, LoRA-target, logits, and greedy-token
identity, at least three counterbalanced four-GPU pairs, load-time improvement,
throughput and VRAM non-regression, cleanup-idle state, semantic
non-inferiority, and one-shot protected OOD non-inferiority.  The current
CPU-only preregistration grants neither artifact creation nor promotion.

## Validation

```text
es-at-scale/.venv/bin/python -m pytest -q test_qwen36_text_only_checkpoint_v78.py
16 passed

python3 build_qwen36_text_only_checkpoint_preregistration_v78.py --check
status=inventory_and_live_fp8_audit_complete_no_artifact_recommended

python3 -m py_compile qwen36_text_only_checkpoint_v78.py \
  build_qwen36_text_only_checkpoint_preregistration_v78.py \
  test_qwen36_text_only_checkpoint_v78.py
```

These checks use no GPU, dataset, protected-evaluation, training, model update,
checkpoint mutation, derivative creation, or site-package modification.
