# Qwen3.6 V73G exact-phase profiler attempt 1

Status: failed closed during real Ray actor initialization on 2026-07-17. This
attempt is immutable and must not be deleted, edited, or reused.

## Result

V73G passed the V73E controller-side worker validation failure and launched all
four `TopologyLLMV40A` Ray actors. Each actor reached vLLM argument parsing and
model-architecture resolution for the configured Qwen3.6-35B-A3B model. Before
model loading, vLLM resolved the V73G worker extension and the extension raised:

`RuntimeError: V73E actor bootstrap environment or import order changed`

The Ray job runtime environment had injected the exact actor flag and guard
hash before `ray.init`, but real Ray/vLLM actor processes import application and
vLLM modules before resolving the worker extension. V73G intentionally did not
install the actor guard from sitecustomize, so its clean-interpreter requirement
was too late for the real actor lifecycle. Synthetic direct-import actors did
not reproduce that preload ordering.

The additive successor must install the actor guard in sitecustomize at actor
process startup when the exact actor flag and guard hash are present. It must
distinguish controller-sitecustomize and actor-sitecustomize receipts, allow
only imports that occurred after the process-start guard, and keep failure
closed for absent/wrong actor flags, hashes, PIDs, or mechanisms. V73G paths may
not be reused.

## Authority and resource accounting

- Protected dev, OOD, holdout, prompt, answer, or semantic content opened: no.
- Actor processes launched: four.
- Model weights loaded: no; failure preceded worker extension resolution and
  model loading.
- Actor CUDA work receipts: 0.
- Checkpoint, snapshot, model update, or promotion performed: no.
- Promotion-charged GPU seconds: 0.
- Reserved wall GPU-seconds: 196.452246116; this is not accepted as measured
  model residency or useful GPU work.
- Final compute-process lists were empty on all four GPUs.

## Sealed artifact identities

- Profile attempt: `463332fe19264c2d302588fa590003d72aa66511d8fc10391bcd5e5bb942a580`
- Profile failure: `95e16d339848bc9dcc8bed6435c968c09d5709d3656bc6face77f23535319997`
- Nsight report: `fdc747f4f0a41990abb25fb8caabe1418b481ca06462e79f5600cc733b7a8dbb`
- Nsight SQLite export: `76a9999f809c5b9e55a9b8461fae0c02fd1a1f4a6c823babb6bba567cdaa1deb`
- Run attempt: `d94194d377e858e3a74bbc1fc425b436a07920d199fdfa6dea68b5b1acc9b9c5`
- Run failure: `6904e092142e85dc3f8d9f090fb08dbc06a1d057d1b45c86cebc97ff6f0ff860`
- Partial phase receipt: `17cfb121e92027a8de0028782a32c818043b31524d156d0308115069d6704d44`
