# S6 V13b middle-late runtime-forwarding retry

V13b retries the exact V13 train-only, alpha-zero five-panel diagnostic after
the first real launch failed closed during layer-plan installation. The failed
attempt is immutable and content-addressed. It generated no diagnostic
responses, wrote no report, and applied no model update.

The failure was a controller wiring error. V13 called V4's original
installation validator, whose frozen runtime table predates the middle-late
plan. The V6 through V11 lineage instead clones that validator with V6's
frozen runtime expectations. V13b makes the same explicit clone for V13 and
adds a regression using the real middle-late plan identity and its 23 runtime
parameters, 142,999,552 elements, and 285,999,104 BF16 bytes.

No recipe, data, panel, perturbation, hardware, generation, aggregation, or
selection surface changes. V13b still uses only the five frozen train panels,
all four TP=1 engines, the same 32 directions and both signs, and alpha zero.
Every update RPC remains forbidden. Validation, OOD, heldout, benchmark, and
evaluation paths remain absent and rejected.

The retry driver requires the exact failed-attempt file/content hashes,
failure phase and message, failed source commit, failed implementation bundle,
and failed recipe hash. It normalizes implementation/retry metadata and
requires the remaining recipe to be exactly equal before constructing a
trainer. The retry uses fresh exclusive attempt and run paths and retains the
same durable failure telemetry as V13.
