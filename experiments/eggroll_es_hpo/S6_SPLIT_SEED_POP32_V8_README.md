# S6 split-seed population-32 v8 result

Both frozen Qwen3.6-35B-A3B middle-late alpha-zero diagnostics completed on
the identical 32-perturbation basis with independent data/bootstrap seeds 43
and 44.  Each persisted journal passes the complete v8-to-v4 provenance
audit.  Both runs used exactly four independent one-GPU actors, repeatedly
reached 100% utilization on every GPU, and had no CUDA co-tenant.

The preregistered same-basis coefficient cosine was 0.427694, below the 0.5
threshold.  V8 therefore fails its response-stability gate.  Seed 43 had raw
domain/anchor cosine 0.112676 and required projection lambda 1.212195; seed 44
had raw cosine -0.100818 and required lambda 1.427391.  The projected
directions retained cosine 0.686320 and 0.516289, respectively, to their raw
domain-score vectors.  These results show material estimator sensitivity even
after doubling the population and aligning the perturbation basis.

Alpha zero remains selected.  No nonzero update was applied, the failed
stability result is not evidence for or against a particular layer location,
and the sealed holdout remains unopened.  The next protocol decomposes
execution, domain-batch, and anchor/reference variance before any LISA-style
architectural comparison or nonzero-alpha gate.

`S6_SPLIT_SEED_POP32_V8_REPORT.json` is the machine-readable coefficient and
train-anchor-geometry report.  It deliberately contains no validation, OOD,
or holdout selection metrics.  Raw journals remain in the ignored `runs/`
workspace and are bound by file and content SHA-256 values in the report.
