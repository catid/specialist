# S6 direction-stability v7

The frozen Qwen3.6-35B-A3B v7 family completed all four alpha-zero runs:
front and middle-late at global seeds 43 and 44, population 16, batch 64,
sigma 0.0003, and four independent one-GPU actors.  Every journal passes the
full v7-to-v4 provenance audit.  Utilization monitoring observed all four GPUs
at 100% during population rollouts and no CUDA co-tenant.

The preregistered population-slot coefficient cosine was 0.068702 for front
and 0.501901 for middle-late against a 0.5 threshold.  Front therefore failed
and middle-late formally passed by 0.001901.  This is a response-shape screen,
not a parameter-space update cosine: seeds 43 and 44 used different
perturbation bases, so equally numbered coefficient slots do not represent the
same parameter perturbations.

The middle-late result is not sufficient evidence for a nonzero update.  Its
raw domain/anchor cosines were -0.141150 and -0.022681, requiring projection
lambdas 1.461160 and 1.355704 to reach the frozen 0.8 anchor cone.  By contrast,
the seed-42 pilot's raw cosine was 0.6921.  The fresh directions are therefore
anchor-dominated and materially inconsistent with the promising pilot
geometry even though the slot heuristic narrowly passed.

Alpha zero remains selected and the sealed holdout remains unopened.  The
next diagnostic increases population size to 32 and freezes one perturbation
basis across two independent data/bootstrap seeds.  That makes coefficient
cosine a same-basis response-stability test before any predeclared nonzero
alpha or architectural comparison is allowed.

`S6_DIRECTION_STABILITY_V7_REPORT.json` is the machine-readable,
coefficient-only report.  It deliberately contains no validation, OOD, or
holdout selection metrics.  Raw journals remain in the ignored `runs/`
workspace and are bound by file and content SHA-256 values in the report.
