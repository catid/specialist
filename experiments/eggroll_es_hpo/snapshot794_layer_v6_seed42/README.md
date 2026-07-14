# S6 edge-split v6 population-16 pilots

All four capacity-matched Qwen3.6-35B-A3B arms completed with seed 42,
population 16, batch 64, sigma 0.0003, and the frozen S6 snapshot.  Every
persisted journal passes the full v6-to-v4 release audit.  All four population
rollouts used four independent one-GPU actors concurrently and were free of
CUDA co-tenants.

No nonzero alpha is release-eligible.  Front produced the largest validation
gain (+0.001390 at alpha 6.25e-6), but its prose point estimate degraded by
-0.001900 and its paired document-bootstrap lower confidence bound was
negative.  Middle-late produced a smaller validation gain (+0.000102 at alpha
3.125e-6) and a positive prose point estimate, but its prose confidence
interval also crossed below zero.  Back was validation-flat, and middle-early
did not improve validation.

Alpha zero therefore remains selected.  The sealed holdout was not opened.
The next preregistered family measures seed-to-seed coefficient-direction
stability for front and middle-late before any further benchmark selection.

The machine-readable aggregate, journal identities, gate definition, and next
experiment are recorded in `summary.json`.  Raw run journals remain in the
ignored `runs/` workspace and are bound by their SHA-256 values in that file.
