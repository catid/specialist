# S6 hierarchical rotating train sampling v14

## Decision

V13's five disjoint panels remain the frozen diagnostic for measuring
data-panel variance. Multi-step EGGROLL-ES training should not permanently
discard every non-representative row from a document, however. V14 therefore
uses a two-stage online design: sample documents first, then sample one row
within each selected document. The exact resulting panel is reused for every
direction and both antithetic signs in that ES iteration, and is refreshed only
between iterations.

This targets the equal-document mean of within-document row means. A page with
27 generated Q&As no longer receives 27 times the weight of a page with one
Q&A, but its useful rows can still rotate into later training steps.

The imbalance is large enough to matter: under uniform row sampling, the 794
rows induce an effective document sample size of only `139.7243` across 310
documents, and the ten largest documents receive `18.64%` of all row-sampling
probability. Uniform document sampling restores the document-level target ESS
to 310 before the fixed per-iteration quotas are applied.

## Frozen policy

The policy accepts only V13's content-addressed 794-row train source. It groups
those rows into 310 documents and assigns each document to its dominant
safety/consent, technique, equipment/material, or resources/general stratum.
Every 56-document iteration contains respectively 9, 16, 6, and 25 documents.

Within a stratum, a keyed permutation is traversed in consecutive cyclic
blocks. The first five iterations are document-disjoint within every stratum;
later iterations cycle without permanently excluding documents. Within each
selected document, a fresh keyed row permutation chooses one row for the
iteration. The ordered 56-row identity is content-addressed and must be
journaled before generation.

For stratum population `N_h` and quota `q_h`, document inclusion is `q_h/N_h`
and the equal-document Horvitz-Thompson contribution weight is `N_h/q_h`.
Selecting one of a document's `m_d` rows has probability `1/m_d`; this makes
the selected reward an unbiased draw of that document's row mean. The
equal-document contribution must not be multiplied by `m_d`. The fixed-quota
weights sum exactly to all 310 documents.

## Common random numbers and refresh boundary

One ES iteration has exactly one frozen optimization panel. Every direction
and its plus/minus evaluations see that exact ordered panel. Refreshing rows by
direction or sign is forbidden because it would add data noise directly to the
finite-difference estimate. The next document block and within-document row
draw are selected only after the iteration is sealed.

For the initial sampling A/B, V13 optimization panels 0--2 remain the crossed
diagnostic. If document balancing improves crossed coefficient cosine, V14 is
the production continuation policy. Stability should be reported per generated
prompt because three 56-row diagnostics cost 31.25 percent more domain scoring
than two 64-row panels.

## Hard replay

Adaptive hard-example replay is disabled (`0.0`) and capped at `0.25` for any
future experiment. It may be enabled only from lagged, out-of-fold train-only
difficulty fixed before the current candidate responses, with exact tier
propensities. Validation, OOD, heldout, or benchmark outcomes may never define
the hard tier. A reasonable later comparison is 80 percent rotating uniform
document coverage plus 20 percent lagged hard documents, but it is not part of
V14 until those prerequisites exist.

## Promotion rule

Do not promote V14 from sampling logic to a nonzero update until the resident
sign equivalence gate passes. Then compare the existing row-based estimator and
the document-first estimator on the same perturbation basis using only frozen
train optimization panels. Require improved median and worst crossed-panel
coefficient cosine, no sign-agreement regression, and agreement on both V13
train-only screen panels before any validation or OOD evaluation. The sealed
holdout remains unopened.

The currently curated dataset lane is intentionally not pulled into this
frozen A/B. After the sampling decision is isolated, rebuild the document frame
once the curated projection is independently promoted; do not change data and
sampling policy inside the same comparison.
