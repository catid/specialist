# S6 antithetic crossed D x A diagnostic v10 result

V10 passed its preregistered four-cell coefficient-stability gate. It was a
diagnostic at alpha exactly zero: no parameter update was selected or applied,
and validation, OOD QA, OOD prose, and heldout outcomes did not participate in
the decision.

## Result

- Minimum pairwise coefficient cosine: `0.7602117278588` (required `>= 0.5`).
- Median pairwise coefficient cosine: `0.7602117278588` (required `>= 0.7`).
- Raw central domain cosine, D43 versus D44: `0.34415939658856237`.
- Raw central anchor cosine, A43 versus A44: `1.0`.
- Within-domain anchor-seed coefficient cosines: `1.0` for D43 and
  `0.9999999999999999` for D44.
- Each of the four cross-domain coefficient comparisons is
  `0.7602117278588`.

The weak raw domain-vector alignment is material and must not be hidden by the
passing final-coefficient gate. The result says that the frozen anchor
constraint produces a stable four-cell coefficient estimate under this
diagnostic; it does not establish that a nonzero update is beneficial. A
nonzero consensus candidate requires a separately frozen protocol and the
normal validation/OOD gates.

## Coverage and runtime evidence

The run completed with 32 base directions, 64 unique signed directions, 128
actual perturb/restore cycles, 128 domain signed scores, and 128 anchor signed
responses. Its final journal has schema
`eggroll-es-anchor-alpha-line-search-v10`, status `complete`, no in-progress
work, and no failure.

Four independent TP=1 actors occupied GPUs 0--3. The utilization guard recorded
134 samples: all four devices were model-loaded in 86, simultaneously active
in 22, and every device reached 100% utilization (including one simultaneous
all-four 100% sample). Peak memory was 82,494 MiB on each device. The main
process exited zero and all four actor processes were gone after teardown.

## Evidence identities

- Report file SHA-256:
  `a1a4528b98cee7c15323e654fe3dd6b57422c3dfba2f7ef23d95f10dee242a7f`.
- Report content SHA-256 before its self field:
  `1cbb920284e096e4ce744844aa098235f9c17dd85ac1e10a9bba19ae6db212a7`.
- Final journal file SHA-256:
  `2708b563034367479da9b25f3fcd8bd556b0c2133f533b3b561fcfd46d9af5ee`.
- Final journal content SHA-256:
  `3e68b1fb925378e31c9c4945de82d33c34f77c6abad585d0415ec456e78d71c7`.
- Crossed diagnostic artifact content SHA-256:
  `e85cc13af630a6201c7e0f4777e439d6c5aab70fc69eb41a56d8337cb8d613b7`.
- Perturbation basis SHA-256:
  `29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`.

The frozen source hashes are recorded in the protocol and verified by the
focused test. The report's implementation bundle
`06090de57809ade221915f74c1b3ebe1ab1da80b7246b66023c2ffa9a4f6fd29`
binds the v10 driver and independent reporter used for the final calculation.
