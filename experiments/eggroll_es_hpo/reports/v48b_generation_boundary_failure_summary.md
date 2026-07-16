# V48B generation-boundary LoRA-ES failure summary

V48B was rejected by its preregistered train-only preservation gate. No adapter
snapshot was retained and no OOD, external evaluation, shadow, or sealed holdout
content was opened.

The projected update at trust-norm ratio `0.5` improved the domain objective by
`+0.000858545157650159` and QA answer log-probability by
`+0.0022618402629583922`, but failed three preservation checks:

- prose LM median paired delta: `-0.0004775705630013505`
- full QA generation F1 median paired delta: `-0.001487740794251291`
- fragile-subset generation F1 median paired delta: `-0.0008279094118506`

Exact-match and nonzero-answer counts were noninferior for both generation
checks. Population reliability was `0.95497`, with split-half Spearman
correlation `0.7381`; all 64 signed actor-state receipts and exact restores
passed. After rejection, all four actor ranks exactly matched canonical master
state SHA-256 `dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192`.

The preregistered follow-up is exact projection backtracking over smaller trust
ratios, reusing this frozen population without paying for another population or
changing any objective data.
