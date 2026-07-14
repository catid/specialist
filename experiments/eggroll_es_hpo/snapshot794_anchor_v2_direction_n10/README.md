# S6 full-model direction diagnostic (N=10)

This package closes the predeclared alpha-zero direction diagnostic on the
byte-frozen S6 snapshot.  Seeds 42--51 used Qwen3.6-35B-A3B, population 16,
64 fixed training examples, 32 prose-anchor examples, sigma 0.0003, and a
minimum domain/anchor cosine of 0.5.  Every run used four TP=1 engines on GPUs
0--3, restored the exact alpha-zero state, and reproduced the strict
validation, OOD-QA, and OOD-prose baselines.

The raw domain/anchor cosine has mean 0.0036823, sample standard deviation
0.2465981, and a two-sided 95% t interval of [-0.1727233, 0.1800879].  Five
directions were positive and five negative; all ten required projection to
the 0.5 cone.  This demonstrates large seed-to-seed direction variance and no
stable positive full-model alignment signal.  These runs never applied an
update and are not alpha-selection or model-quality evidence.  The full-model
family remains closed; the next family is the separately predeclared v4
front/back-versus-middle layer-partition experiment.

`summary.json` binds every source journal, anchor plan, utilization trace, and
reported statistic.  Raw run directories and traces remain ignored working
artifacts; their SHA-256 identities make the compact package auditable.
