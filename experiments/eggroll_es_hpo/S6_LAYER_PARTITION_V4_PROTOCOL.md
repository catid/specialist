# S6 layer-partitioned dense-reward v4 protocol

## Question and frozen comparison

This family tests the owner's front-and-back ensemble prior without changing
the S6 data, validation, or OOD artifacts.  The experimental arm updates dense
weights in layers 0--3 and 36--39; the capacity-matched control updates the
same classes of dense weights in layers 16--23.  Neither arm updates routed
expert tensors.  The sealed heldout split remains forbidden until one fixed
candidate survives replication and every strict guard.

The frozen plan identities are:

- front/back: `experiments/layer_plans/front_back_dense.json`, file SHA-256
  `8e855cbd0d6130278e87b1af348e39dd0f683b8575d9abcb9260f3fe7b29d824`,
  plan SHA-256
  `6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb`
- middle control: `experiments/layer_plans/middle_matched_dense.json`, file
  SHA-256
  `f2b38054e3cdaf41619cce579d3ba2e030fa3cfa87fd42b50543f655ff5f6dc0`,
  plan SHA-256 `b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0`
- model config SHA-256:
  `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`
- front/back checkpoint-to-runtime mapping SHA-256:
  `0a1b84e8ed53ef56c174e7fcac728a4820293505647ab6b9ea02bc86a012b3b1`,
  with runtime-name SHA-256
  `417b3867ba9a56f909d01b1e7bb0b8bb04f903c3ec49438a6675239a7bab270f`
- middle-control checkpoint-to-runtime mapping SHA-256:
  `d6f43de81bb5c41318a38f077b8a3e6272676801752ff68d4772977ac72182f7`,
  with runtime-name SHA-256
  `a7df9257f81c05a3fb3e858209486bd930aad0ddb94d7398e1644b779fb8b70d`
- dense gold-answer reward configuration SHA-256:
  `4941f2e94091b1f8e7ab7b5294ebc6520b80aba1326b7dc6ccea5140a3da5da2`

Each checkpoint plan contains 70 HF tensor units and 285,999,104 parameters.
VLLM losslessly packs each plan into 46 runtime parameters: six selected
linear-attention layers contribute six tensors each and two selected
full-attention layers contribute five each.  The runtime selection must still
contain exactly 285,999,104 parameters (571,998,208 bytes in BF16).  The
mapping, source units, runtime names, shapes, selected identity, immutable
unselected identity, and combined identity must agree on all four engines
before any population perturbation.  Partial packed groups, recurrent
`conv1d`, `shared_expert_gate`/`expert_gate`, and any unplanned tensor are
forbidden.

The domain objective is the mean across examples of each exact gold answer's
mean teacher-forced token log probability.  Prompts use the frozen specialist
template; the answer is appended without an end token.  Tokenizer-prefix
boundary mismatch, truncation, missing selected-token log probabilities,
non-finite values, or a combined length above 1,024 tokens fails the run.
The prose-anchor cone projection remains mandatory.

## Execution stages

1. Run one guarded mechanical smoke per arm with seed 42, population 4, domain
   batch 8, four prose-anchor items, cosine floor 0.5, and targets 0 then
   6.25e-6.  A smoke can validate mechanics but cannot select a model.
2. If and only if both smokes reproduce alpha zero exactly and pass every
   identity/immutability audit, run a paired seed-42 pilot for both arms with
   population 16, domain batch 64, 32 anchor items, cosine floor 0.5, and
   monotonic targets 0, 6.25e-6, 12.5e-6, 25e-6, and 50e-6.
3. Treat pilot states as path-dependent BF16 screening evidence only.  Across
   all arm/alpha states that pass that arm's strict OOD gates, choose the one
   with the largest validation improvement, then the smaller alpha, then the
   lexicographically smaller plan name.  Freeze its alpha before replication
   and test that same alpha in both arms.  If no nonzero state is eligible,
   baseline wins and this family stops.
4. Directly restore alpha zero and test that fixed alpha independently for
   seeds 43--47 in both arms.  No seed-specific alpha changes are allowed.
   Aggregate only after all predeclared replications or an auditable technical
   failure.  Direction-only seeds from the closed full-model family are not
   eligible evidence for this choice.

Every run uses four TP=1 engines on GPUs 0--3 and retains a utilization trace.
Population perturbation, exact restoration, FP32 sharded update, rollback, and
checkpoint export may touch only the selected runtime parameters.  Unselected
weights must retain their frozen origin identity across the complete run.

## Selection gates

A candidate is selectable only if all of the following hold:

- document-disjoint validation improves over its exactly reproduced alpha-zero
  baseline;
- OOD QA exact count, nonzero count, and mean reward do not decrease;
- OOD prose does not decrease, including the predeclared paired-document
  bootstrap lower confidence bound;
- all four workers agree on the complete layer plan, packed mapping, reward
  configuration, coefficient plan, selected and unselected identities, update
  manifest, final state, and commit;
- a direct-alpha replication aggregate, not a monotonic pilot, supports the
  choice without material between-seed instability.

Dataset audit work may continue concurrently, but S6 remains byte-frozen for
this A/B family.  The next curated snapshot is incorporated only after the
family is closed.  Heldout is evaluated once, after the final candidate and
all decision rules are fixed; it is never used for HPO or retry decisions.
