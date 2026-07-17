# CPT → SFT → mirrored-ES curriculum V1: CPU evidence

Date: 2026-07-17

Bead: `specialist-0j5.5`

Status: the CPU input inventories, stage firewall, equal-compute design,
checkpoint/resume contract, preregistration, and adversarial tests are
complete. The Bead remains in progress because the current source-disjoint
contract authorizes only the exact 448-row V434 file; it does not authorize
the new Markdown CPT inventory or the filtered 433-row SFT view. A fresh
four-domain opaque audit is required before launch. Replicated model results
also remain outstanding.

## Sealed stage inputs

### CPT: raw Markdown only

The canonical source registry contains 33 declared direct-training artifacts.
V1 additionally requires a rights-promotion status, which admits 29 source
documents. Each source contributes one contiguous Qwen3.6-token window, never
packed with another document, with a maximum of 4,845 tokens. The exact stage
budget is 99,216 non-padding tokens; the maximum possible source share is
4.884%.

Four legacy artifacts are excluded because their registry promotion gate still
requires rights review:

| Resource | Available tokens |
| --- | ---: |
| Crash Restraint | 24,764 |
| Rope 365 | 19,168 |
| Rope-topia | 2,552 |
| Shibari Atlas | 1,061,134 |

This excludes 1,107,618 tokens, including the corpus that would otherwise
dominate the entire stage. A legacy `direct_training_ready` label cannot
override its recorded rights gap. CPT units are causal next-token Markdown;
they have no user/assistant wrapper and cannot be presented as assistant
answers.

### SFT: verified instructional QA/chat only

The SFT stage starts from the exact registered V434 train bytes and applies the
already sealed quality rule. It retains 433 verified rows in 210 conservative
conflict units and excludes all 15 `qa_resource_index`/canonical-URL-trivia
rows. One pass contains exactly:

- 21,634 masked prompt tokens;
- 11,438 supervised answer tokens; and
- 33,072 non-padding tokens total.

Answers containing chat protocol markers, empty answers, answers above 128
tokens, sequences above 1,024 tokens, unapproved kinds/schemas, raw-Markdown
roles, or URL-trivia flags fail the build. Conflict units receive bounded
equal-unit treatment; source-balancing multipliers are restricted to `[2/3,
3/2]` and the per-unit multiplier cap is `3/2`.

### ES: registered train reward only

All arms use the same fixed `category_stratified` panel from the sampling
contract: 64 unique conflict units, at most 15 rows from one source, eight
directions, and a signed population of 16. One mirrored update therefore costs
1,024 generated rollouts. Both signs share the registered prompt/decode/judge
conditions and use direct pair differences.

## Equal-compute arm design

Every arm targets 14,400 charged GPU-seconds with a 2% relative tolerance and
must show useful activity on physical GPUs 0–3. Native work is exact; charged
GPU seconds are the cross-family equality authority.

| Arm | Stage | Non-padding tokens | Updates | Rollouts | GPU-second target |
| --- | --- | ---: | ---: | ---: | ---: |
| `cpt_sft_es` | CPT | 99,216 | 96 | 0 | 3,600 |
|  | SFT | 99,216 | 48 | 0 | 3,600 |
|  | ES | 0 | 2 | 2,048 | 7,200 |
| `direct_sft_es` | SFT | 198,432 | 96 | 0 | 7,200 |
|  | ES | 0 | 2 | 2,048 | 7,200 |
| `direct_sft` | SFT | 396,864 | 192 | 0 | 14,400 |
| `direct_es` | ES | 0 | 4 | 4,096 | 14,400 |

The no-CPT arm replaces CPT tokens one-for-one with complete SFT passes. The
single-stage controls spend the complete charged-compute target in their own
training family. A hardware throughput preflight must show every fixed stage
fits its allocation. If it misses, the preregistration must be rebuilt before
any model output; work is never adapted after observing quality.

## Checkpoint and contamination firewall

All arms start from the same content-addressed rank-32 LoRA state and immutable
Qwen3.6 base. A cross-stage transition loads only the exact previous model
state and creates a fresh stage-specific optimizer and scheduler. An
interrupted same-stage resume must match model, optimizer, scheduler, RNG, and
dataloader-cursor identities and form one contiguous checkpoint chain.

The runtime receipt validator requires exact stage input-manifest identities,
native token/update/rollout counts, per-stage and total charged GPU seconds,
all-four-GPU activity, and final checkpoint lineage. It rejects unregistered
fallback checkpoints, skipped or duplicated source units, role leakage,
URL-trivia re-entry, protected access, and altered budgets.

The current parent evaluation contract explicitly says any curriculum refresh
must rebuild its document, normalized-URL, raw-lineage, and near-duplicate
audit against the same opaque protected selection. V1 records the required new
input identities but does not fabricate a passing extension. Launch validation
therefore fails closed while CPU preview validation passes.

## Focused evidence

Command, with CUDA hidden:

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  test_curriculum_ablation_v1.py \
  test_build_curriculum_ablation_preregistration_v1.py \
  test_recipe_evaluation_contract_v1.py \
  test_train_sampling_ablation_v1.py
```

Result: `55 passed in 6.30s`.

The 27 new tests cover stage-role leakage, raw documents disguised as answers,
duplicate CPT documents, duplicate SFT facts/rows, duplicate ES conflict
units, URL trivia, rights gaps, source caps, unequal token and rollout budgets,
unequal charged compute, plan tampering, cross-stage checkpoint substitution,
same-stage optimizer/RNG/dataloader resume drift, native-work drift, missing
GPU activity, and protected access. A synthetic opaque extension demonstrates
that valid execution receipts pass without weakening the real preview gate.

## Artifact identities

- Preregistration:
  `experiments/eggroll_es_hpo/preregistrations/cpt_sft_es_curriculum_ablation_v1.json`
- File SHA-256:
  `d4f36022cc41d381e337b19cbfb3a4136b9c62eb31c359f64594e73628a091fb`
- Canonical content SHA-256:
  `72a7fcdbf658313e6c4cd3d4b282b6f2a2483cd271ff0451a776e06ce4d7e75f`
- Parent evaluation-contract content SHA-256:
  `2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad`
- Markdown-registry file SHA-256:
  `633b191178ee378dcd058b56d2a58b7a76d6e05daad154a4eb93c619fc8ee06f`
- SFT source file SHA-256:
  `ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a`
- ES sampling-contract content SHA-256:
  `18cab815193d05de6e7416b17c1ffeae334a6a613f3899faa459cc719144e97f`

No GPU was launched. No model or adapter tensor was loaded or decoded; sealed
model/adapter metadata and adapter file bytes were hashed for identity. No
dev/OOD semantics, protected content, live run directory, or checkpoint was
opened. The CPU builder opened only the candidate Markdown and registered train
content needed to seal input identities.
