# Qwen3.6 EGGROLL / ES-at-Scale HPO

The frozen S3 experiment selected a three-update ES treatment with
`sigma=0.002`, `alpha=0.001`, population 8, batch size 64, seed 42, and 32
generated tokens. It improved mean answer reward numerically on both the
236-item validation split and the sealed 169-item holdout, but the paired
intervals include zero and exact-match counts did not improve. This is a
promising smoke result, not strong evidence that the checkpoint is generally
better.

## Final result

| Split | Base | Treatment | Delta | Exact (base → treatment) | Nonzero (base → treatment) | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (`train`) | 0.097673 | 0.099614 | +0.001941 (+1.99%) | 14 → 14 | 93 → 99 | [-0.000769, +0.004744] |
| Holdout | 0.067436 | 0.072231 | +0.004795 (+7.11%) | 7 → 7 | 54 → 60 | [-0.010359, +0.021285] |

On validation, the treatment won 22 items, lost 15, and tied 199. On holdout,
it won 17, lost 15, and tied 137. The metric is the repository's normalized
answer-token overlap with exact-match credit, not binary accuracy. See
[`final_results.json`](final_results.json),
[`final_validation_comparison.json`](final_validation_comparison.json), and
[`final_holdout_comparison.json`](final_holdout_comparison.json) for the exact
machine-readable results.
The paired comparison files embed aligned per-example reward/exactness records
keyed by prompt/answer hashes. Their wins, ties, means, and bootstrap intervals
therefore remain auditable without the ignored raw generation arrays.

The selected 69,321,427,447-byte checkpoint is stored locally at
`runs/final_selected/checkpoint-es_exact_steps_3/pytorch_model.pth` with SHA-256
`552b8cc091323321cd55647dbb967a165d9c1a8c785bd3544127e9579aecc30b`.
Model weights are intentionally gitignored.

## Final-snapshot HPO

The final grid used the 2,228-row training Arrow artifact with SHA-256
`bb60372725825f2fc81b46b681899ed8b4ba1af79d10ab1e6905bae5fb660f6f`.
The source JSONL hash is
`ea178be3d1052000095cde77a5c4b1b8b93130bb3b16fafdd85d0a107a7edf4d`.
The holdout was not used for HPO or horizon selection.

| Sigma | Alpha | Steps | Validation reward | Delta vs base |
| ---: | ---: | ---: | ---: | ---: |
| 0 (base) | 0 | 0 | 0.097673 | — |
| 0.0003 | 0.00015 | 3 | 0.094959 | -0.002714 |
| 0.0005 | 0.00025 | 3 | 0.095437 | -0.002236 |
| 0.001 | 0.00025 | 3 | 0.094042 | -0.003631 |
| 0.001 | 0.0005 | 3 | 0.093811 | -0.003862 |
| 0.001 | 0.001 | 3 | 0.093349 | -0.004324 |
| **0.002** | **0.001** | **3** | **0.099614** | **+0.001941** |
| 0.002 | 0.001 | 6 | 0.094007 | -0.003666 |

The complete journal is
[`snapshot2228_grid/hpo_results.json`](snapshot2228_grid/hpo_results.json).
The final retrain reproduced the selected validation score exactly before the
holdout was opened.

## Dataset refresh history

Training never read a changing JSONL. Each foreground run consumed an immutable
Arrow snapshot, and manual review tranches were promoted only between A/B
runs.

| Snapshot | Rows | Training Arrow SHA-256 | Relevant result |
| --- | ---: | --- | --- |
| S1 | 2,915 | `807d423724508e5cdee2d9966e5677fb6dcd88fb28d9cc04200c8875725711a3` | Fixed 3-step grid best: 0.098340 at sigma 0.0005 / alpha 0.00025; 6-step probe: 0.101355. |
| S2 | 2,349 | `ca9403139d0094116bb1c86391f7fab8c85bf1fbd8eefe128d55638c22814c63` | Same inherited setting: 0.094837 at 3 steps, 0.098036 at 6 steps. |
| S3 (completed A/B) | 2,228 | `bb60372725825f2fc81b46b681899ed8b4ba1af79d10ab1e6905bae5fb660f6f` | Fresh grid selected sigma 0.002 / alpha 0.001 at 3 steps: 0.099614. |
| S4 (refresh probe) | 1,487 | `ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c` | The inherited S3 winner scored 0.094691 versus 0.097673 baseline; baseline retained. |

The compact run summaries for the S1 one-/six-step and S2 three-/six-step
probes are retained in [`probes/`](probes/). The complete immutable S3 source
JSONL, curation ledger, build report, manifest, and Arrow file are retained in
[`snapshots/s3/`](snapshots/s3/) so a later active-dataset refresh cannot make
the reported A/B cohort disappear.
The hash-guarded S4 transfer probe and self-contained paired comparison are in
[`snapshot1487_probe/`](snapshot1487_probe/). It was deliberately limited to
the prior S3 winner and did not open the holdout.

This instability is itself a finding: ES hyperparameters and the apparent best
horizon changed after manual removal of noisy/contextless rows. Results from
different dataset hashes must not be pooled as if they were replications.

The validation split is not document-disjoint from training. A later manual
audit of the first 300 retained Rope365 rows found 42 training rows whose source
URLs also occur in the `train` validation split. No such URL overlap was found
with the sealed `heldout` split in those tranches. This makes validation/HPO
scores more optimistic and reinforces that the small deltas are exploratory;
the pending Rope365 ledgers will remove those rows in a future snapshot.

## GPU verification

The clean final trace contains 141 samples from startup through checkpoint
serialization. All four GPUs were at least 20% utilized together for 21 samples
and exactly 100% for nine samples. When active, per-GPU utilization averaged
93.3%, 86.1%, 93.0%, and 91.6%; peak memory was 96,639 MiB on GPU 0 and 82,346
MiB on GPUs 1–3.

The faithful upstream design does not keep all devices compute-active in every
phase: population rollouts use all four one-GPU replicas, while evaluation,
the FP32 update, broadcasting, and checkpoint serialization are coordinated or
performed from engine 0. Whole-run means are therefore only 33.1%, 14.4%,
16.7%, and 17.2%, and should not be misrepresented as a utilization-guard pass.
The raw trace is
[`../gpu_utilization_eggroll_es_final.jsonl`](../gpu_utilization_eggroll_es_final.jsonl).
The longer multi-trial HPO trace is retained separately as
[`../gpu_utilization_eggroll_es_hpo.jsonl`](../gpu_utilization_eggroll_es_hpo.jsonl)
(SHA-256 `72317201d8c4713980ceadc8ae12c0bd1450322b83aae1aeedecf0bd17c4dbdc`).
It includes model startup, evaluation, teardown, and idle gaps between trials,
so it is telemetry rather than evidence that a continuous utilization guard
passed.

## Reproduce

Build from the frozen S3 JSONL (not the newer moving dataset), rerun the grid,
and retrain the selected candidate:

```bash
es-at-scale/.venv/bin/python build_eggroll_es_dataset.py \
  --train-jsonl experiments/eggroll_es_hpo/snapshots/s3/train_qa_curated_v1.jsonl \
  --output /tmp/specialist-eggroll-s3
es-at-scale/.venv/bin/python run_eggroll_es_hpo.py \
  --train-dataset /tmp/specialist-eggroll-s3/train \
  --eval-dataset /tmp/specialist-eggroll-s3/eval \
  --output /tmp/specialist-eggroll-s3-hpo --force \
  --steps 3 --population-size 8 --batch-size 64 --max-tokens 32 --seed 42
es-at-scale/.venv/bin/python train_eggroll_es_specialist_s3.py \
  --train-dataset /tmp/specialist-eggroll-s3/train \
  --eval-dataset /tmp/specialist-eggroll-s3/eval \
  --exact-train-steps 3 --skip-baseline-eval --eval-splits train,heldout \
  --sigma 0.002 --alpha 0.001 --population-size 8 \
  --batch-size 64 --mini-batch-size 64 --max-tokens 32 --seed 42 \
  --output-directory /tmp/specialist-eggroll-s3-final \
  --experiment-name final_selected --save-final-checkpoint
```

The upstream `es-at-scale` submodule remains unchanged at
`574a9d134da1ffce2a8bb812019899e5c96b588a`.
The current HPO driver invokes the root trainer, whose only post-S3 behavioral
change is the seed-zero fix; all commands above use seed 42. The explicit final
retrain command uses the root-level byte-exact S3 replay source (kept at the
repository root so its path-derived upstream/model locations remain valid).
