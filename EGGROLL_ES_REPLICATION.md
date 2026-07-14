# EGGROLL / ES-at-Scale replication

This is a direct integration of the shared
[`VsonicV/es-at-scale`](https://github.com/VsonicV/es-at-scale) implementation.
The upstream code is retained unchanged as the `es-at-scale` git submodule at
commit `574a9d134da1ffce2a8bb812019899e5c96b588a`. The local launcher supplies the
specialist dataset and Qwen3.6-35B-A3B model while preserving upstream's
full-parameter, one-sided evolution-strategies algorithm:

- persistent Ray/vLLM replicas;
- seed-reconstructed Gaussian perturbations;
- population reward z-scores;
- an FP32 update accumulator on engine 0; and
- NCCL weight broadcast from engine 0 to every replica after each update.

## Reproduce the environment and data

```bash
git submodule update --init --recursive
./setup_eggroll_es.sh
es-at-scale/.venv/bin/python build_eggroll_es_dataset.py
es-at-scale/.venv/bin/pytest -q test_eggroll_es_specialist.py
```

The builder converts `data/train_qa_curated_v1.jsonl` and
`data/eval_qa_v2.jsonl` to the Hugging Face `DatasetDict` layout expected by
the upstream trainer. The current artifact contains 3,258 training examples,
169 heldout evaluation examples, and 236 train-split evaluation examples.
Generated Arrow data is intentionally gitignored.

## Launch the replicated recipe

The default command uses all four GPUs, one persistent model replica per GPU:

```bash
es-at-scale/.venv/bin/python train_eggroll_es_specialist.py
```

The local defaults are Qwen3.6-35B-A3B, four vLLM engines, population 30,
sigma 0.001, alpha 0.0005, batch/mini-batch 200, 128 output tokens, z-score
reward shaping, and 500 as the upstream `num_iterations` value. The upstream
loop currently performs one more update than that value because its stop test
is `iteration > num_iterations`; this integration leaves that behavior intact.

An evaluation-only validation is:

```bash
es-at-scale/.venv/bin/python train_eggroll_es_specialist.py \
  --n-iterations 0 --mini-batch-size 200 --max-tokens 32 \
  --experiment-name qwen36-specialist-upstream-eval-smoke
```

## Necessary Qwen3.6 runtime compatibility

The upstream package pins vLLM 0.11.0 and Transformers 4.57.6. vLLM 0.11
does not register the local checkpoint's `Qwen3_5MoeForConditionalGeneration`
architecture, so the tested environment uses vLLM 0.25.0, Transformers 5.13.1,
Torch 2.11.0, Ray 2.56.0, and Datasets 5.0.0. No upstream source file is
patched. The launcher confines compatibility changes to engine construction:

- aliases two vLLM network helpers that moved after 0.11;
- uses one in-process vLLM executor inside each one-GPU Ray actor, avoiding a
  nested-Ray shared-memory deadlock introduced by the newer vLLM runtime;
- disables unused image/video processing for this text-only task;
- uses the supported Triton MoE backend to avoid a long first-run CUTLASS JIT;
- reserves 82% of each 96 GiB GPU for vLLM, leaving headroom for ES temporary
  tensors; and
- uses eager execution and a 2,048-token engine context.

These affect runtime compatibility and startup, not the ES objective, noise,
reward shaping, update, or synchronization logic.

## Validation on this host

The end-to-end evaluation completed with heldout pass@1 `0.0674359708` and
train-split pass@1 `0.0982221418`. A separate population-4 smoke run completed
the full perturb, generation, restore, FP32 update, NCCL rebroadcast, and final
checkpoint path. During that end-to-end run every GPU reached 100% utilization;
there were 14 one-second samples where all four were active simultaneously.
The population rollout itself included three consecutive samples with all four
at 100%. The retained monitor artifact is
`experiments/gpu_utilization_eggroll_es_train_smoke.jsonl`.

Upstream evaluation intentionally generates only on engine 0. Its parameter
update also runs on engine 0 before broadcasting to the other engines. Thus all
four GPUs are used concurrently for population rollouts, but the exact upstream
design does not keep all four compute-active during evaluation, the centralized
update, or checkpoint serialization.
