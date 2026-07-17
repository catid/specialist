# Low-Regression Knowledge Fine-Tuning Protocol for Qwen3.6-35B-A3B

## Objective

Fine-tune Qwen3.6-35B-A3B on approximately 1 million tokens of new domain knowledge using LoRA, with the following priorities:

1. Make the model reliably acquire and apply the new knowledge.
2. Preserve the model’s existing reasoning, coding, instruction-following, tool-use, and conversational abilities.
3. Use MLP-only LoRA rather than attention LoRA.
4. Fit within a machine containing four RTX PRO 6000 GPUs.
5. Use supervised fine-tuning as the primary knowledge-transfer mechanism.
6. Use reward-based training only where it provides a clear advantage and has an objective verifier.
7. Produce a merged model or deployment-ready adapter with comprehensive regression measurements.

Assume the GPUs are RTX PRO 6000 Blackwell cards with 96 GB of VRAM each. Confirm this with `nvidia-smi` before selecting the execution topology.

The recommended high-level sequence is:

1. Architecture and memory verification.
2. Small parallel LoRA pilot sweep.
3. One-epoch knowledge SFT with replay data.
4. Verifier-filtered rejection-sampling fine-tuning.
5. Optional short GRPO stage only for objectively verifiable tasks.
6. Comprehensive knowledge and anti-regression evaluation.
7. Adapter-scale selection and final merge.

Do not use RL as the primary mechanism for injecting the million tokens of knowledge.

---

# 1. Core recommendation

Use BF16, MLP-only, expert-aware LoRA.

Train:

* Routed MoE expert MLP weights.
* Shared-expert MLP weights.

Freeze:

* MoE router weights.
* Shared-expert routing/gating weights.
* Attention weights.
* Linear-attention or recurrent-mixing weights.
* Embeddings.
* LM head.
* Vision encoder.
* All normalization parameters unless a specific experiment justifies training them.

Recommended initial LoRA ranks:

* Routed experts: rank 4.
* Shared expert: rank 16.
* LoRA alpha equal to rank.
* LoRA dropout 0.

Recommended main training:

* One epoch over approximately 1 million tokens.
* BF16.
* Sequence length 2,048 initially.
* Packed examples.
* Effective token batch around 2,000 to 4,000 tokens per optimizer update.
* Peak learning rate around 1e-4.
* Five percent warmup.
* Cosine decay.
* Fifteen percent general-behavior replay data.
* Gradient checkpointing enabled.
* Router frozen.

After SFT, perform a verifier-filtered generation and refinement stage rather than immediately performing online RL.

---

# 2. Why SFT should carry the knowledge

Knowledge injection is fundamentally a high-information training problem.

In standard supervised fine-tuning, every target token supplies a training signal. A response containing 500 target tokens can provide hundreds of distinct token-level corrections.

In policy-gradient reinforcement learning, an entire generated response is commonly reduced to one or a small number of scalar rewards or advantages. RL can strongly shape behavior, but it generally provides a much lower-bandwidth learning signal for acquiring a large collection of facts, procedures, definitions, and relationships.

The Thinking Machines LoRA experiments are especially relevant here. Their results suggest:

* LoRA can match full fine-tuning on relatively small post-training datasets when the adapter has enough capacity.
* MLP LoRA can perform approximately as well as combined MLP-plus-attention LoRA in several post-training settings.
* Attention-only LoRA is often weaker than MLP LoRA.
* LoRA can be sensitive to excessively large batches.
* Policy-gradient RL may need surprisingly little adapter rank, possibly because the effective update signal is low dimensional.
* The very low LoRA rank sufficient for RL does not imply that such a low rank is ideal for knowledge-heavy SFT.

Reference:

https://thinkingmachines.ai/blog/lora/

Therefore:

* Use SFT to acquire the knowledge.
* Use reward filtering to improve factual reliability and application.
* Use online RL only to optimize narrowly defined, objectively measurable behavior.

---

# 3. Why BF16 LoRA is preferred over QLoRA

The full 35B parameter model requires approximately 70 GB merely to store BF16 parameters, before accounting for buffers, activations, adapters, and temporary tensors.

A 96 GB GPU may nevertheless fit BF16 LoRA because:

* Base-model gradients are not stored.
* Optimizer states are only stored for LoRA parameters.
* Only a small subset of model parameters is trainable.
* The model is MoE, so activation computation involves only a subset of experts for each token.
* Gradient checkpointing can substantially reduce saved activations.
* A short 2,048-token context is manageable compared with very long-context training.

Use QLoRA only as a fallback when BF16 LoRA does not fit after:

1. Enabling gradient checkpointing.
2. Disabling KV caching.
3. Using batch size one.
4. Reducing sequence length.
5. Removing or excluding the vision encoder.
6. Using efficient attention and MoE kernels.
7. Trying two-GPU FSDP or ZeRO-3.

Reasons not to default to QLoRA:

* The available hardware should provide enough aggregate VRAM.
* MoE quantization support may be less mature than dense-model quantization.
* Quantizing expert tensors can introduce larger behavior changes than desired.
* The goal explicitly prioritizes low regression.
* BF16 gives a cleaner experimental baseline.

QLoRA remains a valid emergency option, but it should not be the first configuration tested.

References:

LoRA:
https://arxiv.org/abs/2106.09685

QLoRA:
https://arxiv.org/abs/2305.14314

Unsloth LoRA hyperparameter guide:
https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

---

# 4. Verify the exact model architecture before attaching LoRA

Do not assume that normal dense-model module targeting will cover the routed experts.

In current Hugging Face implementations of related Qwen MoE architectures, the shared expert may expose ordinary linear modules such as:

* `mlp.shared_expert.gate_proj`
* `mlp.shared_expert.up_proj`
* `mlp.shared_expert.down_proj`

The routed experts may instead be stored as fused parameters such as:

* `mlp.experts.gate_up_proj`
* `mlp.experts.down_proj`

A normal configuration such as:

```python
target_modules = ["gate_proj", "up_proj", "down_proj"]
```

may train the shared expert while entirely missing the routed experts.

Before creating the adapter, print all relevant parameter and module names:

```python
for name, module in model.named_modules():
    if "mlp" in name or "expert" in name:
        print("MODULE", name, type(module).__name__)

for name, parameter in model.named_parameters():
    if "mlp" in name or "expert" in name or "router" in name:
        print(
            "PARAM",
            name,
            tuple(parameter.shape),
            parameter.requires_grad,
        )
```

Inspect the checkpoint configuration:

```python
print(model.config)
print(model.config.model_type)
print(model.config.architectures)
```

Pin the exact Transformers, PEFT, TRL, Accelerate, and Unsloth commits used in the experiment. Model implementation details can change between versions.

Relevant model implementation reference:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

PEFT LoRA documentation:

https://huggingface.co/docs/peft/main/en/developer_guides/lora

Official or expected model card:

https://huggingface.co/Qwen/Qwen3.6-35B-A3B

If the checkpoint’s implementation differs from the linked source, treat the checkpoint’s actual parameter names as authoritative.

---

# 5. Recommended adapter structure

Use separate capacity levels for the routed experts and the shared expert.

Recommended starting values:

```text
Routed expert MLP LoRA:
    rank = 4
    alpha = 4

Shared expert MLP LoRA:
    rank = 16
    alpha = 16

LoRA dropout:
    0.0

Bias training:
    none
```

The routed-expert rank is intentionally small because the adapter is replicated across many experts. Even rank 4 can create hundreds of millions of aggregate trainable LoRA parameters when applied across every layer and every routed expert.

The shared expert processes far more tokens than an individual routed expert, so it receives a higher rank.

The exact trainable parameter count must be printed rather than estimated.

Conceptual PEFT configuration:

```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    # Default rank for normal shared-expert linear modules.
    r=16,
    lora_alpha=16,

    # Match only shared-expert MLP modules.
    target_modules=(
        r".*\.mlp\.shared_expert\."
        r"(gate_proj|up_proj|down_proj)$"
    ),

    # Fused routed-expert parameters.
    target_parameters=[
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ],

    rank_pattern={
        "experts.gate_up_proj": 4,
        "experts.down_proj": 4,
    },

    alpha_pattern={
        "experts.gate_up_proj": 4,
        "experts.down_proj": 4,
    },

    lora_dropout=0.0,
    bias="none",
)
```

This configuration is illustrative. Adjust the names to match the actual checkpoint.

After attaching LoRA, assert the training scope:

```python
trainable = []
unexpected = []

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        trainable.append((name, parameter.numel()))

        allowed = (
            "lora_" in name
            or "base_layer" in name and "lora" in name
        )

        if not allowed:
            unexpected.append(name)

total_trainable = sum(count for _, count in trainable)
total_parameters = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {total_trainable:,}")
print(f"Total parameters:     {total_parameters:,}")
print(f"Trainable fraction:   {total_trainable / total_parameters:.6%}")

for name, count in trainable:
    print(name, count)

assert not unexpected, unexpected
```

Also explicitly assert that none of these are trainable:

```text
router
gate.weight
lm_head
embed_tokens
vision
visual
attention
self_attn
norm
```

Some legitimate LoRA parameter names may contain those strings indirectly, so implement the final assertions based on exact module ownership rather than blindly matching substrings.

---

# 6. Do not train the router initially

Freeze the MoE router during the initial protocol.

Reasons:

* Router drift can alter expert specialization throughout the model.
* A small one-million-token dataset may not contain enough diversity to retrain routing safely.
* Router changes can create broad behavioral regressions even when most model weights are frozen.
* Modified expert outputs can already change later-layer routing indirectly.
* Router training complicates evaluation and attribution.

Even with the router frozen, measure routing behavior before and after fine-tuning.

Track:

* Per-layer expert utilization.
* Router entropy.
* Fraction of tokens assigned to each expert.
* Expert load imbalance.
* Jensen-Shannon divergence between base and adapted routing distributions.
* Frequency of routing collapse or highly dominant experts.

If later experiments show that frozen routing prevents knowledge acquisition, router training can be tested as a separate controlled ablation with a much smaller learning rate.

Do not combine initial expert LoRA experiments with router training.

---

# 7. Dataset construction

The dataset is more important than minor optimizer differences.

## 7.1 Split before generating paraphrases

Split the raw source material by document, entity, event, concept, or atomic fact before generating training examples.

Recommended split:

```text
Training sources:       80%
Validation sources:     10%
Final held-out sources: 10%
```

Do not place a paraphrase of a held-out fact in the training set.

A source-level split is necessary because random example splitting can leak the same fact across train and validation in slightly different wording.

## 7.2 Recommended token allocation

For approximately 1 million total training tokens:

```text
550,000 tokens:
    Closed-book domain questions, explanations, comparisons,
    applications, and procedural tasks.

200,000 tokens:
    Source-grounded answers, evidence-based synthesis,
    conflict resolution, and citation-aware responses.

100,000 tokens:
    High-quality domain continuation or explanatory prose.

150,000 tokens:
    General-behavior replay from the original model.
```

If all 1 million tokens are mandatory domain data, add approximately 100,000 to 150,000 replay tokens rather than deleting domain data.

The final run may therefore contain approximately 1.1 million to 1.15 million tokens.

## 7.3 Represent each important fact in multiple ways

For important knowledge, include several distinct views:

1. Direct explanation.
2. Closed-book question.
3. Application or scenario question.
4. Comparison with a related concept.
5. Causal or mechanistic question.
6. Source-grounded question.
7. Misconception correction.
8. Hard negative or unanswerable form.

Do not rely exclusively on source-in-context examples. A model trained only to answer from supplied text may learn retrieval and extraction without internalizing the knowledge.

## 7.4 Include hard negatives

Approximately 5% to 10% of domain examples should test calibration.

Examples:

* The source does not contain the answer.
* Two sources disagree.
* The question contains a false premise.
* The requested conclusion is not supported.
* A date or version is ambiguous.
* The correct answer requires stating uncertainty.
* The prompt requests a fabricated citation.
* A plausible answer is nevertheless factually false.

These examples should reward:

* Explicit uncertainty.
* Correct qualification.
* Identification of contradictions.
* Refusal to fabricate.
* Requests for missing evidence when genuinely necessary.

## 7.5 Preserve the original interaction format

Use the official chat template for Qwen3.6.

Preserve the model’s expected:

* System-message format.
* User and assistant roles.
* Tool-call format, where relevant.
* Thinking versus non-thinking behavior.
* End-of-turn tokens.
* Special tokens.

Do not add synthetic reasoning traces without validation. Incorrect reasoning traces can teach the model to rationalize wrong answers even when the final answer is correct.

For conversational SFT, mask loss on:

* System tokens.
* User tokens.
* Tool-result tokens when they are input context.
* Padding.

Compute loss primarily on assistant output tokens.

For raw domain continuation examples, use normal causal-language-model loss over the continuation.

---

# 8. Replay data for regression prevention

Construct a replay set representing capabilities that must not regress.

Suggested replay categories:

```text
20% coding and debugging
15% mathematical reasoning
10% structured data and JSON
10% tool use and function calling
10% instruction following
10% ordinary conversation
10% multilingual behavior, if relevant
5% refusal and safety calibration
5% uncertainty and hallucination resistance
5% long-context behavior
```

The replay examples should be high quality.

Preferred sources:

1. Verified examples from a trusted evaluation or internal dataset.
2. Correct original-model responses that pass deterministic checks.
3. Human-written examples.
4. Carefully filtered synthetic examples.

Avoid replaying weak or incorrect generations merely because they came from the base model.

The replay set serves two purposes:

* It preserves important behavior.
* It provides an explicit measurement surface for model drift.

---

# 9. Four-GPU execution strategy

Do not automatically use four-way DDP for the one-million-token SFT run.

Four-way DDP would replicate the full model four times and create a minimum global batch of four sequences. This can reduce the number of optimizer updates, which is undesirable for a small dataset and may worsen LoRA’s large-batch sensitivity.

Use the four GPUs for parallel experiments first.

## 9.1 Initial pilot sweep

Run four independent pilots, each using approximately 150,000 to 200,000 training tokens.

```text
GPU 0:
    Routed rank 2
    Shared rank 16
    Peak LR 1e-4

GPU 1:
    Routed rank 4
    Shared rank 16
    Peak LR 5e-5

GPU 2:
    Routed rank 4
    Shared rank 16
    Peak LR 1e-4

GPU 3:
    Routed rank 4
    Shared rank 16
    Peak LR 2e-4
```

Keep all other variables identical.

Evaluate each checkpoint on:

* Held-out closed-book domain questions.
* Held-out source-grounded questions.
* Domain application questions.
* General replay loss.
* General capability benchmarks.
* Hallucination and abstention tests.
* Router-distribution drift.
* Output format validity.

Select rank 2 instead of rank 4 whenever their results are statistically indistinguishable.

Do not continue the winning pilot into the final run. Restart from the untouched base checkpoint using the selected configuration.

## 9.2 Main-run GPU usage

After selecting the configuration:

```text
GPU 0:
    Full SFT run, seed A.

GPU 1:
    Full SFT run, seed B.

GPU 2:
    Base-model inference, precomputed logits,
    replay generation, or candidate generation.

GPU 3:
    Continuous evaluation and verifier execution.
```

Select the better seed based on the combined knowledge and anti-regression score.

Checkpoint averaging or adapter averaging may be tested only between:

* Adjacent checkpoints.
* Identical adapter structures.
* Runs with similar validation behavior.

Do not average unrelated or diverged adapters blindly.

## 9.3 Fallback when one GPU does not fit

If BF16 LoRA fails to fit on one 96 GB GPU:

1. Verify the vision encoder is not loaded or is fully excluded.
2. Reduce context length to 1,536 or 1,024.
3. Enable more aggressive gradient checkpointing.
4. Use memory-efficient attention.
5. Confirm KV caching is disabled.
6. Reduce shared-expert rank from 16 to 8.
7. Reduce routed rank from 4 to 2.
8. Try two-GPU FSDP or ZeRO-3.
9. Use four-GPU FSDP if necessary.
10. Consider 8-bit optimizer states.
11. Use QLoRA only after the preceding options are tested.

---

# 10. Main SFT hyperparameters

Recommended initial settings:

```text
Precision:
    BF16

Sequence length:
    2,048

Packing:
    Enabled

Per-device sequence batch:
    1

Gradient accumulation:
    1 initially
    2 only if optimization is visibly unstable

Effective token batch:
    Approximately 2,000 to 4,000 tokens per optimizer update

Epochs:
    1.0

Maximum:
    1.5 epochs only if held-out knowledge continues improving
    without general regression

Peak learning rate:
    1e-4 expected default

Learning-rate sweep:
    5e-5
    1e-4
    2e-4

Warmup:
    5% of optimizer steps

Schedule:
    Cosine decay

Final learning rate:
    Approximately 10% of peak

Optimizer:
    AdamW

Optimizer-state precision:
    FP32 by default

Weight decay:
    0.01

Gradient clipping:
    1.0

LoRA dropout:
    0.0

Gradient checkpointing:
    Enabled

KV cache:
    Disabled during training

Router auxiliary loss:
    Disabled unless intentionally studying router training

Checkpoint interval:
    Approximately every 50,000 training tokens
```

The checkpoint interval should be token based rather than epoch based because the entire dataset is small.

Save:

* Adapter weights.
* Optimizer state.
* Scheduler state.
* RNG state.
* Dataset cursor.
* Exact package versions.
* Exact git commits.
* Full configuration.
* Evaluation results.
* Trainable-parameter manifest.

---

# 11. Preservation loss

Start with replay SFT alone.

If the pilot still shows unacceptable general regression, add a KL-based preservation loss on replay examples.

Conceptually:

```text
total_loss =
    domain_supervised_loss
    + replay_supervised_loss
    + lambda_kl * replay_kl_loss
```

Where:

```text
replay_kl_loss =
    KL(base_model_distribution || adapted_model_distribution)
```

Recommended initial value:

```text
lambda_kl = 0.05
```

Possible sweep:

```text
0.0
0.05
0.1
```

Do not apply the preservation KL to new-domain examples. Doing so would explicitly penalize the model for changing its answers in the domain where change is desired.

To avoid simultaneously loading two 35B models into the training process:

1. Run the frozen base model over the replay set.
2. Store selected token logits or compressed top-k distributions.
3. Load those distributions during adapter training.
4. Compute KL only on the replay examples.

Possible storage optimizations:

* Store logits only at assistant target positions.
* Store top-k logits plus an “other” mass.
* Use BF16 or FP16 stored logits.
* Store only the most informative replay examples.
* Apply KL to a sampled subset of replay tokens.

---

# 12. Preferred post-SFT stage: rejection-sampling fine-tuning

After the main SFT run, do not immediately start online RL.

Instead, perform verifier-filtered rejection-sampling fine-tuning.

## 12.1 Candidate generation

For each important training or held-out-style prompt:

* Generate 4 to 8 candidate responses.
* Use moderate sampling diversity.
* Include both thinking and non-thinking modes when relevant.
* Record generation parameters.
* Avoid using final held-out facts during candidate training.

## 12.2 Verification

Use the strongest available verifier.

Preferred verifier types:

1. Unit tests.
2. Exact-answer checks.
3. Formal proof or symbolic checks.
4. Database or source lookup.
5. Schema validation.
6. Tool execution.
7. Citation entailment.
8. Constraint validation.
9. Multiple-model consensus.
10. LLM judge only as a secondary signal.

Reject responses that:

* Contain unsupported factual claims.
* Fabricate citations.
* Fail executable tests.
* Contradict the source.
* Use invalid output formats.
* Answer an unanswerable question confidently.
* Contain correct conclusions supported by incorrect reasoning, when reasoning quality matters.

## 12.3 Refinement dataset

Construct approximately 100,000 to 200,000 tokens of verified refinement data.

Suggested composition:

```text
60% verified domain responses
20% difficult domain failures corrected
20% general replay
```

Train the existing adapter for:

```text
0.25 to 0.5 additional epoch
learning rate 2e-5 to 5e-5
same or smaller token batch
```

This stage provides many of the benefits commonly sought from RL:

* Better factual reliability.
* Better answer selection.
* Improved application of knowledge.
* Reduced hallucination.
* Improved format compliance.

It retains the dense token-level learning signal of SFT and is usually safer than online RL for a small knowledge dataset.

This approach may be described as:

* Rejection-sampling fine-tuning.
* Reward-ranked fine-tuning.
* Verifier-filtered SFT.
* An offline RFT loop.

---

# 13. When true RL is justified

Use online RL only when the reward is objective enough to resist reward hacking.

Good RL tasks:

* Code that must pass tests.
* Mathematical problems with exact answers.
* Structured extraction validated against a schema.
* Tool use with observable success or failure.
* Database-grounded factual retrieval.
* Formal-language generation.
* Constraint satisfaction.
* Simulators with measurable outcomes.

Poor RL tasks:

* “Sound knowledgeable.”
* “Give a good answer.”
* “Write in the preferred style.”
* “Be more intelligent.”
* General factuality judged only by another language model.
* Injecting a large body of new facts.
* Optimizing against one opaque LLM judge.

For weak or subjective rewards, prefer:

* Rejection-sampling SFT.
* Curated preference data.
* A very short DPO-style stage.
* Human review.
* Multi-verifier consensus.

DPO reference:

https://arxiv.org/abs/2305.18290

---

# 14. Optional GRPO protocol

When an objective verifier exists, a short GRPO stage may follow SFT and rejection-sampling refinement.

Before RL:

1. Select the best SFT/refinement adapter.
2. Merge it into a BF16 copy of the base model.
3. Treat the merged model as the new reference policy.
4. Attach a fresh, smaller RL-specific LoRA adapter.

Do not simply leave the SFT adapter attached and disable it for reference-log-probability calculation. In some PEFT and TRL implementations, disabling adapters would recover the original base model rather than the desired SFT reference.

Relevant implementation discussion:

https://github.com/huggingface/trl/pull/6043

Recommended RL adapter:

```text
Routed expert rank:
    1

Shared expert rank:
    4

Alpha:
    Equal to rank

Dropout:
    0
```

Recommended GRPO settings:

```text
Learning rate:
    2e-6 to 5e-6

Default:
    3e-6

Completions per prompt:
    4

Optimizer updates:
    50 to 150

KL coefficient:
    0.02 to 0.05

Maximum completion length:
    1,024 to 2,048 tokens

Loss type:
    dr_grpo when supported and appropriate

Replay:
    One supervised replay/domain minibatch
    approximately every four RL updates
```

Use a nonzero KL penalty because the explicit goal is low regression, even if a framework’s default configuration uses zero KL.

Four-GPU RL topology:

```text
GPU 0:
    Trainable policy with RL LoRA.

GPU 1:
    vLLM or equivalent rollout server.

GPU 2:
    Reward and verifier execution.

GPU 3:
    Frozen evaluation server.
```

Possible alternative:

```text
GPUs 0 and 1:
    FSDP learner.

GPU 2:
    Rollout server.

GPU 3:
    Verifier and evaluation.
```

GRPO references:

TRL GRPO trainer:
https://huggingface.co/docs/trl/main/en/grpo_trainer

DeepSeekMath, which introduced GRPO in this context:
https://arxiv.org/abs/2402.03300

Stop RL immediately when reward improves while any of the following deteriorates:

* General benchmark performance.
* Held-out domain correctness.
* Unsupported-claim rate.
* Response calibration.
* Output length.
* Output entropy.
* Format compliance.
* Router utilization.
* Replay KL.
* Diversity.
* Tool-use reliability.

---

# 15. Do not use DPO to teach the facts

DPO or similar preference optimization can be useful when the available data contains genuine preferred and rejected responses.

It is not the recommended primary knowledge-injection method because:

* Preference pairs contain less direct factual supervision than correct target responses.
* The rejected answer may unintentionally teach false facts.
* Preference optimization can alter response style without improving underlying knowledge.
* The implicit reference-policy pressure complicates new-knowledge acquisition.
* Preference-data quality is difficult to verify.

A small DPO stage may be appropriate after SFT when:

* Human reviewers created reliable preference pairs.
* The preferences concern answer style, calibration, concision, or refusal behavior.
* Both responses are checked for factual validity.
* The stage is short and evaluated for regression.

Do not replace the main SFT stage with DPO.

---

# 16. Evaluation protocol

Create the evaluation suite before training.

Do not change the final held-out suite in response to model failures.

## 16.1 Domain knowledge evaluation

Measure:

* Closed-book direct recall.
* Paraphrased recall.
* Multi-hop reasoning over domain facts.
* Application to new scenarios.
* Comparison and contrast.
* Causal or mechanistic understanding.
* Temporal consistency.
* Source-grounded synthesis.
* Contradiction handling.
* Unanswerable questions.
* False-premise resistance.
* Citation accuracy.
* Confidence calibration.

Report both:

* Exact or verifier-based scores.
* LLM-judge scores, where unavoidable.

Treat deterministic evaluation as higher priority.

## 16.2 General capability evaluation

Include representative tests for:

* Coding.
* Debugging.
* Mathematics.
* General reasoning.
* Instruction following.
* Tool calling.
* JSON and schema adherence.
* Long-context use.
* Multilingual behavior.
* Conversation quality.
* Safety and refusal calibration.
* Uncertainty.
* Hallucination resistance.
* Image understanding if multimodal deployment is expected.

## 16.3 Distributional measurements

Measure:

* Replay-set negative log likelihood.
* Token-level KL between base and adapted outputs.
* Output entropy.
* Mean response length.
* Thinking-token length.
* Refusal rate.
* Unsupported-claim rate.
* Per-layer expert utilization.
* Router entropy.
* Router-distribution divergence.
* Frequency of malformed responses.
* Tool-call validity.

## 16.4 Compare multiple adapter scales

Evaluate the final LoRA adapter at scales:

```text
0.6
0.8
1.0
```

Optionally test:

```text
0.7
0.9
```

A lower adapter scale may preserve most of the new knowledge while reducing general regression.

Choose adapter scale using the held-out composite evaluation, not training loss.

---

# 17. Suggested acceptance gates

Set exact thresholds before training.

Reasonable initial gates:

```text
Domain error:
    At least 20% relative reduction.

Aggregate general benchmark:
    No more than 1 percentage point worse.

Critical individual benchmark:
    No more than 2 percentage points worse.

Replay negative log likelihood:
    No more than approximately 2% worse.

Hallucination rate:
    Must not materially increase.

Unanswerable-question false-answer rate:
    Must not materially increase.

Tool-call validity:
    Must not regress beyond the predefined tolerance.

Router behavior:
    No expert-utilization collapse.

Formatting:
    No increase in malformed chat or tool outputs.

Safety behavior:
    No meaningful degradation on the selected safety set.
```

Use confidence intervals where possible. Do not reject a model because of noise on a tiny evaluation set, and do not accept a model based on a small average gain hiding a major critical regression.

---

# 18. Required ablations

At minimum, compare:

```text
A. Base model, no fine-tuning.

B. Shared-expert-only LoRA:
       shared rank 16.

C. Routed plus shared LoRA:
       routed rank 2,
       shared rank 16.

D. Routed plus shared LoRA:
       routed rank 4,
       shared rank 16.

E. Winning LoRA without replay.

F. Winning LoRA with 15% replay.

G. Winning LoRA with replay and KL anchoring,
   only if ordinary replay is insufficient.

H. SFT-only checkpoint.

I. SFT plus verifier-filtered refinement.

J. Optional SFT plus verifier-filtered refinement plus GRPO.
```

This establishes:

* Whether routed expert training is necessary.
* Whether rank 4 materially improves over rank 2.
* Whether replay prevents regression.
* Whether KL anchoring adds value.
* Whether rejection-sampling refinement improves reliability.
* Whether RL adds anything beyond verified SFT.

---

# 19. Training and evaluation artifacts

For every run, save:

```text
run_config.json
environment.txt
pip_freeze.txt
git_commits.json
model_config.json
adapter_config.json
trainable_parameters.txt
dataset_manifest.json
dataset_hashes.json
training_metrics.jsonl
evaluation_results.json
routing_metrics.json
memory_profile.json
checkpoints/
plots/
samples/
```

The dataset manifest should include:

* Source identifier.
* Split.
* Example type.
* Domain category.
* Token count.
* Whether the example is closed-book.
* Whether it contains source context.
* Whether it is replay.
* Whether it is a hard negative.
* Whether it has a deterministic verifier.
* Generator model, if synthetic.
* Verification status.
* Data lineage.

Do not store sensitive raw source contents in logs unless explicitly permitted.

---

# 20. Memory smoke test

Before the full pilot sweep:

1. Load the model in BF16.
2. Exclude or freeze the vision encoder.
3. Attach the proposed adapter.
4. Create one packed 2,048-token sequence.
5. Run forward and backward.
6. Run one optimizer step.
7. Synchronize CUDA.
8. Record allocated and reserved memory.
9. Repeat for several steps.
10. Confirm there is sufficient headroom for fragmentation and evaluation.

Example instrumentation:

```python
import torch

torch.cuda.reset_peak_memory_stats()

loss = model(**batch).loss
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()

allocated = torch.cuda.max_memory_allocated() / 2**30
reserved = torch.cuda.max_memory_reserved() / 2**30

print(f"Peak allocated: {allocated:.2f} GiB")
print(f"Peak reserved:  {reserved:.2f} GiB")
```

Target at least several GiB of reliable free headroom.

A run that fits with only a few hundred MiB free is not operationally stable.

---

# 21. Implementation order

Use this execution order:

```text
Step 1:
    Pin package and repository versions.

Step 2:
    Inspect exact architecture and parameter names.

Step 3:
    Build immutable train, validation, and held-out splits.

Step 4:
    Construct the replay set.

Step 5:
    Build baseline evaluation results.

Step 6:
    Implement expert-aware LoRA targeting.

Step 7:
    Assert exact trainable parameter scope.

Step 8:
    Run one-GPU BF16 memory smoke test.

Step 9:
    Launch four parallel pilot configurations.

Step 10:
    Evaluate pilots and choose rank and learning rate.

Step 11:
    Restart from the untouched base model.

Step 12:
    Run two full SFT seeds.

Step 13:
    Select the better checkpoint and adapter scale.

Step 14:
    Generate multiple candidate domain answers.

Step 15:
    Verify and filter candidates.

Step 16:
    Run short low-LR refinement SFT.

Step 17:
    Re-evaluate all knowledge and regression gates.

Step 18:
    Run optional GRPO only where objective rewards exist.

Step 19:
    Select final checkpoint and adapter scale.

Step 20:
    Merge adapter into a fresh BF16 checkpoint if deployment
    benefits from eliminating LoRA runtime overhead.

Step 21:
    Re-run final evaluation on the merged model.

Step 22:
    Produce a complete experiment report.
```

---

# 22. Strong final recommendation

For approximately 1 million tokens of new knowledge, the recommended final protocol is:

```text
Model:
    Qwen3.6-35B-A3B

Hardware:
    4x RTX PRO 6000 Blackwell 96 GB

Training precision:
    BF16

Adapter:
    MLP-only expert-aware LoRA

Routed expert rank:
    4 initially
    select rank 2 when it performs equivalently

Shared expert rank:
    16

Router:
    Frozen

Attention:
    Frozen

Vision:
    Frozen or excluded for text-only training

Dataset:
    Approximately 85% domain data
    Approximately 15% general replay

Epochs:
    1.0

Peak learning rate:
    Approximately 1e-4

Context:
    2,048 tokens

Token batch:
    Approximately 2,000 to 4,000 tokens per update

Primary objective:
    Assistant-token supervised cross entropy

Preservation:
    Replay SFT
    optional replay-only KL anchoring

Post-SFT:
    Verifier-filtered rejection-sampling refinement

Online RL:
    Optional
    only for objectively verifiable tasks
    fresh rank-1 routed-expert RL LoRA
    short GRPO run
    small nonzero KL to merged SFT reference
```

The key conceptual rule is:

Use the high-bandwidth supervised objective to teach knowledge. Use verification to select reliable applications of that knowledge. Use RL only to optimize behaviors for which success can be measured objectively.

---

# References

1. Thinking Machines, LoRA research and scaling observations:

https://thinkingmachines.ai/blog/lora/

2. Unsloth LoRA hyperparameter guide:

https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

3. Hugging Face PEFT LoRA documentation:

https://huggingface.co/docs/peft/main/en/developer_guides/lora

4. Hugging Face TRL GRPO trainer documentation:

https://huggingface.co/docs/trl/main/en/grpo_trainer

5. Qwen3.6-35B-A3B model card:

https://huggingface.co/Qwen/Qwen3.6-35B-A3B

6. Related Qwen MoE implementation in Transformers:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

7. LoRA: Low-Rank Adaptation of Large Language Models:

https://arxiv.org/abs/2106.09685

8. QLoRA: Efficient Finetuning of Quantized LLMs:

https://arxiv.org/abs/2305.14314

9. Direct Preference Optimization:

https://arxiv.org/abs/2305.18290

10. DeepSeekMath and Group Relative Policy Optimization:

https://arxiv.org/abs/2402.03300

11. TRL discussion concerning adapter disabling and reference-policy behavior:

https://github.com/huggingface/trl/pull/6043
