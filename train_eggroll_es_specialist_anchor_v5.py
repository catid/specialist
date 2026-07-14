#!/usr/bin/env python3
"""Document-robust frozen-layer EGGROLL-ES v5 trainer.

V5 preserves the complete v4 worker/update path and replaces only the
train-only prose-anchor fitness.  Each population perturbation is scored by a
fixed paired-document bootstrap lower confidence bound relative to the exact
alpha-zero reference.  OOD, validation, and heldout data are never accepted
by this module.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import eggroll_es_robust_anchor as robust_anchor
import train_eggroll_es_specialist_anchor_v4 as anchor_v4


ROOT = Path(__file__).resolve().parent
REQUIRED_ENGINE_COUNT = anchor_v4.REQUIRED_ENGINE_COUNT
WORKER_EXTENSION = anchor_v4.WORKER_EXTENSION
DOCUMENT_LCB_CONFIG_SHA256_V5 = (
    "da49dd210bf5375cc8c96220744695e5f772546fc55c997efa239053c6498cae"
)
if robust_anchor.DOCUMENT_LCB_CONFIG_SHA256 != DOCUMENT_LCB_CONFIG_SHA256_V5:
    raise RuntimeError("v5 document-LCB implementation differs from protocol")
DOCUMENT_LCB_CONFIG_V5 = robust_anchor.document_lcb_config()
_DEFAULT_LAYER_PLAN_BUNDLE = None


def canonical_sha256(value):
    return anchor_v4.canonical_sha256(value)


def coefficient_sha256(seeds, coefficients):
    return anchor_v4.coefficient_sha256(seeds, coefficients)


def file_sha256(path):
    return anchor_v4.file_sha256(path)


def load_anchor_prose(*args, **kwargs):
    return anchor_v4.load_anchor_prose(*args, **kwargs)


def load_frozen_layer_plan_v4(*args, **kwargs):
    return anchor_v4.load_frozen_layer_plan_v4(*args, **kwargs)


def parse_frozen_layer_plan_cli_v4(*args, **kwargs):
    return anchor_v4.parse_frozen_layer_plan_cli_v4(*args, **kwargs)


def set_default_layer_plan_bundle_v5(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    anchor_v4.set_default_layer_plan_bundle_v4(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def summarize_anchor_documents_v5(items, outputs):
    """Reduce vLLM outputs to train-only per-document numeric summaries."""
    if len(items) != len(outputs):
        raise ValueError("v5 anchor item/output counts differ")
    summaries = []
    prompt_token_logprobs = (
        anchor_v4.anchor_v3.anchor_v2.anchor_v1.base.prompt_token_logprobs
    )
    for item, output in zip(items, outputs):
        document_id = item.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("v5 anchor item has no document identity")
        values = prompt_token_logprobs(output, item["prompt_token_ids"])
        if not values:
            raise ValueError("v5 anchor document has no scored tokens")
        token_sum = math.fsum(float(value) for value in values)
        if not math.isfinite(token_sum):
            raise ValueError("v5 anchor document score is non-finite")
        summaries.append({
            "document_id": document_id,
            "scored_token_count": len(values),
            "sum_token_logprob": token_sum,
        })
    return summaries


def _robust_binding_v5(plan, result):
    binding = {
        "schema": "eggroll-es-document-lcb-plan-binding-v1",
        "config_sha256": result["config_sha256"],
        "document_manifest_sha256": result[
            "document_manifest_sha256"
        ],
        "reference_numeric_summary_sha256": result["reference"][
            "numeric_summary_sha256"
        ],
        "population_numeric_summary_sha256": result[
            "population_numeric_summary_sha256"
        ],
        "bootstrap_plan_sha256": result["bootstrap_plan"][
            "plan_sha256"
        ],
        "robust_scores_sha256": result["robust_scores_sha256"],
        "standardized_scores_sha256": result[
            "standardized_scores_sha256"
        ],
        "result_sha256": result["content_sha256_before_self_field"],
        "coefficient_sha256": plan["coefficient_sha256"],
        "projection_sha256": canonical_sha256(plan["projection"]),
        "layer_plan_sha256": plan["frozen_layer_plan_v4"][
            "plan_sha256"
        ],
    }
    binding["robust_plan_sha256"] = canonical_sha256(binding)
    return binding


def validate_robust_plan_v5(plan, *, recompute_numeric=False):
    """Independently bind robust numeric provenance to v4 coefficients."""
    if not isinstance(plan, dict):
        raise RuntimeError("v5 robust plan is missing")
    result = plan.get("document_lcb_anchor_v5")
    binding = plan.get("robust_plan_binding_v5")
    if (
        not isinstance(result, dict)
        or result.get("schema") != robust_anchor.DOCUMENT_LCB_RESULT_SCHEMA
        or result.get("config") != DOCUMENT_LCB_CONFIG_V5
        or result.get("config_sha256")
        != DOCUMENT_LCB_CONFIG_SHA256_V5
        or result.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in result.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v5 robust objective provenance is invalid")
    if recompute_numeric:
        try:
            robust_anchor.validate_document_lcb_result(result)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "v5 robust objective numeric recomputation failed"
            ) from error
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise RuntimeError("v5 robust plan has no complete seed order")
    robust_rows = result.get("robust_scores")
    if not isinstance(robust_rows, list):
        raise RuntimeError("v5 robust scores are missing")
    score_by_seed = {
        row.get("seed"): row.get("score")
        for row in robust_rows if isinstance(row, dict)
    }
    if set(score_by_seed) != set(seeds) or len(score_by_seed) != len(seeds):
        raise RuntimeError("v5 robust scores do not align with seeds")
    ordered_scores = [score_by_seed[seed] for seed in seeds]
    if plan.get("anchor_scores") != ordered_scores:
        raise RuntimeError("v5 plan did not use document-LCB fitness")
    identity_audit = plan.get("identity_audit")
    pre_probe = (
        identity_audit.get("pre_probe")
        if isinstance(identity_audit, dict) else None
    )
    post_probe = (
        identity_audit.get("post_probe")
        if isinstance(identity_audit, dict) else None
    )
    reference_probe = (
        pre_probe.get("document_lcb_anchor_v5")
        if isinstance(pre_probe, dict) else None
    )
    if (
        not isinstance(identity_audit, dict)
        or identity_audit.get("passed") is not True
        or pre_probe != post_probe
        or not isinstance(reference_probe, dict)
        or reference_probe.get("config_sha256")
        != DOCUMENT_LCB_CONFIG_SHA256_V5
        or reference_probe.get("document_count")
        != result.get("reference", {}).get("document_count")
        or reference_probe.get("scored_token_count")
        != result.get("reference", {}).get("scored_token_count")
        or reference_probe.get("document_manifest_sha256")
        != result.get("document_manifest_sha256")
        or reference_probe.get("reference_numeric_summary_sha256")
        != result.get("reference", {}).get("numeric_summary_sha256")
        or reference_probe.get("raw_document_content_persisted") is not False
    ):
        raise RuntimeError(
            "v5 robust reference is not bound to the exact identity probe"
        )
    projection = (
        anchor_v4.anchor_v3.anchor_v2.anchor_v1
        .project_anchor_safe_coefficients(
            plan.get("domain_scores", []),
            ordered_scores,
            min_anchor_cosine=plan.get("projection", {}).get(
                "min_anchor_cosine"
            ),
        )
    )
    if (
        projection.get("coefficients") != plan.get("coefficients")
        or projection.get("diagnostics") != plan.get("projection")
        or coefficient_sha256(seeds, plan.get("coefficients", []))
        != plan.get("coefficient_sha256")
    ):
        raise RuntimeError("v5 robust projection or coefficients changed")
    if result.get("standardization", {}).get("zero_spread") is True and any(
        float(value) != 0.0 for value in plan["coefficients"]
    ):
        raise RuntimeError("v5 zero-spread objective did not fail closed")
    expected_binding = _robust_binding_v5(plan, result)
    if binding != expected_binding:
        raise RuntimeError("v5 robust plan binding changed")
    if any(
        "document_id" in row
        for row in result.get("reference", {}).get("documents", [])
    ):
        raise RuntimeError("v5 persisted a raw document identity")
    return expected_binding


class DocumentLCBAnchoredMixinV5:
    """Retain document variation and use its bootstrap LCB as fitness."""

    def _persist_anchor_plan(self, plan):
        if (
            getattr(self, "_v5_withhold_unbound_plan", False)
            and not isinstance(plan.get("robust_plan_binding_v5"), dict)
        ):
            return None
        return super()._persist_anchor_plan(plan)

    def configure_anchor(self, *args, **kwargs):
        result = super().configure_anchor(*args, **kwargs)
        if self.anchor_items_per_step != len(self.anchor_items):
            raise ValueError("v5 requires every frozen anchor document")
        if len(self.anchor_items) != 128:
            raise ValueError("v5 frozen training anchor must contain 128 rows")
        if self.min_anchor_cosine != 0.8:
            raise ValueError("v5 anchor cosine is frozen at 0.8")
        self._v5_document_lcb_config = dict(DOCUMENT_LCB_CONFIG_V5)
        self._v5_reference_document_summaries = None
        self._v5_pending_robust_result = None
        return result

    def _anchor_sampling_v5(self, iteration):
        return self._sampling_params(
            n=1,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )

    def _identity_probe(
        self, input_batch, domain_sampling, anchor_items, iteration,
    ):
        del domain_sampling
        probe_count = int(getattr(self, "_v4_identity_probe_count", 0)) + 1
        self._v4_identity_probe_count = probe_count
        if probe_count == 2:
            self._v4_pending_population_boundary_audit = (
                self._population_boundary_audit_v4(iteration)
            )
        elif probe_count > 2:
            raise RuntimeError("v5 identity audit performed an extra probe")
        answers = getattr(self, "_v4_identity_target_answers", None)
        if answers is None:
            raise RuntimeError("v5 identity probe has no gold answers")
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, input_batch, answers,
        )
        if min(len(dense_items), len(self.engines)) != REQUIRED_ENGINE_COUNT:
            raise RuntimeError(
                "v5 identity probe domain batch does not cover all four engines"
            )
        dispatch = anchor_v4.anchor_v3.anchor_v2.anchor_v1.dispatch_eval_batch
        dense_outputs = dispatch(
            self.engines,
            [{"prompt_token_ids": item["prompt_token_ids"]}
             for item in dense_items],
            self._dense_sampling_params_v4(iteration),
            self._resolve,
        )
        dense = anchor_v4.score_gold_answer_outputs_v4(
            dense_items, dense_outputs,
        )
        outputs = dispatch(
            self.engines,
            [{"prompt_token_ids": item["prompt_token_ids"]}
             for item in anchor_items],
            self._anchor_sampling_v5(iteration),
            self._resolve,
        )
        summaries = summarize_anchor_documents_v5(anchor_items, outputs)
        summary_identity = (
            robust_anchor.reference_document_summary_identity(summaries)
        )
        if probe_count == 1:
            self._v5_reference_document_summaries = summaries
        elif probe_count == 2:
            if summaries != self._v5_reference_document_summaries:
                raise RuntimeError(
                    "v5 exact-reference document summaries drifted"
                )
        else:
            raise RuntimeError("v5 robust identity probe count changed")
        return {
            "schema": "eggroll-es-train-only-identity-probe-v5",
            "dense_gold_output_sha256": canonical_sha256(dense),
            "anchor_output_sha256": (
                anchor_v4.anchor_v3.anchor_v2.anchor_output_sha256(
                    anchor_items, outputs,
                )
            ),
            "domain_requests": len(dense_items),
            "anchor_requests": len(anchor_items),
            "reward_config_sha256": self._v4_reward_config_sha256,
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "dispatch": "strided_engine_shards_separate_calls",
            "document_lcb_anchor_v5": {
                "config_sha256": DOCUMENT_LCB_CONFIG_SHA256_V5,
                **summary_identity,
                "raw_document_content_persisted": False,
            },
        }

    def _evaluate_population_with_anchor(
        self, seeds, input_batch, target_batch, domain_sampling_params,
        anchor_items, iteration,
    ):
        del domain_sampling_params
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, input_batch, target_batch,
        )
        dense_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in dense_items
        ]
        dense_sampling = self._dense_sampling_params_v4(iteration)
        anchor_sampling = self._anchor_sampling_v5(iteration)
        anchor_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in anchor_items
        ]
        seeds_perf = {}
        population_documents = {}
        results = []
        for start in range(0, len(seeds), len(self.engines)):
            engine_batch = seeds[start:start + len(self.engines)]
            if len(engine_batch) != len(self.engines):
                raise ValueError("partial v5 population wave would idle a GPU")
            dense_batches = None
            anchor_batches = None
            try:
                self._resolve([
                    self.engines[index].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(int(seed), self.sigma, False),
                    )
                    for index, seed in enumerate(engine_batch)
                ])
                dense_batches = self._resolve([
                    self.engines[index].generate.remote(
                        list(dense_prompts), dense_sampling, use_tqdm=False,
                    )
                    for index, _ in enumerate(engine_batch)
                ])
                if anchor_items:
                    anchor_batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(anchor_prompts), anchor_sampling,
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(engine_batch)
                    ])
            finally:
                self._restore_all_engines_exact()
            if len(dense_batches) != len(engine_batch):
                raise ValueError("v5 dense population engine count changed")
            if anchor_items and len(anchor_batches) != len(engine_batch):
                raise ValueError("v5 anchor population engine count changed")
            for index, seed in enumerate(engine_batch):
                dense = anchor_v4.score_gold_answer_outputs_v4(
                    dense_items, dense_batches[index],
                )
                example_rewards = [
                    item["mean_answer_token_logprob"]
                    for item in dense["examples"]
                ]
                seeds_perf[int(seed)] = {
                    "avg_reward": dense["mean_example_mean_logprob"],
                    "rewards": example_rewards,
                    "results": dense["examples"],
                    "dense_gold_reward_v4": dense,
                }
                results.append({
                    "seed": int(seed),
                    "avg_reward": dense["mean_example_mean_logprob"],
                })
                if anchor_items:
                    outputs = anchor_batches[index]
                    if len(outputs) != len(anchor_items):
                        raise ValueError(
                            "v5 population engine changed anchor request count"
                        )
                    population_documents[int(seed)] = (
                        summarize_anchor_documents_v5(anchor_items, outputs)
                    )
        if anchor_items:
            reference = self._v5_reference_document_summaries
            if not isinstance(reference, list):
                raise RuntimeError("v5 has no exact-reference anchor summary")
            robust_result = robust_anchor.score_population_document_lcbs(
                reference,
                [
                    {"seed": int(seed),
                     "documents": population_documents[int(seed)]}
                    for seed in seeds
                ],
            )
            score_by_seed = {
                row["seed"]: row["score"]
                for row in robust_result["robust_scores"]
            }
            anchor_scores = {
                int(seed): score_by_seed[int(seed)] for seed in seeds
            }
            self._v5_pending_robust_result = robust_result
        else:
            anchor_scores = {}
        return seeds_perf, anchor_scores, results

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        self._v5_reference_document_summaries = None
        self._v5_pending_robust_result = None
        self._v5_withhold_unbound_plan = True
        try:
            plan = super().estimate_step_coefficients(
                iteration, seeds, input_text, target_text,
            )
        finally:
            self._v5_withhold_unbound_plan = False
        result = self._v5_pending_robust_result
        if not isinstance(result, dict):
            raise RuntimeError("v5 population produced no robust objective")
        plan["document_lcb_anchor_v5"] = result
        plan["robust_plan_binding_v5"] = _robust_binding_v5(plan, result)
        validate_robust_plan_v5(plan, recompute_numeric=True)
        self._persist_anchor_plan(plan)
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        validate_robust_plan_v5(plan, recompute_numeric=True)
        return super().apply_seed_coefficients(plan, target_alpha)


def load_trainer(layer_plan_bundle=None):
    """Load the unchanged v4 worker/update path with v5 coordinator logic."""
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured_bundle = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    parent = anchor_v4.load_trainer(captured_bundle)

    class DocumentLCBFrozenLayerTrainerV5(
        DocumentLCBAnchoredMixinV5, parent,
    ):
        pass

    return DocumentLCBFrozenLayerTrainerV5
