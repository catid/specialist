#!/usr/bin/env python3
"""V43I-derived LoRA-ES runtime with direct train generation-boundary F1."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from contextlib import contextmanager
from pathlib import Path

import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_base_generation_evidence_v48b as evidence_runtime
import run_lora_es_multi_anchor_v43i as v43i
import seal_lora_es_generation_boundary_subset_v48b as subset_sealer


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v48b_matched_lora_es_generation_boundary_pop8"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_report_v48b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v48b.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_step1_v48b").resolve()
CALIBRATION_ARTIFACT = (RUN_DIR / "numeric_calibration_v48b.json").resolve()
ANCHOR_CALIBRATION_ARTIFACT = (RUN_DIR / "anchor_calibration_v48b.json").resolve()
RELIABILITY_ARTIFACT = (RUN_DIR / "population_reliability_v48b.json").resolve()
POST_UPDATE_ARTIFACT = (RUN_DIR / "post_update_consensus_v48b.json").resolve()
CANDIDATE_GATE_ARTIFACT = (RUN_DIR / "candidate_gate_v48b.json").resolve()
ABORT_ARTIFACT = (RUN_DIR / "exact_abort_v48b.json").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_generation_boundary_pop8_v48b.json"
).resolve()
SUBSET = subset_sealer.OUTPUT
EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B = (
    "bc58bed21f60f6665b666b12c11d084b0e681b72e160918c871866b31d23ca99"
)
_PREPARED_FRAGILE = None
_SEALED_SUBSET = None

_ORIGINAL_FUSED_REQUESTS = v43i.fused.fused_requests_v43i
_ORIGINAL_ANCHOR_ONLY = v43i.fused.anchor_only_requests_v43i
_ORIGINAL_SCORE_FUSED = v43i.fused.score_fused_outputs_v43i
_ORIGINAL_CANDIDATE_GATE = v43i.fused.candidate_gate_v43i
_ORIGINAL_PREPARE = v43i._prepare


def _compact(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_train_bundle_v48b(*_args) -> dict:
    rows, membership = evidence_runtime.load_train_inputs_v48b()
    weights = [1.0 / (208 * row["row_count"]) for row in rows]
    if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-15, rel_tol=0.0):
        raise RuntimeError("v48b equal-unit weights changed")
    result = {
        "schema": "eggroll-es-equal-unit-train-bundle-v48b",
        "dataset": {
            "path": str(evidence_runtime.TRAIN_DATASET),
            "file_sha256": evidence_runtime.EXPECTED_TRAIN_SHA256,
            "rows": 448,
            "ordered_row_sha256": v43i.v40a.canonical_sha256(
                [row["row_sha256"] for row in rows]
            ),
        },
        "train_membership": {
            "path": str(evidence_runtime.MEMBERSHIP),
            "file_sha256": evidence_runtime.EXPECTED_MEMBERSHIP_SHA256,
            "content_sha256": evidence_runtime.EXPECTED_MEMBERSHIP_CONTENT_SHA256,
            "ordered_membership_sha256": membership[
                "ordered_membership_sha256"
            ],
        },
        "questions": [row["question"] for row in rows],
        "answers": [row["answer"] for row in rows],
        "weights": weights,
        "row_sha256": [row["row_sha256"] for row in rows],
        "conflict_units": 208,
        "weight_identity_sha256": v43i.v40a.canonical_sha256([{
            "row_sha256": row["row_sha256"],
            "unit_identity_sha256": row["unit_identity_sha256"],
            "unit_rows": row["row_count"],
        } for row in rows]),
        "unit_membership_v48b": [{
            "row_sha256": row["row_sha256"],
            "unit_identity_sha256": row["unit_identity_sha256"],
            "row_count": row["row_count"],
        } for row in rows],
    }
    result["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(result)
    return result


def augment_unit_membership_v48b(bundle: dict) -> dict:
    if (
        bundle.get("content_sha256_before_self_field")
        != EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
        or len(bundle.get("unit_membership_v48b", [])) != 448
    ):
        raise RuntimeError("v48b train bundle membership changed")
    result = dict(bundle)
    result["unit_membership_v43i"] = [{
        "unit_identity_sha256": item["unit_identity_sha256"],
        "row_count": item["row_count"],
    } for item in bundle["unit_membership_v48b"]]
    result["unit_membership_sha256_v43i"] = v43i.v40a.canonical_sha256([{
        "row_sha256": row_sha,
        "unit_identity_sha256": item["unit_identity_sha256"],
        "row_count": item["row_count"],
    } for row_sha, item in zip(
        bundle["row_sha256"], bundle["unit_membership_v48b"], strict=True,
    )])
    return result


def load_subset_v48b(path: Path, file_sha: str, content_sha: str) -> dict:
    path = Path(path).resolve()
    if v43i.v40a.file_sha256(path) != file_sha:
        raise RuntimeError("v48b subset file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    subset = value.get("subset", {})
    items = subset.get("items", [])
    request_rows = [item.get("row_sha256") for item in items]
    request_units = [item.get("unit_identity_sha256") for item in items]
    source = value.get("source", {})
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or v43i.v40a.canonical_sha256(_compact(value)) != content_sha
        or value.get("schema") != "sealed-train-generation-boundary-subset-v48b"
        or value.get("status") != "complete_before_population_launch"
        or value.get("selected_rows") != 64
        or value.get("selected_conflict_units") != 64
        or value.get("question_answer_or_generation_text_persisted") is not False
        or value.get("protected_semantics_opened") is not False
        or value.get("shadow_ood_holdout_or_benchmark_opened") is not False
        or subset.get("schema") != "train-generation-boundary-subset-v48a"
        or subset.get("status")
        != "selected_once_from_prepopulation_base_evidence"
        or subset.get("content_sha256_before_self_field")
        != boundary.canonical_sha256_v48a(_compact(subset))
        or len(items) != 64
        or [item.get("request_index") for item in items] != list(range(64))
        or len(set(request_rows)) != 64
        or len(set(request_units)) != 64
        or request_rows != subset.get("request_order_row_sha256")
        or subset.get("request_order_sha256")
        != boundary.canonical_sha256_v48a(request_rows)
        or value.get("request_order_sha256")
        != subset.get("request_order_sha256")
        or value.get("common_random_generation_params")
        != boundary.GENERATION_PARAMS_V48A
        or subset.get("common_random_generation_params")
        != boundary.GENERATION_PARAMS_V48A
        or subset.get("teacher_forced_domain_sampling_changed") is not False
        or subset.get("rows_duplicated_or_oversampled_in_domain_objective")
        is not False
        or any(
            not isinstance(source.get(key), str)
            or len(source[key]) != 64
            for key in (
                "evidence_file_sha256", "evidence_content_sha256",
                "evidence_report_file_sha256",
                "evidence_report_content_sha256",
            )
        )
    ):
        raise RuntimeError("v48b subset content changed")
    return value


def implementation_bindings_v48b(subset_file_sha: str) -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_generation_boundary_preregistration_v48b.py",
        "tests": ROOT / "test_lora_es_generation_boundary_runtime_v48b.py",
        "v43i_runtime": Path(v43i.__file__).resolve(),
        "boundary_runtime": Path(boundary.__file__).resolve(),
        "subset_sealer": Path(subset_sealer.__file__).resolve(),
        "base_evidence_runtime": Path(evidence_runtime.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_v43i.py",
        "train_dataset": evidence_runtime.TRAIN_DATASET,
        "train_membership": evidence_runtime.MEMBERSHIP,
        "source_weights": v43i.SOURCE_WEIGHTS,
        "source_config": v43i.SOURCE_CONFIG,
        "source_manifest": v43i.SOURCE_MANIFEST,
        "staged_weights": v43i.STAGED_WEIGHTS,
        "staged_config": v43i.STAGED_CONFIG,
        "staged_manifest": v43i.STAGED_MANIFEST,
        "prose_anchor": v43i.PROSE_ANCHOR,
        "prose_report": v43i.PROSE_REPORT,
        "qa_anchor": v43i.QA_ANCHOR,
        "qa_report": v43i.QA_REPORT,
        "model_config": v43i.v40a.MODEL / "config.json",
        "model_index": v43i.v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v43i.v40a.TUNED_FILE,
    }
    result = {label: v43i.v40a.file_sha256(path) for label, path in paths.items()}
    result["subset"] = subset_file_sha
    result["model_shards_content_sha256"] = v43i.v40a.MODEL_SHARDS_CONTENT_SHA256
    return result


def load_preregistration_v48b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if v43i.v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v48b preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    recipe = value.get("recipe", {})
    if (
        content != args.preregistration_content_sha256
        or v43i.v40a.canonical_sha256(_compact(value)) != content
        or value.get("schema")
        != "matched-lora-es-generation-boundary-preregistration-v48b"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("gpu_launch_authorized") is not True
        or value.get("protected_semantic_access_authorized") is not False
        or value.get("shadow_ood_holdout_or_benchmark_authorized") is not False
        or recipe.get("dataset") != str(evidence_runtime.TRAIN_DATASET)
        or recipe.get("dataset_sha256") != evidence_runtime.EXPECTED_TRAIN_SHA256
        or recipe.get("train_bundle_content_sha256")
        != EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
        or recipe.get("membership") != str(evidence_runtime.MEMBERSHIP)
        or recipe.get("subset") != str(SUBSET)
        or recipe.get("fused_requests_per_population_actor_state") != 608
        or recipe.get("population_size") != v43i.POPULATION_SIZE
        or recipe.get("seeds") != v43i.SEEDS
        or recipe.get("sigma") != v43i.SIGMA
        or recipe.get("alpha") != v43i.ALPHA
        or recipe.get("worker_extension") != v43i.WORKER_EXTENSION
    ):
        raise RuntimeError("v48b preregistration contract changed")
    subset = load_subset_v48b(
        Path(recipe["subset"]), recipe["subset_file_sha256"],
        recipe["subset_content_sha256"],
    )
    if (
        value.get("implementation_bindings")
        != implementation_bindings_v48b(recipe["subset_file_sha256"])
        or subset["request_order_sha256"] != recipe["request_order_sha256"]
    ):
        raise RuntimeError("v48b implementation or subset binding changed")
    global _SEALED_SUBSET
    _SEALED_SUBSET = subset
    return value


def prepare_v48b(trainer, bundle: dict, anchor_bundle: dict):
    dense, requests, panel, full = _ORIGINAL_PREPARE(
        trainer, bundle, anchor_bundle
    )
    if _SEALED_SUBSET is None:
        raise RuntimeError("v48b subset was not loaded before preparation")
    by_sha = {
        row_sha: (question, answer)
        for row_sha, question, answer in zip(
            bundle["row_sha256"], bundle["questions"], bundle["answers"],
            strict=True,
        )
    }
    items = []
    for index, selected in enumerate(_SEALED_SUBSET["subset"]["items"]):
        row_sha = selected["row_sha256"]
        if row_sha not in by_sha or selected["request_index"] != index:
            raise RuntimeError("v48b subset row missing from exact train bundle")
        question, answer = by_sha[row_sha]
        prompt = v43i.v40a.base.specialist_template(question)
        token_ids = v43i.fused._encode(
            trainer.tokenizer, prompt, v43i.fused.MAX_QA_TOKENS_V43I,
            "V48B fragile generation prompt",
        )
        items.append({
            "row_sha256": row_sha,
            "unit_identity_sha256": selected["unit_identity_sha256"],
            "answer": answer,
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": boundary.canonical_sha256_v48a(token_ids),
        })
    global _PREPARED_FRAGILE
    _PREPARED_FRAGILE = items
    return dense, requests, panel, full


def fused_requests_v48b(domain_requests: list[dict], anchors: dict) -> dict:
    plan = _ORIGINAL_FUSED_REQUESTS(domain_requests, anchors)
    if _PREPARED_FRAGILE is None:
        raise RuntimeError("v48b fragile prompts were not prepared")
    start = len(plan["requests"])
    plan["requests"].extend({
        "prompt_token_ids": item["prompt_token_ids"]
    } for item in _PREPARED_FRAGILE)
    plan["slices"]["fragile_generation"] = [start, len(plan["requests"])]
    if len(plan["requests"]) != len(domain_requests) + 3 * anchors["documents"] + 64:
        raise RuntimeError("v48b fused request coverage changed")
    return plan


def anchor_only_requests_v48b(anchors: dict) -> dict:
    return _ORIGINAL_FUSED_REQUESTS([], anchors)


def sampling_params_for_plan_v48b(plan: dict) -> list:
    teacher = v43i._teacher_sampling_params()
    generation = v43i._generation_sampling_params()
    generation_slices = [plan["slices"]["qa_generation"]]
    if "fragile_generation" in plan["slices"]:
        generation_slices.append(plan["slices"]["fragile_generation"])
    return [
        generation if any(start <= index < stop for start, stop in generation_slices)
        else teacher
        for index in range(len(plan["requests"]))
    ]


def score_fused_outputs_v48b(
    plan: dict, outputs: list, anchors: dict, dense_anchor_module,
    *, domain_scorer=None,
) -> dict:
    if "fragile_generation" not in plan["slices"]:
        return _ORIGINAL_SCORE_FUSED(
            plan, outputs, anchors, dense_anchor_module,
            domain_scorer=domain_scorer,
        )
    start, stop = plan["slices"]["fragile_generation"]
    if stop != len(outputs) or stop - start != 64 or _PREPARED_FRAGILE is None:
        raise RuntimeError("v48b fragile fused output coverage changed")
    base_plan = {
        "requests": plan["requests"][:start],
        "slices": {
            key: value for key, value in plan["slices"].items()
            if key != "fragile_generation"
        },
    }
    result = _ORIGINAL_SCORE_FUSED(
        base_plan, outputs[:start], anchors, dense_anchor_module,
        domain_scorer=domain_scorer,
    )
    metrics = []
    for item, output in zip(_PREPARED_FRAGILE, outputs[start:stop], strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise RuntimeError("v48b fragile completion multiplicity changed")
        prediction = v43i.fused._extract_answer(str(generated[0].text))
        f1 = v43i.fused._f1(prediction, item["answer"])
        answer_tokens = v43i.fused._tokens(item["answer"])
        metrics.append({
            "row_sha256": item["row_sha256"],
            "prediction_sha256": hashlib.sha256(
                prediction.encode("utf-8")
            ).hexdigest(),
            "f1": f1,
            "exact": int(
                bool(answer_tokens)
                and v43i.fused._tokens(prediction) == answer_tokens
            ),
            "nonzero": int(f1 > 0.0),
        })
    result["fragile_generation"] = boundary.score_fragile_items_v48a(
        _SEALED_SUBSET["subset"], metrics
    )
    return result


def compact_signed_score_v48b(score: dict) -> dict:
    result = v43i._compact_signed_score(score)
    result["fragile_generation"] = score["fragile_generation"]
    return result


def replicated_population_v48b(
    trainer, bundle, dense_items, requests, anchors: dict, master_sha: str,
) -> dict:
    numeric = v43i.numeric
    plan = fused_requests_v48b(requests, anchors)
    params = sampling_params_for_plan_v48b(plan)
    if len(plan["requests"]) != 608 or len(params) != 608:
        raise RuntimeError("v48b preregistered fused population size changed")
    scores = {label: [[None] * numeric.SIGNED_REPLICATES_V43G
                      for _ in range(v43i.POPULATION_SIZE)]
              for label in ("plus", "minus")}
    perturbations = {label: [[None] * numeric.SIGNED_REPLICATES_V43G
                             for _ in range(v43i.POPULATION_SIZE)]
                     for label in ("plus", "minus")}
    restorations, assignments, receipts = [], [], []
    for direction in range(v43i.POPULATION_SIZE):
        assignment = numeric.complete_actor_assignments_v43g(direction)
        assignments.append(assignment)
        for label, sign in (("plus", 1), ("minus", -1)):
            values = trainer._resolve([
                trainer.engines[actor].collective_rpc.remote(
                    "materialize_antithetic_adapter_v41a",
                    args=(v43i.SEEDS[direction], v43i.SIGMA, sign, master_sha),
                ) for actor in range(4)
            ])
            if len(values) != 4 or any(len(value) != 1 for value in values):
                raise RuntimeError("v48b perturbation actor coverage changed")
            certificates = [value[0] for value in values]
            batches = None
            try:
                batches = trainer._resolve([
                    trainer.engines[actor].generate.remote(
                        plan["requests"], params, use_tqdm=False,
                        lora_request=v43i._lora_request(),
                    ) for actor in range(4)
                ])
            finally:
                restored = v43i.v40a._rpc_all(
                    trainer, "restore_adapter_master_v41a"
                )
                restorations.extend(restored)
            if len(batches) != 4 or any(len(batch) != 608 for batch in batches):
                raise RuntimeError("v48b complete signed actor block is incomplete")
            for actor_rank, batch in enumerate(batches):
                scored = score_fused_outputs_v48b(
                    plan, batch, anchors, v43i.anchor_v4,
                    domain_scorer=lambda outputs: v43i.score_batch_detailed_v43i(
                        bundle, dense_items, outputs,
                    ),
                )
                scores[label][direction][actor_rank] = {
                    "actor_rank": actor_rank,
                    **compact_signed_score_v48b(scored),
                }
                perturbations[label][direction][actor_rank] = certificates[actor_rank]
                receipts.append({
                    "direction": direction, "sign": label,
                    "actor_rank": actor_rank,
                    "subset_content_sha256": _SEALED_SUBSET["subset"][
                        "content_sha256_before_self_field"
                    ],
                    "request_order_sha256": _SEALED_SUBSET[
                        "request_order_sha256"
                    ],
                    "generation_params": dict(boundary.GENERATION_PARAMS_V48A),
                })
    if any(item is None for label in scores.values()
           for direction in label for item in direction):
        raise RuntimeError("v48b signed score matrix incomplete")
    if any(item["restored_identity"]["sha256"] != master_sha
           for item in restorations):
        raise RuntimeError("v48b signed exact restore changed master")
    for label in ("plus", "minus"):
        for direction in range(v43i.POPULATION_SIZE):
            certs = perturbations[label][direction]
            if (
                len({v43i.v40a.canonical_sha256(item["candidate_identity"])
                     for item in certs}) != 1
                or len({item["materialization"]["runtime_values_sha256"]
                        for item in certs}) != 1
            ):
                raise RuntimeError("v48b replicated signed state differs")
    common = boundary.assert_common_random_plan_v48a(
        receipts, _SEALED_SUBSET["subset"]
    )
    paths = {
        "domain": ("domain", "aggregate", "equal_unit_mean"),
        "prose_lm": ("prose_lm", "mean_token_logprob"),
        "qa_answer_logprob": ("qa_answer_logprob", "mean_example_logprob"),
        "fragile_generation_f1": (
            "fragile_generation", "equal_conflict_unit_mean_f1",
        ),
    }
    sign_scores = {}
    for objective, path in paths.items():
        sign_scores[objective] = {
            label: [[float(v43i._nested_value(
                scores[label][direction][rep], path,
            )) for rep in range(numeric.SIGNED_REPLICATES_V43G)]
                    for direction in range(v43i.POPULATION_SIZE)]
            for label in ("plus", "minus")
        }
    direct = boundary.direct_generation_objective_v48a(
        sign_scores["domain"], sign_scores["fragile_generation_f1"],
        sign_scores["prose_lm"], sign_scores["qa_answer_logprob"],
    )
    return {
        "schema": "fused-generation-boundary-population-v48b",
        "assignments": assignments,
        "signed_scores": scores,
        "objective_sign_scores": sign_scores,
        "objective_fitness": direct["objective_fitness"],
        "central_replicates": v43i._central_replicates(sign_scores["domain"]),
        "coefficients": direct["projection"]["coefficients"],
        "unconstrained_domain_coefficients": direct["objective_fitness"][
            "domain"
        ]["coefficients"],
        "projection": direct["projection"],
        "direct_generation_boundary_objective": direct,
        "common_random_plan": common,
        "fused_requests_per_actor_state": 608,
        "perturbation_certificates": perturbations,
        "restoration_certificate_count": len(restorations),
        "all_exact_restores_passed": True,
    }


def candidate_gate_v48b(reference, candidate, calibration):
    result = _ORIGINAL_CANDIDATE_GATE(reference, candidate, calibration)
    before = [item["fragile_generation"] for item in reference]
    after = [item["fragile_generation"] for item in candidate]
    checks = {
        "fragile_generation_f1_noninferiority": statistics.median([
            right["equal_conflict_unit_mean_f1"]
            - left["equal_conflict_unit_mean_f1"]
            for left, right in zip(before, after, strict=True)
        ]) >= 0.0,
        "fragile_generation_exact_noninferiority": statistics.median([
            right["exact_count"] - left["exact_count"]
            for left, right in zip(before, after, strict=True)
        ]) >= 0.0,
        "fragile_generation_nonzero_noninferiority": statistics.median([
            right["nonzero_count"] - left["nonzero_count"]
            for left, right in zip(before, after, strict=True)
        ]) >= 0.0,
    }
    result = dict(result)
    result["schema"] = "uncommitted-generation-boundary-candidate-gate-v48b"
    result["checks"] = {**result["checks"], **checks}
    result["passed"] = all(result["checks"].values())
    result["fragile_generation_checks"] = checks
    result["content_sha256"] = boundary.canonical_sha256_v48a({
        key: item for key, item in result.items() if key != "content_sha256"
    })
    return result


@contextmanager
def patched_v43i_v48b():
    globals_to_patch = {
        "EXPERIMENT": EXPERIMENT, "RUN_DIR": RUN_DIR, "ATTEMPT": ATTEMPT,
        "REPORT": REPORT, "GPU_LOG": GPU_LOG, "SNAPSHOT": SNAPSHOT,
        "CALIBRATION_ARTIFACT": CALIBRATION_ARTIFACT,
        "ANCHOR_CALIBRATION_ARTIFACT": ANCHOR_CALIBRATION_ARTIFACT,
        "RELIABILITY_ARTIFACT": RELIABILITY_ARTIFACT,
        "POST_UPDATE_ARTIFACT": POST_UPDATE_ARTIFACT,
        "CANDIDATE_GATE_ARTIFACT": CANDIDATE_GATE_ARTIFACT,
        "ABORT_ARTIFACT": ABORT_ARTIFACT,
        "DATASET": evidence_runtime.TRAIN_DATASET,
        "DATASET_SHA256": evidence_runtime.EXPECTED_TRAIN_SHA256,
        "SPLIT_MANIFEST": evidence_runtime.MEMBERSHIP,
        "SPLIT_MANIFEST_SHA256": evidence_runtime.EXPECTED_MEMBERSHIP_SHA256,
        "TRAIN_BUNDLE_SHA256": EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B,
        "load_preregistration": load_preregistration_v48b,
        "augment_unit_membership_v43i": augment_unit_membership_v48b,
        "_prepare": prepare_v48b,
        "_sampling_params_for_plan": sampling_params_for_plan_v48b,
        "_replicated_population": replicated_population_v48b,
    }
    module_saved = {name: getattr(v43i, name) for name in globals_to_patch}
    for name, value in globals_to_patch.items():
        setattr(v43i, name, value)
    fused_patch = {
        "fused_requests_v43i": fused_requests_v48b,
        "anchor_only_requests_v43i": anchor_only_requests_v48b,
        "score_fused_outputs_v43i": score_fused_outputs_v48b,
        "candidate_gate_v43i": candidate_gate_v48b,
    }
    fused_saved = {name: getattr(v43i.fused, name) for name in fused_patch}
    for name, value in fused_patch.items():
        setattr(v43i.fused, name, value)
    old_loader = v43i.equal_v38.load_equal_unit_train_bundle
    v43i.equal_v38.load_equal_unit_train_bundle = load_train_bundle_v48b
    try:
        yield
    finally:
        v43i.equal_v38.load_equal_unit_train_bundle = old_loader
        for name, value in fused_saved.items():
            setattr(v43i.fused, name, value)
        for name, value in module_saved.items():
            setattr(v43i, name, value)


def main(argv: list[str] | None = None) -> int:
    args = v43i.parser().parse_args(argv)
    prereg = load_preregistration_v48b(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "subset_content_sha256": prereg["recipe"][
                "subset_content_sha256"
            ],
            "train_semantics_loaded": False,
            "model_or_gpu_loaded": False,
            "protected_semantic_access_count": 0,
            "shadow_ood_holdout_or_benchmark_opened": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    with patched_v43i_v48b():
        code = v43i.main(argv)
    if code == 0 and REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report.pop("content_sha256_before_self_field", None)
        report["schema"] = "matched-lora-es-generation-boundary-report-v48b"
        report["status"] = "complete_generation_boundary_precommit_gated_state"
        report["recipe"]["fused_requests_per_population_actor_state"] = 608
        report["recipe"]["signed_sequence_presentations"] = (
            2 * v43i.numeric.SIGNED_REPLICATES_V43G
            * v43i.POPULATION_SIZE * 608
        )
        report["recipe"]["fragile_generation_documents"] = 64
        report["recipe"]["generated_f1_direct_primary_objective"] = True
        report["recipe"]["common_random_fragile_request_order"] = True
        report["protected_semantics_opened"] = False
        report["shadow_ood_holdout_or_benchmark_opened"] = False
        report["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(report)
        temporary = REPORT.with_name(f".{REPORT.name}.rewrite")
        if temporary.exists():
            raise FileExistsError(temporary)
        v43i.v40a.atomic_json(temporary, report)
        temporary.replace(REPORT)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
