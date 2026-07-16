#!/usr/bin/env python3
"""Four-GPU alpha-zero paired evaluator calibration for V61C.

The live path evaluates only the sealed 64-unit ranking panel and four sparse
exact sentinels at one unchanged V434 LoRA state.  Reference/candidate labels
are counterbalanced aliases for that identical state.  No perturbation,
candidate materialization, adapter update, selection, promotion, holdback, or
protected-data access is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import build_lora_es_paired_null_inputs_v61c as inputs
import lora_es_nested_population_v52 as design_v52
import lora_es_paired_null_calibration_v61c as analysis
import qa_quality
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_nested_population_v52 as runtime_v52


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v61c_v434_identical_state_paired_evaluator_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "paired_null_evidence_v61c.json").resolve()
ANALYSIS = (RUN_DIR / "paired_null_analysis_v61c.json").resolve()
REPORT = (RUN_DIR / "paired_null_report_v61c.json").resolve()
FAILURE = (RUN_DIR / "failure_v61c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61c.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v434_identical_state_paired_evaluator_calibration_v61c.json"
).resolve()
STAGED_DATASET = inputs.OUTPUT_DATASET
STAGED_PANEL = inputs.OUTPUT_PANEL
STAGED_DATASET_FILE_SHA256 = (
    "9c1b7f69595cf70ef045259e2097c39546e9f1d84a6b0870fcb14e987655079a"
)
STAGED_PANEL_FILE_SHA256 = (
    "92e0c6160bfc7884a00be4c34c427685dcb2bf5a6aa8c3820f5c53e225f8091c"
)
STAGED_PANEL_CONTENT_SHA256 = (
    "ca0a947e6437c0d84360176087b0a9dab12b79cf6ba1be8f965b24e9f4ec7ba4"
)
WORKER_EXTENSION_V61C = runtime_v52.WORKER_EXTENSION_V52


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def implementation_bindings_v61c() -> dict:
    paths = {
        "runtime_v61c": Path(__file__).resolve(),
        "preregistration_builder_v61c": (
            ROOT / "build_lora_es_paired_null_preregistration_v61c.py"
        ),
        "input_builder_v61c": Path(inputs.__file__).resolve(),
        "analysis_v61c": Path(analysis.__file__).resolve(),
        "tests_v61c": ROOT / "test_lora_es_paired_null_calibration_v61c.py",
        "qa_parser": Path(qa_quality.__file__).resolve(),
        "runtime_v61a": Path(runtime_v61a.__file__).resolve(),
        "design_v52": Path(design_v52.__file__).resolve(),
        "runtime_v52": Path(runtime_v52.__file__).resolve(),
        "anchor_v4": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
        "worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
        "worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
        "worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
    }
    return {
        "code_file_sha256": {
            key: runtime_v61a.file_sha256_v61a(path)
            for key, path in paths.items()
        },
        "pinned_v434_identities_without_reopening": (
            runtime_v61a.implementation_bindings_v61a()[
                "pinned_v52_artifact_identities_without_reopening"
            ]
        ),
        "staged_dataset_file_sha256": STAGED_DATASET_FILE_SHA256,
        "staged_panel_file_sha256": STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": STAGED_PANEL_CONTENT_SHA256,
        "train_semantics_model_gpu_or_protected_paths_opened_to_build_bindings": False,
    }


def _artifacts_v61c() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE),
        "analysis": str(ANALYSIS),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def load_preregistration_v61c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if runtime_v61a.file_sha256_v61a(path) != args.preregistration_sha256:
        raise RuntimeError("v61c preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    recipe = value.get("fixed_calibration_recipe", {})
    access = value.get("access_contract", {})
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or analysis.canonical_sha256_v61c(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "v61c-v434-identical-state-paired-evaluator-preregistration"
        or value.get("status")
        != "preregistered_before_v61c_train_semantics_model_or_gpu_access"
        or value.get("gpu_launch_authorized") is not True
        or value.get("selection_hpo_update_or_promotion_authorized") is not False
        or value.get("eval_ood_shadow_or_holdout_access_authorized") is not False
        or access.get("only_live_semantic_paths_may_open")
        != [str(STAGED_DATASET), str(STAGED_PANEL)]
        or access.get("full_v52_train_or_membership_may_open_live") is not False
        or recipe.get("staged_dataset_file_sha256")
        != STAGED_DATASET_FILE_SHA256
        or recipe.get("staged_panel_file_sha256") != STAGED_PANEL_FILE_SHA256
        or recipe.get("staged_panel_content_sha256")
        != STAGED_PANEL_CONTENT_SHA256
        or recipe.get("rows") != analysis.ROWS_V61C
        or recipe.get("ranking_units") != analysis.RANKING_UNITS_V61C
        or recipe.get("exact_sentinel_units")
        != analysis.EXACT_SENTINEL_UNITS_V61C
        or recipe.get("actors") != analysis.ACTORS_V61C
        or recipe.get("sequential_periods") != analysis.PERIODS_V61C
        or recipe.get("label_plan") != analysis.LABEL_PLAN_V61C
        or recipe.get("request_type_order") != analysis.REQUEST_TYPE_ORDER_V61C
        or recipe.get("pair_periods")
        != [list(pair) for pair in analysis.PAIR_PERIODS_V61C]
        or recipe.get("common_generation_seed")
        != analysis.COMMON_GENERATION_SEED_V61C
        or recipe.get("generation_params_without_seed")
        != analysis.GENERATION_PARAMS_WITHOUT_SEED_V61C
        or recipe.get("teacher_forced_params_without_seed")
        != analysis.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C
        or recipe.get("alpha") != 0.0
        or recipe.get("canonical_fp32_master_sha256")
        != design_v52.MASTER_SHA256_V52
        or recipe.get("bf16_runtime_values_sha256")
        != design_v52.MASTER_RUNTIME_SHA256_V52
        or recipe.get("worker_extension") != WORKER_EXTENSION_V61C
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("implementation_bindings") != implementation_bindings_v61c()
        or value.get("artifacts") != _artifacts_v61c()
        or value.get("raw_question_answer_or_generation_text_may_be_persisted")
        is not False
    ):
        raise RuntimeError("v61c preregistration contract changed")
    return value


def _read_self_hashed_panel_v61c() -> dict:
    if runtime_v61a.file_sha256_v61a(STAGED_PANEL) != STAGED_PANEL_FILE_SHA256:
        raise RuntimeError("v61c staged panel file changed")
    panel = json.loads(STAGED_PANEL.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in panel.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        panel.get("content_sha256_before_self_field")
        != STAGED_PANEL_CONTENT_SHA256
        or analysis.canonical_sha256_v61c(compact)
        != STAGED_PANEL_CONTENT_SHA256
        or panel.get("schema") != "v61c-paired-null-calibration-panel"
        or panel.get("status") != "sealed_cpu_only_before_v61c_preregistration"
        or panel.get("ranking_units") != 64
        or panel.get("exact_sentinel_units") != 4
        or panel.get("holdback_units_in_runtime_dataset") != 0
        or panel.get("holdback_documents_in_runtime_dataset") != 0
        or panel.get("protected_semantics_opened") is not False
        or panel.get("source", {}).get("preview_file_sha256")
        != inputs.PREVIEW_FILE_SHA256
        or panel.get("source", {}).get("preview_content_sha256")
        != inputs.PREVIEW_CONTENT_SHA256
        or panel.get("adaptive_design_provenance") != {
            "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
            "future_candidate_outcomes_used_for_panel_selection": False,
            "protected_or_holdback_outcomes_used": False,
            "train_only_adaptive_design": True,
        }
    ):
        raise RuntimeError("v61c staged panel content changed")
    return panel


def load_staged_inputs_v61c() -> tuple[list[dict], dict]:
    panel = _read_self_hashed_panel_v61c()
    if runtime_v61a.file_sha256_v61a(STAGED_DATASET) != STAGED_DATASET_FILE_SHA256:
        raise RuntimeError("v61c staged dataset changed")
    raw_rows = [
        json.loads(line) for line in STAGED_DATASET.read_text(
            encoding="utf-8"
        ).splitlines() if line
    ]
    items = panel.get("items", [])
    if len(raw_rows) != 68 or len(items) != 68:
        raise RuntimeError("v61c staged row coverage changed")
    prepared = []
    for request_index, (row, item) in enumerate(zip(raw_rows, items, strict=True)):
        row_sha = inputs.row_sha256(row)
        pair = qa_quality.qa_pair_from_record(row)
        if (
            item.get("request_index") != request_index
            or item.get("row_sha256") != row_sha
            or item.get("role")
            != ("ranking" if request_index < 64 else "exact_sentinel")
            or pair is None
            or not isinstance(item.get("unit_identity_sha256"), str)
            or len(item["unit_identity_sha256"]) != 64
        ):
            raise RuntimeError("v61c staged row/panel identity changed")
        question, answer = pair
        prepared.append({
            "request_index": request_index,
            "row_sha256": row_sha,
            "unit_identity_sha256": item["unit_identity_sha256"],
            "role": item["role"],
            "question": question,
            "answer": answer,
        })
    if (
        len({item["row_sha256"] for item in prepared}) != 68
        or len({item["unit_identity_sha256"] for item in prepared}) != 68
        or panel.get("request_order_row_sha256")
        != [item["row_sha256"] for item in prepared]
    ):
        raise RuntimeError("v61c staged order or conflict-unit coverage changed")
    return prepared, panel


def _verify_adapter_artifacts_v61c() -> dict:
    observed = {
        "source_weights": runtime_v61a.file_sha256_v61a(
            design_v52.SOURCE_WEIGHTS_V52
        ),
        "source_config": runtime_v61a.file_sha256_v61a(
            design_v52.SOURCE_CONFIG_V52
        ),
        "staged_weights": runtime_v61a.file_sha256_v61a(
            design_v52.STAGED_WEIGHTS_V52
        ),
        "staged_config": runtime_v61a.file_sha256_v61a(
            design_v52.STAGED_CONFIG_V52
        ),
        "staged_manifest": runtime_v61a.file_sha256_v61a(
            design_v52.STAGED_MANIFEST_V52
        ),
    }
    expected = {
        "source_weights": design_v52.SOURCE_WEIGHTS_SHA256_V52,
        "source_config": design_v52.SOURCE_CONFIG_SHA256_V52,
        "staged_weights": design_v52.STAGED_WEIGHTS_SHA256_V52,
        "staged_config": design_v52.STAGED_CONFIG_SHA256_V52,
        "staged_manifest": design_v52.STAGED_MANIFEST_FILE_SHA256_V52,
    }
    if observed != expected:
        raise RuntimeError("v61c V434 adapter artifacts changed")
    return observed


def _make_trainer_v61c(prereg: dict, prior):
    v40a = prior.v40a
    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = (
        EXPERIMENT, RUN_DIR, WORKER_EXTENSION_V61C,
    )
    try:
        trainer = prior.v40c.make_trainer_v40c(prereg)
    except BaseException:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def _lora_request_v61c(prior):
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v434_identical_state_paired_evaluator_v61c", 1,
        str(design_v52.STAGED_V52), base_model_name=str(prior.v40a.MODEL),
    )


def _generation_params_v61c():
    from vllm import SamplingParams
    return SamplingParams(
        seed=analysis.COMMON_GENERATION_SEED_V61C,
        **analysis.GENERATION_PARAMS_WITHOUT_SEED_V61C,
    )


def _teacher_params_v61c():
    from vllm import SamplingParams
    return SamplingParams(
        seed=analysis.COMMON_GENERATION_SEED_V61C,
        **analysis.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C,
    )


def score_generation_batch_v61c(rows, outputs, fused) -> list[dict]:
    if len(rows) != 68 or len(outputs) != 68:
        raise RuntimeError("v61c generation coverage changed")
    metrics = []
    for row, output in zip(rows, outputs, strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise RuntimeError("v61c completion multiplicity changed")
        prediction = fused._extract_answer(str(generated[0].text))
        f1 = float(fused._f1(prediction, row["answer"]))
        expected = fused._tokens(row["answer"])
        metric = {
            "f1": f1,
            "exact": int(bool(expected) and fused._tokens(prediction) == expected),
            "nonzero": int(f1 > 0.0),
        }
        analysis._generation_metric_v61c(metric)
        metrics.append(metric)
    return metrics


def score_teacher_batch_v61c(dense_items, outputs, anchor_v4) -> list[dict]:
    if len(dense_items) != 68 or len(outputs) != 68:
        raise RuntimeError("v61c teacher-forced coverage changed")
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    examples = dense.get("examples", [])
    if len(examples) != 68:
        raise RuntimeError("v61c dense example coverage changed")
    metrics = []
    for request_index, example in enumerate(examples):
        if example.get("example_index") != request_index:
            raise RuntimeError("v61c dense example order changed")
        metric = {
            "mean_answer_token_logprob": float(
                example["mean_answer_token_logprob"]
            ),
            "answer_token_count": int(example["answer_token_count"]),
            "numeric_example_sha256": analysis.canonical_sha256_v61c(example),
        }
        analysis._teacher_metric_v61c(metric)
        metrics.append(metric)
    return metrics


def _assert_v434_certificates_v61c(certificates: list[dict]) -> dict:
    if len(certificates) != 4:
        raise RuntimeError("v61c certificate actor coverage changed")
    masters = [item.get("current_identity", {}) for item in certificates]
    runtime_hashes = {
        item.get("materialization", {}).get("runtime_values_sha256")
        for item in certificates
    }
    if (
        any(item.get("schema") != "canonical-lora-state-certificate-v52"
            or item.get("transaction_state_quiescent") is not True
            or item.get("active_perturbation") is not None
            or item.get("pending_update") is not None
            or item.get("active_plan_id") is not None
            for item in certificates)
        or
        len({analysis.canonical_sha256_v61c(item) for item in masters}) != 1
        or any(item.get("sha256") != design_v52.MASTER_SHA256_V52
               for item in masters)
        or runtime_hashes != {design_v52.MASTER_RUNTIME_SHA256_V52}
    ):
        raise RuntimeError("v61c exact V434 state changed")
    return {
        "canonical_fp32_master_sha256": masters[0]["sha256"],
        "canonical_master_identity_sha256": analysis.canonical_sha256_v61c(
            masters[0]
        ),
        "four_actor_certificate_sha256": analysis.canonical_sha256_v61c(
            certificates
        ),
        "bf16_runtime_values_sha256": next(iter(runtime_hashes)),
    }


def build_evidence_v61c(rows, periods, state_receipts: list[dict]) -> dict:
    if (
        len(rows) != 68 or len(periods) != 4 or len(state_receipts) != 4
        or any(len(period.get("generation", [])) != 4 for period in periods)
        or any(len(period.get("teacher_forced", [])) != 4 for period in periods)
        or any(len(batch) != 68 for period in periods
               for key in ("generation", "teacher_forced")
               for batch in period[key])
    ):
        raise RuntimeError("v61c evidence coverage changed")
    evidence_rows = []
    for request_index, row in enumerate(rows):
        evidence_rows.append({
            "request_index": request_index,
            "row_sha256": row["row_sha256"],
            "unit_identity_sha256": row["unit_identity_sha256"],
            "role": row["role"],
            "periods": [{
                "period_index": period_index,
                "request_type_order": list(
                    analysis.REQUEST_TYPE_ORDER_V61C[str(period_index)]
                ),
                "actors": [{
                    "actor_rank": actor_rank,
                    "label": analysis.LABEL_PLAN_V61C[str(actor_rank)][
                        period_index
                    ],
                    "generation": periods[period_index]["generation"][
                        actor_rank
                    ][request_index],
                    "teacher_forced": periods[period_index]["teacher_forced"][
                        actor_rank
                    ][request_index],
                } for actor_rank in range(4)],
            } for period_index in range(4)],
        })
    value = {
        "schema": "v61c-identical-state-paired-evaluator-evidence",
        "status": "complete_alpha_zero_no_update_characterization",
        "staged_dataset_file_sha256": STAGED_DATASET_FILE_SHA256,
        "staged_panel_file_sha256": STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": STAGED_PANEL_CONTENT_SHA256,
        "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
        "row_count": 68, "ranking_units": 64, "exact_sentinel_units": 4,
        "actor_count": 4, "period_count": 4,
        "label_plan": dict(analysis.LABEL_PLAN_V61C),
        "request_type_order": dict(analysis.REQUEST_TYPE_ORDER_V61C),
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V61C,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V61C
        ),
        "teacher_forced_params_without_seed": dict(
            analysis.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C
        ),
        "state_receipts": state_receipts,
        "numeric_state_receipts_sha256": analysis.canonical_sha256_v61c(
            state_receipts
        ),
        "rows": evidence_rows,
        "numeric_actor_period_manifest_sha256": analysis.canonical_sha256_v61c(
            evidence_rows
        ),
        "alpha": 0.0,
        "adapter_update_or_candidate_materialization_performed": False,
        "holdback_semantics_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v61c(value)
    )
    analysis.validate_evidence_v61c(value)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v61c(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v61c dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "staged_train_rows_opened": 0,
            "full_v52_train_or_membership_opened": False,
            "model_or_gpu_loaded": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v61c live path requires --execute")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v61c requires fresh artifact paths")

    import run_lora_es_generation_boundary_v48b as v48b
    prior = v48b.v43i
    v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = runtime_v61a.self_hashed_v61a({
        "schema": "v61c-identical-state-paired-evaluator-attempt",
        "status": "launching_alpha_zero_characterization_only",
        "phase": (
            "after_gpu_inventory_preflight_before_staged_train_semantics_"
            "model_load_or_gpu_compute"
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "gpu_inventory_preflight_performed": True,
        "model_loaded_or_gpu_compute_started": False,
        "full_v52_train_membership_holdback_or_protected_opened": False,
    })
    runtime_v61a.atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event(); failures: queue.Queue = queue.Queue()
    phase = v40a.Phase(); started = time.monotonic()
    try:
        adapter_identities = _verify_adapter_artifacts_v61c()
        rows, panel = load_staged_inputs_v61c()
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v61c(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = prior.prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        ); monitor.start()
        request = _lora_request_v61c(prior)
        phase.value = "activate_v434_lora_slot"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v61c four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master"
        installations = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(design_v52.SOURCE_WEIGHTS_V52),
            str(design_v52.SOURCE_CONFIG_V52),
            design_v52.SOURCE_WEIGHTS_SHA256_V52,
            design_v52.SOURCE_CONFIG_SHA256_V52,
        ))
        installed = _assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )
        dense_items = prior.anchor_v4.prepare_gold_answer_items_v4(
            trainer.tokenizer, prompts, [row["answer"] for row in rows],
        )
        teacher_requests = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in dense_items
        ]
        periods = []
        state_receipts = []
        for period_index in range(4):
            before = _assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            period = {"generation": None, "teacher_forced": None}
            for request_type in analysis.REQUEST_TYPE_ORDER_V61C[
                str(period_index)
            ]:
                phase.value = f"period_{period_index}_{request_type}_all_actors"
                if request_type == "generation":
                    batches = trainer._resolve([
                        engine.generate.remote(
                            prompts, _generation_params_v61c(), use_tqdm=False,
                            lora_request=request,
                        ) for engine in trainer.engines
                    ])
                    if len(batches) != 4 or any(len(batch) != 68 for batch in batches):
                        raise RuntimeError("v61c four-actor generation coverage changed")
                    period[request_type] = [
                        score_generation_batch_v61c(rows, batch, prior.fused)
                        for batch in batches
                    ]
                elif request_type == "teacher_forced":
                    batches = trainer._resolve([
                        engine.generate.remote(
                            teacher_requests, _teacher_params_v61c(),
                            use_tqdm=False, lora_request=request,
                        ) for engine in trainer.engines
                    ])
                    if len(batches) != 4 or any(len(batch) != 68 for batch in batches):
                        raise RuntimeError("v61c four-actor teacher coverage changed")
                    period[request_type] = [
                        score_teacher_batch_v61c(
                            dense_items, batch, prior.anchor_v4
                        ) for batch in batches
                    ]
                else:
                    raise RuntimeError("v61c request order changed")
            after = _assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            if before != after or before != installed:
                raise RuntimeError("v61c adapter state changed across a period")
            periods.append(period)
            state_receipts.append({
                "period_index": period_index,
                "before": before,
                "after": after,
                "identical_v434_state": True,
            })
        evidence = build_evidence_v61c(rows, periods, state_receipts)
        null_analysis = analysis.build_analysis_v61c(evidence)

        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v61c GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown(); idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        runtime_v61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime_v61a.atomic_json_v61a(ANALYSIS, null_analysis)
        report = runtime_v61a.self_hashed_v61a({
            "schema": "v61c-identical-state-paired-evaluator-report",
            "status": "complete_content_free_alpha_zero_characterization_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "adapter_artifact_identities": adapter_identities,
            "panel_file_sha256": STAGED_PANEL_FILE_SHA256,
            "panel_content_sha256": STAGED_PANEL_CONTENT_SHA256,
            "panel_document_block_audit": panel["document_block_audit"],
            "master_state_receipt": installed,
            "installations": installations,
            "state_receipts_sha256": evidence["numeric_state_receipts_sha256"],
            "actor_identities": actor_ids,
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
                "rows": 68, "actors": 4, "periods": 4,
                "generation_completions": 1088,
                "teacher_forced_requests": 1088,
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
                "content_sha256": null_analysis[
                    "content_sha256_before_self_field"
                ],
                "teacher_forced_logprob_primary_eligible": null_analysis[
                    "noise_scale_comparison"
                ]["teacher_forced_logprob_primary_eligible"],
                "exact_sentinel_passed": null_analysis[
                    "exact_sentinel"
                ]["passed"],
            },
            "gpu_activity": gpu, "cleanup": cleanup, "final_gpu_idle": idle,
            "gpu_log_file_sha256": runtime_v61a.file_sha256_v61a(GPU_LOG),
            "alpha": 0.0,
            "adapter_update_or_candidate_materialization_performed": False,
            "full_v52_train_membership_or_holdback_opened": False,
            "raw_question_answer_or_generation_text_persisted": False,
            "protected_semantics_opened": False,
        })
        runtime_v61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": runtime_v61a.file_sha256_v61a(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "evidence_file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
            "evidence_content_sha256": evidence["content_sha256_before_self_field"],
            "analysis_file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
            "analysis_content_sha256": null_analysis[
                "content_sha256_before_self_field"
            ],
            "teacher_forced_logprob_primary_eligible": null_analysis[
                "noise_scale_comparison"
            ]["teacher_forced_logprob_primary_eligible"],
            "exact_sentinel_passed": null_analysis["exact_sentinel"]["passed"],
            "all_four_gpus_attributed_positive": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None: monitor.join(timeout=10)
        if not FAILURE.exists():
            failure = runtime_v61a._sanitize_failure_v61a(error)
            failure.update({
                "schema": "v61c-identical-state-paired-evaluator-failure",
                "full_v52_train_membership_holdback_or_protected_opened": False,
                "adapter_update_or_candidate_materialization_performed": False,
            })
            runtime_v61a.atomic_json_v61a(
                FAILURE, runtime_v61a.self_hashed_v61a(failure)
            )
        raise
    finally:
        if trainer is not None:
            try: v40a.base.close_trainer(trainer)
            except Exception: pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            prior.v40a.EXPERIMENT, prior.v40a.RUN_DIR, prior.v40a.WORKER_EXTENSION = saved


if __name__ == "__main__":
    raise SystemExit(main())
