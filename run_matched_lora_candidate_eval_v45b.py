#!/usr/bin/env python3
"""V45B: add successful V43G to sealed OOD-first matched evaluation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as prior
import stage_v43g_adapter_vllm_v45b as v43g_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V45B = (
    "base_a", "base_b", "base_c", "base_d", "base_e", "base_f"
)
CANDIDATE_ARMS_V45B = prior.CANDIDATE_ARMS_V45A + ("lora_es_v43g",)
ARMS_V45B = BASE_ARMS_V45B + CANDIDATE_ARMS_V45B
STAGED_BY_ARM_V45B = dict(prior.STAGED_BY_ARM_V45A)
STAGED_BY_ARM_V45B["lora_es_v43g"] = v43g_stage.OUTPUT
ADAPTER_IDS_V45B = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V45B)
}

EXPERIMENT = "v45b_matched_lora_hpo_v43g_ood_eligible_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v45b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v45b.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_hpo_v43g_ood_eligible_eval_v45b.json"
).resolve()
V45A_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_hpo_earlystop_ood_eligible_eval_v45a.json"
).resolve()
V45A_PREREG_FILE_SHA256 = (
    "76ee0170a157155e02eb8924d04075e6a2b82bbc604ff4779072ef340e409385"
)
V45A_PREREG_CONTENT_SHA256 = (
    "a962b29a0dcdd8af91bbe1e9d775a93e09596734aa972b5d7d96ec9d887af2c0"
)
V45A_REPORT = prior.REPORT
V45A_REPORT_FILE_SHA256 = (
    "64f0edab64c66aa208a03759f759a3fe62d3bc886e6fce55180abe7b57cb919d"
)
V45A_REPORT_CONTENT_SHA256 = (
    "419c5855dbbaf7365d259a22d588e58ce593d340efb502b1c2682067d9b57c7b"
)


def arm_wave_plan_v45b() -> tuple[tuple[tuple[str, int], ...], ...]:
    return (
        (("base_a", 0), ("base_b", 1), ("base_c", 2), ("base_d", 3)),
        (("base_e", 0), ("base_f", 1),
         ("sft_v42b_step16", 2), ("sft_v42b_step32", 3)),
        (("sft_v42b", 0), ("sft_v42c", 1),
         ("sft_v42d", 2), ("lora_es_v43d", 3)),
        (("sft_v42e", 0), ("sft_v42f", 1),
         ("sft_v42g", 2), ("lora_es_v43g", 3)),
    )


@contextmanager
def patched_prior_constants_v45b():
    saved = {
        "BASE": prior.BASE_ARMS_V45A,
        "CANDIDATE": prior.CANDIDATE_ARMS_V45A,
        "ARMS": prior.ARMS_V45A,
        "STAGED": prior.STAGED_BY_ARM_V45A,
        "IDS": prior.ADAPTER_IDS_V45A,
        "wave": prior.arm_wave_plan_v45a,
    }
    prior.BASE_ARMS_V45A = BASE_ARMS_V45B
    prior.CANDIDATE_ARMS_V45A = CANDIDATE_ARMS_V45B
    prior.ARMS_V45A = ARMS_V45B
    prior.STAGED_BY_ARM_V45A = STAGED_BY_ARM_V45B
    prior.ADAPTER_IDS_V45A = ADAPTER_IDS_V45B
    prior.arm_wave_plan_v45a = arm_wave_plan_v45b
    try:
        yield
    finally:
        prior.BASE_ARMS_V45A = saved["BASE"]
        prior.CANDIDATE_ARMS_V45A = saved["CANDIDATE"]
        prior.ARMS_V45A = saved["ARMS"]
        prior.STAGED_BY_ARM_V45A = saved["STAGED"]
        prior.ADAPTER_IDS_V45A = saved["IDS"]
        prior.arm_wave_plan_v45a = saved["wave"]


def staged_adapter_bindings_v45b() -> dict:
    with patched_prior_constants_v45b():
        value = prior.staged_adapter_bindings_v45a()
    if tuple(value) != CANDIDATE_ARMS_V45B:
        raise RuntimeError("V45B staged candidate order changed")
    return value


def prior_preregistration_v45b() -> dict:
    if core.file_sha256(V45A_PREREG) != V45A_PREREG_FILE_SHA256:
        raise RuntimeError("V45B V45A preregistration file changed")
    value = json.loads(V45A_PREREG.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V45A_PREREG_CONTENT_SHA256
        or core.canonical_sha256(compact) != V45A_PREREG_CONTENT_SHA256
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("cpu_preflight_expected", {}).get(
            "holdout_or_heldout_opened"
        ) is not False
    ):
        raise RuntimeError("V45B V45A preregistration content changed")
    return value


def prior_result_v45b() -> dict:
    if core.file_sha256(V45A_REPORT) != V45A_REPORT_FILE_SHA256:
        raise RuntimeError("V45B V45A aggregate file changed")
    value = json.loads(V45A_REPORT.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V45A_REPORT_CONTENT_SHA256
        or core.canonical_sha256(compact) != V45A_REPORT_CONTENT_SHA256
        or value.get("status") != "complete_aggregate_only_no_heldout_access"
        or value.get("selection", {}).get("selected_arm") != "sft_v42g"
        or value.get("final_gate", {}).get("passed") is not True
        or value.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("V45B V45A aggregate content changed")
    return {
        "path": str(V45A_REPORT),
        "file_sha256": V45A_REPORT_FILE_SHA256,
        "content_sha256": V45A_REPORT_CONTENT_SHA256,
        "selected_arm": "sft_v42g",
        "strict_gate_passed": True,
        "heldout_or_holdout_opened": False,
        "aggregate_only_no_raw_semantics_inspected_for_v45b": True,
    }


def implementation_bindings_v45b() -> dict:
    source_paths = {
        "source_weights": v43g_stage.SOURCE / "adapter_model.safetensors",
        "source_config": v43g_stage.SOURCE / "adapter_config.json",
        "source_report": v43g_stage.REPORT,
        "source_preregistration": v43g_stage.PREREG,
        "source_attempt": v43g_stage.ATTEMPT,
        "source_calibration": v43g_stage.CALIBRATION,
        "source_population_reliability": v43g_stage.RELIABILITY,
        "source_post_update_consensus": v43g_stage.CONSENSUS,
        "staged_weights": v43g_stage.OUTPUT / "adapter_model.safetensors",
        "staged_config": v43g_stage.OUTPUT / "adapter_config.json",
        "stage_manifest": v43g_stage.OUTPUT / "stage_manifest_v44a.json",
    }
    return {
        "runtime": core.file_sha256(Path(__file__).resolve()),
        "v45a_runtime": core.file_sha256(Path(prior.__file__).resolve()),
        "v45a_preregistration_file_sha256": V45A_PREREG_FILE_SHA256,
        "v45a_preregistration_content_sha256": V45A_PREREG_CONTENT_SHA256,
        "v45a_aggregate": prior_result_v45b(),
        "v45a_implementation": prior.implementation_bindings_v45a(),
        "v43g_staging_runtime": core.file_sha256(
            Path(v43g_stage.__file__).resolve()
        ),
        "v43g_source_seal": v43g_stage.source_seal_v45b(),
        "v43g_artifacts": {
            label: core.file_sha256(path) for label, path in source_paths.items()
        },
    }


def load_preregistration_v45b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V45B preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256(compact)
        or value.get("schema")
        != "matched-lora-v43g-ood-eligible-eval-preregistration-v45b"
        or value.get("status")
        != "preregistered_before_fresh_v43g_extended_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45b_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V45B)
        or value.get("candidate_arms") != list(CANDIDATE_ARMS_V45B)
        or value.get("staged_adapters") != staged_adapter_bindings_v45b()
        or value.get("implementation_bindings") != implementation_bindings_v45b()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V45B preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    prior.environment.environment_bindings_v44b()
    saved = {
        "EXPERIMENT": core.EXPERIMENT, "RUN_DIR": core.RUN_DIR,
        "ATTEMPT": core.ATTEMPT, "RAW": core.RAW,
        "GPU_LOG": core.GPU_LOG, "REPORT": core.REPORT,
        "load": core.load_preregistration,
        "preflight": core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A,
        "select": core.select_candidate_v44a,
        "eval_qa": core.evaluate_qa_v44a,
        "eval_prose": core.evaluate_prose_v44a,
    }
    with patched_prior_constants_v45b():
        state = prior.EvaluationStateV45A()

        def evaluate_qa(trainer, bundle, raw_sink, label):
            result = saved["eval_qa"](trainer, bundle, raw_sink, label)
            baseline = result["base_a"]
            if any(result[arm] != baseline for arm in BASE_ARMS_V45B[1:]):
                raise RuntimeError(f"V45B six-base equivalence failed on {label}")
            if label == "ood_qa":
                state.ood_qa = result
            return result

        def evaluate_prose(trainer, rows, raw_sink):
            aggregate, detailed = saved["eval_prose"](trainer, rows, raw_sink)
            baseline = aggregate["base_a"]
            if any(aggregate[arm] != baseline for arm in BASE_ARMS_V45B[1:]):
                raise RuntimeError("V45B six-base prose equivalence failed")
            final = prior.finalize_selection_v45a(
                state.shadow, state.ood_qa, detailed, raw_sink
            )
            final.pop("all_three_base_duplicates_required_exact", None)
            final.update({
                "all_six_base_duplicates_required_exact": True,
                "padding_base_arms": ["base_d", "base_e", "base_f"],
                "padding_base_arms_excluded_from_candidate_eligibility_and_ranking": True,
            })
            state.selection.clear()
            state.selection.update(final)
            return aggregate, detailed

        core.EXPERIMENT = EXPERIMENT
        core.RUN_DIR = RUN_DIR
        core.ATTEMPT = ATTEMPT
        core.RAW = RAW
        core.GPU_LOG = GPU_LOG
        core.REPORT = REPORT
        core.load_preregistration = load_preregistration_v45b
        core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = (
            prior.parser_fix.protected_preflight_v44c
        )
        core.select_candidate_v44a = state.provisional
        core.evaluate_qa_v44a = evaluate_qa
        core.evaluate_prose_v44a = evaluate_prose
        try:
            with prior.patched_candidate_globals_v45a():
                return core.main(argv)
        finally:
            core.EXPERIMENT = saved["EXPERIMENT"]
            core.RUN_DIR = saved["RUN_DIR"]
            core.ATTEMPT = saved["ATTEMPT"]
            core.RAW = saved["RAW"]
            core.GPU_LOG = saved["GPU_LOG"]
            core.REPORT = saved["REPORT"]
            core.load_preregistration = saved["load"]
            core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = saved["preflight"]
            core.select_candidate_v44a = saved["select"]
            core.evaluate_qa_v44a = saved["eval_qa"]
            core.evaluate_prose_v44a = saved["eval_prose"]


if __name__ == "__main__":
    raise SystemExit(main())
