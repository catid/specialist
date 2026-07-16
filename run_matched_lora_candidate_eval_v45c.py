#!/usr/bin/env python3
"""V45C: add matched 6e-5 V42H to sealed V45B OOD-first evaluation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45b as prior
import stage_v42h_adapter_vllm_v45c as v42h_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V45C = tuple(f"base_{letter}" for letter in "abcdefghi")
CANDIDATE_ARMS_V45C = prior.CANDIDATE_ARMS_V45B + ("sft_v42h",)
ARMS_V45C = BASE_ARMS_V45C + CANDIDATE_ARMS_V45C
STAGED_BY_ARM_V45C = dict(prior.STAGED_BY_ARM_V45B)
STAGED_BY_ARM_V45C["sft_v42h"] = v42h_stage.OUTPUT
ADAPTER_IDS_V45C = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V45C)
}
PADDING_BASE_ARMS_V45C = tuple(f"base_{letter}" for letter in "defghi")

EXPERIMENT = "v45c_matched_lora_hpo_v42h_v43g_ood_eligible_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v45c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v45c.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_hpo_v42h_v43g_ood_eligible_eval_v45c.json"
).resolve()
V45B_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_hpo_v43g_ood_eligible_eval_v45b.json"
).resolve()
V45B_PREREG_FILE_SHA256 = (
    "c16782e3f9ca66d9438d4194b23fde2277fdc5af624d359a56ef6ef7e10896b9"
)
V45B_PREREG_CONTENT_SHA256 = (
    "e4097a05599fb80509384daf4cdbb2415fa7495dae5832a6ee44fc5b2e123a05"
)
V45B_REPORT = prior.REPORT
V45B_REPORT_FILE_SHA256 = (
    "cd24ac92b97efa2940594c1929b1b02f611c50812830ecff2b13b02a5a15dbf1"
)
V45B_REPORT_CONTENT_SHA256 = (
    "779e69731374a5a1c4300ecce27b8aced82e21dfa1e5f81f16bdcde05a65941e"
)


def arm_wave_plan_v45c() -> tuple[tuple[tuple[str, int], ...], ...]:
    return (
        (("base_a", 0), ("base_b", 1), ("base_c", 2), ("base_d", 3)),
        (("base_e", 0), ("base_f", 1), ("base_g", 2), ("base_h", 3)),
        (("base_i", 0), ("sft_v42b_step16", 1),
         ("sft_v42b_step32", 2), ("sft_v42b", 3)),
        (("sft_v42c", 0), ("sft_v42d", 1),
         ("lora_es_v43d", 2), ("sft_v42e", 3)),
        (("sft_v42f", 0), ("sft_v42g", 1),
         ("lora_es_v43g", 2), ("sft_v42h", 3)),
    )


@contextmanager
def patched_prior_constants_v45c():
    saved = {
        "BASE": prior.BASE_ARMS_V45B, "CANDIDATE": prior.CANDIDATE_ARMS_V45B,
        "ARMS": prior.ARMS_V45B, "STAGED": prior.STAGED_BY_ARM_V45B,
        "IDS": prior.ADAPTER_IDS_V45B, "wave": prior.arm_wave_plan_v45b,
    }
    prior.BASE_ARMS_V45B = BASE_ARMS_V45C
    prior.CANDIDATE_ARMS_V45B = CANDIDATE_ARMS_V45C
    prior.ARMS_V45B = ARMS_V45C
    prior.STAGED_BY_ARM_V45B = STAGED_BY_ARM_V45C
    prior.ADAPTER_IDS_V45B = ADAPTER_IDS_V45C
    prior.arm_wave_plan_v45b = arm_wave_plan_v45c
    try:
        yield
    finally:
        prior.BASE_ARMS_V45B = saved["BASE"]
        prior.CANDIDATE_ARMS_V45B = saved["CANDIDATE"]
        prior.ARMS_V45B = saved["ARMS"]
        prior.STAGED_BY_ARM_V45B = saved["STAGED"]
        prior.ADAPTER_IDS_V45B = saved["IDS"]
        prior.arm_wave_plan_v45b = saved["wave"]


def staged_adapter_bindings_v45c() -> dict:
    with patched_prior_constants_v45c():
        value = prior.staged_adapter_bindings_v45b()
    if tuple(value) != CANDIDATE_ARMS_V45C:
        raise RuntimeError("V45C staged candidate order changed")
    return value


def prior_preregistration_v45c() -> dict:
    if core.file_sha256(V45B_PREREG) != V45B_PREREG_FILE_SHA256:
        raise RuntimeError("V45C V45B preregistration file changed")
    value = json.loads(V45B_PREREG.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V45B_PREREG_CONTENT_SHA256
        or core.canonical_sha256(compact) != V45B_PREREG_CONTENT_SHA256
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45b_revision")
        is not False
    ):
        raise RuntimeError("V45C V45B preregistration content changed")
    return value


def prior_result_v45c() -> dict:
    if core.file_sha256(V45B_REPORT) != V45B_REPORT_FILE_SHA256:
        raise RuntimeError("V45C V45B aggregate file changed")
    value = json.loads(V45B_REPORT.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    v43g = value.get("selection", {}).get("per_arm_gate_table", {}).get(
        "lora_es_v43g", {}
    )
    if (
        value.get("content_sha256_before_self_field")
        != V45B_REPORT_CONTENT_SHA256
        or core.canonical_sha256(compact) != V45B_REPORT_CONTENT_SHA256
        or value.get("status") != "complete_aggregate_only_no_heldout_access"
        or value.get("selection", {}).get("selected_arm") != "sft_v42g"
        or value.get("final_gate", {}).get("passed") is not True
        or value.get("heldout_or_holdout_opened") is not False
        or v43g.get("eligible") is not False
    ):
        raise RuntimeError("V45C V45B aggregate content changed")
    return {
        "path": str(V45B_REPORT), "file_sha256": V45B_REPORT_FILE_SHA256,
        "content_sha256": V45B_REPORT_CONTENT_SHA256,
        "selected_arm": "sft_v42g", "strict_gate_passed": True,
        "v43g_eligible": False, "heldout_or_holdout_opened": False,
        "aggregate_only_no_raw_semantics_inspected_for_v45c": True,
    }


def implementation_bindings_v45c() -> dict:
    paths = {
        "source_weights": v42h_stage.SOURCE / "adapter_model.safetensors",
        "source_config": v42h_stage.SOURCE / "adapter_config.json",
        "source_report": v42h_stage.REPORT,
        "staged_weights": v42h_stage.OUTPUT / "adapter_model.safetensors",
        "staged_config": v42h_stage.OUTPUT / "adapter_config.json",
        "stage_manifest": v42h_stage.OUTPUT / "stage_manifest_v44a.json",
    }
    return {
        "runtime": core.file_sha256(Path(__file__).resolve()),
        "v45b_runtime": core.file_sha256(Path(prior.__file__).resolve()),
        "v45b_preregistration_file_sha256": V45B_PREREG_FILE_SHA256,
        "v45b_preregistration_content_sha256": V45B_PREREG_CONTENT_SHA256,
        "v45b_aggregate": prior_result_v45c(),
        "v45b_implementation": prior.implementation_bindings_v45b(),
        "v42h_staging_runtime": core.file_sha256(
            Path(v42h_stage.__file__).resolve()
        ),
        "v42h_source_seal": v42h_stage.source_seal_v45c(),
        "v42h_artifacts": {
            label: core.file_sha256(path) for label, path in paths.items()
        },
    }


def load_preregistration_v45c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V45C preregistration file identity changed")
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
        != "matched-lora-v42h-v43g-ood-eligible-preregistration-v45c"
        or value.get("status")
        != "preregistered_before_fresh_v42h_extended_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45c_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V45C)
        or value.get("candidate_arms") != list(CANDIDATE_ARMS_V45C)
        or value.get("staged_adapters") != staged_adapter_bindings_v45c()
        or value.get("implementation_bindings") != implementation_bindings_v45c()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V45C preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    prior.prior.environment.environment_bindings_v44b()
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
    with patched_prior_constants_v45c():
        with prior.patched_prior_constants_v45b():
            state = prior.prior.EvaluationStateV45A()

            def evaluate_qa(trainer, bundle, raw_sink, label):
                result = saved["eval_qa"](trainer, bundle, raw_sink, label)
                baseline = result["base_a"]
                if any(result[arm] != baseline for arm in BASE_ARMS_V45C[1:]):
                    raise RuntimeError(
                        f"V45C nine-base equivalence failed on {label}"
                    )
                if label == "ood_qa":
                    state.ood_qa = result
                return result

            def evaluate_prose(trainer, rows, raw_sink):
                aggregate, detailed = saved["eval_prose"](
                    trainer, rows, raw_sink
                )
                baseline = aggregate["base_a"]
                if any(
                    aggregate[arm] != baseline for arm in BASE_ARMS_V45C[1:]
                ):
                    raise RuntimeError("V45C nine-base prose equivalence failed")
                final = prior.prior.finalize_selection_v45a(
                    state.shadow, state.ood_qa, detailed, raw_sink
                )
                final.pop("all_three_base_duplicates_required_exact", None)
                final.update({
                    "all_nine_base_duplicates_required_exact": True,
                    "padding_base_arms": list(PADDING_BASE_ARMS_V45C),
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
            core.load_preregistration = load_preregistration_v45c
            core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = (
                prior.prior.parser_fix.protected_preflight_v44c
            )
            core.select_candidate_v44a = state.provisional
            core.evaluate_qa_v44a = evaluate_qa
            core.evaluate_prose_v44a = evaluate_prose
            try:
                with prior.prior.patched_candidate_globals_v45a():
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
