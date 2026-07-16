#!/usr/bin/env python3
"""V45D compact SFT boundary: G final, H early stops, and I final."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as prior
import stage_v42h_earlystop_adapters_vllm_v45d as h_stage
import stage_v42i_adapter_vllm_v45d as i_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V45D = ("base_a", "base_b", "base_c", "base_d")
CANDIDATE_ARMS_V45D = (
    "sft_v42g", "sft_v42h_step16", "sft_v42h_step32", "sft_v42i"
)
ARMS_V45D = BASE_ARMS_V45D + CANDIDATE_ARMS_V45D
STAGED_BY_ARM_V45D = {
    "sft_v42g": prior.STAGED_BY_ARM_V45A["sft_v42g"],
    "sft_v42h_step16": h_stage.SOURCE_SPECS_V45D[
        "sft_v42h_step16"
    ]["output"],
    "sft_v42h_step32": h_stage.SOURCE_SPECS_V45D[
        "sft_v42h_step32"
    ]["output"],
    "sft_v42i": i_stage.OUTPUT,
}
ADAPTER_IDS_V45D = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V45D)
}
EXPERIMENT = "v45d_matched_lora_sft_boundary_ood_eligible_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v45d.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v45d.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_boundary_ood_eligible_eval_v45d.json"
).resolve()
V45C_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_hpo_v42h_v43g_ood_eligible_eval_v45c.json"
).resolve()
V45C_PREREG_FILE_SHA256 = (
    "46cdf515c2d3e00eae7bf9c591ee24b3658159dce4a1a4eae07a78d30c524add"
)
V45C_PREREG_CONTENT_SHA256 = (
    "2e76445e3284ac34622dfbd6a23cd0f2bb0186dcfae073b6cc37c184c304ee0a"
)
V45C_REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_hpo_v42h_v43g_ood_eligible_eval_v45c.json"
).resolve()
V45C_REPORT_FILE_SHA256 = (
    "2081b319a777a5c9bbe5129cd123b7661b50c9915594f277ca7676486fa09daf"
)
V45C_REPORT_CONTENT_SHA256 = (
    "3a1737796e442ae99a464ce128aa2721ef8c91516ba84ac97f13489e53dcadb0"
)


def arm_wave_plan_v45d() -> tuple[tuple[tuple[str, int], ...], ...]:
    return (
        (("base_a", 0), ("base_b", 1), ("base_c", 2), ("base_d", 3)),
        (("sft_v42g", 0), ("sft_v42h_step16", 1),
         ("sft_v42h_step32", 2), ("sft_v42i", 3)),
    )


@contextmanager
def patched_prior_constants_v45d():
    saved = {
        "BASE": prior.BASE_ARMS_V45A,
        "CANDIDATE": prior.CANDIDATE_ARMS_V45A,
        "ARMS": prior.ARMS_V45A,
        "STAGED": prior.STAGED_BY_ARM_V45A,
        "IDS": prior.ADAPTER_IDS_V45A,
        "wave": prior.arm_wave_plan_v45a,
    }
    prior.BASE_ARMS_V45A = BASE_ARMS_V45D
    prior.CANDIDATE_ARMS_V45A = CANDIDATE_ARMS_V45D
    prior.ARMS_V45A = ARMS_V45D
    prior.STAGED_BY_ARM_V45A = STAGED_BY_ARM_V45D
    prior.ADAPTER_IDS_V45A = ADAPTER_IDS_V45D
    prior.arm_wave_plan_v45a = arm_wave_plan_v45d
    try:
        yield
    finally:
        prior.BASE_ARMS_V45A = saved["BASE"]
        prior.CANDIDATE_ARMS_V45A = saved["CANDIDATE"]
        prior.ARMS_V45A = saved["ARMS"]
        prior.STAGED_BY_ARM_V45A = saved["STAGED"]
        prior.ADAPTER_IDS_V45A = saved["IDS"]
        prior.arm_wave_plan_v45a = saved["wave"]


def staged_adapter_bindings_v45d() -> dict:
    with patched_prior_constants_v45d():
        value = prior.staged_adapter_bindings_v45a()
    if tuple(value) != CANDIDATE_ARMS_V45D:
        raise RuntimeError("V45D staged candidate order changed")
    return value


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def prior_preregistration_v45d() -> dict:
    if core.file_sha256(V45C_PREREG) != V45C_PREREG_FILE_SHA256:
        raise RuntimeError("V45D V45C preregistration file changed")
    value = json.loads(V45C_PREREG.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != V45C_PREREG_CONTENT_SHA256
        or _compact_sha(value) != V45C_PREREG_CONTENT_SHA256
        or value.get("heldout_or_holdout_access_authorized") is not False
    ):
        raise RuntimeError("V45D V45C preregistration content changed")
    return value


def prior_result_v45d() -> dict:
    if core.file_sha256(V45C_REPORT) != V45C_REPORT_FILE_SHA256:
        raise RuntimeError("V45D V45C aggregate file changed")
    value = json.loads(V45C_REPORT.read_text())
    table = value.get("selection", {}).get("per_arm_gate_table", {})
    if (
        value.get("content_sha256_before_self_field")
        != V45C_REPORT_CONTENT_SHA256
        or _compact_sha(value) != V45C_REPORT_CONTENT_SHA256
        or value.get("status")
        != "complete_aggregate_only_no_heldout_access"
        or value.get("selection", {}).get("selected_arm") != "sft_v42g"
        or table.get("sft_v42g", {}).get("eligible") is not True
        or table.get("sft_v42h", {}).get("eligible") is not False
        or table.get("sft_v42h", {}).get("ood_qa", {}).get(
            "exact_count_delta"
        ) != -1
        or value.get("final_gate", {}).get("passed") is not True
        or value.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("V45D V45C aggregate content changed")
    return {
        "path": str(V45C_REPORT),
        "file_sha256": V45C_REPORT_FILE_SHA256,
        "content_sha256": V45C_REPORT_CONTENT_SHA256,
        "selected_arm": "sft_v42g",
        "v42g_eligible": True,
        "v42h_final_eligible": False,
        "v42h_final_ood_exact_count_delta": -1,
        "v42h_final_shadow_generated_equal_unit_mean_reward": value[
            "shadow"
        ]["metrics"]["sft_v42h"]["generated_equal_unit_mean_reward"],
        "heldout_or_holdout_opened": False,
        "aggregate_only_no_raw_semantics_inspected_for_v45d": True,
    }


def implementation_bindings_v45d() -> dict:
    artifact_paths = {}
    for arm, directory in STAGED_BY_ARM_V45D.items():
        artifact_paths[f"{arm}_staged_weights"] = (
            directory / "adapter_model.safetensors"
        )
        artifact_paths[f"{arm}_staged_config"] = (
            directory / "adapter_config.json"
        )
        artifact_paths[f"{arm}_stage_manifest"] = (
            directory / "stage_manifest_v44a.json"
        )
    for arm, spec in h_stage.SOURCE_SPECS_V45D.items():
        artifact_paths[f"{arm}_source_weights"] = (
            spec["source"] / "adapter_model.safetensors"
        )
        artifact_paths[f"{arm}_source_config"] = (
            spec["source"] / "adapter_config.json"
        )
        artifact_paths[f"{arm}_trainer_state"] = (
            spec["source"] / "trainer_state.json"
        )
    artifact_paths.update({
        "sft_v42i_source_weights": i_stage.SOURCE / "adapter_model.safetensors",
        "sft_v42i_source_config": i_stage.SOURCE / "adapter_config.json",
        "sft_v42i_source_report": i_stage.REPORT,
    })
    return {
        "runtime": core.file_sha256(Path(__file__).resolve()),
        "v45a_ood_first_runtime": core.file_sha256(Path(prior.__file__).resolve()),
        "source_faithful_parser_runtime": core.file_sha256(
            Path(prior.parser_fix.__file__).resolve()
        ),
        "environment": prior.environment.environment_bindings_v44b(),
        "v42h_earlystop_staging_runtime": core.file_sha256(
            Path(h_stage.__file__).resolve()
        ),
        "v42i_staging_runtime": core.file_sha256(Path(i_stage.__file__).resolve()),
        "artifacts": {
            label: core.file_sha256(path)
            for label, path in artifact_paths.items()
        },
        "v42h_source_audits": {
            arm: h_stage.audit_source_v45d(arm)["trainer_state_binding_v45d"]
            for arm in h_stage.SOURCE_SPECS_V45D
        },
        "v42i_source_audited": len(i_stage.audit_source_v45d()["records"]) == 70,
        "v45c_aggregate_finding": prior_result_v45d(),
    }


def load_preregistration_v45d(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V45D preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "matched-lora-sft-boundary-ood-eligible-preregistration-v45d"
        or value.get("status")
        != "preregistered_before_fresh_sft_boundary_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45d_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V45D)
        or value.get("candidate_arms") != list(CANDIDATE_ARMS_V45D)
        or value.get("staged_adapters") != staged_adapter_bindings_v45d()
        or value.get("implementation_bindings") != implementation_bindings_v45d()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V45D preregistration content changed")
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
    with patched_prior_constants_v45d():
        state = prior.EvaluationStateV45A()

        def evaluate_qa(trainer, bundle, raw_sink, label):
            result = saved["eval_qa"](trainer, bundle, raw_sink, label)
            baseline = result["base_a"]
            if any(result[arm] != baseline for arm in BASE_ARMS_V45D[1:]):
                raise RuntimeError(
                    f"V45D four-base equivalence failed on {label}"
                )
            if label == "ood_qa":
                state.ood_qa = result
            return result

        def evaluate_prose(trainer, rows, raw_sink):
            aggregate, detailed = saved["eval_prose"](trainer, rows, raw_sink)
            baseline = aggregate["base_a"]
            if any(aggregate[arm] != baseline for arm in BASE_ARMS_V45D[1:]):
                raise RuntimeError("V45D four-base prose equivalence failed")
            final = prior.finalize_selection_v45a(
                state.shadow, state.ood_qa, detailed, raw_sink
            )
            final.pop("all_three_base_duplicates_required_exact", None)
            final["all_four_base_duplicates_required_exact"] = True
            final["padding_base_arms"] = []
            final["padding_base_arms_excluded_from_candidate_ranking"] = True
            state.selection.clear()
            state.selection.update(final)
            return aggregate, detailed

        core.EXPERIMENT, core.RUN_DIR = EXPERIMENT, RUN_DIR
        core.ATTEMPT, core.RAW = ATTEMPT, RAW
        core.GPU_LOG, core.REPORT = GPU_LOG, REPORT
        core.load_preregistration = load_preregistration_v45d
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
