#!/usr/bin/env python3
"""V45E replicated I/J/K SFT boundary with OOD-first logical selection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first
import run_matched_lora_candidate_eval_v45d as prior
import stage_candidate_adapters_vllm_v44a as canonical_stage
import stage_v42i_adapter_vllm_v45d as i_stage
import stage_v42j_adapter_vllm_v45e as j_stage
import stage_v42k_adapter_vllm_v45e as k_stage


ROOT = Path(__file__).resolve().parent
PRIMARY_BASE_ARMS_V45E = ("base_a", "base_b", "base_c", "base_d")
PADDING_BASE_ARMS_V45E = ("base_e", "base_f")
BASE_ARMS_V45E = PRIMARY_BASE_ARMS_V45E + PADDING_BASE_ARMS_V45E
LOGICAL_REPLICAS_V45E = {
    "sft_v42i": ("sft_v42i_replica_a", "sft_v42i_replica_b"),
    "sft_v42j": ("sft_v42j_replica_a", "sft_v42j_replica_b"),
    "sft_v42k": ("sft_v42k_replica_a", "sft_v42k_replica_b"),
}
LOGICAL_CANDIDATES_V45E = tuple(LOGICAL_REPLICAS_V45E)
CANDIDATE_ARMS_V45E = tuple(
    arm for pair in LOGICAL_REPLICAS_V45E.values() for arm in pair
)
STAGED_BY_LOGICAL_V45E = {
    "sft_v42i": i_stage.OUTPUT,
    "sft_v42j": j_stage.OUTPUT,
    "sft_v42k": k_stage.OUTPUT,
}
STAGED_BY_ARM_V45E = {
    replica: STAGED_BY_LOGICAL_V45E[logical]
    for logical, replicas in LOGICAL_REPLICAS_V45E.items()
    for replica in replicas
}
ADAPTER_IDS_V45E = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V45E)
}
ARMS_V45E = (
    *PRIMARY_BASE_ARMS_V45E,
    *LOGICAL_REPLICAS_V45E["sft_v42i"],
    *LOGICAL_REPLICAS_V45E["sft_v42j"],
    *LOGICAL_REPLICAS_V45E["sft_v42k"],
    *PADDING_BASE_ARMS_V45E,
)
EXPERIMENT = "v45e_matched_lora_sft_ijk_replicated_ood_eligible_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v45e.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v45e.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_ijk_replicated_ood_eligible_eval_v45e.json"
).resolve()
V45D_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_boundary_ood_eligible_eval_v45d.json"
).resolve()
V45D_PREREG_FILE_SHA256 = (
    "bc06e84381b35e8f477baf1b0df0a846e9b51e0ca7a210279b79fe0b62812d25"
)
V45D_PREREG_CONTENT_SHA256 = (
    "20c6608e77e4b6de79ff9110092cb458501dfa13682d36e9d554f1f8cc2dc04f"
)
V45D_REPORT = prior.REPORT
V45D_REPORT_FILE_SHA256 = (
    "a38ef3b9e947f2b6a6328f9eea679d7031a515ce59c8d0804144ae6d587985be"
)
V45D_REPORT_CONTENT_SHA256 = (
    "f39a12fc33e55c957b540033220eb1ff82d0e1459e962ee30f591024d0b6a9a8"
)


def arm_wave_plan_v45e() -> tuple[tuple[tuple[str, int], ...], ...]:
    return (
        (("base_a", 0), ("base_b", 1), ("base_c", 2), ("base_d", 3)),
        (("sft_v42i_replica_a", 0), ("sft_v42i_replica_b", 1),
         ("sft_v42j_replica_a", 2), ("sft_v42j_replica_b", 3)),
        (("sft_v42k_replica_a", 0), ("sft_v42k_replica_b", 1),
         ("base_e", 2), ("base_f", 3)),
    )


@contextmanager
def patched_candidate_globals_v45e():
    saved = {
        "BASE": core.BASE_ARMS, "CANDIDATE": core.CANDIDATE_ARMS,
        "ARMS": core.ARMS, "STAGED": core.STAGED_BY_ARM,
        "IDS": core.ADAPTER_IDS_V44A, "ENGINE": core.ENGINE_INDEX_BY_ARM_V44A,
        "wave": core.arm_wave_plan_v44a,
    }
    core.BASE_ARMS = BASE_ARMS_V45E
    core.CANDIDATE_ARMS = CANDIDATE_ARMS_V45E
    core.ARMS = ARMS_V45E
    core.STAGED_BY_ARM = STAGED_BY_ARM_V45E
    core.ADAPTER_IDS_V44A = ADAPTER_IDS_V45E
    core.ENGINE_INDEX_BY_ARM_V44A = {
        arm: engine for wave in arm_wave_plan_v45e() for arm, engine in wave
    }
    core.arm_wave_plan_v44a = arm_wave_plan_v45e
    try:
        yield
    finally:
        core.BASE_ARMS = saved["BASE"]
        core.CANDIDATE_ARMS = saved["CANDIDATE"]
        core.ARMS = saved["ARMS"]
        core.STAGED_BY_ARM = saved["STAGED"]
        core.ADAPTER_IDS_V44A = saved["IDS"]
        core.ENGINE_INDEX_BY_ARM_V44A = saved["ENGINE"]
        core.arm_wave_plan_v44a = saved["wave"]


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def prior_preregistration_v45e() -> dict:
    if core.file_sha256(V45D_PREREG) != V45D_PREREG_FILE_SHA256:
        raise RuntimeError("V45E V45D preregistration file changed")
    value = json.loads(V45D_PREREG.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != V45D_PREREG_CONTENT_SHA256
        or _compact_sha(value) != V45D_PREREG_CONTENT_SHA256
        or value.get("heldout_or_holdout_access_authorized") is not False
    ):
        raise RuntimeError("V45E V45D preregistration content changed")
    return value


def prior_result_v45e() -> dict:
    if core.file_sha256(V45D_REPORT) != V45D_REPORT_FILE_SHA256:
        raise RuntimeError("V45E V45D aggregate file changed")
    value = json.loads(V45D_REPORT.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != V45D_REPORT_CONTENT_SHA256
        or _compact_sha(value) != V45D_REPORT_CONTENT_SHA256
        or value.get("status")
        != "complete_aggregate_only_no_heldout_access"
        or value.get("selection", {}).get("selected_arm") != "sft_v42i"
        or value.get("selection", {}).get("per_arm_gate_table", {}).get(
            "sft_v42i", {}
        ).get("eligible") is not True
        or value.get("final_gate", {}).get("passed") is not True
        or value.get("base_duplicate_equivalence", {}).get("all_splits")
        is not True
        or value.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("V45E V45D aggregate content changed")
    return {
        "path": str(V45D_REPORT), "file_sha256": V45D_REPORT_FILE_SHA256,
        "content_sha256": V45D_REPORT_CONTENT_SHA256,
        "selected_arm": "sft_v42i", "strict_gate_passed": True,
        "heldout_or_holdout_opened": False,
        "aggregate_only_no_raw_semantics_inspected_for_v45e": True,
    }


def canonical_stage_binding_v45e(logical: str) -> dict:
    directory = STAGED_BY_LOGICAL_V45E[logical]
    manifest_path = directory / "stage_manifest_v44a.json"
    value = json.loads(manifest_path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != _compact_sha(value)
        or value.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
        or value.get("status") != "complete_cpu_only_key_transform"
        or value.get("arm") != logical
        or value.get("implementation", {}).get("path")
        != str(Path(canonical_stage.__file__).resolve())
        or value.get("implementation", {}).get("file_sha256")
        != core.file_sha256(Path(canonical_stage.__file__).resolve())
        or value.get("transformed_identity", {}).get("tensor_count") != 70
        or value.get("transformed_identity", {}).get("elements") != 4_528_128
        or value.get("transformed_identity", {}).get(
            "all_tensor_bytes_preserved_exactly"
        ) is not True
        or value.get("dataset_or_evaluation_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
    ):
        raise RuntimeError(f"V45E canonical stage changed: {logical}")
    artifact = value["artifact"]
    result = {
        "logical_candidate": logical, "directory": str(directory),
        "weights_file_sha256": core.file_sha256(
            directory / "adapter_model.safetensors"
        ),
        "adapter_config_file_sha256": core.file_sha256(
            directory / "adapter_config.json"
        ),
        "manifest_file_sha256": core.file_sha256(manifest_path),
        "manifest_content_sha256": content,
        "transformed_identity_sha256": value[
            "transformed_identity"
        ]["sha256"],
        "target_namespace": artifact["target_namespace"],
        "tensor_count": 70, "elements": 4_528_128,
        "tensor_bytes_preserved_exactly": True,
    }
    if (
        result["weights_file_sha256"] != artifact["weights_file_sha256"]
        or result["adapter_config_file_sha256"]
        != artifact["adapter_config_file_sha256"]
    ):
        raise RuntimeError(f"V45E canonical staged files changed: {logical}")
    return result


def replica_stage_bindings_v45e() -> dict:
    logical = {
        name: canonical_stage_binding_v45e(name)
        for name in LOGICAL_CANDIDATES_V45E
    }
    return {
        replica: {
            **logical[name], "replica_arm": replica,
            "adapter_id": ADAPTER_IDS_V45E[replica],
        }
        for name, replicas in LOGICAL_REPLICAS_V45E.items()
        for replica in replicas
    }


def implementation_bindings_v45e() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "ood_first_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(ood_first.parser_fix.__file__).resolve(),
        "environment_runtime": Path(ood_first.environment.__file__).resolve(),
        "v42i_staging_runtime": Path(i_stage.__file__).resolve(),
        "v42j_staging_runtime": Path(j_stage.__file__).resolve(),
        "v42k_staging_runtime": Path(k_stage.__file__).resolve(),
        "canonical_staging_runtime": Path(canonical_stage.__file__).resolve(),
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
    }
    modules = {"sft_v42i": i_stage, "sft_v42j": j_stage, "sft_v42k": k_stage}
    for logical, module in modules.items():
        paths[f"{logical}_source_weights"] = module.SOURCE / "adapter_model.safetensors"
        paths[f"{logical}_source_config"] = module.SOURCE / "adapter_config.json"
        paths[f"{logical}_source_report"] = module.REPORT
        paths[f"{logical}_staged_weights"] = module.OUTPUT / "adapter_model.safetensors"
        paths[f"{logical}_staged_config"] = module.OUTPUT / "adapter_config.json"
        paths[f"{logical}_stage_manifest"] = module.OUTPUT / "stage_manifest_v44a.json"
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["v45d_aggregate_finding"] = prior_result_v45e()
    return result


def assert_replica_equivalence_v45e(metrics: dict, raw: dict | None,
                                     label: str) -> dict:
    baseline = metrics["base_a"]
    if any(metrics[arm] != baseline for arm in BASE_ARMS_V45E[1:]):
        raise RuntimeError(f"V45E six-base equivalence failed on {label}")
    logical = {}
    for name, (left, right) in LOGICAL_REPLICAS_V45E.items():
        metrics_equal = metrics[left] == metrics[right]
        raw_equal = raw is None or raw[left] == raw[right]
        if not metrics_equal or not raw_equal:
            raise RuntimeError(f"V45E {name} replica equivalence failed on {label}")
        logical[name] = {
            "replicas": [left, right], "metrics_exact": True,
            "raw_numeric_outputs_exact": raw_equal,
        }
    return {
        "label": label, "all_six_base_outputs_exact": True,
        "four_primary_and_two_padding_bases": True,
        "logical_candidates": logical, "all_logical_replicas_exact": True,
    }


def logical_selection_key_v45e(shadow: dict, logical: str) -> tuple:
    representative = LOGICAL_REPLICAS_V45E[logical][0]
    metrics = shadow[representative]
    return (
        metrics["generated_equal_unit_mean_reward"],
        metrics["generated_exact_count"],
        metrics["generated_nonzero_count"],
        metrics["teacher_forced_equal_unit_mean_answer_logprob"],
        LOGICAL_CANDIDATES_V45E.index(logical),
    )


def finalize_selection_v45e(shadow: dict, ood_qa: dict, prose: dict,
                            raw_sink: dict, equivalence: dict) -> dict:
    table = {}
    base_qa, base_prose = ood_qa["base_a"], prose["base_a"]
    for logical, (representative, _replica) in LOGICAL_REPLICAS_V45E.items():
        qa_gate = core.v39a.qa_ood_gate(base_qa, ood_qa[representative])
        qa_gate.update(ood_first.paired_qa_bootstrap_v45a(
            raw_sink["ood_qa"]["base_a"], raw_sink["ood_qa"][representative]
        ))
        prose_gate = core.v39a.prose_gate(base_prose, prose[representative])
        counters = shadow[representative]["protocol_leak_counters"]
        base_counters = shadow["base_a"]["protocol_leak_counters"]
        protocol_safe = all(
            counters[key] <= base_counters[key] for key in base_counters
        )
        table[logical] = {
            "representative_replica": representative,
            "replicas": list(LOGICAL_REPLICAS_V45E[logical]),
            "replica_equivalence_required_and_passed": True,
            "ood_qa": qa_gate, "ood_prose": prose_gate,
            "no_protocol_or_leak_counter_increase": protocol_safe,
            "eligible": qa_gate["passed"] and prose_gate["passed"] and protocol_safe,
        }
    eligible = tuple(
        logical for logical in LOGICAL_CANDIDATES_V45E
        if table[logical]["eligible"]
    )
    selected_logical = (
        max(eligible, key=lambda arm: logical_selection_key_v45e(shadow, arm))
        if eligible else None
    )
    selected_arm = (
        "base_a" if selected_logical is None
        else LOGICAL_REPLICAS_V45E[selected_logical][0]
    )
    improved = (
        selected_logical is not None
        and logical_selection_key_v45e(shadow, selected_logical)[:-1]
        > (
            shadow["base_a"]["generated_equal_unit_mean_reward"],
            shadow["base_a"]["generated_exact_count"],
            shadow["base_a"]["generated_nonzero_count"],
            shadow["base_a"]["teacher_forced_equal_unit_mean_answer_logprob"],
        )
    )
    return {
        "selected_arm": selected_arm,
        "selected_candidate_arm": None if selected_logical is None else selected_arm,
        "selected_logical_candidate": selected_logical,
        "eligible_logical_candidates": list(eligible),
        "ineligible_logical_candidates": [
            arm for arm in LOGICAL_CANDIDATES_V45E if arm not in eligible
        ],
        "per_logical_candidate_gate_table": table,
        "ood_eligible_set_constructed_before_shadow_ranking": True,
        "rule": (
            "require exact replica equivalence; OOD-gate each logical candidate "
            "using replica A; then rank only eligible logical candidates on "
            "frozen shadow metrics"
        ),
        "shadow_improvement_gate_passed": improved,
        "no_protocol_or_leak_counter_increase": (
            True if selected_logical is None
            else table[selected_logical]["no_protocol_or_leak_counter_increase"]
        ),
        "replica_equivalence": equivalence,
        "all_six_base_duplicates_required_exact": True,
        "primary_base_arms": list(PRIMARY_BASE_ARMS_V45E),
        "padding_base_arms": list(PADDING_BASE_ARMS_V45E),
        "padding_base_arms_excluded_from_eligibility_and_ranking": True,
    }


class EvaluationStateV45E:
    def __init__(self):
        self.shadow = self.ood_qa = None
        self.equivalence = {}
        self.selection = None

    def provisional(self, shadow: dict) -> dict:
        self.shadow = shadow
        self.selection = {
            "selected_arm": "base_a", "selected_candidate_arm": None,
            "selected_logical_candidate": None,
            "eligible_logical_candidates": [],
            "ineligible_logical_candidates": list(LOGICAL_CANDIDATES_V45E),
            "rule": "pending OOD eligibility and replica equivalence",
            "shadow_improvement_gate_passed": False,
            "no_protocol_or_leak_counter_increase": True,
        }
        return self.selection


def load_preregistration_v45e(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V45E preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "matched-lora-sft-ijk-replicated-ood-preregistration-v45e"
        or value.get("status")
        != "preregistered_before_fresh_replicated_boundary_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45e_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V45E)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES_V45E)
        or value.get("replica_staged_adapters") != replica_stage_bindings_v45e()
        or value.get("implementation_bindings") != implementation_bindings_v45e()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V45E preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    ood_first.environment.environment_bindings_v44b()
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
    state = EvaluationStateV45E()

    def evaluate_qa(trainer, bundle, raw_sink, label):
        result = saved["eval_qa"](trainer, bundle, raw_sink, label)
        state.equivalence[label] = assert_replica_equivalence_v45e(
            result, raw_sink[label], label
        )
        if label == "ood_qa":
            state.ood_qa = result
        return result

    def evaluate_prose(trainer, rows, raw_sink):
        aggregate, detailed = saved["eval_prose"](trainer, rows, raw_sink)
        state.equivalence["ood_prose"] = assert_replica_equivalence_v45e(
            aggregate, raw_sink["ood_prose"], "ood_prose"
        )
        final = finalize_selection_v45e(
            state.shadow, state.ood_qa, detailed, raw_sink, state.equivalence
        )
        state.selection.clear()
        state.selection.update(final)
        return aggregate, detailed

    core.EXPERIMENT, core.RUN_DIR = EXPERIMENT, RUN_DIR
    core.ATTEMPT, core.RAW = ATTEMPT, RAW
    core.GPU_LOG, core.REPORT = GPU_LOG, REPORT
    core.load_preregistration = load_preregistration_v45e
    core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = (
        ood_first.parser_fix.protected_preflight_v44c
    )
    core.select_candidate_v44a = state.provisional
    core.evaluate_qa_v44a = evaluate_qa
    core.evaluate_prose_v44a = evaluate_prose
    try:
        with patched_candidate_globals_v45e():
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
