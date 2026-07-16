#!/usr/bin/env python3
"""V46C replicated V42I SFT versus accepted V43J LoRA-ES, OOD first."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first
import run_matched_lora_candidate_eval_v45e as parent
import stage_candidate_adapters_vllm_v44a as canonical_stage
import stage_v42i_adapter_vllm_v45d as sft_stage
import stage_v43j_adapter_vllm_v46c as es_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V46C = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS_V46C = {
    "sft_v42i": ("sft_v42i_a", "sft_v42i_b"),
    "lora_es_v43j": ("lora_es_v43j_a", "lora_es_v43j_b"),
}
LOGICAL_CANDIDATES_V46C = tuple(LOGICAL_REPLICAS_V46C)
CANDIDATE_ARMS_V46C = tuple(
    arm for replicas in LOGICAL_REPLICAS_V46C.values() for arm in replicas
)
ARMS_V46C = BASE_ARMS_V46C + CANDIDATE_ARMS_V46C
STAGED_BY_LOGICAL_V46C = {
    "sft_v42i": sft_stage.OUTPUT,
    "lora_es_v43j": es_stage.OUTPUT,
}
STAGED_BY_ARM_V46C = {
    replica: STAGED_BY_LOGICAL_V46C[logical]
    for logical, replicas in LOGICAL_REPLICAS_V46C.items()
    for replica in replicas
}
ADAPTER_IDS_V46C = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V46C)
}
EXPERIMENT = "v46c_lora_es_v43j_vs_sft_v42i_replicated_ood_first_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v46c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v46c.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "lora_es_v43j_vs_sft_v42i_replicated_ood_first_eval_v46c.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v43j_vs_sft_v42i_replicated_ood_first_eval_v46c.json"
).resolve()
PARENT_PREREGISTRATION = parent.ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_ijk_replicated_ood_eligible_eval_v45e.json"
)
PARENT_PREREGISTRATION_FILE_SHA256 = (
    "e7ca432c68de6f751ffecf03595c1b5debb7d3adb3db8244ab144938c14fb34e"
)
PARENT_PREREGISTRATION_CONTENT_SHA256 = (
    "47c335dc6003a648b8eeb96f5b7f3ed6922a94c5502c93d6dc9e1426caeeed2a"
)
STAGE_EXPECTED_V46C = {
    "sft_v42i": {
        "arm": "sft_v42i",
        "weights": "79207dd2c0b46aaef4af5933aaac9fbbaf837db91241ab9d352e652b5c53afad",
        "config": "0e8060efd40772233390f3f97ace489e473b2bc76572e7566b83afe3dd83cc51",
        "manifest_file": "f3bc58058032fd3cc00176ddda9de861d2cc3a989b8ce2e7499e15f1adb6d0c5",
        "manifest_content": "040f13fbddcb16b15d8771f736cedac1a89a5b987805c442da8e03445cc1c838",
        "transformed_identity": "d185cbe52414054759188334fd38b96dbda601957bc8931256fd1e3c0fe71041",
    },
    "lora_es_v43j": {
        "arm": "lora_es_v43j",
        "weights": "77fc7afc71e9f0462cd8204dcba14a471edc2017a819d23639814d4b48b29055",
        "config": "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52",
        "manifest_file": "e6c9face492298963ce4b992a25d5559eb058124b4edc08015851ffa28cefa6d",
        "manifest_content": "8619f831f0b53684e1cea09ea8496cdbdb6bc7a34d23f38f52a6f621e1ddf606",
        "transformed_identity": "3edee28f8af2eb8110df43fcba3c7f55629873090d42dca126da67d54634233b",
    },
}
RANK_FIELDS_V46C = (
    "generated_equal_unit_mean_reward",
    "generated_exact_count",
    "generated_nonzero_count",
    "teacher_forced_equal_unit_mean_answer_logprob",
)


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def parent_preregistration_v46c() -> dict:
    if core.file_sha256(PARENT_PREREGISTRATION) != (
        PARENT_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("V46C V45E parent preregistration file changed")
    value = json.loads(PARENT_PREREGISTRATION.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256_before_self_field")
        != PARENT_PREREGISTRATION_CONTENT_SHA256
        or _compact_sha(value) != PARENT_PREREGISTRATION_CONTENT_SHA256
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45e_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V46C V45E parent preregistration content changed")
    return value


def arm_wave_plan_v46c() -> tuple[tuple[tuple[str, int], ...], ...]:
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS_V46C)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS_V46C)),
    )


@contextmanager
def patched_candidate_globals_v46c():
    saved = {
        "BASE": core.BASE_ARMS,
        "CANDIDATE": core.CANDIDATE_ARMS,
        "ARMS": core.ARMS,
        "STAGED": core.STAGED_BY_ARM,
        "IDS": core.ADAPTER_IDS_V44A,
        "ENGINE": core.ENGINE_INDEX_BY_ARM_V44A,
        "wave": core.arm_wave_plan_v44a,
    }
    core.BASE_ARMS = BASE_ARMS_V46C
    core.CANDIDATE_ARMS = CANDIDATE_ARMS_V46C
    core.ARMS = ARMS_V46C
    core.STAGED_BY_ARM = STAGED_BY_ARM_V46C
    core.ADAPTER_IDS_V44A = ADAPTER_IDS_V46C
    core.ENGINE_INDEX_BY_ARM_V44A = {
        arm: engine for wave in arm_wave_plan_v46c() for arm, engine in wave
    }
    core.arm_wave_plan_v44a = arm_wave_plan_v46c
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


def canonical_stage_binding_v46c(logical: str) -> dict:
    directory = STAGED_BY_LOGICAL_V46C[logical]
    expected = STAGE_EXPECTED_V46C[logical]
    manifest_path = directory / "stage_manifest_v44a.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = value.get("artifact", {})
    transformed = value.get("transformed_identity", {})
    observed = {
        "logical_candidate": logical,
        "directory": str(directory),
        "weights_file_sha256": core.file_sha256(
            directory / "adapter_model.safetensors"
        ),
        "adapter_config_file_sha256": core.file_sha256(
            directory / "adapter_config.json"
        ),
        "manifest_file_sha256": core.file_sha256(manifest_path),
        "manifest_content_sha256": value.get(
            "content_sha256_before_self_field"
        ),
        "transformed_identity_sha256": transformed.get("sha256"),
        "target_namespace": artifact.get("target_namespace"),
        "tensor_count": transformed.get("tensor_count"),
        "elements": transformed.get("elements"),
        "tensor_bytes_preserved_exactly": transformed.get(
            "all_tensor_bytes_preserved_exactly"
        ),
    }
    if (
        value.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
        or value.get("status") != "complete_cpu_only_key_transform"
        or value.get("arm") != expected["arm"]
        or _compact_sha(value) != expected["manifest_content"]
        or observed["weights_file_sha256"] != expected["weights"]
        or observed["adapter_config_file_sha256"] != expected["config"]
        or observed["manifest_file_sha256"] != expected["manifest_file"]
        or observed["manifest_content_sha256"] != expected["manifest_content"]
        or observed["transformed_identity_sha256"]
        != expected["transformed_identity"]
        or observed["target_namespace"]
        != canonical_stage.TARGET_PREFIX_V44A + "*"
        or observed["tensor_count"] != 70
        or observed["elements"] != 4_528_128
        or observed["tensor_bytes_preserved_exactly"] is not True
        or value.get("dataset_or_evaluation_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
        or value.get("gpu_accessed") is not False
    ):
        raise RuntimeError(f"V46C canonical staged artifact changed: {logical}")
    return observed


def replica_stage_bindings_v46c() -> dict:
    logical = {
        name: canonical_stage_binding_v46c(name)
        for name in LOGICAL_CANDIDATES_V46C
    }
    return {
        replica: {
            **logical[name],
            "replica_arm": replica,
            "adapter_id": ADAPTER_IDS_V46C[replica],
        }
        for name, replicas in LOGICAL_REPLICAS_V46C.items()
        for replica in replicas
    }


def implementation_bindings_v46c() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_vs_sft_ood_first_preregistration_v46c.py",
        "tests": ROOT / "test_lora_es_vs_sft_ood_first_v46c.py",
        "core_runtime": Path(core.__file__).resolve(),
        "ood_first_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(
            ood_first.parser_fix.__file__
        ).resolve(),
        "environment_runtime": Path(ood_first.environment.__file__).resolve(),
        "canonical_staging_runtime": Path(canonical_stage.__file__).resolve(),
        "sft_v42i_staging_runtime": Path(sft_stage.__file__).resolve(),
        "lora_es_v43j_staging_runtime": Path(es_stage.__file__).resolve(),
        "sft_v42i_source_weights": sft_stage.SOURCE / "adapter_model.safetensors",
        "sft_v42i_source_config": sft_stage.SOURCE / "adapter_config.json",
        "sft_v42i_source_report": sft_stage.REPORT,
        "lora_es_v43j_source_weights": es_stage.SOURCE / "adapter_model.safetensors",
        "lora_es_v43j_source_config": es_stage.SOURCE / "adapter_config.json",
        "lora_es_v43j_source_report": es_stage.REPORT,
        "lora_es_v43j_candidate_gate": es_stage.CANDIDATE_GATE,
        "lora_es_v43j_candidate_consensus": es_stage.CANDIDATE_CONSENSUS,
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
        "parent_v45e_preregistration": PARENT_PREREGISTRATION,
    }
    for logical, directory in STAGED_BY_LOGICAL_V46C.items():
        paths[f"{logical}_staged_weights"] = (
            directory / "adapter_model.safetensors"
        )
        paths[f"{logical}_staged_config"] = directory / "adapter_config.json"
        paths[f"{logical}_stage_manifest"] = (
            directory / "stage_manifest_v44a.json"
        )
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["v43j_source_seal"] = es_stage.source_seal_v46c()
    result["sft_v42i_source_audited"] = (
        len(sft_stage.audit_source_v45d()["records"]) == 70
    )
    return result


def assert_exact_bases_v46c(metrics: dict, label: str) -> dict:
    baseline = metrics["base_a"]
    if any(metrics[arm] != baseline for arm in BASE_ARMS_V46C[1:]):
        raise RuntimeError(f"V46C four-base exact equivalence failed on {label}")
    return {
        "label": label,
        "base_arms": list(BASE_ARMS_V46C),
        "all_four_base_outputs_exact": True,
    }


def mean_shadow_metrics_v46c(shadow: dict, logical: str) -> dict:
    replicas = LOGICAL_REPLICAS_V46C[logical]
    return {
        field: sum(float(shadow[arm][field]) for arm in replicas) / 2.0
        for field in RANK_FIELDS_V46C
    }


def logical_selection_key_v46c(shadow: dict, logical: str) -> tuple:
    mean = mean_shadow_metrics_v46c(shadow, logical)
    return tuple(mean[field] for field in RANK_FIELDS_V46C) + (
        -LOGICAL_CANDIDATES_V46C.index(logical),
    )


def _replica_gate_v46c(
    shadow: dict,
    ood_qa: dict,
    prose: dict,
    raw_sink: dict,
    arm: str,
) -> dict:
    qa_gate = core.v39a.qa_ood_gate(ood_qa["base_a"], ood_qa[arm])
    qa_gate.update(ood_first.paired_qa_bootstrap_v45a(
        raw_sink["ood_qa"]["base_a"], raw_sink["ood_qa"][arm]
    ))
    prose_gate = core.v39a.prose_gate(prose["base_a"], prose[arm])
    counters = shadow[arm]["protocol_leak_counters"]
    base_counters = shadow["base_a"]["protocol_leak_counters"]
    protocol_safe = all(
        counters[key] <= base_counters[key] for key in base_counters
    )
    return {
        "arm": arm,
        "ood_qa": qa_gate,
        "ood_prose": prose_gate,
        "no_protocol_or_leak_counter_increase": protocol_safe,
        "eligible": qa_gate["passed"] and prose_gate["passed"] and protocol_safe,
    }


def finalize_selection_v46c(
    shadow: dict, ood_qa: dict, prose: dict, raw_sink: dict,
    base_equivalence: dict,
) -> dict:
    table = {}
    for logical, replicas in LOGICAL_REPLICAS_V46C.items():
        replica_gates = [
            _replica_gate_v46c(shadow, ood_qa, prose, raw_sink, arm)
            for arm in replicas
        ]
        table[logical] = {
            "replicas": list(replicas),
            "replica_gates": replica_gates,
            "both_replicas_independently_ood_eligible": all(
                gate["eligible"] for gate in replica_gates
            ),
            "candidate_metric_or_generation_bit_equality_required": False,
            "mean_replicated_shadow_metrics": mean_shadow_metrics_v46c(
                shadow, logical
            ),
        }
        table[logical]["eligible"] = table[logical][
            "both_replicas_independently_ood_eligible"
        ]
    eligible = tuple(
        logical for logical in LOGICAL_CANDIDATES_V46C
        if table[logical]["eligible"]
    )
    selected_logical = (
        max(eligible, key=lambda item: logical_selection_key_v46c(shadow, item))
        if eligible else None
    )
    selected_arm = (
        "base_a" if selected_logical is None
        else LOGICAL_REPLICAS_V46C[selected_logical][0]
    )
    baseline_key = tuple(
        shadow["base_a"][field] for field in RANK_FIELDS_V46C
    )
    selected_key = (
        None if selected_logical is None
        else logical_selection_key_v46c(shadow, selected_logical)[:-1]
    )
    improved = selected_key is not None and selected_key > baseline_key
    return {
        "selected_arm": selected_arm,
        "selected_candidate_arm": None if selected_logical is None else selected_arm,
        "selected_logical_candidate": selected_logical,
        "eligible_logical_candidates": list(eligible),
        "ineligible_logical_candidates": [
            item for item in LOGICAL_CANDIDATES_V46C if item not in eligible
        ],
        "per_logical_candidate_gate_table": table,
        "ood_eligible_set_constructed_before_shadow_ranking": True,
        "rule": (
            "require both replicas to independently pass OOD QA, OOD prose, "
            "and protocol gates; then rank eligible logical candidates by the "
            "mean of their two replicated shadow metric vectors"
        ),
        "shadow_improvement_gate_passed": improved,
        "no_protocol_or_leak_counter_increase": (
            True if selected_logical is None else all(
                gate["no_protocol_or_leak_counter_increase"]
                for gate in table[selected_logical]["replica_gates"]
            )
        ),
        "base_duplicate_equivalence": base_equivalence,
        "all_four_base_duplicates_required_exact": True,
        "candidate_replicas_required_bit_exact": False,
        "candidate_replica_count_per_logical_arm": 2,
        "mean_replicated_shadow_ranking": True,
        "exact_tie_order": list(LOGICAL_CANDIDATES_V46C),
    }


class EvaluationStateV46C:
    def __init__(self):
        self.shadow = self.ood_qa = None
        self.base_equivalence = {}
        self.selection = None

    def provisional(self, shadow: dict) -> dict:
        self.shadow = shadow
        self.selection = {
            "selected_arm": "base_a",
            "selected_candidate_arm": None,
            "selected_logical_candidate": None,
            "eligible_logical_candidates": [],
            "ineligible_logical_candidates": list(LOGICAL_CANDIDATES_V46C),
            "rule": "pending independent replica OOD gates",
            "shadow_improvement_gate_passed": False,
            "no_protocol_or_leak_counter_increase": True,
        }
        return self.selection


def load_preregistration_v46c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V46C preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "lora-es-v43j-vs-sft-v42i-replicated-ood-first-preregistration-v46c"
        or value.get("status")
        != "preregistered_before_fresh_replicated_ood_first_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v46c_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V46C)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES_V46C)
        or value.get("replica_staged_adapters") != replica_stage_bindings_v46c()
        or value.get("staged_adapters") != replica_stage_bindings_v46c()
        or value.get("implementation_bindings") != implementation_bindings_v46c()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V46C preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    ood_first.environment.environment_bindings_v44b()
    saved = {
        "EXPERIMENT": core.EXPERIMENT,
        "RUN_DIR": core.RUN_DIR,
        "ATTEMPT": core.ATTEMPT,
        "RAW": core.RAW,
        "GPU_LOG": core.GPU_LOG,
        "REPORT": core.REPORT,
        "load": core.load_preregistration,
        "preflight": core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A,
        "select": core.select_candidate_v44a,
        "eval_qa": core.evaluate_qa_v44a,
        "eval_prose": core.evaluate_prose_v44a,
    }
    state = EvaluationStateV46C()

    def evaluate_qa(trainer, bundle, raw_sink, label):
        result = saved["eval_qa"](trainer, bundle, raw_sink, label)
        state.base_equivalence[label] = assert_exact_bases_v46c(result, label)
        if label == "ood_qa":
            state.ood_qa = result
        return result

    def evaluate_prose(trainer, rows, raw_sink):
        aggregate, detailed = saved["eval_prose"](trainer, rows, raw_sink)
        state.base_equivalence["ood_prose"] = assert_exact_bases_v46c(
            aggregate, "ood_prose"
        )
        final = finalize_selection_v46c(
            state.shadow, state.ood_qa, detailed, raw_sink,
            state.base_equivalence,
        )
        state.selection.clear()
        state.selection.update(final)
        return aggregate, detailed

    core.EXPERIMENT, core.RUN_DIR = EXPERIMENT, RUN_DIR
    core.ATTEMPT, core.RAW = ATTEMPT, RAW
    core.GPU_LOG, core.REPORT = GPU_LOG, REPORT
    core.load_preregistration = load_preregistration_v46c
    core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = (
        ood_first.parser_fix.protected_preflight_v44c
    )
    core.select_candidate_v44a = state.provisional
    core.evaluate_qa_v44a = evaluate_qa
    core.evaluate_prose_v44a = evaluate_prose
    try:
        with patched_candidate_globals_v46c():
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
