#!/usr/bin/env python3
"""Matched SFT/ES evaluator with OOD-eligible-set-first selection."""

from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44b as environment
import run_matched_lora_candidate_eval_v44c as parser_fix
import stage_candidate_adapters_vllm_v45a as staging


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V45A = ("base_a", "base_b", "base_c")
CANDIDATE_ARMS_V45A = (
    "sft_v42b_step16", "sft_v42b_step32",
    "sft_v42b", "sft_v42c", "sft_v42d", "lora_es_v43d",
    "sft_v42e", "sft_v42f", "sft_v42g",
)
ARMS_V45A = BASE_ARMS_V45A + CANDIDATE_ARMS_V45A
STAGED_BY_ARM_V45A = {
    arm: staging.STAGED_BY_ARM_V45A[arm] for arm in CANDIDATE_ARMS_V45A
}
ADAPTER_IDS_V45A = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V45A)
}

EXPERIMENT = "v45a_matched_lora_hpo_earlystop_ood_eligible_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v45a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v45a.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_hpo_earlystop_ood_eligible_eval_v45a.json"
).resolve()
V44C_REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44c.json"
).resolve()
V44C_REPORT_FILE_SHA256 = (
    "3b3a55b9b208fd563825f09c36f855d74f05cad2bf9201467fb454a08105b076"
)
V44C_REPORT_CONTENT_SHA256 = (
    "ff2b4571cf81a9d73fa890e9a179a087b1af912d6f16b16d3c3e81e13534864c"
)


def arm_wave_plan_v45a() -> tuple[tuple[tuple[str, int], ...], ...]:
    """Three full waves: every GPU gets one identical-batch request per wave."""
    return (
        (("base_a", 0), ("base_b", 1), ("base_c", 2),
         ("sft_v42b_step16", 3)),
        (("sft_v42b_step32", 0), ("sft_v42b", 1),
         ("sft_v42c", 2), ("sft_v42d", 3)),
        (("lora_es_v43d", 0), ("sft_v42e", 1),
         ("sft_v42f", 2), ("sft_v42g", 3)),
    )


@contextmanager
def patched_candidate_globals_v45a():
    saved = {
        "BASE_ARMS": core.BASE_ARMS,
        "CANDIDATE_ARMS": core.CANDIDATE_ARMS,
        "ARMS": core.ARMS,
        "STAGED_BY_ARM": core.STAGED_BY_ARM,
        "ADAPTER_IDS": core.ADAPTER_IDS_V44A,
        "ENGINE_INDEX": core.ENGINE_INDEX_BY_ARM_V44A,
        "wave": core.arm_wave_plan_v44a,
    }
    core.BASE_ARMS = BASE_ARMS_V45A
    core.CANDIDATE_ARMS = CANDIDATE_ARMS_V45A
    core.ARMS = ARMS_V45A
    core.STAGED_BY_ARM = STAGED_BY_ARM_V45A
    core.ADAPTER_IDS_V44A = ADAPTER_IDS_V45A
    core.ENGINE_INDEX_BY_ARM_V44A = {
        arm: engine for wave in arm_wave_plan_v45a() for arm, engine in wave
    }
    core.arm_wave_plan_v44a = arm_wave_plan_v45a
    try:
        yield
    finally:
        core.BASE_ARMS = saved["BASE_ARMS"]
        core.CANDIDATE_ARMS = saved["CANDIDATE_ARMS"]
        core.ARMS = saved["ARMS"]
        core.STAGED_BY_ARM = saved["STAGED_BY_ARM"]
        core.ADAPTER_IDS_V44A = saved["ADAPTER_IDS"]
        core.ENGINE_INDEX_BY_ARM_V44A = saved["ENGINE_INDEX"]
        core.arm_wave_plan_v44a = saved["wave"]


def staged_adapter_bindings_v45a() -> dict:
    with patched_candidate_globals_v45a():
        value = core.staged_adapter_bindings_v44a()
    if tuple(value) != CANDIDATE_ARMS_V45A:
        raise RuntimeError("V45A staged candidate order changed")
    return value


def v44c_finding_provenance_v45a() -> dict:
    if core.file_sha256(V44C_REPORT) != V44C_REPORT_FILE_SHA256:
        raise RuntimeError("V45A V44C report file changed")
    value = json.loads(V44C_REPORT.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V44C_REPORT_CONTENT_SHA256
        or core.canonical_sha256(compact) != V44C_REPORT_CONTENT_SHA256
        or value.get("heldout_or_holdout_opened") is not False
        or value.get("selection", {}).get("selected_arm") != "sft_v42b"
        or value.get("ood_qa_gate", {}).get("passed") is not False
    ):
        raise RuntimeError("V45A V44C selection finding changed")
    return {
        "path": str(V44C_REPORT),
        "file_sha256": V44C_REPORT_FILE_SHA256,
        "content_sha256": V44C_REPORT_CONTENT_SHA256,
        "heldout_or_holdout_opened": False,
        "finding": (
            "shadow-first V42B selection failed OOD while a lower-shadow arm "
            "was point-safe; V45A therefore filters every arm through frozen "
            "OOD gates before shadow ranking"
        ),
        "selection_metrics_observed_before_v45a_preregistration": True,
    }


def implementation_bindings_v45a() -> dict:
    extra = {}
    for arm, spec in staging.SOURCE_SPECS_V45A.items():
        extra[f"{arm}_source_weights"] = core.file_sha256(
            spec["source"] / "adapter_model.safetensors"
        )
        extra[f"{arm}_source_config"] = core.file_sha256(
            spec["source"] / "adapter_config.json"
        )
        extra[f"{arm}_source_report"] = core.file_sha256(spec["report"])
        if "trainer_state_sha256" in spec:
            extra[f"{arm}_source_trainer_state"] = core.file_sha256(
                spec["source"] / "trainer_state.json"
            )
        extra[f"{arm}_staged_weights"] = core.file_sha256(
            spec["output"] / "adapter_model.safetensors"
        )
        extra[f"{arm}_staged_config"] = core.file_sha256(
            spec["output"] / "adapter_config.json"
        )
        extra[f"{arm}_stage_manifest"] = core.file_sha256(
            spec["output"] / "stage_manifest_v44a.json"
        )
    return {
        "runtime": core.file_sha256(Path(__file__).resolve()),
        "core_and_original_four_candidates": core.implementation_bindings_v44a(),
        "source_faithful_parser_runtime": core.file_sha256(
            Path(parser_fix.__file__).resolve()
        ),
        "environment_runtime": core.file_sha256(
            Path(environment.__file__).resolve()
        ),
        "v45_staging_runtime": core.file_sha256(Path(staging.__file__).resolve()),
        "worker_extension": core.file_sha256(
            ROOT / "eggroll_es_worker_lora_topology_v40a.py"
        ),
        "environment": environment.environment_bindings_v44b(),
        "additional_candidate_artifacts": extra,
        "v44c_finding": v44c_finding_provenance_v45a(),
    }


def paired_qa_bootstrap_v45a(base_rows: list[dict], candidate_rows: list[dict],
                              samples: int = core.BOOTSTRAP_SAMPLES) -> dict:
    if len(base_rows) != len(candidate_rows) or not base_rows:
        raise RuntimeError("V45A paired OOD QA rows changed")
    reward_deltas, exact_deltas = [], []
    for baseline, candidate in zip(base_rows, candidate_rows, strict=True):
        if baseline["item_sha256"] != candidate["item_sha256"]:
            raise RuntimeError("V45A paired OOD QA identities changed")
        reward_deltas.append(float(candidate["reward"]) - float(baseline["reward"]))
        exact_deltas.append(
            int(candidate["format"] == "exact") - int(baseline["format"] == "exact")
        )
    rng = random.Random(core.BOOTSTRAP_SEED)
    reward_samples, exact_samples = [], []
    count = len(reward_deltas)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        reward_samples.append(math.fsum(reward_deltas[i] for i in indices) / count)
        exact_samples.append(math.fsum(exact_deltas[i] for i in indices) / count)
    return {
        "reward_mean_delta_paired_item_bootstrap_95_ci": [
            core.base.linear_percentile(reward_samples, 0.025),
            core.base.linear_percentile(reward_samples, 0.975),
        ],
        "exact_rate_delta_paired_item_bootstrap_95_ci": [
            core.base.linear_percentile(exact_samples, 0.025),
            core.base.linear_percentile(exact_samples, 0.975),
        ],
        "bootstrap": {
            "unit": "ood_qa_item", "item_count": count,
            "samples": samples, "seed": core.BOOTSTRAP_SEED,
            "percentiles": [0.025, 0.975],
            "ci_is_informational_point_gates_are_preregistered": True,
        },
    }


def choose_eligible_candidate_v45a(shadow_metrics: dict,
                                    gate_table: dict) -> dict:
    def selection_key(arm: str) -> tuple:
        metrics = shadow_metrics[arm]
        return (
            metrics["generated_equal_unit_mean_reward"],
            metrics["generated_exact_count"],
            metrics["generated_nonzero_count"],
            metrics["teacher_forced_equal_unit_mean_answer_logprob"],
            ADAPTER_IDS_V45A.get(arm, 0),
        )
    eligible = tuple(
        arm for arm in CANDIDATE_ARMS_V45A if gate_table[arm]["eligible"]
    )
    selected = (
        max(eligible, key=selection_key) if eligible else "base_a"
    )
    candidate = shadow_metrics[selected]
    baseline = shadow_metrics["base_a"]
    protocol_safe = selected == "base_a" or gate_table[selected][
        "no_protocol_or_leak_counter_increase"
    ]
    improved = (
        selected != "base_a"
        and selection_key(selected)[:-1] > selection_key("base_a")[:-1]
        and protocol_safe
    )
    return {
        "selected_arm": selected,
        "selected_candidate_arm": None if selected == "base_a" else selected,
        "eligible_arms": list(eligible),
        "ineligible_arms": [
            arm for arm in CANDIDATE_ARMS_V45A if arm not in eligible
        ],
        "rule": (
            "filter every candidate on OOD QA mean+exact point non-degradation, "
            "OOD prose point+paired-document-bootstrap-LCB non-degradation, and "
            "shadow protocol safety; rank only eligible arms lexicographically "
            "on frozen shadow metrics; frozen candidate order breaks exact ties"
        ),
        "ood_eligible_set_constructed_before_shadow_ranking": True,
        "per_arm_gate_table": gate_table,
        "shadow_improvement_gate_passed": improved,
        "no_protocol_or_leak_counter_increase": protocol_safe,
        "all_three_base_duplicates_required_exact": True,
    }


def finalize_selection_v45a(shadow_metrics: dict, ood_qa_metrics: dict,
                            prose_details: dict, raw_sink: dict) -> dict:
    baseline_qa = ood_qa_metrics["base_a"]
    baseline_prose = prose_details["base_a"]
    base_raw = raw_sink["ood_qa"]["base_a"]
    table = {}
    for arm in CANDIDATE_ARMS_V45A:
        qa_gate = core.v39a.qa_ood_gate(baseline_qa, ood_qa_metrics[arm])
        qa_gate.update(paired_qa_bootstrap_v45a(
            base_raw, raw_sink["ood_qa"][arm]
        ))
        prose_gate = core.v39a.prose_gate(baseline_prose, prose_details[arm])
        counters = shadow_metrics[arm]["protocol_leak_counters"]
        base_counters = shadow_metrics["base_a"]["protocol_leak_counters"]
        protocol_safe = all(
            counters[key] <= base_counters[key] for key in base_counters
        )
        eligible = qa_gate["passed"] and prose_gate["passed"] and protocol_safe
        table[arm] = {
            "ood_qa": qa_gate,
            "ood_prose": prose_gate,
            "no_protocol_or_leak_counter_increase": protocol_safe,
            "eligible": eligible,
        }
    return choose_eligible_candidate_v45a(shadow_metrics, table)


class EvaluationStateV45A:
    def __init__(self):
        self.shadow = self.ood_qa = None
        self.selection = None

    def provisional(self, shadow_metrics: dict) -> dict:
        self.shadow = shadow_metrics
        self.selection = {
            "selected_arm": "base_a", "selected_candidate_arm": None,
            "eligible_arms": [], "ineligible_arms": list(CANDIDATE_ARMS_V45A),
            "rule": "pending preregistered OOD-eligible-set construction",
            "shadow_improvement_gate_passed": False,
            "no_protocol_or_leak_counter_increase": True,
        }
        return self.selection


def load_preregistration_v45a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V45A preregistration file identity changed")
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
        != "matched-lora-ood-eligible-eval-preregistration-v45a"
        or value.get("status")
        != "preregistered_before_fresh_ood_eligible_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V45A)
        or value.get("candidate_arms") != list(CANDIDATE_ARMS_V45A)
        or value.get("staged_adapters") != staged_adapter_bindings_v45a()
        or value.get("implementation_bindings") != implementation_bindings_v45a()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V45A preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    environment.environment_bindings_v44b()
    state = EvaluationStateV45A()
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

    def evaluate_qa(trainer, bundle, raw_sink, label):
        result = saved["eval_qa"](trainer, bundle, raw_sink, label)
        if not (result["base_a"] == result["base_b"] == result["base_c"]):
            raise RuntimeError(f"V45A three-base equivalence failed on {label}")
        if label == "ood_qa":
            state.ood_qa = result
        return result

    def evaluate_prose(trainer, rows, raw_sink):
        aggregate, detailed = saved["eval_prose"](trainer, rows, raw_sink)
        if not (
            aggregate["base_a"] == aggregate["base_b"] == aggregate["base_c"]
        ):
            raise RuntimeError("V45A three-base prose equivalence failed")
        final = finalize_selection_v45a(
            state.shadow, state.ood_qa, detailed, raw_sink
        )
        state.selection.clear()
        state.selection.update(final)
        return aggregate, detailed

    core.EXPERIMENT = EXPERIMENT
    core.RUN_DIR = RUN_DIR
    core.ATTEMPT = ATTEMPT
    core.RAW = RAW
    core.GPU_LOG = GPU_LOG
    core.REPORT = REPORT
    core.load_preregistration = load_preregistration_v45a
    core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = parser_fix.protected_preflight_v44c
    core.select_candidate_v44a = state.provisional
    core.evaluate_qa_v44a = evaluate_qa
    core.evaluate_prose_v44a = evaluate_prose
    try:
        with patched_candidate_globals_v45a():
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
