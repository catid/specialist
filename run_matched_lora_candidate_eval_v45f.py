#!/usr/bin/env python3
"""V45F: conservative per-replica OOD consensus after V45E exact failure."""

from __future__ import annotations

import json
import math
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first
import run_matched_lora_candidate_eval_v45e as prior


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v45f_matched_lora_sft_ijk_replica_consensus_ood_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "aggregate_receipts_no_raw_semantics_v45f.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v45f.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_ijk_replica_consensus_ood_eval_v45f.json"
).resolve()
V45E_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_ijk_replicated_ood_eligible_eval_v45e.json"
).resolve()
V45E_PREREG_FILE_SHA256 = (
    "e7ca432c68de6f751ffecf03595c1b5debb7d3adb3db8244ab144938c14fb34e"
)
V45E_PREREG_CONTENT_SHA256 = (
    "47c335dc6003a648b8eeb96f5b7f3ed6922a94c5502c93d6dc9e1426caeeed2a"
)
V45E_FAILURE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v45e_matched_lora_sft_ijk_replicated_ood_eligible_eval/"
    "failure_v44a.json"
).resolve()
V45E_ATTEMPT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v45e_matched_lora_sft_ijk_replicated_ood_eligible_eval.attempt.json"
).resolve()
V45E_GPU_LOG = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v45e_matched_lora_sft_ijk_replicated_ood_eligible_eval/"
    "gpu_activity_v45e.jsonl"
).resolve()
V45E_FAILURE_FILE_SHA256 = (
    "da3037e88675de8d593f75ae5124093ba83136e3e96e9dbdf323b1ff1f965e27"
)
V45E_FAILURE_CONTENT_SHA256 = (
    "80ea753043a56238bf903d83947e06d1d22c3cbde4f5236c14d4f349036ffd58"
)
V45E_ATTEMPT_FILE_SHA256 = (
    "57bd6d4a73f86a53ae036316f79801de38eb91a68dfe06760369a7cf88666484"
)
V45E_ATTEMPT_CONTENT_SHA256 = (
    "8af154e2402bc3ab546192d475462ac237a0fd7933956011a0ad8627c609f9a8"
)
V45E_GPU_LOG_FILE_SHA256 = (
    "49c4359fdcf8b49ec9bb882f306439211b29dafd04391fd1fde138ddd4bcf1b6"
)


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def prior_preregistration_v45f() -> dict:
    if core.file_sha256(V45E_PREREG) != V45E_PREREG_FILE_SHA256:
        raise RuntimeError("V45F V45E preregistration file changed")
    value = json.loads(V45E_PREREG.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != V45E_PREREG_CONTENT_SHA256
        or _compact_sha(value) != V45E_PREREG_CONTENT_SHA256
        or value.get("heldout_or_holdout_access_authorized") is not False
    ):
        raise RuntimeError("V45F V45E preregistration content changed")
    return value


def failure_evidence_v45f() -> dict:
    if (
        core.file_sha256(V45E_FAILURE) != V45E_FAILURE_FILE_SHA256
        or core.file_sha256(V45E_ATTEMPT) != V45E_ATTEMPT_FILE_SHA256
        or core.file_sha256(V45E_GPU_LOG) != V45E_GPU_LOG_FILE_SHA256
    ):
        raise RuntimeError("V45F V45E failure artifact changed")
    failure = json.loads(V45E_FAILURE.read_text())
    attempt = json.loads(V45E_ATTEMPT.read_text())
    if (
        failure.get("content_sha256_before_self_field")
        != V45E_FAILURE_CONTENT_SHA256
        or _compact_sha(failure) != V45E_FAILURE_CONTENT_SHA256
        or failure.get("schema") != "matched-lora-candidate-eval-failure-v44a"
        or failure.get("type") != "RuntimeError"
        or failure.get("message")
        != "V45E sft_v42i replica equivalence failed on shadow"
        or failure.get("protected_semantic_access_count") != 4
        or failure.get("heldout_or_holdout_opened") is not False
        or attempt.get("content_sha256_before_self_field")
        != V45E_ATTEMPT_CONTENT_SHA256
        or _compact_sha(attempt) != V45E_ATTEMPT_CONTENT_SHA256
        or attempt.get("preregistration_file_sha256")
        != V45E_PREREG_FILE_SHA256
        or attempt.get("preregistration_content_sha256")
        != V45E_PREREG_CONTENT_SHA256
        or attempt.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("V45F V45E failure evidence content changed")
    return {
        "failure_path": str(V45E_FAILURE),
        "failure_file_sha256": V45E_FAILURE_FILE_SHA256,
        "failure_content_sha256": V45E_FAILURE_CONTENT_SHA256,
        "attempt_path": str(V45E_ATTEMPT),
        "attempt_file_sha256": V45E_ATTEMPT_FILE_SHA256,
        "attempt_content_sha256": V45E_ATTEMPT_CONTENT_SHA256,
        "gpu_log_path": str(V45E_GPU_LOG),
        "gpu_log_file_sha256": V45E_GPU_LOG_FILE_SHA256,
        "failed_rule": "cross-GPU candidate replica bit-exact shadow equality",
        "protected_parser_preflight_semantic_access_count": 4,
        "protected_parser_preflight_loaded_all_splits_before_model_creation": True,
        "failure_occurred_during_shadow_scoring_before_ood_metrics": True,
        "replacement_rule": (
            "both replicas independently pass every OOD and protocol gate; "
            "rank eligible logical candidates on replicated means"
        ),
        "heldout_or_holdout_opened": False,
        "raw_semantics_inspected_for_revision": False,
    }


def implementation_bindings_v45f() -> dict:
    return {
        "runtime": core.file_sha256(Path(__file__).resolve()),
        "v45e_runtime": core.file_sha256(Path(prior.__file__).resolve()),
        "v45e_preregistration_file_sha256": V45E_PREREG_FILE_SHA256,
        "v45e_preregistration_content_sha256": V45E_PREREG_CONTENT_SHA256,
        "v45e_bound_implementation": prior.implementation_bindings_v45e(),
        "v45e_failure_evidence": failure_evidence_v45f(),
    }


def assert_base_equivalence_v45f(metrics: dict, label: str) -> dict:
    baseline = metrics["base_a"]
    if any(metrics[arm] != baseline for arm in prior.BASE_ARMS_V45E[1:]):
        raise RuntimeError(f"V45F six-base equivalence failed on {label}")
    return {
        "label": label, "all_six_base_outputs_exact": True,
        "primary_base_arms": list(prior.PRIMARY_BASE_ARMS_V45E),
        "padding_base_arms": list(prior.PADDING_BASE_ARMS_V45E),
    }


def replicated_summary_v45f(metrics: dict, replicas: tuple[str, str],
                            keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        values_by_replica = {
            replica: float(metrics[replica][key]) for replica in replicas
        }
        values = list(values_by_replica.values())
        low, high = min(values), max(values)
        result[key] = {
            "values_by_replica": values_by_replica,
            "mean": math.fsum(values) / len(values),
            "min": low, "max": high, "range": high - low,
        }
    return result


def all_replica_ranges_v45f(shadow: dict, ood_qa: dict,
                            ood_prose: dict) -> dict:
    result = {}
    qa_keys = (
        "generated_equal_unit_mean_reward", "generated_row_mean_reward",
        "generated_exact_count", "generated_nonzero_count",
        "teacher_forced_equal_unit_mean_answer_logprob",
    )
    for logical, replicas in prior.LOGICAL_REPLICAS_V45E.items():
        result[logical] = {
            "shadow": replicated_summary_v45f(shadow, replicas, qa_keys),
            "ood_qa": replicated_summary_v45f(ood_qa, replicas, qa_keys),
            "ood_prose": replicated_summary_v45f(
                ood_prose, replicas, ("mean_token_logprob",)
            ),
        }
    return result


def logical_mean_selection_key_v45f(ranges: dict, logical: str) -> tuple:
    shadow = ranges[logical]["shadow"]
    return (
        shadow["generated_equal_unit_mean_reward"]["mean"],
        shadow["generated_exact_count"]["mean"],
        shadow["generated_nonzero_count"]["mean"],
        shadow["teacher_forced_equal_unit_mean_answer_logprob"]["mean"],
        prior.LOGICAL_CANDIDATES_V45E.index(logical),
    )


def finalize_selection_v45f(shadow: dict, ood_qa: dict, prose: dict,
                            raw_sink: dict, base_equivalence: dict) -> dict:
    ranges = all_replica_ranges_v45f(shadow, ood_qa, prose)
    table = {}
    base_qa, base_prose = ood_qa["base_a"], prose["base_a"]
    for logical, replicas in prior.LOGICAL_REPLICAS_V45E.items():
        replica_gates = {}
        for replica in replicas:
            qa_gate = core.v39a.qa_ood_gate(base_qa, ood_qa[replica])
            qa_gate.update(ood_first.paired_qa_bootstrap_v45a(
                raw_sink["ood_qa"]["base_a"], raw_sink["ood_qa"][replica]
            ))
            prose_gate = core.v39a.prose_gate(base_prose, prose[replica])
            counters = shadow[replica]["protocol_leak_counters"]
            base_counters = shadow["base_a"]["protocol_leak_counters"]
            protocol_safe = all(
                counters[key] <= base_counters[key] for key in base_counters
            )
            replica_gates[replica] = {
                "ood_qa": qa_gate, "ood_prose": prose_gate,
                "no_protocol_or_leak_counter_increase": protocol_safe,
                "eligible": (
                    qa_gate["passed"] and prose_gate["passed"] and protocol_safe
                ),
            }
        eligible = all(item["eligible"] for item in replica_gates.values())
        table[logical] = {
            "replica_gates": replica_gates,
            "both_replicas_independently_eligible": eligible,
            "eligible": eligible,
            "replica_metric_ranges": ranges[logical],
        }
    eligible = tuple(
        logical for logical in prior.LOGICAL_CANDIDATES_V45E
        if table[logical]["eligible"]
    )
    selected_logical = (
        max(eligible, key=lambda arm: logical_mean_selection_key_v45f(ranges, arm))
        if eligible else None
    )
    selected_arm = (
        "base_a" if selected_logical is None
        else prior.LOGICAL_REPLICAS_V45E[selected_logical][0]
    )
    base_key = (
        shadow["base_a"]["generated_equal_unit_mean_reward"],
        shadow["base_a"]["generated_exact_count"],
        shadow["base_a"]["generated_nonzero_count"],
        shadow["base_a"]["teacher_forced_equal_unit_mean_answer_logprob"],
    )
    improved = (
        selected_logical is not None
        and logical_mean_selection_key_v45f(ranges, selected_logical)[:-1]
        > base_key
    )
    return {
        "selected_arm": selected_arm,
        "selected_candidate_arm": None if selected_logical is None else selected_arm,
        "selected_logical_candidate": selected_logical,
        "eligible_logical_candidates": list(eligible),
        "ineligible_logical_candidates": [
            arm for arm in prior.LOGICAL_CANDIDATES_V45E if arm not in eligible
        ],
        "per_logical_candidate_consensus_gate_table": table,
        "all_replica_metric_ranges": ranges,
        "ood_eligible_set_constructed_before_shadow_ranking": True,
        "logical_eligibility_requires_both_replicas": True,
        "rule": (
            "gate each replica independently on OOD QA reward+exact, OOD prose "
            "point+LCB, and protocol safety; a logical candidate is eligible "
            "only if both pass; rank eligible logical candidates on replicated "
            "mean shadow reward, exact, nonzero, then teacher logprob"
        ),
        "shadow_improvement_gate_passed": improved,
        "no_protocol_or_leak_counter_increase": (
            True if selected_logical is None else all(
                gate["no_protocol_or_leak_counter_increase"]
                for gate in table[selected_logical]["replica_gates"].values()
            )
        ),
        "base_equivalence": base_equivalence,
        "all_six_base_duplicates_required_exact": True,
        "padding_base_arms": list(prior.PADDING_BASE_ARMS_V45E),
        "padding_base_arms_excluded_from_eligibility_and_ranking": True,
        "candidate_replica_bit_exact_equivalence_required": False,
        "raw_questions_answers_or_generations_persisted": False,
    }


class EvaluationStateV45F:
    def __init__(self):
        self.shadow = self.ood_qa = None
        self.base_equivalence = {}
        self.selection = None

    def provisional(self, shadow: dict) -> dict:
        self.shadow = shadow
        self.selection = {
            "selected_arm": "base_a", "selected_candidate_arm": None,
            "selected_logical_candidate": None,
            "eligible_logical_candidates": [],
            "ineligible_logical_candidates": list(prior.LOGICAL_CANDIDATES_V45E),
            "rule": "pending per-replica OOD consensus",
            "shadow_improvement_gate_passed": False,
            "no_protocol_or_leak_counter_increase": True,
        }
        return self.selection


def load_preregistration_v45f(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V45F preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "matched-lora-sft-ijk-replica-consensus-preregistration-v45f"
        or value.get("status")
        != "preregistered_before_fresh_replica_consensus_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v45f_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(prior.ARMS_V45E)
        or value.get("logical_candidates")
        != list(prior.LOGICAL_CANDIDATES_V45E)
        or value.get("replica_staged_adapters")
        != prior.replica_stage_bindings_v45e()
        or value.get("implementation_bindings") != implementation_bindings_v45f()
        or value.get("v45e_failure_evidence") != failure_evidence_v45f()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V45F preregistration content changed")
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
    state = EvaluationStateV45F()

    def evaluate_qa(trainer, bundle, raw_sink, label):
        result = saved["eval_qa"](trainer, bundle, raw_sink, label)
        state.base_equivalence[label] = assert_base_equivalence_v45f(result, label)
        if label == "ood_qa":
            state.ood_qa = result
        return result

    def evaluate_prose(trainer, rows, raw_sink):
        aggregate, detailed = saved["eval_prose"](trainer, rows, raw_sink)
        state.base_equivalence["ood_prose"] = assert_base_equivalence_v45f(
            aggregate, "ood_prose"
        )
        final = finalize_selection_v45f(
            state.shadow, state.ood_qa, detailed, raw_sink,
            state.base_equivalence,
        )
        state.selection.clear()
        state.selection.update(final)
        # Bootstrap and gates are complete. Persist only aggregate selection and
        # access receipts; never persist questions, answers, or generations.
        raw_sink.clear()
        raw_sink.update({
            "schema": "aggregate-receipts-no-raw-semantics-v45f",
            "raw_questions_answers_or_generations_persisted": False,
        })
        return aggregate, detailed

    core.EXPERIMENT, core.RUN_DIR = EXPERIMENT, RUN_DIR
    core.ATTEMPT, core.RAW = ATTEMPT, RAW
    core.GPU_LOG, core.REPORT = GPU_LOG, REPORT
    core.load_preregistration = load_preregistration_v45f
    core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = (
        ood_first.parser_fix.protected_preflight_v44c
    )
    core.select_candidate_v44a = state.provisional
    core.evaluate_qa_v44a = evaluate_qa
    core.evaluate_prose_v44a = evaluate_prose
    try:
        with prior.patched_candidate_globals_v45e():
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
