#!/usr/bin/env python3
"""CPU-only fail-closed template for a future V43I-versus-SFT evaluation.

This revision intentionally cannot launch.  It freezes the four-GPU arm
schedule, exact duplicate checks, protected single-access firewall, and
OOD-eligibility-before-shadow-ranking rule while both candidate identities are
still unknown.  A later sealed revision must supply and bind those identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V46B = ("base_a", "base_b", "base_c", "base_d")
CANDIDATE_REPLICA_ARMS_V46B = (
    "sft_boundary_winner_a", "sft_boundary_winner_b",
    "lora_es_v43i_a", "lora_es_v43i_b",
)
ARMS_V46B = BASE_ARMS_V46B + CANDIDATE_REPLICA_ARMS_V46B
CANDIDATE_FAMILIES_V46B = {
    "sft_boundary_winner": (
        "sft_boundary_winner_a", "sft_boundary_winner_b",
    ),
    "lora_es_v43i": ("lora_es_v43i_a", "lora_es_v43i_b"),
}
FAMILY_TIE_ORDER_V46B = ("sft_boundary_winner", "lora_es_v43i")
EXPERIMENT = "v46b_lora_es_v43i_vs_sft_boundary_ood_first_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v46b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v46b.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "lora_es_v43i_vs_sft_boundary_ood_first_eval_v46b.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v43i_vs_sft_boundary_ood_first_template_v46b.json"
).resolve()
V43I_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_multi_anchor_v43i.json"
).resolve()
V45D_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_boundary_ood_eligible_eval_v45d.json"
).resolve()
V43I_PREREGISTRATION_FILE_SHA256 = (
    "00c545926b217a64acabbc541f3e92e071a1a199dbabef121383c788f574272e"
)
V43I_PREREGISTRATION_CONTENT_SHA256 = (
    "086d94f1b69732a9a0d7913c8bab7789b15f64131f125ba4381eea3bcc228c5a"
)
V45D_PREREGISTRATION_FILE_SHA256 = (
    "bc06e84381b35e8f477baf1b0df0a846e9b51e0ca7a210279b79fe0b62812d25"
)
V45D_PREREGISTRATION_CONTENT_SHA256 = (
    "20c6608e77e4b6de79ff9110092cb458501dfa13682d36e9d554f1f8cc2dc04f"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def arm_wave_plan_v46b() -> tuple[tuple[tuple[str, int], ...], ...]:
    """Two full waves keep all four GPUs occupied in the future evaluation."""
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS_V46B)),
        tuple((arm, index) for index, arm in enumerate(
            CANDIDATE_REPLICA_ARMS_V46B
        )),
    )


def pending_candidate_seals_v46b() -> dict:
    return {
        "lora_es_v43i": {
            "status": "pending_exact_post_v43i_run_seal",
            "required": {
                "train_report_path": None,
                "train_report_file_sha256": None,
                "train_report_content_sha256": None,
                "snapshot_directory": None,
                "snapshot_weights_sha256": None,
                "snapshot_config_sha256": None,
                "canonical_adapter_identity_sha256": None,
                "staged_directory": None,
                "staged_weights_sha256": None,
                "staged_config_sha256": None,
                "stage_manifest_sha256": None,
            },
        },
        "sft_boundary_winner": {
            "status": "pending_fixed_v45d_winner_and_exact_artifact_seal",
            "required": {
                "v45d_report_path": None,
                "v45d_report_file_sha256": None,
                "v45d_report_content_sha256": None,
                "selected_arm": None,
                "selected_source_weights_sha256": None,
                "selected_source_config_sha256": None,
                "selected_source_report_file_sha256": None,
                "selected_source_report_content_sha256": None,
                "staged_directory": None,
                "staged_weights_sha256": None,
                "staged_config_sha256": None,
                "stage_manifest_sha256": None,
            },
        },
    }


def implementation_bindings_v46b() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_vs_sft_ood_first_template_v46b.py",
        "tests": ROOT / "test_lora_es_vs_sft_ood_first_template_v46b.py",
        "protected_firewall_runtime": Path(core.__file__).resolve(),
        "ood_first_selection_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_preflight_runtime": Path(
            ood_first.parser_fix.__file__
        ).resolve(),
        "v43i_preregistration": V43I_PREREGISTRATION,
        "v45d_preregistration": V45D_PREREGISTRATION,
    }
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    if (
        result["v43i_preregistration"]
        != V43I_PREREGISTRATION_FILE_SHA256
        or result["v45d_preregistration"]
        != V45D_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("V46B parent preregistration identity changed")
    return result


def _assert_exact_replicas(values: dict, label: str) -> None:
    baseline = values[BASE_ARMS_V46B[0]]
    if any(values[arm] != baseline for arm in BASE_ARMS_V46B[1:]):
        raise RuntimeError(f"V46B four-base exact equivalence failed on {label}")
    for family, replicas in CANDIDATE_FAMILIES_V46B.items():
        if values[replicas[0]] != values[replicas[1]]:
            raise RuntimeError(
                f"V46B {family} two-GPU exact equivalence failed on {label}"
            )


def _selection_key_v46b(metrics: dict, family: str) -> tuple:
    arm = CANDIDATE_FAMILIES_V46B[family][0]
    value = metrics[arm]
    return (
        value["generated_equal_unit_mean_reward"],
        value["generated_exact_count"],
        value["generated_nonzero_count"],
        value["teacher_forced_equal_unit_mean_answer_logprob"],
        FAMILY_TIE_ORDER_V46B.index(family),
    )


def choose_eligible_family_v46b(
    shadow_metrics: dict, family_gate_table: dict,
) -> dict:
    """Rank shadow metrics only after the OOD-eligible set is immutable."""
    eligible = tuple(
        family for family in FAMILY_TIE_ORDER_V46B
        if family_gate_table[family]["eligible"]
    )
    selected_family = (
        max(eligible, key=lambda item: _selection_key_v46b(
            shadow_metrics, item,
        )) if eligible else None
    )
    selected_arm = (
        "base_a" if selected_family is None
        else CANDIDATE_FAMILIES_V46B[selected_family][0]
    )
    baseline_key = (
        shadow_metrics["base_a"]["generated_equal_unit_mean_reward"],
        shadow_metrics["base_a"]["generated_exact_count"],
        shadow_metrics["base_a"]["generated_nonzero_count"],
        shadow_metrics["base_a"][
            "teacher_forced_equal_unit_mean_answer_logprob"
        ],
    )
    selected_key = (
        None if selected_family is None
        else _selection_key_v46b(shadow_metrics, selected_family)[:-1]
    )
    improved = selected_key is not None and selected_key > baseline_key
    return {
        "selected_arm": selected_arm,
        "selected_family": selected_family,
        "eligible_families": list(eligible),
        "ineligible_families": [
            family for family in FAMILY_TIE_ORDER_V46B
            if family not in eligible
        ],
        "family_gate_table": family_gate_table,
        "rule": (
            "freeze each family OOD-QA, OOD-prose, and protocol eligibility "
            "before reading family shadow values for lexicographic ranking"
        ),
        "ood_eligible_set_constructed_before_shadow_ranking": True,
        "shadow_improvement_gate_passed": improved,
        "all_four_base_duplicates_required_exact": True,
        "both_candidate_families_require_two_gpu_exact_replicas": True,
    }


def finalize_selection_v46b(
    shadow_metrics: dict,
    ood_qa_metrics: dict,
    prose_details: dict,
    raw_sink: dict,
) -> dict:
    for label, values in (
        ("shadow", shadow_metrics),
        ("ood_qa", ood_qa_metrics),
        ("ood_prose", prose_details),
    ):
        _assert_exact_replicas(values, label)
    baseline_qa = ood_qa_metrics["base_a"]
    baseline_prose = prose_details["base_a"]
    table = {}
    for family, replicas in CANDIDATE_FAMILIES_V46B.items():
        arm = replicas[0]
        if raw_sink["ood_qa"][replicas[0]] != raw_sink["ood_qa"][replicas[1]]:
            raise RuntimeError(f"V46B {family} raw OOD-QA replicas changed")
        qa_gate = core.v39a.qa_ood_gate(baseline_qa, ood_qa_metrics[arm])
        qa_gate.update(ood_first.paired_qa_bootstrap_v45a(
            raw_sink["ood_qa"]["base_a"], raw_sink["ood_qa"][arm],
        ))
        prose_gate = core.v39a.prose_gate(baseline_prose, prose_details[arm])
        counters = shadow_metrics[arm]["protocol_leak_counters"]
        base_counters = shadow_metrics["base_a"]["protocol_leak_counters"]
        protocol_safe = all(
            counters[key] <= base_counters[key] for key in base_counters
        )
        table[family] = {
            "ood_qa": qa_gate,
            "ood_prose": prose_gate,
            "no_protocol_or_leak_counter_increase": protocol_safe,
            "eligible": (
                qa_gate["passed"] and prose_gate["passed"] and protocol_safe
            ),
            "eligibility_finalized_before_shadow_ranking": True,
        }
    return choose_eligible_family_v46b(shadow_metrics, table)


def load_preregistration_v46b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V46B template file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256(compact)
        or value.get("schema")
        != "lora-es-v43i-vs-sft-ood-first-eval-template-v46b"
        or value.get("status")
        != "blocked_pending_v43i_and_v45d_exact_candidate_seals"
        or value.get("gpu_launch_authorized") is not False
        or value.get("evaluation_launch_authorized") is not False
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_v46b_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V46B)
        or value.get("candidate_replica_arms")
        != list(CANDIDATE_REPLICA_ARMS_V46B)
        or value.get("candidate_artifact_seals")
        != pending_candidate_seals_v46b()
        or value.get("implementation_bindings") != implementation_bindings_v46b()
    ):
        raise RuntimeError("V46B fail-closed template contract changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    value = load_preregistration_v46b(args)
    summary = {
        "schema": value["schema"],
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"],
        "launch_authorized": False,
        "gpu_launched": False,
        "model_runtime_loaded": False,
        "protected_semantic_access_count": 0,
        "protected_paths_opened": [],
        "heldout_or_holdout_opened": False,
        "missing_candidate_seals": ["lora_es_v43i", "sft_boundary_winner"],
    }
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise RuntimeError(
        "V46B is an unlaunchable CPU template; seal exact V43I snapshot and "
        "fixed V45D winner identities in a new preregistered revision first"
    )


if __name__ == "__main__":
    raise SystemExit(main())
