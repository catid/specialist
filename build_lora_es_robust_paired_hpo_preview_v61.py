#!/usr/bin/env python3
"""Freeze a launch-ineligible V61 paired/block-bootstrap HPO preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_robust_paired_hpo_v61 as design


ROOT = Path(__file__).resolve().parent
V61A_STRATA = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61a_v434_train_only_baseline_census/baseline_census_strata_v61a.json"
).resolve()
V61B_ANALYSIS = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61b_v434_common_seed_repeat_census/common_seed_repeat_analysis_v61b.json"
).resolve()
V61B_REPORT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61b_v434_common_seed_repeat_census/common_seed_repeat_report_v61b.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_paired_block_bootstrap_v61_preview.json"
).resolve()


def file_sha256_v61(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_sealed_v61(path: Path, file_sha: str, content_sha: str) -> dict:
    if file_sha256_v61(path) != file_sha:
        raise RuntimeError(f"v61 sealed aggregate file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or design.canonical_sha256_v61(compact) != content_sha
    ):
        raise RuntimeError(f"v61 sealed aggregate content changed: {path}")
    return value


def implementation_bindings_v61() -> dict:
    paths = {
        "pure_design": Path(design.__file__).resolve(),
        "preview_builder": Path(__file__).resolve(),
        "preview_tests": ROOT / "test_lora_es_robust_paired_hpo_v61.py",
        "v52_design": Path(design_v52.__file__).resolve(),
    }
    return {key: file_sha256_v61(path) for key, path in paths.items()}


def build_preview_v61() -> dict:
    strata = _read_sealed_v61(
        V61A_STRATA, design.V61A_STRATA_FILE_SHA256,
        design.V61A_STRATA_CONTENT_SHA256,
    )
    v61b = _read_sealed_v61(
        V61B_ANALYSIS, design.V61B_ANALYSIS_FILE_SHA256,
        design.V61B_ANALYSIS_CONTENT_SHA256,
    )
    report = _read_sealed_v61(
        V61B_REPORT, design.V61B_REPORT_FILE_SHA256,
        design.V61B_REPORT_CONTENT_SHA256,
    )
    if (
        v61b.get("schema") != "v61b-v434-common-seed-repeat-census-analysis"
        or v61b.get("status") != "complete_characterization_only"
        or v61b.get("raw_question_answer_or_generation_text_persisted") is not False
        or v61b.get("selection_update_or_promotion_performed") is not False
        or v61b.get("eval_ood_shadow_or_holdout_opened") is not False
        or report.get("status") != "complete_content_free_characterization_sealed"
        or report.get("evidence", {}).get("file_sha256")
        != design.V61B_EVIDENCE_FILE_SHA256
        or report.get("evidence", {}).get("content_sha256")
        != design.V61B_EVIDENCE_CONTENT_SHA256
        or report.get("gpu_activity", {}).get("all_four_attributed_positive") is not True
        or report.get("final_gpu_idle", {}).get("all_four_compute_process_lists_empty") is not True
    ):
        raise RuntimeError("v61 V61B aggregate/report contract changed")
    panels = design.build_panels_v61(strata)
    within = v61b["within_actor_pass_repeat"]
    actor_repeat_means = [
        item["mean_absolute_f1_delta"] for item in within["actors"]
    ]
    if (
        within["all_actor_row_comparisons"]["comparisons"] != 1792
        or within["all_actor_row_comparisons"]["f1_absolute_delta_gt_counts"]
        != {"1e-12": 654, "0.01": 543, "0.05": 229, "0.1": 96, "0.25": 7}
        or not all(0.018 <= item <= 0.022 for item in actor_repeat_means)
    ):
        raise RuntimeError("v61 V61B repeat calibration changed")
    state_count = 2 * len(design_v52.P16_SEEDS_V52)
    ranking_completions = (
        state_count * design.RANKING_UNITS_V61 * design.ACTORS_V61
        * design.PASSES_PER_STATE_V61
    )
    value = {
        "schema": "matched-lora-es-paired-block-bootstrap-hpo-preview-v61",
        "status": "cpu_only_preview_frozen_launch_ineligible",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": False,
        "reason_launch_ineligible": (
            "V61A preregistered stable-exact 8/4/2 gate failed at 3/3/0; "
            "this preview does not relax or override that gate"
        ),
        "new_launchable_preregistration_required": True,
        "scientific_change": (
            "Replace absolute signed-state generation ranks with antithetic, "
            "within-actor/pass paired conflict-unit bootstrap deltas and an "
            "explicit continuous instability penalty."
        ),
        "source_evidence": {
            "v61a_strata": {
                "path": str(V61A_STRATA),
                "file_sha256": design.V61A_STRATA_FILE_SHA256,
                "content_sha256": design.V61A_STRATA_CONTENT_SHA256,
                "eligibility_gate_passed": False,
            },
            "v61b_evidence_bound_not_opened": {
                "file_sha256": design.V61B_EVIDENCE_FILE_SHA256,
                "content_sha256": design.V61B_EVIDENCE_CONTENT_SHA256,
            },
            "v61b_analysis": {
                "path": str(V61B_ANALYSIS),
                "file_sha256": design.V61B_ANALYSIS_FILE_SHA256,
                "content_sha256": design.V61B_ANALYSIS_CONTENT_SHA256,
            },
            "v61b_report": {
                "path": str(V61B_REPORT),
                "file_sha256": design.V61B_REPORT_FILE_SHA256,
                "content_sha256": design.V61B_REPORT_CONTENT_SHA256,
            },
        },
        "panels": panels,
        "fixed_model_optimizer_recipe": {
            "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "source_v434": str(design_v52.SOURCE_V52),
            "source_weights_sha256": design_v52.SOURCE_WEIGHTS_SHA256_V52,
            "canonical_master_sha256": design_v52.MASTER_SHA256_V52,
            "layers_and_lora_rank_changed_from_v52": False,
            "population_size": 16,
            "seeds": list(design_v52.P16_SEEDS_V52),
            "sigma": 0.0048,
            "alpha": design_v52.ALPHA_V52,
            "scale_order": list(design_v52.SCALE_ORDER_V52),
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": 4,
            "passes_per_signed_state": 2,
            "common_generation_seed_per_actor_pass": 2_026_071_601,
            "signed_state_order": "direction0_plus_then_minus_through_direction15",
            "ranking_generation_completions": ranking_completions,
            "ranking_generation_completions_per_gpu": ranking_completions // 4,
            "exact_abort_after_each_candidate": True,
            "master_commit_authorized": False,
        },
        "paired_population_fitness": {
            "pairing_keys": [
                "direction", "unit_identity_sha256", "actor_rank", "pass_index",
            ],
            "plus_minus_state_adjacency_required": True,
            "bootstrap_block": (
                "conservative conflict unit connected over document, URL, "
                "lineage, and semantic identity"
            ),
            "bootstrap_replicates": design.BOOTSTRAP_REPLICATES_V61,
            "bootstrap_seed": design.BOOTSTRAP_SEED_V61,
            "one_sided_alpha": design.BOOTSTRAP_ALPHA_V61,
            "generation_composite_weights": dict(
                design.GENERATION_COMPOSITE_WEIGHTS_V61
            ),
            "stability_penalty_weights": dict(design.STABILITY_WEIGHTS_V61),
            "exact_in_population_composite": False,
            "exact_reason": (
                "only four any-exact units; exact is a sparse sentinel and "
                "full-census hard gate, not a bootstrap ranking fiction"
            ),
            "directional_coefficient": (
                "standardized paired plus-minus robust_generation_fitness"
            ),
            "primary_direction_weights": {
                "paired_robust_generation": 0.65,
                "equal_conflict_unit_domain": 0.35,
            },
            "projection_halfspaces": [
                "paired_robust_generation",
                "negative_continuous_instability",
                "equal_conflict_unit_domain",
                "prose_document_logprob_lcb",
                "qa_answer_logprob_lcb",
                "qa_generation_f1_lcb",
            ],
            "zero_spread_or_failed_halfspace_action": "abort_before_update",
        },
        "calibration_from_v61b": {
            "within_actor_repeat_mean_absolute_f1_delta_range": [
                min(actor_repeat_means), max(actor_repeat_means)
            ],
            "all_actor_row_comparison_threshold_counts": within[
                "all_actor_row_comparisons"
            ]["f1_absolute_delta_gt_counts"],
            "frozen_continuous_bands": [1e-12, 0.01, 0.05, 0.1, 0.25],
            "candidate_instability_maximum_mean_absolute_f1": (
                max(actor_repeat_means) * 1.10
            ),
            "same_seed_removed_variance": False,
        },
        "strict_train_only_candidate_gates": {
            "ranking_panel": [
                "paired conflict-unit bootstrap F1 delta LCB >= 0",
                "paired nonzero delta LCB >= 0",
                "continuous stability improvement LCB >= 0",
            ],
            "untouched_holdback_50_units": [
                "never opened during population direction construction",
                "paired conflict-unit bootstrap F1 delta LCB >= 0",
                "paired nonzero delta LCB >= 0",
                "candidate instability <= calibrated V61B ceiling",
            ],
            "sparse_exact_sentinel_4_units": [
                "candidate total exact actor-pass count >= reference",
                "candidate exact actor-pass count >= reference per unit",
                "three all-exact reference units remain represented",
            ],
            "one_time_full_448_row_four_actor_two_pass_gate": [
                "paired conflict-unit bootstrap F1 delta LCB >= 0",
                "paired nonzero delta LCB >= 0",
                "candidate total exact count >= reference",
                "all-actor exact rows >= 3 and any-actor exact rows >= 4",
                "continuous instability <= calibrated V61B ceiling",
            ],
            "retained_v52_anchors": [
                "domain point improvement",
                "prose document-block logprob noninferiority",
                "QA answer-logprob noninferiority",
                "QA generated-F1/nonzero noninferiority",
            ],
            "candidate_snapshot_only_after_every_gate": True,
            "candidate_never_committed_to_master": True,
        },
        "protected_phase_contract": {
            "eval_ood_shadow_terminal_access_authorized": False,
            "protected_path_open_count": 0,
            "separate_future_preregistration_required": True,
        },
        "implementation_bindings": implementation_bindings_v61(),
        "raw_question_answer_or_generation_text_persisted": False,
        "train_semantics_opened_by_preview_builder": False,
        "model_or_gpu_accessed_by_preview_builder": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = design.canonical_sha256_v61(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists(): raise FileExistsError(output)
    value = build_preview_v61()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v61(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": False,
        "train_semantics_model_gpu_or_protected_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
