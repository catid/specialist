#!/usr/bin/env python3
"""Build the prospective high-rep V65B alpha-zero preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import build_lora_es_ranking64_alpha_zero_preregistration_v65a as base
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65b as design


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v65b_ranking64_alpha_zero_high_rep_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
RANKING_PANEL_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v65b_ranking64_alpha_zero_high_rep_panel.json"
).resolve()
PREREGISTRATION_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "ranking64_alpha_zero_high_rep_calibration_v65b.json"
).resolve()

R1_RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v65a_r1_ranking64_alpha_zero_calibration"
).resolve()
R1_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "ranking64_alpha_zero_calibration_v65a_r1.json"
).resolve()
R1_ANALYSIS = (R1_RUN_DIR / "ranking64_alpha_zero_analysis_v65a_r1.json")
R1_EVIDENCE = (R1_RUN_DIR / "ranking64_alpha_zero_evidence_v65a_r1.json")
R1_FINALIZED = (R1_RUN_DIR / "ranking64_alpha_zero_finalized_v65a_r1.json")
R1_PREREGISTRATION_FILE_SHA256 = (
    "6a236f691b74468403c115c72082237604a9e8a9754e06b7c5d9c5a0f5497e24"
)
R1_PREREGISTRATION_CONTENT_SHA256 = (
    "8f6f8a5e0d07355164e84e8b0563c8d1484336b9a4b4dd3eb1cf81421f89c89e"
)
R1_ANALYSIS_FILE_SHA256 = (
    "8d4a6e0600d11fb250b7e6003ed5408df6a9a85bfbf6f9d3a56081e67412eb25"
)
R1_ANALYSIS_CONTENT_SHA256 = (
    "7d3af98967ef7de61b4969ec7f9f9e44b07ca17d0f604cb4a1955daa8ed11f66"
)
R1_EVIDENCE_FILE_SHA256 = (
    "29dacec8522ea797dfae2249ee61e54497cb3116db7e2f911e3511e38d95c960"
)
R1_EVIDENCE_CONTENT_SHA256 = (
    "54fc306a29a7f54b05f9feff0beda5fd9115f91d527c49ab297284f616a9650d"
)
R1_FINALIZED_FILE_SHA256 = (
    "83e11e97d5deae1aa3c7c2ce88399a27bad86effb1f9dd8955e330a2de05ce2f"
)
R1_FINALIZED_CONTENT_SHA256 = (
    "9472ce138866451012173c042b9e0708b4f44aa7ca6549a095901f8e5b389dd8"
)
R1_REQUEST_PROMPT_TOKEN_IDS_SHA256 = (
    "79541fcb77b728d57e55e98c0c4d593e4300806b1b3c96936eb3d27eeea40116"
)

DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65B = {
    "numeric_analysis_v65b": ROOT / (
        "lora_es_ranking64_alpha_zero_calibration_v65b.py"
    ),
    "preregistration_builder_v65b": Path(__file__).resolve(),
    "runtime_v65b": ROOT / (
        "run_lora_es_ranking64_alpha_zero_calibration_v65b.py"
    ),
    "worker_v65b": ROOT / "eggroll_es_worker_lora_v65b.py",
    "finalizer_v65b": ROOT / (
        "finalize_lora_es_ranking64_alpha_zero_v65b.py"
    ),
    "numeric_tests_v65b": ROOT / (
        "test_lora_es_ranking64_alpha_zero_calibration_v65b.py"
    ),
    "finalizer_tests_v65b": ROOT / (
        "test_finalize_lora_es_ranking64_alpha_zero_v65b.py"
    ),
    "r1_numeric_analysis": ROOT / (
        "lora_es_ranking64_alpha_zero_calibration_v65a.py"
    ),
    "r1_runtime_reused_primitives": ROOT / (
        "run_lora_es_ranking64_alpha_zero_calibration_v65a.py"
    ),
    "r1_worker_parent": ROOT / "eggroll_es_worker_lora_v65a.py",
    "v52_state_design": ROOT / "lora_es_nested_population_v52.py",
    "v65_hash_panel_design": ROOT / "lora_es_robust_sampling_population_v65.py",
    **{
        f"v64_runtime__{name}": Path(path).resolve()
        for name, path in sorted(
            base.runtime64.WORKER_EXECUTION_PATHS_V64.items()
        )
    },
}
REQUIRED_IMPLEMENTATION_BINDING_KEYS_V65B = frozenset(
    f"entry__{name}" for name in DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65B
)


def json_payload_v65b(value: dict) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("ascii")


def payload_sha256_v65b(value: dict) -> str:
    return hashlib.sha256(json_payload_v65b(value)).hexdigest()


def exact_int_v65b(value: object, expected: int) -> bool:
    """Reject bool/float aliases in sealed integer-valued contracts."""
    return type(value) is int and value == expected


def implementation_bindings_v65b(
    entry_paths: dict[str, Path] | None = None,
) -> dict:
    return base.common65.implementation_bindings_v65(
        entry_paths or DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65B
    )


def artifacts_v65b() -> dict:
    return {
        "run_directory": str(RUN_DIR),
        "attempt": str(RUN_DIR.parent / f".{EXPERIMENT}.attempt.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v65b.jsonl"),
        "evidence": str(RUN_DIR / "ranking64_high_rep_evidence_v65b.json"),
        "analysis": str(RUN_DIR / "ranking64_high_rep_analysis_v65b.json"),
        "report": str(RUN_DIR / "ranking64_high_rep_report_v65b.json"),
        "failure": str(RUN_DIR / "failure_v65b.json"),
        "finalized": str(RUN_DIR / "ranking64_high_rep_finalized_v65b.json"),
    }


def _read_exact(path: Path, file_sha256: str, content_sha256: str) -> dict:
    """Read and verify one immutable byte snapshot of an R1 artifact."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != file_sha256:
        raise RuntimeError(f"v65b sealed R1 file changed: {path}")

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(
                    f"v65b duplicate JSON key in sealed R1 artifact: {key}"
                )
            value[key] = item
        return value

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"v65b non-UTF-8 sealed R1 artifact: {path}") from error
    value = json.loads(text, object_pairs_hook=reject_duplicates)
    if (
        not isinstance(value, dict)
        or value.get("content_sha256_before_self_field") != content_sha256
        or base.population65.self_content_sha256_v65(value) != content_sha256
    ):
        raise RuntimeError(f"v65b sealed R1 content changed: {path}")
    return value


def r1_planning_binding_v65b() -> dict:
    prereg = _read_exact(
        R1_PREREGISTRATION,
        R1_PREREGISTRATION_FILE_SHA256,
        R1_PREREGISTRATION_CONTENT_SHA256,
    )
    analysis = _read_exact(
        R1_ANALYSIS, R1_ANALYSIS_FILE_SHA256, R1_ANALYSIS_CONTENT_SHA256,
    )
    evidence = _read_exact(
        R1_EVIDENCE, R1_EVIDENCE_FILE_SHA256, R1_EVIDENCE_CONTENT_SHA256,
    )
    finalized = _read_exact(
        R1_FINALIZED, R1_FINALIZED_FILE_SHA256, R1_FINALIZED_CONTENT_SHA256,
    )
    primary = analysis.get("primary_cluster_bootstrap", {})
    intervals = primary.get("intervals", {})
    temporal = primary.get("temporal_pair_joint_composite_intervals", {})
    gate = analysis.get("required_alpha_zero_gate", {})
    r1_recipe = prereg.get("fixed_calibration_recipe", {})
    r1_numeric = prereg.get("numeric_analysis_contract", {})
    r1_required_gates = r1_numeric.get("required_gates", {})
    r1_gate_checks = gate.get("checks", {})
    expected_panel, _expected_sources = base.sealed_source_bindings_v65a()
    r1_panel = prereg.get("ranking_panel", {})
    paired = design.v65a.paired_replicas_v65a(evidence.get("scored_periods"))
    rebuilt_analysis = design.v65a.analyze_scored_periods_v65a(
        evidence.get("scored_periods")
    )
    rebuilt_analysis.pop("content_sha256_before_self_field", None)
    rebuilt_analysis["source_evidence_content_sha256"] = (
        R1_EVIDENCE_CONTENT_SHA256
    )
    rebuilt_analysis["content_sha256_before_self_field"] = (
        design.v65a.canonical_sha256_v65a(rebuilt_analysis)
    )
    expected_metric_identities = [
        (index, item.get("row_sha256"), item.get("unit_identity_sha256"))
        for index, item in enumerate(expected_panel.get("items", []))
    ]
    first_metrics = (
        evidence.get("scored_periods", [[[]]])[0][0]
        if evidence.get("scored_periods") else []
    )
    observed_metric_identities = [
        (metric.get("request_index"), metric.get("row_sha256"),
         metric.get("unit_identity_sha256"))
        for metric in first_metrics if isinstance(metric, dict)
    ]
    source_hashes = finalized.get("source_hashes", {})
    verification = finalized.get("verification", {})
    observed_final = finalized.get(
        "observed_numeric_outcome_without_authorization", {}
    )
    frozen_final = finalized.get("frozen_non_authorization", {})

    def canonical_equal(left, right):
        try:
            return (
                design.v65a.canonical_sha256_v65a(left)
                == design.v65a.canonical_sha256_v65a(right)
            )
        except (TypeError, ValueError):
            return False
    r1_indices = design.v65a.frozen_bootstrap_indices_v65a()
    r1_index_sha256 = hashlib.sha256(
        r1_indices.astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest()
    pair0_f1_units = paired[:, :, 0, 0].mean(axis=1)
    pair0_f1 = design.v65a._bootstrap_interval_v65a(
        pair0_f1_units, r1_indices,
    )
    cells = paired[..., 0].reshape(64, 8)
    within_variance = float(np.var(cells, axis=1, ddof=1).mean())
    unit_mean_variance = float(np.var(cells.mean(axis=1), ddof=1))
    between_variance = max(
        unit_mean_variance - within_variance / 8.0, 0.0,
    )
    r1_halfwidth = intervals.get("generated_f1_delta", {}).get("halfwidth")
    projected_random_effects = float(r1_halfwidth * math.sqrt(
        (between_variance + within_variance / 144.0)
        / (between_variance + within_variance / 8.0)
    ))
    projected_pair0_joint = float(
        temporal.get("pair_0", {}).get("halfwidth") / math.sqrt(18.0)
    )
    projected_pair0_f1 = float(pair0_f1["halfwidth"] / math.sqrt(18.0))
    naive_iid_minimum = float(
        8.0 * (r1_halfwidth / design.MAX_PRIMARY_CI_HALFWIDTH_V65B) ** 2
    )
    if (
        prereg.get("schema")
        != "v65a-ranking64-alpha-zero-calibration-preregistration"
        or analysis.get("schema") != "v65a-ranking64-alpha-zero-analysis"
        or not canonical_equal(analysis, rebuilt_analysis)
        or rebuilt_analysis.get("content_sha256_before_self_field")
        != R1_ANALYSIS_CONTENT_SHA256
        or evidence.get("panel_content_sha256")
        != prereg.get("ranking_panel", {}).get("content_sha256")
        or evidence.get("panel_content_sha256")
        != expected_panel.get("content_sha256_before_self_field")
        or not canonical_equal(
            observed_metric_identities, expected_metric_identities,
        )
        or len(observed_metric_identities) != 64
        or evidence.get("authorized_input_receipt", {}).get(
            "request_prompt_token_ids_sha256"
        ) != R1_REQUEST_PROMPT_TOKEN_IDS_SHA256
        or not exact_int_v65b(analysis.get("paired_replicas_per_unit"), 8)
        or not exact_int_v65b(evidence.get("paired_replicas_per_unit"), 8)
        or paired.shape != (64, 4, 2, 3)
        or design.R1_PLANNING_OBSERVATIONS_V65B[
            "paired_replicas_per_unit"
        ] != 8
        or intervals.get("generated_f1_delta", {}).get("halfwidth")
        != design.R1_PLANNING_OBSERVATIONS_V65B["generated_f1_halfwidth"]
        or gate.get("maximum_primary_ci_halfwidth")
        != design.R1_PLANNING_OBSERVATIONS_V65B["generated_f1_limit"]
        or gate.get("maximum_actor_leave_one_out_shift")
        != design.R1_PLANNING_OBSERVATIONS_V65B[
            "maximum_actor_leave_one_out_shift"
        ]
        or design.MAX_PRIMARY_CI_HALFWIDTH_V65B
        != r1_required_gates.get("maximum_primary_ci_halfwidth_inclusive")
        or design.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B
        != r1_required_gates.get(
            "maximum_actor_leave_one_out_shift_inclusive"
        )
        or design.BOOTSTRAP_ALPHA_V65B != r1_numeric.get("one_sided_alpha")
        or design.COMPOSITE_WEIGHTS_V65B
        != r1_numeric.get("joint_composite_weights")
        or design.COMMON_GENERATION_SEED_V65B
        != r1_recipe.get("common_generation_seed")
        or design.GENERATION_PARAMS_WITHOUT_SEED_V65B
        != r1_recipe.get("generation_params_without_seed")
        or design.ENGINE_CONTROLS_V65B
        != r1_recipe.get("runtime_determinism_controls")
        or r1_index_sha256
        != r1_numeric.get("bootstrap_index_matrix_sha256")
        or r1_gate_checks != {
            "actor_leave_one_out_shift_within_v62b_limit": True,
            "generated_f1_primary_ci_halfwidth_within_v62b_limit": False,
            "generated_f1_primary_interval_contains_zero": True,
            "joint_composite_interval_contains_zero": True,
            "stability_improvement_interval_contains_zero": True,
        }
        or temporal.get("pair_0", {}).get("null_radius")
        != design.R1_PLANNING_OBSERVATIONS_V65B[
            "pair_0_joint_composite_null_radius"
        ]
        or temporal.get("pair_1", {}).get("null_radius")
        != design.R1_PLANNING_OBSERVATIONS_V65B[
            "pair_1_joint_composite_null_radius"
        ]
        or temporal.get("pair_0", {}).get("halfwidth")
        != design.R1_PLANNING_OBSERVATIONS_V65B[
            "pair_0_joint_composite_halfwidth"
        ]
        or gate.get("passed") is not False
        or pair0_f1.get("halfwidth")
        != design.R1_PLANNING_OBSERVATIONS_V65B[
            "pair_0_generated_f1_halfwidth"
        ]
        or not math.isclose(
            naive_iid_minimum,
            design.R1_PLANNING_OBSERVATIONS_V65B[
                "naive_iid_minimum_paired_replicas"
            ],
            rel_tol=0.0, abs_tol=1e-15,
        )
        or not math.isclose(
            projected_random_effects,
            design.R1_PLANNING_OBSERVATIONS_V65B[
                "prospective_random_effects_halfwidth_at_144"
            ],
            rel_tol=0.0, abs_tol=1e-18,
        )
        or not math.isclose(
            projected_pair0_f1,
            design.R1_PLANNING_OBSERVATIONS_V65B[
                "prospective_pair_0_generated_f1_halfwidth_at_72"
            ],
            rel_tol=0.0, abs_tol=1e-18,
        )
        or not math.isclose(
            projected_pair0_joint,
            design.R1_PLANNING_OBSERVATIONS_V65B[
                "prospective_pair_0_joint_halfwidth_at_72"
            ],
            rel_tol=0.0, abs_tol=1e-18,
        )
        or r1_panel.get("content_sha256")
        != expected_panel.get("content_sha256_before_self_field")
        or r1_panel.get("request_order_sha256")
        != expected_panel.get("request_order_sha256")
        or r1_panel.get("unit_order_sha256")
        != expected_panel.get("unit_order_sha256")
        or r1_panel.get("units") != expected_panel.get("ranking_units")
        or r1_panel.get("hash_only") is not True
        or finalized.get("v65_population_launch_authorized") is not False
        or finalized.get("schema")
        != "v65a-ranking64-alpha-zero-independent-finalizer"
        or finalized.get("status")
        != "complete_numeric_only_observation_v65_still_unauthorized"
        or source_hashes.get("preregistration") != {
            "file_sha256": R1_PREREGISTRATION_FILE_SHA256,
            "content_sha256": R1_PREREGISTRATION_CONTENT_SHA256,
        }
        or source_hashes.get("evidence") != {
            "file_sha256": R1_EVIDENCE_FILE_SHA256,
            "content_sha256": R1_EVIDENCE_CONTENT_SHA256,
        }
        or source_hashes.get("analysis") != {
            "file_sha256": R1_ANALYSIS_FILE_SHA256,
            "content_sha256": R1_ANALYSIS_CONTENT_SHA256,
        }
        or verification.get(
            "stored_analysis_exactly_equals_independent_numeric_rebuild"
        ) is not True
        or not canonical_equal(
            observed_final.get("primary_cluster_bootstrap"),
            analysis.get("primary_cluster_bootstrap"),
        )
        or not canonical_equal(
            observed_final.get("actor_influence"),
            analysis.get("actor_influence"),
        )
        or not canonical_equal(
            observed_final.get("required_alpha_zero_gate", {}).get("checks"),
            gate.get("checks"),
        )
        or observed_final.get("required_alpha_zero_gate", {}).get("passed")
        is not gate.get("passed")
        or frozen_final.get("failed_gate_reinterpreted_or_relaxed") is not False
        or frozen_final.get("thresholds_changed_after_outcome") is not False
        or finalized.get("frozen_non_authorization", {}).get(
            "thresholds_changed_after_outcome"
        ) is not False
    ):
        raise RuntimeError("v65b finalized R1 planning source changed")

    def binding(path, file_sha, content_sha):
        return {
            "path": str(path), "file_sha256": file_sha,
            "content_sha256": content_sha,
        }

    return {
        "schema": "v65b-r1-prospective-sample-size-planning-binding",
        "r1_preregistration": binding(
            R1_PREREGISTRATION, R1_PREREGISTRATION_FILE_SHA256,
            R1_PREREGISTRATION_CONTENT_SHA256,
        ),
        "r1_analysis": binding(
            R1_ANALYSIS, R1_ANALYSIS_FILE_SHA256, R1_ANALYSIS_CONTENT_SHA256,
        ),
        "r1_numeric_evidence": binding(
            R1_EVIDENCE, R1_EVIDENCE_FILE_SHA256, R1_EVIDENCE_CONTENT_SHA256,
        ),
        "r1_finalized": binding(
            R1_FINALIZED, R1_FINALIZED_FILE_SHA256,
            R1_FINALIZED_CONTENT_SHA256,
        ),
        "observations": dict(design.R1_PLANNING_OBSERVATIONS_V65B),
        "request_prompt_token_ids_sha256": (
            R1_REQUEST_PROMPT_TOKEN_IDS_SHA256
        ),
        "recomputed_planning_inputs": {
            "within_unit_cell_variance": within_variance,
            "between_unit_variance_floor": between_variance,
            "random_effects_formula": (
                "H_R1*sqrt((between+within/144)/(between+within/8))"
            ),
            "naive_iid_formula": "8*(H_R1/fixed_limit)^2",
            "pair0_f1_formula": "R1_pair0_f1_halfwidth/sqrt(18)",
            "pair0_joint_heuristic_formula": (
                "R1_pair0_joint_halfwidth/sqrt(18)"
            ),
            "pair0_joint_projection_is_nongating_heuristic_because_stability_is_nonlinear": True,
            "naive_iid_minimum_paired_replicas": naive_iid_minimum,
            "projected_random_effects_halfwidth_at_144": (
                projected_random_effects
            ),
            "projected_pair0_generated_f1_halfwidth_at_72": (
                projected_pair0_f1
            ),
            "projected_pair0_joint_halfwidth_at_72": projected_pair0_joint,
        },
        "use": {
            "prospective_sample_size_planning_only": True,
            "label_schedule_redesigned_prospectively_to_remove_time_confound": True,
            "threshold_relaxation": False,
            "failed_outcome_reinterpretation": False,
            "bound_transfer_from_failed_r1": False,
            "v65_population_authority": False,
        },
    }


def build_preregistration_v65b(
    panel: dict, sources: dict, implementation_bindings: dict,
    *, ranking_panel_output: Path = RANKING_PANEL_OUTPUT,
) -> dict:
    expected = implementation_bindings_v65b()
    expected_panel, expected_sources = base.sealed_source_bindings_v65a()
    if (
        panel != expected_panel
        or sources != expected_sources
        or panel.get("schema") != "v65-robust-sampling-ranking-panel"
        or not exact_int_v65b(panel.get("ranking_units"), 64)
        or design.v65a.self_content_sha256_v65a(panel)
        != panel.get("content_sha256_before_self_field")
        or len(panel.get("items", [])) != 64
        or len({
            item.get("row_sha256") for item in panel.get("items", [])
            if isinstance(item, dict)
        }) != 64
        or len({
            item.get("unit_identity_sha256") for item in panel.get("items", [])
            if isinstance(item, dict)
        }) != 64
        or panel.get("question_answer_or_generation_text_persisted") is not False
        or panel.get("protected_semantics_opened") is not False
        or implementation_bindings != expected
        or not REQUIRED_IMPLEMENTATION_BINDING_KEYS_V65B.issubset(expected)
    ):
        raise RuntimeError("v65b preregistration input changed")
    panel_path = Path(ranking_panel_output).resolve()
    schedule = design.validate_schedule_v65b()
    value = {
        "schema": "v65b-ranking64-high-rep-alpha-zero-preregistration",
        "status": "sealed_before_v65b_semantics_model_ray_or_gpu_access",
        "purpose": (
            "Run one fixed higher-rep exact-64 alpha-zero calibration after "
            "R1 failed its unchanged precision gate. R1 is used only for "
            "prospective sample-size and counterbalance planning."
        ),
        "authorization": {
            "gpu_launch": True,
            "alpha_zero_high_rep_calibration": True,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": 4,
            "tensor_parallel_size_per_actor": 1,
            "adapter_update": False,
            "candidate": False,
            "hpo_population": False,
            "holdback_sentinel_ood_protected_terminal": False,
            "promotion": False,
            "v65_population": False,
        },
        "r1_prospective_planning": r1_planning_binding_v65b(),
        "source_evidence": sources,
        "ranking_panel": {
            "path": str(panel_path),
            "file_sha256": payload_sha256_v65b(panel),
            "content_sha256": panel["content_sha256_before_self_field"],
            "units": 64,
            "request_order_sha256": panel["request_order_sha256"],
            "unit_order_sha256": panel["unit_order_sha256"],
            "hash_only": True,
        },
        "access_contract": {
            "live_semantic_dataset_path": str(base.population65.V61C_ROWS),
            "decode_exactly_first_64_ranking_rows": True,
            "decode_row_64_or_later": False,
            "ranking_prefix_bytes": design.v65a.RANKING_PREFIX_BYTES_V65A,
            "ranking_prefix_sha256": design.v65a.RANKING_PREFIX_SHA256_V65A,
            "request_prompt_token_ids_sha256": (
                R1_REQUEST_PROMPT_TOKEN_IDS_SHA256
            ),
            "source_file_size_metadata_bytes": (
                design.v65a.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
            ),
            "raw_prompt_answer_or_generation_text_may_be_persisted": False,
            "full_file_read_or_hash_live": False,
        },
        "fixed_recipe": {
            "base_model": base.common65.base_model_binding_v65(),
            "v434_adapter": base.common65.v434_binding_v65(),
            "alpha": 0.0,
            "sigma_or_direction": None,
            "rows": 64,
            "same_exact_64_unit_and_request_order_as_r1": True,
            "actors": 4,
            "warmup_periods": 8,
            "warmups_run_in_fresh_v65b_process": True,
            "r1_warmup_or_engine_state_transferred": False,
            "warmup_outputs_scored_or_persisted": False,
            "scored_periods": 72,
            "adjacent_blocks": 36,
            "four_period_superblocks": 18,
            "paired_replicas_per_unit": 144,
            "warmup_generation_completions": 2_048,
            "scored_generation_completions": 18_432,
            "total_generation_completions": 20_480,
            "schedule": schedule,
            "label_plan": dict(design.LABEL_PLAN_V65B),
            "forward_subset_cycle": [
                list(value) for value in design.FORWARD_SUBSETS_V65B
            ],
            "temporal_pass_blocks": {
                key: list(value)
                for key, value in design.TEMPORAL_PASS_BLOCKS_V65B.items()
            },
            "run_half_blocks": {
                key: list(value)
                for key, value in design.RUN_HALF_BLOCKS_V65B.items()
            },
            "epoch_blocks": {
                key: list(value)
                for key, value in design.TEMPORAL_EPOCH_BLOCKS_V65B.items()
            },
            "generation_seed": design.COMMON_GENERATION_SEED_V65B,
            "generation_params_without_seed": dict(
                design.GENERATION_PARAMS_WITHOUT_SEED_V65B
            ),
            "runtime_determinism_controls": dict(
                design.ENGINE_CONTROLS_V65B
            ),
            "fixed_actor_rpc_generation_and_construction_timeouts": {
                "schema": (
                    "v65b-fixed-construction-actor-rpc-generation-and-"
                    "pool-shutdown-timeouts"
                ),
                "seconds": 300.0,
                "whole_trainer_construction_watchdog_seconds": 660.0,
                "reward_pool_shutdown_timeout_seconds": 12.0,
                "gpu_preflight_watchdog_seconds": 30.0,
                "nvidia_smi_subprocess_timeout_seconds": 10.0,
                "strict_cleanup_and_idle_watchdog_seconds": 120.0,
                "final_ray_shutdown_watchdog_seconds": 30.0,
                "scope": [
                    "whole_trainer_construction_watchdog",
                    "placement_group_ready_waits",
                    "constructor_collective_rpc_waits",
                    "actor_identity_waits", "collective_rpc_waits",
                    "four_actor_generation_waits",
                    "reward_pool_terminate_and_join",
                    "gpu_preflight_and_nvidia_smi_queries",
                    "strict_cleanup_and_four_gpu_idle_proof",
                    "final_unconditional_ray_shutdown",
                    "terminal_artifact_lock_waits",
                ],
                "timeout_retry_drop_reorder_or_early_stop": False,
                "timeout_invalidates_calibration": True,
                "timeout_enters_strict_engine_cleanup_and_four_gpu_idle_proof": True,
                "successful_run_requires_all_after_generation_and_final_exact_state_receipts": True,
            },
            "exact_master_rematerialization": {
                "rpc": "rematerialize_exact_master_v65b",
                "four_actor_certificate_sha256": (
                    "00a49e786aa1e8b4bed0e0241235a4be118d757af5ee27ef479498e08af60000"
                ),
                "slot_write_before_every_warmup_and_scored_period": True,
                "period_slot_write_receipts_required": 80,
                "read_only_rpc": "read_only_exact_master_slot_v65b",
                "read_only_live_slot_receipts_required": 160,
                "read_only_edges": [
                    "before_generation", "after_generation",
                ],
                "after_generation_receipt_may_write_or_reset_slot": False,
            },
            "gpu_hardware_health_monitor": {
                "sample_interval_seconds": 0.5,
                "controller_failure_poll_seconds": 0.25,
                "initial_cycle_ready_timeout_seconds": 30.0,
                "generation_wait_aborts_on_monitor_failure_before_actor_deadline": True,
                "maximum_consecutive_sample_cycle_gap_seconds": 2.0,
                "sample_fields": [
                    "physical_gpu", "expected_and_compute_pids",
                    "utilization_percent", "memory_used_mib",
                    "temperature_c", "power_draw_mw",
                    "nvml_current_clock_event_reasons_bitmask",
                ],
                "foreign_compute_pid_fails_closed_in_every_phase": True,
                "generation_phase_count": 80,
                "forbidden_generation_clock_event_reason_mask": 232,
                "forbidden_generation_reasons": {
                    "hardware_slowdown": 8,
                    "software_thermal_slowdown": 32,
                    "hardware_thermal_slowdown": 64,
                    "hardware_external_power_brake": 128,
                },
                "diagnostic_nonfailing_reasons": {
                    "idle": 1,
                    "application_or_user_clocks": 2,
                    "software_power_cap": 4,
                    "sync_boost": 16,
                    "display_clock_setting": 256,
                },
                "each_generation_phase_requires_attributed_positive_activity_on_all_four_gpus": True,
                "each_generation_phase_requires_a_simultaneous_positive_four_gpu_cycle": True,
                "thermal_or_hardware_slowdown_observations_required": 0,
            },
            "report_wall_clock_reconciliation_tolerance_seconds": 5.0,
            "adaptive_retry_drop_reorder_or_early_stop": False,
        },
        "numeric_contract": {
            "primary_resampled_axis": "conflict_unit_only",
            "unit_clusters": 64,
            "replicas_preserved_inside_each_unit": 144,
            "blocks_or_completions_treated_as_independent_units": False,
            "bootstrap_replicates": design.BOOTSTRAP_REPLICATES_V65B,
            "bootstrap_seed": design.BOOTSTRAP_SEED_V65B,
            "bootstrap_index_matrix_sha256": (
                design.BOOTSTRAP_INDEX_MATRIX_SHA256_V65B
            ),
            "one_sided_alpha": design.BOOTSTRAP_ALPHA_V65B,
            "maximum_primary_ci_halfwidth_unchanged": (
                design.MAX_PRIMARY_CI_HALFWIDTH_V65B
            ),
            "maximum_actor_leave_one_out_shift_unchanged": (
                design.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B
            ),
            "pooled_f1_joint_stability_intervals_must_contain_zero": True,
            "early_and_late_position_f1_intervals_must_contain_zero": True,
            "early_and_late_position_joint_intervals_must_contain_zero": True,
            "early_and_late_position_f1_halfwidths_use_same_fixed_limit": True,
            "first_and_second_run_half_intervals_must_contain_zero": True,
            "first_and_second_run_half_f1_halfwidths_use_same_fixed_limit": True,
            "all_six_epoch_intervals_sealed_as_non_gating_diagnostics": True,
            "orientation_effect_interval_must_contain_zero": True,
            "early_minus_late_joint_interval_must_contain_zero": True,
            "superblock_leave_one_out_sealed_as_non_gating_diagnostic": True,
            "B_C_pass_definition": (
                "max(early_position.joint_composite.null_radius,"
                "late_position.joint_composite.null_radius)"
            ),
            "threshold_relaxation_or_failed_outcome_reinterpretation": False,
        },
        "runtime": dict(design52.RUNTIME_V52),
        "required_python": str(design52.REQUIRED_PYTHON_V52),
        "implementation_bindings": implementation_bindings,
        "implementation_closure_manifest_sha256": (
            design.v65a.canonical_sha256_v65a({
                key: binding["file_sha256"]
                for key, binding in sorted(implementation_bindings.items())
            })
        ),
        "artifacts": artifacts_v65b(),
        "success_directly_authorizes_v65_population": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        design.v65a.canonical_sha256_v65a(value)
    )
    return value


def build_v65b(
    *, ranking_panel_output: Path = RANKING_PANEL_OUTPUT,
) -> tuple[dict, dict]:
    panel, sources = base.sealed_source_bindings_v65a()
    implementation = implementation_bindings_v65b()
    return panel, build_preregistration_v65b(
        panel, sources, implementation,
        ranking_panel_output=ranking_panel_output,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-panel-output", default=str(RANKING_PANEL_OUTPUT))
    parser.add_argument("--preregistration-output", default=str(
        PREREGISTRATION_OUTPUT
    ))
    args = parser.parse_args(argv)
    panel_path = Path(args.ranking_panel_output).resolve()
    prereg_path = Path(args.preregistration_output).resolve()
    panel, prereg = build_v65b(ranking_panel_output=panel_path)
    base.common65._exclusive_write_pair_v65(
        panel_path, json_payload_v65b(panel),
        prereg_path, json_payload_v65b(prereg),
    )
    print(json.dumps({
        "ranking_panel": str(panel_path),
        "preregistration": str(prereg_path),
        "ranking_panel_file_sha256": base.population65.file_sha256_v65(
            panel_path
        ),
        "preregistration_file_sha256": base.population65.file_sha256_v65(
            prereg_path
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
