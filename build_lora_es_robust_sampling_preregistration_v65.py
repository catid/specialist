#!/usr/bin/env python3
"""Build the hash-only panel and measurement-only V65 preregistration.

The builder reads only already-sealed, numeric/hash-only aggregate evidence.
It does not read the staged semantic dataset, any holdback/sentinel row, the
base-model directory, or a protected path.  The output is deterministic for
the same repository bytes and requested output paths; no wall-clock field is
part of either sealed artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path

import lora_es_nested_population_v52 as design52
import lora_es_pre_hpo_alpha_zero_calibration_v62b as calibration62b
import lora_es_robust_sampling_population_v65 as design65
import lora_es_v59_vs_v434_robust_confirmation_v64 as design64


ROOT = Path(__file__).resolve().parent
RANKING_PANEL_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v65_robust_sampling_ranking_panel.json"
).resolve()
PREREGISTRATION_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_robust_sampling_population_v65.json"
).resolve()
EXPERIMENT = "v65_lora_es_robust_sampling_population"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()

V62B_FINALIZED = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v62b_v434_pre_hpo_alpha_zero_generation_calibration/"
    "alpha_zero_finalized_v62b.json"
).resolve()
V62B_FINALIZED_FILE_SHA256 = design64.V62B_FINALIZED_FILE_SHA256_V64
V62B_FINALIZED_CONTENT_SHA256 = design64.V62B_FINALIZED_CONTENT_SHA256_V64
V62B_FINALIZER_COMMIT = design64.V62B_FINALIZER_COMMIT_V64

STAGED_DATASET_V61C = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v61c_paired_null_calibration_rows.jsonl"
).resolve()
STAGED_DATASET_FILE_SHA256_V61C = (
    "9c1b7f69595cf70ef045259e2097c39546e9f1d84a6b0870fcb14e987655079a"
)
STAGED_PANEL_V61C = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v61c_paired_null_calibration_panel.json"
).resolve()
STAGED_PANEL_FILE_SHA256_V61C = (
    "92e0c6160bfc7884a00be4c34c427685dcb2bf5a6aa8c3820f5c53e225f8091c"
)
STAGED_PANEL_CONTENT_SHA256_V61C = (
    "ca0a947e6437c0d84360176087b0a9dab12b79cf6ba1be8f965b24e9f4ec7ba4"
)
STAGED_RANKING_PREFIX_BYTES_V65 = 136_848
STAGED_RANKING_PREFIX_SHA256_V65 = (
    "8259894003268a2fafed6a9a66ce3e604d5eb76cdf19a1c1c759e5ffc5916c70"
)
V61C_EVIDENCE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61c_v434_identical_state_paired_evaluator_calibration/"
    "paired_null_evidence_v61c.json"
).resolve()
V61C_EVIDENCE_FILE_SHA256 = (
    "5be0a46ef0051c760b89d535cc252eeb1c9a6b2c700c209799049191615fa3dc"
)
V61C_EVIDENCE_CONTENT_SHA256 = (
    "15b7d74ea9b003d03ad4ba7667936ac80fac121cbbc28e4ced2c1cd9f57c7fa8"
)
V61C_ANALYSIS = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61c_v434_identical_state_paired_evaluator_calibration/"
    "paired_null_analysis_v61c.json"
).resolve()
V61C_ANALYSIS_FILE_SHA256 = (
    "b7588ccb58ac9ae6a196ce2605cc0b637b962170832470edc5b3095f07fafaeb"
)
V61C_ANALYSIS_CONTENT_SHA256 = (
    "93732923303da1201949c4619690b72ec3e34482ddc1c00da740cec1d0254563"
)
ADAPTIVE_PANEL_OVERLAP_MANIFEST_SHA256_V65 = (
    "4b16e58a918c119190cf46f91820eb5fd4dfa23dfa35b766f134f273d61af42c"
)

BASE_MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
BASE_MODEL_SEAL = (
    ROOT / "experiments/eggroll_es_hpo/S6_V23A_INSERTION_MODEL_SEAL.json"
).resolve()
BASE_MODEL_SEAL_COMMIT = "00471e1c44b78813d02f0a8895c8e014a75dcc4e"
BASE_MODEL_SEAL_FILE_SHA256 = (
    "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440"
)
BASE_MODEL_SEAL_CONTENT_SHA256 = (
    "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
)
BASE_MODEL_WEIGHT_SHARD_MANIFEST_SHA256 = (
    "af8ea3a900c04e97d2d8e3146b8e23be5ee3e6548dea20440020b2f43ee6656e"
)
BASE_MODEL_NON_WEIGHT_MANIFEST_SHA256 = (
    "f53b938b97ac06c075d697eaec662695b5349f344f79808cbaa86218f2b057e1"
)
BASE_MODEL_ALL_FILES_FINGERPRINT_SHA256 = (
    "1a21a765e374e266037b3b7e5313a62a0de8ca37c00c0462b67c21af7e21f61e"
)

COMMON_GENERATION_SEED_V65 = 2_026_071_601
GENERATION_PARAMS_WITHOUT_SEED_V65 = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "n": 1,
    "detokenize": True,
}

DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65 = {
    "runtime_v65": ROOT / "run_lora_es_robust_sampling_population_v65.py",
    "population_design_v65": Path(design65.__file__).resolve(),
    "cpu_scoring_worker_v65": (
        ROOT / "eggroll_es_worker_robust_sampling_v65.py"
    ),
    "preregistration_builder_v65": Path(__file__).resolve(),
    "tests_v65": ROOT / "test_lora_es_robust_sampling_population_v65.py",
    "base_model_byte_receipt_runtime_v64": (
        ROOT / "run_lora_es_v59_vs_v434_robust_confirmation_v64.py"
    ),
    "generation_scorer_runtime_v48b": (
        ROOT / "run_lora_es_generation_boundary_v48b.py"
    ),
    "population_runtime_v52": ROOT / "run_lora_es_nested_population_v52.py",
    "population_worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
    "transaction_worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
    "transition_runtime_v51": (
        ROOT / "run_lora_es_transition_microbenchmark_v51.py"
    ),
    "canonical_state_worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
    "staged_request_runtime_v61c": (
        ROOT / "run_lora_es_paired_null_calibration_v61c.py"
    ),
}


def json_payload_v65(value: dict) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("ascii")


def payload_sha256_v65(value: dict) -> str:
    return hashlib.sha256(json_payload_v65(value)).hexdigest()


def _local_import_closure_v65(entry_paths: dict[str, Path]) -> dict[str, str]:
    """Hash every repository-local Python file statically reachable here.

    Dynamic worker/scorer imports are roots in ``entry_paths``.  External
    packages are intentionally ignored.  This makes the implementation seal
    complete without importing the not-yet-authorized live runner.
    """
    pending = [Path(path).resolve() for path in entry_paths.values()]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        observed.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            raise RuntimeError(f"v65 cannot parse implementation file {path}") from error
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0 and node.module
            ):
                modules.add(node.module.split(".", 1)[0])
        for module in modules:
            candidate = (ROOT / f"{module}.py").resolve()
            if candidate.is_file() and candidate not in observed:
                pending.append(candidate)
    return {
        str(path.relative_to(ROOT)): design65.file_sha256_v65(path)
        for path in sorted(observed)
    }


def implementation_bindings_v65(
    entry_paths: dict[str, Path] | None = None,
) -> dict:
    paths = {
        key: Path(path).resolve() for key, path in (
            entry_paths or DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65
        ).items()
    }
    entries = {
        f"entry__{key}": {
            "path": str(path),
            "file_sha256": design65.file_sha256_v65(path),
        }
        for key, path in sorted(paths.items())
    }
    closure = _local_import_closure_v65(paths)
    bound_paths = {Path(row["path"]).resolve() for row in entries.values()}
    for relative, digest in closure.items():
        path = (ROOT / relative).resolve()
        if path not in bound_paths:
            entries[f"closure__{relative}"] = {
                "path": str(path),
                "file_sha256": digest,
            }
    return dict(sorted(entries.items()))


def sealed_source_bindings_v65() -> tuple[dict, dict]:
    """Verify only frozen aggregate/hash artifacts and build the V65 panel."""
    preview = design65.read_exact_self_hashed_v65(
        design65.PREVIEW_V61,
        design65.PREVIEW_V61_FILE_SHA256,
        design65.PREVIEW_V61_CONTENT_SHA256,
    )
    panel = design65.build_ranking_panel_v65(preview)

    staged_panel = design65.read_exact_self_hashed_v65(
        design65.V61C_PANEL,
        design65.V61C_PANEL_FILE_SHA256,
        design65.V61C_PANEL_CONTENT_SHA256,
    )
    staged_items = staged_panel.get("items", [])
    projected = [{
        "request_index": item.get("request_index"),
        "row_sha256": item.get("row_sha256"),
        "unit_identity_sha256": item.get("unit_identity_sha256"),
    } for item in staged_items[:design65.RANKING_UNITS_V65]]
    expected_projection = [{
        "request_index": item["request_index"],
        "row_sha256": item["row_sha256"],
        "unit_identity_sha256": item["unit_identity_sha256"],
    } for item in panel["items"]]
    if (
        staged_panel.get("schema") != "v61c-paired-null-calibration-panel"
        or len(staged_items) != 68
        or projected != expected_projection
        or any(item.get("role") != "ranking" for item in staged_items[:64])
        or any(
            item.get("role") != "exact_sentinel"
            for item in staged_items[64:]
        )
        or staged_panel.get("holdback_units_in_runtime_dataset") != 0
        or staged_panel.get("holdback_documents_in_runtime_dataset") != 0
        or staged_panel.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v65 V61C staged ranking-prefix contract changed")

    arm = design65.read_exact_self_hashed_v65(
        design65.V53_SIGMA_ARM,
        design65.V53_SIGMA_ARM_FILE_SHA256,
        design65.V53_SIGMA_ARM_CONTENT_SHA256,
    )
    integrity = arm.get("perturbation_integrity", {})
    reliability = arm.get("reliability", {})
    restored = arm.get("post_arm_exact_master_state", {})
    random_plan = arm.get("common_random_plan", {})
    ordered_v53_identities = design65.expected_v53_state_identities_v65(arm)
    if (
        arm.get("schema") != "sigma-discrimination-arm-v53"
        or arm.get("sigma") != design65.SIGMA_V65
        or arm.get("population_size") != design65.POPULATION_SIZE_V65
        or arm.get("passed") is not True
        or arm.get("projection_performed") is not False
        or arm.get("optimizer_update_or_train_gate_opened") is not False
        or arm.get("protected_semantics_opened") is not False
        or arm.get("sealed_holdout_opened") is not False
        or reliability.get("passed") is not True
        or integrity.get("states") != 32
        or integrity.get("unique_nonmaster_fp32_candidates") != 32
        or integrity.get("unique_nonmaster_bf16_candidates") != 32
        or integrity.get("candidate_identity_inventory_sha256")
        != design65.V53_CANDIDATE_IDENTITY_INVENTORY_SHA256
        or integrity.get("runtime_identity_inventory_sha256")
        != design65.V53_RUNTIME_IDENTITY_INVENTORY_SHA256
        or integrity.get("direct_from_pinned_master") is not True
        or integrity.get("exact_antithetic_seed_sign_coverage") is not True
        or integrity.get("four_actor_consensus_before_scoring") is not True
        or restored.get("all_four_bf16_runtime_identities_exact") is not True
        or restored.get(
            "all_four_canonical_fp32_master_identities_exact"
        ) is not True
        or restored.get("controller_transaction_quiescent") is not True
        or restored.get("worker_transactions_quiescent") is not True
        or random_plan.get("request_order_sha256")
        != design52.REQUEST_ORDER_SHA256_V52
    ):
        raise RuntimeError("v65 exact V53 sigma=0.0048 arm changed")

    v61c_finalized = design65.read_exact_self_hashed_v65(
        design65.V61C_FINALIZED,
        design65.V61C_FINALIZED_FILE_SHA256,
        design65.V61C_FINALIZED_CONTENT_SHA256,
    )
    v61c_evidence = design65.read_exact_self_hashed_v65(
        V61C_EVIDENCE,
        V61C_EVIDENCE_FILE_SHA256,
        V61C_EVIDENCE_CONTENT_SHA256,
    )
    v61c_analysis = design65.read_exact_self_hashed_v65(
        V61C_ANALYSIS,
        V61C_ANALYSIS_FILE_SHA256,
        V61C_ANALYSIS_CONTENT_SHA256,
    )
    v61c_sources = v61c_finalized.get("source_hashes", {})
    primary_null = v61c_analysis.get("ranking_bootstrap", {}).get(
        "primary_conflict_unit_cluster_bootstrap", {}
    )
    f1_null = primary_null.get("intervals", {}).get(
        "generated_f1_delta", {}
    )
    if (
        v61c_finalized.get("schema")
        != "v61c-paired-null-independent-finalizer"
        or v61c_finalized.get("status")
        != "complete_numeric_only_evidence_verified_hpo_unauthorized"
        or v61c_finalized.get("protected_semantics_opened") is not False
        or v61c_sources.get("evidence") != {
            "file_sha256": V61C_EVIDENCE_FILE_SHA256,
            "content_sha256": V61C_EVIDENCE_CONTENT_SHA256,
        }
        or v61c_sources.get("analysis") != {
            "file_sha256": V61C_ANALYSIS_FILE_SHA256,
            "content_sha256": V61C_ANALYSIS_CONTENT_SHA256,
        }
        or v61c_sources.get("panel") != {
            "file_sha256": design65.V61C_PANEL_FILE_SHA256,
            "content_sha256": design65.V61C_PANEL_CONTENT_SHA256,
        }
        or v61c_evidence.get("protected_semantics_opened") is not False
        or v61c_analysis.get("schema")
        != "v61c-identical-state-paired-evaluator-null-analysis"
        or v61c_analysis.get("status") != "complete_characterization_only"
        or v61c_analysis.get("holdback_semantics_opened") is not False
        or v61c_analysis.get("protected_semantics_opened") is not False
        or primary_null.get("resampled_axis") != "conflict_unit"
        or primary_null.get(
            "within_unit_actor_pair_replicas_preserved_and_averaged"
        ) != 8
        or f1_null.get("halfwidth")
        != design65.V61C_RANKING_F1_NULL_HALFWIDTH
        or f1_null.get("contains_zero") is not True
    ):
        raise RuntimeError("v65 V61C null-calibration contract changed")

    calibration = design65.read_exact_self_hashed_v65(
        V62B_FINALIZED,
        V62B_FINALIZED_FILE_SHA256,
        V62B_FINALIZED_CONTENT_SHA256,
    )
    eligibility = calibration.get("calibration_eligibility_observation", {})
    frozen = calibration.get("frozen_non_authorization", {})
    v434_receipts = calibration.get("verification", {}).get(
        "v434_state_receipts", {}
    )
    if (
        calibration.get("schema")
        != "v62b-pre-hpo-alpha-zero-independent-finalizer"
        or calibration.get("status")
        != "complete_numeric_only_eligibility_observed_hpo_unauthorized"
        or eligibility.get(
            "eligible_for_later_separately_preregistered_hpo_work"
        ) is not True
        or eligibility.get("failed_gate_count") != 0
        or eligibility.get("passed_gate_count") != 3
        or eligibility.get("hpo_population_launch_or_update_authorized")
        is not False
        or eligibility.get("protected_access_authorized") is not False
        or frozen.get("hpo_population_launch_or_update_authorized") is not False
        or frozen.get(
            "holdback_ood_shadow_terminal_or_protected_access_authorized"
        ) is not False
        or calibration.get("protected_semantics_opened") is not False
        or calibration.get(
            "raw_question_answer_prediction_or_generation_text_persisted"
        ) is not False
        or v434_receipts.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or v434_receipts.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
    ):
        raise RuntimeError("v65 sealed V62B eligibility calibration changed")

    # The model seal is aggregate-only.  Do not read any model-directory byte.
    model_seal = design65.read_exact_self_hashed_v65(
        BASE_MODEL_SEAL,
        BASE_MODEL_SEAL_FILE_SHA256,
        BASE_MODEL_SEAL_CONTENT_SHA256,
    )
    base_arm = model_seal.get("arms", {}).get("base_middle_late", {})
    if (
        base_arm.get("path") != str(BASE_MODEL)
        or base_arm.get("shard_count") != 26
        or base_arm.get("all_files_fingerprint_sha256")
        != BASE_MODEL_ALL_FILES_FINGERPRINT_SHA256
    ):
        raise RuntimeError("v65 committed base-model seal changed")

    sources = {
        "v61_hash_only_preview": {
            "path": str(design65.PREVIEW_V61),
            "file_sha256": design65.PREVIEW_V61_FILE_SHA256,
            "content_sha256": design65.PREVIEW_V61_CONTENT_SHA256,
            "panel_manifest_sha256": preview["panels"][
                "panel_manifest_sha256"
            ],
            "launch_authorized_by_source": False,
            "future_candidate_outcomes_used_for_selection": False,
            "protected_semantics_opened": False,
        },
        "v53_exact_sigma_0p0048_arm": {
            "path": str(design65.V53_SIGMA_ARM),
            "file_sha256": design65.V53_SIGMA_ARM_FILE_SHA256,
            "content_sha256": design65.V53_SIGMA_ARM_CONTENT_SHA256,
            "sigma": design65.SIGMA_V65,
            "population_size": design65.POPULATION_SIZE_V65,
            "passed": True,
            "candidate_identity_inventory_sha256": (
                design65.V53_CANDIDATE_IDENTITY_INVENTORY_SHA256
            ),
            "runtime_identity_inventory_sha256": (
                design65.V53_RUNTIME_IDENTITY_INVENTORY_SHA256
            ),
            "ordered_state_identities": ordered_v53_identities,
            "ordered_state_identities_sha256": (
                design65.canonical_sha256_v65(ordered_v53_identities)
            ),
            "signed_score_receipts_sha256": arm[
                "signed_score_receipts_sha256"
            ],
            "signed_scores_sha256": arm["signed_scores_sha256"],
            "projection_or_update_authorized_by_source": False,
            "protected_semantics_opened": False,
        },
        "v61c_eight_replica_null_calibration": {
            "finalized": {
                "path": str(design65.V61C_FINALIZED),
                "file_sha256": design65.V61C_FINALIZED_FILE_SHA256,
                "content_sha256": design65.V61C_FINALIZED_CONTENT_SHA256,
                "launch_or_update_authorized_by_source": False,
            },
            "evidence": {
                "path": str(V61C_EVIDENCE),
                "file_sha256": V61C_EVIDENCE_FILE_SHA256,
                "content_sha256": V61C_EVIDENCE_CONTENT_SHA256,
            },
            "analysis": {
                "path": str(V61C_ANALYSIS),
                "file_sha256": V61C_ANALYSIS_FILE_SHA256,
                "content_sha256": V61C_ANALYSIS_CONTENT_SHA256,
            },
            "panel": {
                "path": str(design65.V61C_PANEL),
                "file_sha256": design65.V61C_PANEL_FILE_SHA256,
                "content_sha256": design65.V61C_PANEL_CONTENT_SHA256,
            },
            "primary_resampled_axis": "conflict_unit",
            "within_unit_actor_pair_replicas_preserved_and_averaged": 8,
            "ranking_f1_null_halfwidth": (
                design65.V61C_RANKING_F1_NULL_HALFWIDTH
            ),
            "protected_semantics_opened": False,
        },
        "v62b_finalized_calibration": {
            "path": str(V62B_FINALIZED),
            "finalizer_commit": V62B_FINALIZER_COMMIT,
            "file_sha256": V62B_FINALIZED_FILE_SHA256,
            "content_sha256": V62B_FINALIZED_CONTENT_SHA256,
            "eligible_for_later_separately_preregistered_hpo_work": True,
            "passed_gate_count": 3,
            "failed_gate_count": 0,
            "launch_or_update_authorized_by_source": False,
            "protected_access_authorized_by_source": False,
        },
    }
    return panel, sources


def base_model_binding_v65() -> dict:
    return {
        "path": str(BASE_MODEL),
        "committed_seal": {
            "path": str(BASE_MODEL_SEAL),
            "commit": BASE_MODEL_SEAL_COMMIT,
            "file_sha256": BASE_MODEL_SEAL_FILE_SHA256,
            "content_sha256": BASE_MODEL_SEAL_CONTENT_SHA256,
        },
        "top_level_file_count": 40,
        "weight_shard_count": 26,
        "weight_shard_manifest_sha256": (
            BASE_MODEL_WEIGHT_SHARD_MANIFEST_SHA256
        ),
        "non_weight_file_count": 14,
        "non_weight_manifest_sha256": BASE_MODEL_NON_WEIGHT_MANIFEST_SHA256,
        "all_top_level_files_fingerprint_sha256": (
            BASE_MODEL_ALL_FILES_FINGERPRINT_SHA256
        ),
        "full_byte_receipt_implementation": (
            "run_lora_es_v59_vs_v434_robust_confirmation_v64"
        ),
        "builder_read_model_directory_bytes": False,
    }


def v434_binding_v65() -> dict:
    return {
        "source": {
            "path": str(design52.SOURCE_V52),
            "weights_file_sha256": design52.SOURCE_WEIGHTS_SHA256_V52,
            "config_file_sha256": design52.SOURCE_CONFIG_SHA256_V52,
        },
        "staged": {
            "path": str(design52.STAGED_V52),
            "weights_file_sha256": design52.STAGED_WEIGHTS_SHA256_V52,
            "config_file_sha256": design52.STAGED_CONFIG_SHA256_V52,
            "manifest_file_sha256": (
                design52.STAGED_MANIFEST_FILE_SHA256_V52
            ),
            "manifest_content_sha256": (
                design52.STAGED_MANIFEST_CONTENT_SHA256_V52
            ),
            "transformed_identity_sha256": (
                design52.STAGED_TRANSFORMED_IDENTITY_SHA256_V52
            ),
            "ordered_values_sha256": design52.STAGED_ORDERED_VALUES_SHA256_V52,
        },
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "canonical_ordered_key_sha256": design52.MASTER_ORDERED_KEY_SHA256_V52,
        "runtime_assignment_sha256": design52.RUNTIME_ASSIGNMENT_SHA256_V52,
        "builder_reopened_adapter_bytes": False,
    }


def dataset_binding_v65() -> dict:
    return {
        "staged_semantic_rows": {
            "path": str(STAGED_DATASET_V61C),
            "historical_sealed_full_file_sha256_bound_not_reverified_live": (
                STAGED_DATASET_FILE_SHA256_V61C
            ),
            "physical_rows_in_sealed_file": 68,
            "authorized_zero_based_line_interval": [0, 63],
            "authorized_ranking_rows": 64,
            "authorized_exact_prefix_byte_count": (
                STAGED_RANKING_PREFIX_BYTES_V65
            ),
            "authorized_exact_prefix_sha256": (
                STAGED_RANKING_PREFIX_SHA256_V65
            ),
            "live_read_primitive": (
                "one exact os.pread of the authorized prefix byte count"
            ),
            "full_file_hash_verification_or_full_file_read_live": False,
            "decoded_json_lines_live": 64,
            "line_64_or_later_read_or_decoded_live": False,
        },
        "source_v61c_hash_numeric_panel": {
            "path": str(STAGED_PANEL_V61C),
            "file_sha256": STAGED_PANEL_FILE_SHA256_V61C,
            "content_sha256": STAGED_PANEL_CONTENT_SHA256_V61C,
            "physical_items": 68,
            "first_64_roles": "ranking",
            "last_4_roles": "exact_sentinel_not_semantically_opened",
            "may_open_live_for_hash_identity_mapping": True,
        },
        "source_train_artifacts_bound_but_may_not_open_live": {
            "train_dataset_path": str(design52.TRAIN_DATASET_V52),
            "train_dataset_file_sha256": design52.DATASET_SHA256_V52,
            "train_bundle_content_sha256": design52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
            "membership_path": str(design52.TRAIN_MEMBERSHIP_V52),
            "membership_file_sha256": design52.MEMBERSHIP_SHA256_V52,
            "membership_content_sha256": design52.MEMBERSHIP_CONTENT_SHA256_V52,
        },
        "builder_read_raw_or_semantic_dataset_rows": False,
        "live_train_holdback_or_sentinel_rows_authorized": False,
    }


def build_preregistration_v65(
    panel: dict,
    sources: dict,
    implementation_bindings: dict,
    *,
    ranking_panel_output: Path = RANKING_PANEL_OUTPUT,
    preregistration_output: Path = PREREGISTRATION_OUTPUT,
) -> dict:
    """Pure preregistration construction from already verified bindings."""
    if (
        panel.get("schema") != "v65-robust-sampling-ranking-panel"
        or panel.get("ranking_units") != design65.RANKING_UNITS_V65
        or panel.get("question_answer_or_generation_text_persisted") is not False
        or panel.get("protected_semantics_opened") is not False
        or set(sources) != {
            "v61_hash_only_preview", "v53_exact_sigma_0p0048_arm",
            "v61c_eight_replica_null_calibration",
            "v62b_finalized_calibration",
        }
        or not implementation_bindings
        or any(
            not isinstance(binding, dict)
            or set(binding) != {"path", "file_sha256"}
            for binding in implementation_bindings.values()
        )
    ):
        raise RuntimeError("v65 pure preregistration inputs changed")
    panel_output = Path(ranking_panel_output).resolve()
    preregistration_path = Path(preregistration_output).resolve()
    panel_file_sha256 = payload_sha256_v65(panel)
    state_grid = design65.state_derivations_v65()
    value = {
        "schema": "v65-lora-es-robust-sampling-population-preregistration",
        "status": (
            "sealed_before_v65_train_semantics_model_ray_or_gpu_access"
        ),
        "specific_v65_four_gpu_population_measurement_authorized": True,
        "prior_evidence_or_eligibility_alone_authorizes_launch": False,
        "builder_or_dry_run_performed_model_or_cuda_compute_launch": False,
        "purpose": (
            "Measure the exact reliable V53 sigma=0.0048 P16 antithetic "
            "population twice per signed state on V61's frozen 64-unit "
            "train-only ranking panel, using the V62B-calibrated four-actor "
            "generation evaluator. Persist numeric/hash-only measurements and "
            "stop without projection, update, holdback, sentinel, OOD, "
            "protected, or terminal access."
        ),
        "authorization": {
            "authority_origin": "this_specific_v65_preregistration_only",
            "gpu_launch": True,
            "population_generation_measurement": True,
            "gpu_population_measurement": True,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": 4,
            "tensor_parallel_size_per_actor": 1,
            "all_four_actors_generate_every_signed_state_pass": True,
            "projection": False,
            "optimizer_update": False,
            "optimizer_or_adapter_update": False,
            "candidate_snapshot": False,
            "candidate_snapshot_or_promotion": False,
            "promotion": False,
            "train_holdback": False,
            "exact_sentinel": False,
            "unused_reserve": False,
            "ood_shadow": False,
            "ood_shadow_or_benchmark": False,
            "protected_semantics": False,
            "terminal_holdout": False,
            "terminal_or_sealed_holdout": False,
        },
        "scientific_scope": {
            "measurement_only_population_pass": True,
            "ranking_panel_only": True,
            "future_candidate_outcomes_used_for_panel_selection": False,
            "exact_v53_sigma_arm_and_perturbation_identities_reused": True,
            "sigma_was_train_adapted_on_partially_overlapping_v52_v53_panel": True,
            "v61_partitions_are_unopened_within_v65_only": True,
            "v61_holdback_sentinel_or_reserve_globally_candidate_unexposed": False,
            "v65_result_can_serve_future_promotion_without_redesign": False,
            "v61c_eight_replica_null_calibration_reused": True,
            "v62b_evaluator_calibration_reused": True,
            "direction_coefficients_are_numeric_measurements_only": True,
            "direction_coefficients_authorize_projection_or_update": False,
            "measurement_success_authorizes_follow_on_launch": False,
        },
        "historical_adaptive_exposure": {
            "manifest_sha256": ADAPTIVE_PANEL_OVERLAP_MANIFEST_SHA256_V65,
            "v52_v53_panel_overlap_with_v61_partitions": {
                "ranking": {"conflict_units": 21, "selected_rows": 19},
                "holdback": {"conflict_units": 17, "selected_rows": 15},
                "exact_sentinel": {"conflict_units": 1, "selected_rows": 1},
                "unused_reserve": {"conflict_units": 25, "selected_rows": 17},
            },
            "sigma_train_adapted_on_partially_overlapping_panel": True,
            "v61_partitions_disjoint_within_current_v65_runtime": True,
            "globally_candidate_unexposed_claimed": False,
            "future_promotion_requires_panel_redesign": True,
        },
        "source_evidence": sources,
        "ranking_panel": {
            "path": str(panel_output),
            "file_sha256": panel_file_sha256,
            "content_sha256": panel["content_sha256_before_self_field"],
            "units": design65.RANKING_UNITS_V65,
            "request_order_sha256": panel["request_order_sha256"],
            "unit_order_sha256": panel["unit_order_sha256"],
            "hash_only": True,
            "question_answer_or_generation_text_persisted": False,
        },
        "fixed_measurement_recipe": {
            "base_model": base_model_binding_v65(),
            "v434_adapter": v434_binding_v65(),
            "dataset": dataset_binding_v65(),
            "population_size": design65.POPULATION_SIZE_V65,
            "seeds": list(design65.SEEDS_V65),
            "sigma": design65.SIGMA_V65,
            "unique_exact_v53_signed_states": 32,
            "unique_exact_v53_states": 32,
            "state_occurrences": design65.STATE_COUNT_V65,
            "scheduled_state_occurrences": design65.STATE_COUNT_V65,
            "passes_per_signed_state": (
                design65.PASSES_PER_SIGNED_STATE_V65
            ),
            "state_grid_sha256": design65.canonical_sha256_v65(state_grid),
            "state_schedule": state_grid,
            "state_order": (
                "direction_major_plus0_minus0_minus1_plus1_counterbalanced"
            ),
            "ordered_exact_v53_state_identities_sha256": sources[
                "v53_exact_sigma_0p0048_arm"
            ]["ordered_state_identities_sha256"],
            "exact_identity_occurrences": 64,
            "unique_fp32_and_bf16_identity_pairs": 32,
            "required_transition_rpc": (
                "transition_antithetic_from_pinned_master_v51"
            ),
            "v52_discovery_only_transition_used": False,
            "direct_candidate_derivation_from_pinned_fp32_master": True,
            "exact_candidate_identity_and_four_actor_consensus_before_score": True,
            "intermediate_master_restore_between_occurrences": False,
            "single_exact_master_restore_after_all_64_occurrences": True,
            "final_worker_transaction_quiescence_required": True,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": design65.ACTORS_V65,
            "tensor_parallel_size_per_actor": 1,
            "ranking_units_per_actor_call": design65.RANKING_UNITS_V65,
            "unscored_master_warmup": {
                "periods": 4,
                "state": "exact unchanged V434 pinned master",
                "actors": 4,
                "ranking_requests_per_actor_period": 64,
                "discarded_generation_completions": 1_024,
                "occurs_before_every_signed_state": True,
                "exact_master_receipt_before_and_after_each_period": True,
                "raw_outputs_scored_or_persisted": False,
                "generation_metrics_computed_or_persisted": False,
                "adaptive_retry_drop_reorder_or_early_stop": False,
            },
            "scored_generation_completions": (
                design65.GENERATION_COMPLETIONS_V65
            ),
            "total_generation_completions": 17_408,
            "scored_generation_completions_per_gpu": (
                design65.GENERATION_COMPLETIONS_V65 // design65.ACTORS_V65
            ),
            "common_generation_seed_per_actor_pass": (
                COMMON_GENERATION_SEED_V65
            ),
            "generation_seed": COMMON_GENERATION_SEED_V65,
            "generation_params_without_seed": dict(
                GENERATION_PARAMS_WITHOUT_SEED_V65
            ),
            "fixed_engine_controls": {
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "async_scheduling": False,
                "max_num_seqs": 68,
                "scheduling_policy": "fcfs",
                "VLLM_BATCH_INVARIANT": False,
            },
            "v62a_v62b_calibrated_runtime_controls": dict(
                calibration62b.RUNTIME_CONTROLS_V62B
            ),
            "sanitized_live_engine_config_receipt": {
                "required_from_every_actor": True,
                "actors": 4,
                "allowed_fields": [
                    "actor_rank", "physical_gpu_id", "pid",
                    "enable_prefix_caching", "enforce_eager",
                    "async_scheduling", "max_num_seqs",
                    "scheduling_policy", "VLLM_BATCH_INVARIANT",
                ],
                "raw_model_config_or_semantic_text_persisted": False,
                "all_fields_must_exactly_equal_fixed_engine_controls": True,
            },
            "runtime": dict(design52.RUNTIME_V52),
            "required_python": str(design52.REQUIRED_PYTHON_V52),
            "metric_order": list(design65.METRIC_ORDER_V65),
            "raw_generation_discarded_after_numeric_hash_scoring": True,
        },
        "population_analysis_contract": {
            "pairing_keys": [
                "direction", "unit_identity_sha256", "actor_rank",
                "pass_index",
            ],
            "plus_minus_pairing_within_each_direction": True,
            "within_unit_actor_pass_replicas_preserved_and_averaged": 8,
            "paired_replicas_per_unit_preserved_and_averaged": 8,
            "bootstrap_resampled_axis": "conflict_unit_only",
            "resampled_axis": "conflict_unit_only",
            "single_replica_per_resampled_unit_sampling": False,
            "bootstrap_replicates": design65.BOOTSTRAP_REPLICATES_V65,
            "bootstrap_seed": design65.BOOTSTRAP_SEED_V65,
            "bootstrap_index_matrix_sha256": hashlib.sha256(
                design65.frozen_bootstrap_indices_v65().astype(
                    "<i8", copy=False,
                ).tobytes(order="C")
            ).hexdigest(),
            "one_sided_alpha": design65.BOOTSTRAP_ALPHA_V65,
            "generation_composite_weights": dict(
                design65.design61.GENERATION_COMPOSITE_WEIGHTS_V61
            ),
            "stability_penalty_weights": dict(
                design65.design61.STABILITY_WEIGHTS_V61
            ),
            "robust_generation_direction_coefficients": (
                "unit_norm_of_population_mean_centered_robust_fitness"
            ),
            "stability_lcb_direction_coefficients": (
                "unit_norm_of_population_mean_centered_stability_lcb"
            ),
            "discriminability_gate": {
                "split_pass_spearman_minimum_inclusive": (
                    design65.MINIMUM_SPLIT_PASS_SPEARMAN_V65
                ),
                "split_pass_centered_cosine_minimum_inclusive": (
                    design65.MINIMUM_SPLIT_PASS_CENTERED_COSINE_V65
                ),
                "v61c_ranking_f1_null_halfwidth": (
                    design65.V61C_RANKING_F1_NULL_HALFWIDTH
                ),
                "direction_population_standard_deviation_strictly_greater_than": (
                    design65.MINIMUM_DIRECTION_SPREAD_V65
                ),
                "required_check_keys": [
                    "split_pass_spearman_at_least_0p50",
                    "split_pass_centered_cosine_at_least_0p50",
                    "direction_spread_strictly_above_twice_v61c_null_halfwidth",
                    "stability_direction_spread_strictly_positive",
                ],
                "failed_gate_action": (
                    "persist numeric diagnostics with coefficient fields null"
                ),
            },
            "exact_used_for_population_ranking": False,
            "projection_or_update_performed": False,
        },
        "access_contract": {
            "builder_may_read": [
                "sealed V61 hash/numeric preview",
                "sealed V53 numeric sigma arm",
                "sealed V61C hash/numeric panel, evidence, analysis, and finalizer",
                "sealed V62B numeric finalizer",
                "committed aggregate base-model seal",
                "repository implementation source bytes",
            ],
            "builder_or_dry_run_may_read_raw_dataset_or_semantic_rows": False,
            "builder_or_dry_run_may_read_base_model_directory_bytes": False,
            "live_semantic_dataset_path": str(STAGED_DATASET_V61C),
            "decode_exactly_first_64_v61c_ranking_rows": True,
            "decode_v61c_row_64_or_later": False,
            "ranking_prefix_bytes": STAGED_RANKING_PREFIX_BYTES_V65,
            "ranking_prefix_sha256": STAGED_RANKING_PREFIX_SHA256_V65,
            "live_semantic_zero_based_line_interval": [0, 63],
            "live_semantic_exact_prefix_byte_count": (
                STAGED_RANKING_PREFIX_BYTES_V65
            ),
            "live_semantic_exact_prefix_sha256": (
                STAGED_RANKING_PREFIX_SHA256_V65
            ),
            "live_semantic_reader_uses_one_exact_prefix_pread": True,
            "live_full_jsonl_hash_verification_or_full_read": False,
            "live_semantic_reader_must_not_read_or_decode_line_64_or_later": True,
            "v61c_hash_numeric_panel_may_open_live": True,
            "full_train_dataset_or_membership_may_open_live": False,
            "train_holdback_exact_sentinel_or_unused_reserve_may_open": False,
            "ood_shadow_benchmark_protected_or_terminal_may_open": False,
            "optimizer_master_projection_or_update_state_may_open": False,
            "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
            "numeric_hash_only_evidence_required": True,
        },
        "implementation_bindings": implementation_bindings,
        "implementation_closure_summary": {
            "file_count": len(implementation_bindings),
            "manifest_sha256": design65.canonical_sha256_v65({
                key: binding["file_sha256"]
                for key, binding in sorted(implementation_bindings.items())
            }),
            "live_runner_imported_or_executed_by_builder": False,
            "base_model_byte_receipt_implementation_bound_not_executed": True,
        },
        "artifacts": {
            "run_directory": str(RUN_DIR),
            "attempt": str(RUN_DIR.parent / f".{EXPERIMENT}.attempt.json"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v65.jsonl"),
            "evidence": str(RUN_DIR / "robust_sampling_evidence_v65.json"),
            "analysis": str(RUN_DIR / "robust_sampling_analysis_v65.json"),
            "report": str(RUN_DIR / "robust_sampling_report_v65.json"),
            "failure": str(RUN_DIR / "failure_v65.json"),
        },
        "required_integrity_gates": {
            "all_sealed_source_file_and_content_hashes_exact": True,
            "ranking_panel_file_content_and_order_hashes_exact": True,
            "full_base_model_byte_receipt_before_load_and_after_cleanup": True,
            "v434_source_staged_master_and_runtime_identities_exact": True,
            "exclusive_idle_four_gpu_preflight": True,
            "exactly_four_tp1_actor_pid_and_runtime_identities": True,
            "all_four_gpus_attributed_positive_each_generation_phase": True,
            "four_unscored_all_actor_master_warmups_complete_before_population": True,
            "warmup_exact_master_receipts_before_and_after_each_period": True,
            "warmup_1024_completions_discarded_without_scoring_or_persistence": True,
            "all_four_live_engine_config_receipts_match_v62a_v62b_controls": True,
            "all_64_occurrences_use_v51_direct_pinned_master_transition_rpc": True,
            "exactly_32_unique_nonmaster_fp32_and_bf16_identity_pairs": True,
            "all_four_actor_candidate_identities_exact_before_score": True,
            "one_exact_final_master_restore_and_quiescence_after_population": True,
            "all_64_state_occurrences_and_16384_scored_completions_included": True,
            "total_generation_completions_exactly_17408": True,
            "all_eight_actor_pass_replicas_preserved_before_unit_bootstrap": True,
            "split_pass_reliability_and_v61c_noise_gate_applied": True,
            "no_retry_drop_reorder_early_stop_or_adaptive_selection": True,
            "strict_four_engine_cleanup_and_final_idle": True,
            "numeric_hash_only_evidence": True,
            "projection_update_candidate_holdback_sentinel_ood_protected_"
            "terminal_access_zero": True,
        },
        "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
        "projection_update_or_candidate_performed": False,
        "train_holdback_or_exact_sentinel_opened": False,
        "ood_shadow_benchmark_protected_or_terminal_opened": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        design65.canonical_sha256_v65(value)
    )
    return value


def build_v65(
    *,
    ranking_panel_output: Path = RANKING_PANEL_OUTPUT,
    preregistration_output: Path = PREREGISTRATION_OUTPUT,
    implementation_entry_paths: dict[str, Path] | None = None,
) -> tuple[dict, dict]:
    panel, sources = sealed_source_bindings_v65()
    implementation = implementation_bindings_v65(
        implementation_entry_paths
    )
    preregistration = build_preregistration_v65(
        panel, sources, implementation,
        ranking_panel_output=ranking_panel_output,
        preregistration_output=preregistration_output,
    )
    return panel, preregistration


def _exclusive_write_pair_v65(
    first_path: Path, first_payload: bytes,
    second_path: Path, second_payload: bytes,
) -> None:
    paths = [Path(first_path).resolve(), Path(second_path).resolve()]
    if paths[0] == paths[1] or any(path.exists() for path in paths):
        raise FileExistsError("v65 outputs must be two distinct fresh paths")
    temporaries: list[Path] = []
    linked: list[Path] = []
    try:
        for path, payload in zip(paths, (first_payload, second_payload), strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.tmp-", dir=path.parent,
            )
            temporary = Path(temporary_name)
            temporaries.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in zip(temporaries, paths, strict=True):
            os.link(temporary, path)
            linked.append(path)
    except BaseException:
        for path in linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranking-panel-output", default=str(RANKING_PANEL_OUTPUT),
    )
    parser.add_argument(
        "--preregistration-output", default=str(PREREGISTRATION_OUTPUT),
    )
    args = parser.parse_args(argv)
    panel_output = Path(args.ranking_panel_output).resolve()
    preregistration_output = Path(args.preregistration_output).resolve()
    panel, preregistration = build_v65(
        ranking_panel_output=panel_output,
        preregistration_output=preregistration_output,
    )
    _exclusive_write_pair_v65(
        panel_output, json_payload_v65(panel),
        preregistration_output, json_payload_v65(preregistration),
    )
    print(json.dumps({
        "ranking_panel": str(panel_output),
        "ranking_panel_file_sha256": design65.file_sha256_v65(panel_output),
        "ranking_panel_content_sha256": panel[
            "content_sha256_before_self_field"
        ],
        "preregistration": str(preregistration_output),
        "preregistration_file_sha256": design65.file_sha256_v65(
            preregistration_output
        ),
        "preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "specific_v65_four_gpu_population_measurement_authorized": True,
        "builder_raw_semantics_model_cuda_or_protected_accessed": False,
        "projection_update_holdback_sentinel_ood_or_terminal_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
