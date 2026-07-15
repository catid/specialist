#!/usr/bin/env python3
"""Build compact immutable negative evidence for completed V34C."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V34D = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v34c_production_v401_replacement_fraction_hpo_basis20261015."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_V34D = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v34c_production_v401_replacement_fraction_hpo_basis20261015/"
    "production_v401_replacement_fraction_report_v34c.json"
)
OUTPUT_RELATIVE_PATH_V34D = (
    "experiments/eggroll_es_hpo/"
    "S6_V34D_V34C_PRODUCTION_V401_REPLACEMENT_FRACTION_NEGATIVE_EVIDENCE.json"
)
ATTEMPT_PATH_V34D = ROOT / ATTEMPT_RELATIVE_PATH_V34D
REPORT_PATH_V34D = ROOT / REPORT_RELATIVE_PATH_V34D
OUTPUT_PATH_V34D = ROOT / OUTPUT_RELATIVE_PATH_V34D

ATTEMPT_FILE_SHA256_V34D = (
    "66e32c7528b196f8f62cc06c684cb646cdebaf2a016df53a4d7983a8a26ede6f"
)
ATTEMPT_CONTENT_SHA256_V34D = (
    "17793b2a328c9f711d494bffff85888b7a081fe9612e789216fcaa1fc4194dd6"
)
REPORT_FILE_SHA256_V34D = (
    "8dc9bcec82c52a1a961e6e7596059425d42e8003281e97cc30f424fc1a6e3825"
)
REPORT_CONTENT_SHA256_V34D = (
    "1163bd079d692c48a83fce845825907467c98a8800660c5e4d92b4c6c107c79d"
)
COMMITTED_SOURCE_CERTIFICATE_SHA256_V34D = (
    "980321c40dd9fb750b9eda1716ad8106247ef0a18e2117c024bc145c346e00d3"
)
IMPLEMENTATION_BUNDLE_SHA256_V34D = (
    "62799c09d85cf62c5428bb541836726086b04f97c6440036f4ab4e0f7ab8563c"
)
RECIPE_CONTENT_SHA256_V34D = (
    "e1196c730be0cddb82af953a49120a659ab39ecb4a3927a288b806319de1b2e7"
)
RUNTIME_ENVIRONMENT_CERTIFICATE_SHA256_V34D = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_MODEL_AUDIT_SHA256_V34D = (
    "fcfeda342954ba12e80c4ac82a0040f279c6bdcb3c003649ffa9c9abd737e2c5"
)
PRELAUNCH_IDLE_CERTIFICATE_SHA256_V34D = (
    "63436f339c5aa2430289d200b59ecd6043298981e79dcb7c89d0346be8a78de6"
)
FINAL_IDLE_CERTIFICATE_SHA256_V34D = (
    "66a8efe7fbd5abc4e9423ccb5dd0616a72a52c569695ace5de0834d3d2d575e8"
)
POSTCLEANUP_BINDING_RECHECK_SHA256_V34D = (
    "266988823212d6671108ddd1151832261e89826f3e6e87374188f4d82862cfa3"
)
CONFIGURATION_CONTENT_SHA256_V34D = (
    "e87e8b5214c96433676a8c42ac59f247be75b9b9ec6bf1fe60cc18ff227fd2ef"
)
PREANALYSIS_AUDIT_CONTENT_SHA256_V34D = (
    "387002798060145fe1ec38140bbb26f646a53c06626395dba1ddaa8578fe6f9b"
)
SUMMARY_CONTENT_SHA256_V34D = (
    "af2d793cd9650bb748de509c6a038a4f4312197674b9158c8190af78237885d6"
)
FIXED_SEQUENCE_CONTENT_SHA256_V34D = (
    "f3e614d8a4ffbad202c739ef8b28c2ccf43f7ca0472f2bc6eb1876c64024ccdb"
)
GATE_CONTENT_SHA256_V34D = (
    "ae92bb2637a8d89bf3362ab3c4c42af2337a735e4b3f5575ad3f19daeca471f4"
)

V34B_COMMIT_V34D = "b254d4bdae0bb3fcb98d015c155393df9cca2d5d"
V34C_COMMIT_V34D = "756b782e33cbf5cff27c68c3827d86d0ca6d5150"
V34B_SOURCE_FILES_V34D = {
    "frame_builder_v34b": {
        "relative_path": "build_eggroll_es_v401_replacement_fraction_frame_v34b.py",
        "file_sha256": "936efe422f560fd49f4f6bfa775465c786e9b3d2299189c4343f38ae1ede1774",
    },
    "frame_test_v34b": {
        "relative_path": "test_build_eggroll_es_v401_replacement_fraction_frame_v34b.py",
        "file_sha256": "b1a8c7098302c844966dc056e191f193a3cfd908f17558e2c591ebdfc6f17ff7",
    },
    "frame_v34b": {
        "relative_path": (
            "experiments/eggroll_es_hpo/train_panel_sampling_v34b/"
            "production_v401_replacement_fraction_panels_v34b.json"
        ),
        "file_sha256": "832bbea07d08c487621e2dc88dfb8ebffc4b05d888badbbe5eb0fd71124efde3",
    },
    "preregistration_module_v34b": {
        "relative_path": "eggroll_es_v401_replacement_fraction_preregistration_v34b.py",
        "file_sha256": "5ebb49b951c61e144cda1f6dd06d23b4b1f746cd500a58059f77fc8aec48f41b",
    },
    "preregistration_test_v34b": {
        "relative_path": "test_eggroll_es_v401_replacement_fraction_preregistration_v34b.py",
        "file_sha256": "1499e8d5aff293fcdafc14896c89cc3506c691a1468ac39a0b7d15a16296b13d",
    },
    "preregistration_v34b": {
        "relative_path": (
            "experiments/eggroll_es_hpo/"
            "S6_V401_REPLACEMENT_FRACTION_HPO_V34B_PREREGISTRATION.json"
        ),
        "file_sha256": "b852730872621fe9259087dd681ebf8854f985e8caa9208f0e8257a1d07de91b",
    },
    "mechanics_v34b": {
        "relative_path": "train_eggroll_es_v401_replacement_fraction_v34b.py",
        "file_sha256": "5a59d618ba690f354c0564a52a391602b6f7a207a076523eac5ae262ba50c183",
    },
    "mechanics_test_v34b": {
        "relative_path": "test_train_eggroll_es_v401_replacement_fraction_v34b.py",
        "file_sha256": "d6d1b87d53b18f9c0e69087c9e02cc2445e1cfb81d9acaf83535b6ec261bb3f9",
    },
    "cpu_runner_v34b": {
        "relative_path": "run_eggroll_es_v401_replacement_fraction_v34b.py",
        "file_sha256": "4cfa9e0da6038b2ba5f3998c8597827e715b130e0bda7a2ca4c7145bf58cd02e",
    },
    "cpu_runner_test_v34b": {
        "relative_path": "test_run_eggroll_es_v401_replacement_fraction_v34b.py",
        "file_sha256": "fc3e789b9d1a69f94f8c02e70cbb422ceee503bfb7d740930277703928066e45",
    },
}
V34C_SOURCE_FILES_V34D = {
    "gpu_runtime_v34c": {
        "relative_path": "run_eggroll_es_v401_replacement_fraction_v34c.py",
        "file_sha256": "c9686c15fe2b187908ab8f916ab3d4feb7b2b52b148585f9bed7bdbc0fa43b83",
    },
    "gpu_runtime_test_v34c": {
        "relative_path": "test_run_eggroll_es_v401_replacement_fraction_v34c.py",
        "file_sha256": "d09235124e09f07b1683ec74ba2b8c08986fbe079ecf70bd5ab25adf6cc82b4f",
    },
}
PREREGISTRATION_IDENTITY_V34D = {
    "relative_path": (
        "experiments/eggroll_es_hpo/"
        "S6_V401_REPLACEMENT_FRACTION_HPO_V34B_PREREGISTRATION.json"
    ),
    "file_sha256": "b852730872621fe9259087dd681ebf8854f985e8caa9208f0e8257a1d07de91b",
    "content_sha256": "8ebf4b34b693e459c2b59f331e0768eb116842dbd61c24202302794b3e3b439b",
}
PANEL_IDENTITIES_V34D = {
    "content_free_frame": {
        "relative_path": (
            "experiments/eggroll_es_hpo/train_panel_sampling_v34b/"
            "production_v401_replacement_fraction_panels_v34b.json"
        ),
        "file_sha256": "832bbea07d08c487621e2dc88dfb8ebffc4b05d888badbbe5eb0fd71124efde3",
        "content_sha256": "2b2a5f9b66b59401f7ca1fffd4d901933d02b84b580356dc50b7b874da18dc7b",
        "runtime_frame_content_sha256": (
            "a4f290bcdece10de81997d680c07475266896c7140ed394ede990b0e93d98c0e"
        ),
    },
    "transient_panel_bundle": {
        "content_sha256": "9d9824dfb0051ab8ed39d2a1d01ad22baa77c94305a9c76d434b38c41aca2f6c",
    },
}
CANDIDATE_MANIFEST_IDENTITY_V34D = {
    "relative_path": "experiments/eggroll_es_hpo/dataset_candidates/v401/manifest.json",
    "freeze_commit": "59dfe718a914be8b37e05ff9daa822ab467d18a4",
    "file_sha256": "1013032a1a4c21a2ece6e80e0930aecee8430639e07eaf48c2ac7701708e8f52",
    "content_sha256": "42304107e89119c10c545dc79b4f85ab08bd4b8b78efec710d286987a3e8a5af",
    "candidate_file_sha256": (
        "8e29826dd389171c69f5eb6f43781f900345974c3d4d11274268e86c6145693b"
    ),
}
PRODUCTION_IDENTITY_V34D = {
    "relative_path": "data/train_qa_curated_v1.jsonl",
    "source_commit": "a21de35748054c3ae8737a767606234952f9561e",
    "file_sha256": "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507",
}
MODEL_AND_LAYER_IDENTITIES_V34D = {
    "model": {
        "relative_path": "models/Qwen3.6-35B-A3B",
        "config_sha256": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "index_sha256": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    },
    "layer_plan": {
        "relative_path": "experiments/layer_plans/v23a_base_middle_late_dense.json",
        "file_sha256": "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747",
        "content_sha256": "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f",
    },
}

ATTEMPT_KEYS_V34D = {
    "all_four_gpus_idle_after_cleanup", "checkpoint_written",
    "committed_source_certificate_sha256", "content_sha256_before_self_field",
    "dataset_promotion_applied", "evaluation_opened",
    "final_idle_certificate_sha256", "implementation_bundle_sha256",
    "live_model_audit_sha256", "model_update_applied", "nontrain_surface_opened",
    "phase", "prelaunch_idle_certificate_sha256", "recipe_sha256",
    "report_binding", "runtime_environment_certificate_sha256", "schema", "status",
}
REPORT_KEYS_V34D = {
    "all_four_gpus_idle_after_cleanup", "checkpoint_written",
    "committed_source_certificate_sha256", "configuration",
    "content_sha256_before_self_field", "dataset_promotion_applied",
    "direct_action_taken", "evaluation_opened", "final_idle_certificate_sha256",
    "gate", "implementation_bundle_sha256", "live_model_audit_sha256",
    "model_update_applied", "nontrain_surface_opened",
    "postcleanup_binding_recheck_sha256", "preanalysis_runtime_audit",
    "prelaunch_idle_certificate_sha256", "recipe_sha256",
    "runtime_environment_certificate_sha256", "schema", "status", "summary",
}
FORBIDDEN_PAYLOAD_KEYS_V34D = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "unit_scores", "responses", "coefficients",
    "bootstrap_replicates", "bootstrap_draws", "row_content", "row_index",
    "row_sha256", "document_sha256", "unit_id", "unit_ids", "pids",
    "timings", "memory_samples", "traceback",
}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    result = copy.deepcopy(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json_object(path, label):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"V34D {label} path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V34D {label} must be a JSON object")
    return value


def _verify_self(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V34D {label} self hash changed",
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v34d(value):
    overlap = FORBIDDEN_PAYLOAD_KEYS_V34D & set(_recursive_keys(value))
    _require(
        not overlap,
        f"V34D evidence contains forbidden detailed payload keys: {sorted(overlap)}",
    )


def _validate_attempt_v34d(attempt, report):
    _require(set(attempt) == ATTEMPT_KEYS_V34D, "V34D attempt schema keys changed")
    _verify_self(attempt, ATTEMPT_CONTENT_SHA256_V34D, "attempt")
    _require(
        attempt.get("schema") == "eggroll-es-v34c-durable-launch-attempt"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_compact_report_and_final_gpu_cleanup"
        and attempt.get("committed_source_certificate_sha256")
        == COMMITTED_SOURCE_CERTIFICATE_SHA256_V34D
        and attempt.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V34D
        and attempt.get("recipe_sha256") == RECIPE_CONTENT_SHA256_V34D
        and attempt.get("runtime_environment_certificate_sha256")
        == RUNTIME_ENVIRONMENT_CERTIFICATE_SHA256_V34D
        and attempt.get("live_model_audit_sha256")
        == LIVE_MODEL_AUDIT_SHA256_V34D
        and attempt.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_CERTIFICATE_SHA256_V34D
        and attempt.get("final_idle_certificate_sha256")
        == FINAL_IDLE_CERTIFICATE_SHA256_V34D,
        "V34D attempt source, implementation, recipe, model, or lifecycle binding changed",
    )
    _require(
        attempt.get("report_binding") == {
            "path": str(REPORT_PATH_V34D.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V34D,
            "content_sha256": REPORT_CONTENT_SHA256_V34D,
        },
        "V34D attempt report binding changed",
    )
    cross_keys = (
        "committed_source_certificate_sha256", "implementation_bundle_sha256",
        "recipe_sha256", "runtime_environment_certificate_sha256",
        "live_model_audit_sha256", "prelaunch_idle_certificate_sha256",
        "final_idle_certificate_sha256",
    )
    _require(
        all(report.get(key) == attempt.get(key) for key in cross_keys),
        "V34D attempt and report bindings disagree",
    )


def _validate_configuration_v34d(report):
    configuration = report.get("configuration", {})
    _verify_self(configuration, CONFIGURATION_CONTENT_SHA256_V34D, "configuration")
    _require(
        set(configuration) == {
            "schema", "all_four_tp1_engines", "alpha_zero",
            "content_sha256_before_self_field", "device_identity_sha256",
            "installation_sha256", "panel_bundle_content_sha256",
            "seed_certificate_sha256", "selected_reference_identity_sha256",
            "update_checkpoint_evaluation_surfaces_closed",
        }
        and configuration.get("schema")
        == "eggroll-es-v401-replacement-fraction-configuration-v34c"
        and configuration.get("all_four_tp1_engines") is True
        and configuration.get("alpha_zero") is True
        and configuration.get("update_checkpoint_evaluation_surfaces_closed") is True
        and configuration.get("device_identity_sha256")
        == "a62153ba381c77400ecdaf2601326e174365def247f28e9ec00fba085e0f6a30"
        and configuration.get("installation_sha256")
        == "182b2d2f69d2860fea1c620da61721d2845a39b14ffd4ec8fe4e31ca09daaa61"
        and configuration.get("panel_bundle_content_sha256")
        == PANEL_IDENTITIES_V34D["transient_panel_bundle"]["content_sha256"]
        and configuration.get("seed_certificate_sha256")
        == "de6d18568bd93b2c3a453c9e2e637fdf72c0ab017d774e826e812e61adb77fa7"
        and configuration.get("selected_reference_identity_sha256")
        == "8123122857a895cd94e8f45119ed30e770536517e0a1d0606b9a45432da6fd8f",
        "V34D four-GPU configuration or origin identity changed",
    )
    return configuration


def _validate_preanalysis_v34d(report):
    audit = report.get("preanalysis_runtime_audit", {})
    _verify_self(audit, PREANALYSIS_AUDIT_CONTENT_SHA256_V34D, "preanalysis audit")
    _require(
        set(audit) == {
            "schema", "all_four_tp1_engines_both_sources_every_wave",
            "content_sha256_before_self_field", "dense_result_commitments_sha256",
            "fixed_request_identity_sha256", "fraction_specific_request_count",
            "full_context_guard", "full_context_request_count", "model_update_applied",
            "perturbed_request_count", "population_boundary_sha256",
            "raw_scores_or_semantic_payloads_persisted", "signed_schedule_sha256",
            "synchronized_signed_wave_count", "token_audit_sha256",
            "total_generation_request_count", "wave_activity_and_restore_sha256",
        }
        and audit.get("schema")
        == "eggroll-es-v401-replacement-fraction-preanalysis-audit-v34c"
        and audit.get("all_four_tp1_engines_both_sources_every_wave") is True
        and audit.get("synchronized_signed_wave_count") == 32
        and audit.get("perturbed_request_count") == 49_920
        and audit.get("full_context_request_count") == 4_680
        and audit.get("fraction_specific_request_count") == 0
        and audit.get("total_generation_request_count") == 54_600
        and audit.get("model_update_applied") is False
        and audit.get("raw_scores_or_semantic_payloads_persisted") is False
        and audit.get("dense_result_commitments_sha256")
        == "89415913905439f9b4013f3b7af41f4d3e39ecd4d03227c9ee98962a63f53e0c"
        and audit.get("fixed_request_identity_sha256")
        == "ce6546c845dbde7020b5d3d1e37652ce892ba3f727ba1da5e93dcd13834fb6d6"
        and audit.get("population_boundary_sha256")
        == "0fafb9834bbc724795b72a2bfa7da36c3539f3ee093995180f8c3f7bd2087988"
        and audit.get("signed_schedule_sha256")
        == "18e24bad74170825fe3612324b35d1d9712f3dae359a7c525f36ef1286f2e8b9"
        and audit.get("token_audit_sha256")
        == "3ee889b5dc15f2daa46c2af87bcd1f1d15eb0eaefdd57a13e9f98c97b135b7ed"
        and audit.get("wave_activity_and_restore_sha256")
        == "9366f559c58207e3d1731d7733f8e50009bfe600f87007d7fa1531397e641a89",
        "V34D 32-wave, request, activity, or boundary audit changed",
    )
    guard = audit.get("full_context_guard", {})
    phase = {
        "dense_commitments_sha256": (
            "98b72f96cde7941bace3ffdb9aa90825cc71a5d436cfb62aaddae12542780ccf"
        ),
        "score_arrays_sha256": (
            "9cccc3c1c0626bc329838a875c04b72dd04e3c25e3d01988a232d977e8866093"
        ),
    }
    exact = {
        "all_dense_result_commitments_exact": True,
        "all_source_engine_panel_score_arrays_exact": True,
    }
    _require(
        set(guard) == {
            "a_b_exact", "a_c_exact", "excluded_from_fraction_analysis",
            "phase_a", "phase_b", "phase_c",
        }
        and guard.get("a_b_exact") == exact
        and guard.get("a_c_exact") == exact
        and guard.get("excluded_from_fraction_analysis") is True
        and all(guard.get(name) == phase for name in (
            "phase_a", "phase_b", "phase_c"
        )),
        "V34D full-context A-B-C exact guard changed",
    )
    return audit


def _expected_fraction_endpoints_v34d():
    return {
        "aggregate_to_optimization_cosine_median": {
            "familywise_lcb": -0.007115963038491676,
            "fraction_minus_production": -0.0010475757231054317,
            "noninferiority_margin": 0.0,
        },
        "aggregate_to_optimization_cosine_worst": {
            "familywise_lcb": -0.007732728118699607,
            "fraction_minus_production": -0.0036341577519395374,
            "noninferiority_margin": 0.0,
        },
        "aggregate_to_optimization_sign_agreement_median": {
            "familywise_lcb": -0.03125,
            "fraction_minus_production": 0.0,
            "noninferiority_margin": 0.0,
        },
        "aggregate_to_optimization_sign_agreement_worst": {
            "familywise_lcb": -0.03125,
            "fraction_minus_production": 0.0,
            "noninferiority_margin": 0.0,
        },
        "optimization_pairwise_cosine_median": {
            "familywise_lcb": -0.006948961205803455,
            "fraction_minus_production": -0.0029125659578429774,
            "noninferiority_margin": 0.0,
        },
        "optimization_pairwise_cosine_worst": {
            "familywise_lcb": -0.007046657959264781,
            "fraction_minus_production": -0.0025744247860692293,
            "noninferiority_margin": 0.0,
        },
        "optimization_pairwise_sign_agreement_median": {
            "familywise_lcb": -0.03125,
            "fraction_minus_production": 0.0,
            "noninferiority_margin": 0.0,
        },
        "optimization_pairwise_sign_agreement_worst": {
            "familywise_lcb": -0.03125,
            "fraction_minus_production": 0.0,
            "noninferiority_margin": 0.0,
        },
        "train_screen_cosine_median": {
            "familywise_lcb": -0.003918169789458088,
            "fraction_minus_production": 0.0020034967993220465,
            "noninferiority_margin": 0.0,
        },
        "train_screen_cosine_worst": {
            "familywise_lcb": -0.006005723865678034,
            "fraction_minus_production": 0.002938824557453368,
            "noninferiority_margin": 0.0,
        },
        "train_screen_sign_agreement_median": {
            "familywise_lcb": -0.03125,
            "fraction_minus_production": 0.0078125,
            "noninferiority_margin": 0.0,
        },
        "train_screen_sign_agreement_worst": {
            "familywise_lcb": -0.03125,
            "fraction_minus_production": 0.015625,
            "noninferiority_margin": 0.0,
        },
    }


def _validate_fixed_sequence_v34d(summary):
    analysis = summary.get("fixed_sequence_analysis", {})
    _verify_self(analysis, FIXED_SEQUENCE_CONTENT_SHA256_V34D, "fixed sequence")
    _require(
        set(analysis) == {
            "schema", "bootstrap", "content_sha256_before_self_field",
            "fraction_specific_model_requests", "largest_consecutively_passing_fraction",
            "persisted_response_vectors_unit_scores_coefficients_or_draws",
            "production_compact_estimator_sha256", "stopped_at_first_failure",
            "tested_fractions", "untested_fractions_after_first_failure",
        }
        and analysis.get("schema")
        == "eggroll-es-v401-replacement-fraction-fixed-sequence-analysis-v34b"
        and analysis.get("fraction_specific_model_requests") == 0
        and analysis.get("largest_consecutively_passing_fraction") == 0.0
        and analysis.get("stopped_at_first_failure") is True
        and analysis.get(
            "persisted_response_vectors_unit_scores_coefficients_or_draws"
        ) is False
        and analysis.get("production_compact_estimator_sha256")
        == "35384d94bd0c9eb803e151b07604d793430f8dd2b1a7c8d9c65db6d66f82380d"
        and analysis.get("untested_fractions_after_first_failure")
        == [0.1, 0.2, 0.4, 1.0],
        "V34D fixed-sequence stop or higher-fraction status changed",
    )
    _require(
        analysis.get("bootstrap") == {
            "draw_plan_sha256": (
                "458d4bdaf9e8f990258712e561699c630ab8e1091f9919979492081c283d5dec"
            ),
            "one_sided_bonferroni_quantile": 0.004166666666666667,
            "quantile_method": "linear",
            "raw_draws_or_replicates_persisted": False,
            "repetitions": 50_000,
            "seed": 20_261_016,
        },
        "V34D bootstrap aggregate changed",
    )
    tested = analysis.get("tested_fractions", [])
    _require(len(tested) == 1, "V34D tested fraction count changed")
    fraction = tested[0]
    endpoints = fraction.get("endpoints", {})
    _require(
        set(fraction) == {
            "all_12_familywise_lcbs_nonnegative",
            "all_12_point_deltas_nonnegative", "endpoints", "fraction",
            "fraction_compact_estimator_sha256", "pass",
        }
        and fraction.get("fraction") == 0.05
        and fraction.get("pass") is False
        and fraction.get("all_12_familywise_lcbs_nonnegative") is False
        and fraction.get("all_12_point_deltas_nonnegative") is False
        and fraction.get("fraction_compact_estimator_sha256")
        == "aca47da56eca4302791d2e25cde8493a7f9890b362bcd0a132cc9c8b64a74a43"
        and endpoints == _expected_fraction_endpoints_v34d()
        and len(endpoints) == 12
        and all(item["familywise_lcb"] < 0.0 for item in endpoints.values()),
        "V34D exact 5-percent metrics or familywise LCB failures changed",
    )
    return analysis


def _validate_summary_v34d(report):
    summary = report.get("summary", {})
    _verify_self(summary, SUMMARY_CONTENT_SHA256_V34D, "summary")
    _require(
        set(summary) == {
            "schema", "contains_dataset_rows_questions_answers_document_or_eval_content",
            "contains_unit_scores_response_vectors_coefficients_bootstrap_draws_or_replicates",
            "content_sha256_before_self_field", "fixed_sequence_analysis",
            "frame_content_sha256", "preregistration_content_sha256",
            "runtime_integrity",
        }
        and summary.get("schema")
        == "eggroll-es-v401-replacement-fraction-compact-summary-v34b"
        and summary.get(
            "contains_dataset_rows_questions_answers_document_or_eval_content"
        ) is False
        and summary.get(
            "contains_unit_scores_response_vectors_coefficients_bootstrap_draws_or_replicates"
        ) is False
        and summary.get("frame_content_sha256")
        == PANEL_IDENTITIES_V34D["content_free_frame"]["content_sha256"]
        and summary.get("preregistration_content_sha256")
        == PREREGISTRATION_IDENTITY_V34D["content_sha256"],
        "V34D compact summary scope or frozen identities changed",
    )
    expected_integrity = {
        "all_four_tp1_engines_every_signed_wave": True,
        "all_integrity_audits_passed": True,
        "all_thirty_two_signed_waves_complete": True,
        "base_layer_and_unselected_origin_audits_passed": True,
        "both_sources_every_direction_and_sign": True,
        "counterbalanced_source_order_complete": True,
        "exact_reference_restored_after_each_signed_wave": True,
        "failure_cleanup_and_final_all_gpu_idle_passed": True,
        "fresh_exclusive_paths_and_committed_clean_source_passed": True,
        "population_boundary_selected_and_unselected_audits_passed": True,
        "pre_post_full_context_reference_probes_equal": True,
        "same_resident_perturbation_both_sources": True,
        "source_and_preregistration_hashes_rechecked": True,
    }
    _require(
        summary.get("runtime_integrity") == expected_integrity,
        "V34D activity, cleanup, origin, or source integrity changed",
    )
    _validate_fixed_sequence_v34d(summary)
    return summary


def _validate_closed_authority_v34d(attempt, report):
    gate = report.get("gate", {})
    _verify_self(gate, GATE_CONTENT_SHA256_V34D, "authorization gate")
    _require(
        set(gate) == {
            "schema", "checkpoint_write_authorized",
            "content_sha256_before_self_field", "decision",
            "direct_dataset_promotion_authorized",
            "largest_consecutively_passing_fraction", "model_update_authorized",
            "stopped_at_first_failure", "tested_fraction_count",
            "validation_heldout_ood_or_benchmark_evaluation_authorized",
        }
        and gate.get("schema")
        == "eggroll-es-v401-replacement-fraction-gate-v34b"
        and gate.get("decision") == "retain_production_no_fraction_authorized"
        and gate.get("largest_consecutively_passing_fraction") == 0.0
        and gate.get("stopped_at_first_failure") is True
        and gate.get("tested_fraction_count") == 1
        and all(gate.get(key) is False for key in (
            "checkpoint_write_authorized", "direct_dataset_promotion_authorized",
            "model_update_authorized",
            "validation_heldout_ood_or_benchmark_evaluation_authorized",
        )),
        "V34D closed authorization gate changed",
    )
    side_effect_keys = (
        "checkpoint_written", "dataset_promotion_applied", "evaluation_opened",
        "model_update_applied", "nontrain_surface_opened",
    )
    _require(
        all(attempt.get(key) is False for key in side_effect_keys)
        and all(report.get(key) is False for key in side_effect_keys)
        and report.get("direct_action_taken") is False
        and report.get("all_four_gpus_idle_after_cleanup") is True
        and attempt.get("all_four_gpus_idle_after_cleanup") is True,
        "V34D forbidden side effect, direct action, or cleanup state changed",
    )
    return gate


def validate_bound_artifacts_v34d():
    _require(
        file_sha256(ATTEMPT_PATH_V34D) == ATTEMPT_FILE_SHA256_V34D,
        "V34D attempt file hash changed",
    )
    _require(
        file_sha256(REPORT_PATH_V34D) == REPORT_FILE_SHA256_V34D,
        "V34D report file hash changed",
    )
    attempt = _load_json_object(ATTEMPT_PATH_V34D, "attempt")
    report = _load_json_object(REPORT_PATH_V34D, "report")
    _require(set(report) == REPORT_KEYS_V34D, "V34D report schema keys changed")
    _verify_self(report, REPORT_CONTENT_SHA256_V34D, "report")
    _require(
        report.get("schema")
        == "eggroll-es-v401-replacement-fraction-report-v34c"
        and report.get("status") == "completed_train_only_alpha_zero_no_update"
        and report.get("postcleanup_binding_recheck_sha256")
        == POSTCLEANUP_BINDING_RECHECK_SHA256_V34D,
        "V34D report schema, completion, or postcleanup binding changed",
    )
    _validate_attempt_v34d(attempt, report)
    _validate_configuration_v34d(report)
    _validate_preanalysis_v34d(report)
    _validate_summary_v34d(report)
    _validate_closed_authority_v34d(attempt, report)
    return attempt, report


def build_negative_evidence_v34d():
    attempt, report = validate_bound_artifacts_v34d()
    configuration = report["configuration"]
    audit = report["preanalysis_runtime_audit"]
    summary = report["summary"]
    analysis = summary["fixed_sequence_analysis"]
    fraction = analysis["tested_fractions"][0]
    guard = audit["full_context_guard"]
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-negative-evidence-v34d",
        "status": "sealed_completed_compact_negative_evidence",
        "artifacts": {
            "durable_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V34D,
                "file_sha256": ATTEMPT_FILE_SHA256_V34D,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V34D,
            },
            "compact_report": {
                "relative_path": REPORT_RELATIVE_PATH_V34D,
                "file_sha256": REPORT_FILE_SHA256_V34D,
                "content_sha256": REPORT_CONTENT_SHA256_V34D,
            },
        },
        "frozen_bindings": {
            "committed_source_certificate_sha256": attempt[
                "committed_source_certificate_sha256"
            ],
            "implementation_bundle_sha256": attempt[
                "implementation_bundle_sha256"
            ],
            "recipe_content_sha256": attempt["recipe_sha256"],
            "runtime_environment_certificate_sha256": attempt[
                "runtime_environment_certificate_sha256"
            ],
            "live_model_audit_sha256": attempt["live_model_audit_sha256"],
            "prelaunch_idle_certificate_sha256": attempt[
                "prelaunch_idle_certificate_sha256"
            ],
            "final_idle_certificate_sha256": attempt[
                "final_idle_certificate_sha256"
            ],
            "postcleanup_binding_recheck_sha256": report[
                "postcleanup_binding_recheck_sha256"
            ],
            "configuration_content_sha256": CONFIGURATION_CONTENT_SHA256_V34D,
            "preanalysis_audit_content_sha256": (
                PREANALYSIS_AUDIT_CONTENT_SHA256_V34D
            ),
            "summary_content_sha256": SUMMARY_CONTENT_SHA256_V34D,
            "fixed_sequence_content_sha256": FIXED_SEQUENCE_CONTENT_SHA256_V34D,
            "authorization_gate_content_sha256": GATE_CONTENT_SHA256_V34D,
        },
        "transitively_frozen_source_contracts": {
            "binding_basis": {
                "committed_source_certificate_sha256": (
                    COMMITTED_SOURCE_CERTIFICATE_SHA256_V34D
                ),
                "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V34D,
                "recipe_content_sha256": RECIPE_CONTENT_SHA256_V34D,
                "postcleanup_binding_recheck_sha256": (
                    POSTCLEANUP_BINDING_RECHECK_SHA256_V34D
                ),
            },
            "v34b_commit": V34B_COMMIT_V34D,
            "v34b_source_files": copy.deepcopy(V34B_SOURCE_FILES_V34D),
            "v34c_commit": V34C_COMMIT_V34D,
            "v34c_source_files": copy.deepcopy(V34C_SOURCE_FILES_V34D),
            "preregistration": copy.deepcopy(PREREGISTRATION_IDENTITY_V34D),
            "panels": copy.deepcopy(PANEL_IDENTITIES_V34D),
            "candidate_manifest": copy.deepcopy(CANDIDATE_MANIFEST_IDENTITY_V34D),
            "production": copy.deepcopy(PRODUCTION_IDENTITY_V34D),
            "model_and_layer_plan": copy.deepcopy(MODEL_AND_LAYER_IDENTITIES_V34D),
        },
        "aggregate_execution": {
            "synchronized_signed_wave_count": audit[
                "synchronized_signed_wave_count"
            ],
            "physical_gpu_count": 4,
            "all_four_tp1_engines_both_sources_every_wave": True,
            "perturbed_request_count": audit["perturbed_request_count"],
            "full_context_request_count": audit["full_context_request_count"],
            "fraction_specific_request_count": audit[
                "fraction_specific_request_count"
            ],
            "total_generation_request_count": audit[
                "total_generation_request_count"
            ],
            "configuration": {
                "content_sha256": CONFIGURATION_CONTENT_SHA256_V34D,
                "device_identity_sha256": configuration[
                    "device_identity_sha256"
                ],
                "installation_sha256": configuration["installation_sha256"],
                "selected_reference_identity_sha256": configuration[
                    "selected_reference_identity_sha256"
                ],
                "alpha_zero": True,
            },
            "activity_and_restore_sha256": audit[
                "wave_activity_and_restore_sha256"
            ],
            "all_runtime_integrity_audits_passed": True,
            "origin_and_population_boundary_audits_passed": True,
            "a_b_exact": copy.deepcopy(guard["a_b_exact"]),
            "a_c_exact": copy.deepcopy(guard["a_c_exact"]),
            "phase_a_b_c_commitments_identical": True,
            "full_context_guard_excluded_from_fraction_analysis": True,
            "failure_cleanup_and_final_all_gpu_idle_passed": True,
        },
        "fixed_sequence_result": {
            "production_fraction": 0.0,
            "tested_fraction_count": 1,
            "tested_fraction": fraction["fraction"],
            "tested_fraction_passed": fraction["pass"],
            "endpoint_count": len(fraction["endpoints"]),
            "familywise_lcb_failure_count": 12,
            "all_12_familywise_lcbs_failed": True,
            "all_12_point_deltas_nonnegative": False,
            "exact_endpoints": copy.deepcopy(fraction["endpoints"]),
            "fraction_compact_estimator_sha256": fraction[
                "fraction_compact_estimator_sha256"
            ],
            "stopped_at_first_failure": True,
            "largest_consecutively_passing_fraction": 0.0,
            "untested_fractions_after_first_failure": copy.deepcopy(
                analysis["untested_fractions_after_first_failure"]
            ),
            "no_higher_fraction_result_inferred": True,
            "fraction_specific_model_request_count": 0,
            "bootstrap": copy.deepcopy(analysis["bootstrap"]),
        },
        "decision": {
            "retain_production_at_fraction_0_0": True,
            "replacement_fraction_adoption_authority": False,
            "model_update_authority": False,
            "checkpoint_write_authority": False,
            "validation_heldout_ood_or_benchmark_evaluation_authority": False,
            "dataset_promotion_authority": False,
            "nontrain_reuse_authority": False,
        },
        "side_effects": {
            "direct_action_taken": False,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
            "nontrain_surface_opened": False,
        },
        "input_scope": {
            "compact_attempt_and_report_aggregates_only": True,
            "source_contract_constants_only": True,
            "panel_candidate_manifest_or_dataset_content_opened": False,
            "raw_stdout_or_runtime_logs_opened": False,
            "evaluation_validation_heldout_ood_or_benchmark_content_opened": False,
            "detailed_scores_responses_coefficients_or_bootstrap_draws_persisted": False,
            "gpu_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    _assert_compact_v34d(value)
    return value


def _exclusive_write_json_v34d(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V34D.resolve():
        raise ValueError("V34D evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V34D evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_negative_evidence_v34d()
    if not args.dry_run:
        _exclusive_write_json_v34d(OUTPUT_PATH_V34D, value)
    print(json.dumps({
        "schema": "eggroll-es-v401-replacement-fraction-negative-evidence-build-v34d",
        "content_sha256": value["content_sha256_before_self_field"],
        "retain_production_at_fraction_0_0": True,
        "tested_fraction": 0.05,
        "tested_fraction_passed": False,
        "higher_fractions_inferred": False,
        "direct_action_authorized": False,
        "gpu_launched": False,
    }, indent=2, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
