#!/usr/bin/env python3
"""Seal the additive V73B same-live-reward V71/V72 calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import build_lora_es_v71_v72_live_calibration_preregistration_v73 as v73
import run_lora_es_mirrored_calibration_v66 as v66


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_same_live_calibration_v73b.json"
).resolve()
RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v73b_lora_es_same_live_qwen36_calibration"
).resolve()
FAILED_V73_RUN = v73.RUN

FAILED_V73_FILES = {
    "failure": FAILED_V73_RUN / "failure_v73.json",
    "actor_cuda_work_log": (
        FAILED_V73_RUN / "actor_cuda_work_receipts_v73.jsonl"
    ),
    "host_process_summary": FAILED_V73_RUN / "host_process_summary_v73.json",
    "gpu_log": FAILED_V73_RUN / "gpu_activity_v73.jsonl",
}


def artifacts_v73b() -> dict[str, str]:
    stem = "v73b_lora_es_same_live_qwen36_calibration"
    return {
        "attempt": str(RUN.parent / f".{stem}.attempt.json"),
        "run_directory": str(RUN),
        "gpu_log": str(RUN / "gpu_activity_v73b.jsonl"),
        "actor_cuda_work_log": str(
            RUN / "actor_cuda_work_receipts_v73b.jsonl"
        ),
        "host_process_samples": str(RUN / "host_process_samples_v73b.jsonl"),
        "host_process_summary": str(RUN / "host_process_summary_v73b.json"),
        "population": str(RUN / "mirrored_population_v73b.json"),
        "update": str(RUN / "pair_difference_update_v73b.json"),
        "audit_traffic": str(RUN / "exact_audit_traffic_v73b.json"),
        "equivalence": str(RUN / "same_live_equivalence_v73b.json"),
        "report": str(RUN / "mirrored_calibration_report_v73b.json"),
        "failure": str(RUN / "failure_v73b.json"),
    }


def failed_v73_observation() -> dict:
    paths = {key: path.resolve() for key, path in FAILED_V73_FILES.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"failed V73 {name} evidence is absent")
    failure = json.loads(paths["failure"].read_text(encoding="ascii"))
    if (
        failure.get("schema")
        != "v73-v71-v72-qwen36-calibration-failure"
        or failure.get("type") != "RuntimeError"
        or failure.get("message")
        != "v73 population differs from accepted V66d: "
        "['signed_rewards', 'signed_reward_sha256']"
        or failure.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or failure.get("cleanup_failure") is not None
        or failure.get("checkpoint_snapshot_or_promotion_performed") is not False
        or failure.get("protected_dev_ood_or_holdout_opened") is not False
    ):
        raise RuntimeError("failed V73 observation semantics changed")
    actor_rows = [
        json.loads(line)
        for line in paths["actor_cuda_work_log"].read_text(
            encoding="ascii"
        ).splitlines()
        if line
    ]
    if (
        len(actor_rows) != 16
        or len({row.get("work_id") for row in actor_rows}) != 16
        or {row.get("physical_gpu_id") for row in actor_rows}
        != {0, 1, 2, 3}
        or any(
            row.get("output_cardinality", {}).get("request_outputs") != 64
            or row.get("output_cardinality", {}).get("samples") != 64
            for row in actor_rows
        )
    ):
        raise RuntimeError("failed V73 actor work coverage changed")
    return {
        "failure": {
            "path": str(paths["failure"]),
            "file_sha256": v66.file_sha256_v66(paths["failure"]),
        },
        "actor_cuda_work_log": {
            "path": str(paths["actor_cuda_work_log"]),
            "file_sha256": v66.file_sha256_v66(paths["actor_cuda_work_log"]),
            "rows": 16,
            "physical_gpu_ids": [0, 1, 2, 3],
        },
        "host_process_summary": {
            "path": str(paths["host_process_summary"]),
            "file_sha256": v66.file_sha256_v66(paths["host_process_summary"]),
        },
        "gpu_log": {
            "path": str(paths["gpu_log"]),
            "file_sha256": v66.file_sha256_v66(paths["gpu_log"]),
        },
        "observed_population_difference_fields": [
            "signed_rewards", "signed_reward_sha256",
        ],
        "all_other_v73_population_equivalence_fields_exact": True,
        "numeric_live_reward_values_recoverable_from_failed_artifacts": False,
        "reason_values_absent": (
            "the failed population writer rejected before persistence and raw "
            "questions, answers, and outputs are intentionally never persisted"
        ),
        "cleanup_idle": True,
    }


def build_preregistration_v73b() -> dict:
    result = v73.build_preregistration_v73()
    result.pop("content_sha256_before_self_field", None)
    result["schema"] = (
        "lora-es-v71-v72-qwen36-same-live-calibration-"
        "preregistration-v73b"
    )
    result["status"] = (
        "sealed_cpu_only_after_failed_v73_before_v73b_train_model_ray_gpu_"
        "or_protected_access"
    )
    result["purpose"] = (
        "Replace V73's invalid cross-run floating reward bit-equality gate "
        "with exact same-live-vector equivalence between the canonical V66 "
        "compiler and an independent one-pass compiler, followed by the V72 "
        "distributed executor, while retaining every candidate, audit, "
        "acceptance, abort, telemetry, and cleanup fail-closed boundary."
    )
    result["artifacts"] = artifacts_v73b()
    result["failed_v73_observation"] = failed_v73_observation()
    result["fixed_recipe"]["integration_v73b"] = {
        "dependency_order": [
            "accepted_v66d_control_and_failed_v73_identity",
            "four_exclusive_gpu_preflight",
            "v72_install_and_one_bank_receipt",
            "exact_candidate_runtime_restore_and_work_metadata_equivalence",
            "finite_complete_live_reward_matrix_with_exact_assignment_metadata",
            "v71_candidate_exact_audits_before_reward_use",
            "rank_local_population_reward_acceptance",
            "canonical_v66_and_independent_one_pass_compilers_same_live_vector",
            "exact_compiler_result_and_coefficient_digest_equivalence",
            "rank_local_update_acceptance_and_v72_execute",
            "four_actor_live_candidate_and_runtime_identity_consensus",
            "exact_four_actor_abort_to_original_master",
            "audit_traffic_and_quiescent_one_bank_receipts",
            "host_telemetry_flush_before_actor_cleanup",
            "four_gpu_cleanup_idle_certificate",
        ],
        "reward_acceptance_surface": {
            "historical_reward_floats": "diagnostic_only_not_an_acceptance_gate",
            "historical_reward_digest": "diagnostic_only_not_an_acceptance_gate",
            "assignment_metadata": "exact_to_accepted_v66d",
            "matrix_coverage": "exactly_one_finite_reward_per_pair_and_sign",
            "no_post_hoc_numeric_tolerance": True,
        },
        "update_equivalence_surface": {
            "input": "the_identical_live_reward_object",
            "canonical_compiler": "eggroll_es_mirrored_v66.pair_difference_update_v66",
            "independent_compiler": "one_pass_pair_table_v73b",
            "compiler_output": "whole_mapping_exact",
            "distributed_executor": "v72_all_four_actor_exact_identity_consensus",
            "historical_update_identity": "diagnostic_only",
        },
        "commit_or_checkpoint_authorized": False,
        "population_acceptance_tokens_are_rank_local": True,
        "update_acceptance_tokens_are_rank_local": True,
        "unknown_or_partial_rpc": "exact_abort_or_terminal_poison",
    }
    # Keep the inherited key because the immutable V73 integration context
    # reads only its learning-rate-independent telemetry fields.  Override its
    # obsolete semantic claim so no consumer can infer historical bit equality.
    result["fixed_recipe"]["integration_v73"][
        "candidate_reward_update_equivalence"
    ] = "superseded_by_integration_v73b_same_live_vector"
    result["acceptance"].update({
        "signed_rewards_bit_exact_to_accepted_v66d": False,
        "historical_reward_values_are_diagnostic_only": True,
        "live_reward_assignment_metadata_exact_to_accepted_v66d": True,
        "live_reward_matrix_complete_finite_and_unique": True,
        "canonical_and_independent_compilers_receive_same_live_object": True,
        "canonical_and_independent_compiler_outputs_whole_mapping_exact": True,
        "four_actor_live_candidate_and_runtime_identity_consensus": True,
        "historical_candidate_update_identity_required": False,
        "same_live_candidate_is_nonzero_and_differs_from_master": True,
    })
    result["beads"] = [
        "specialist-0j5.29", "specialist-0j5.19",
        "specialist-0j5.21", "specialist-0j5.20",
    ]
    bindings = result["implementation_bindings"]
    bindings["v73_failed_runner"] = bindings.pop("runner")
    bindings["v73_failed_builder"] = bindings.pop("builder")
    paths = {
        "builder": Path(__file__).resolve(),
        "runner": ROOT / "run_lora_es_v71_v72_same_live_calibration_v73b.py",
        "v73_adapter": ROOT / "run_lora_es_v71_v72_live_calibration_v73.py",
    }
    bindings.update({
        name: {
            "path": str(path.resolve()),
            "file_sha256": v66.file_sha256_v66(path),
        }
        for name, path in paths.items()
    })
    result["content_sha256_before_self_field"] = (
        v66.mirrored.canonical_sha256_v66(result)
    )
    return result


def write_preregistration_v73b(path=OUTPUT) -> dict:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    value = build_preregistration_v73b()
    payload = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2,
                   sort_keys=True) + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    value = build_preregistration_v73b()
    if args.check:
        if (
            not output.is_file()
            or json.loads(output.read_text(encoding="ascii")) != value
        ):
            raise RuntimeError("v73b preregistration is absent or stale")
    else:
        write_preregistration_v73b(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": v66.file_sha256_v66(output) if output.is_file() else None,
        "content_sha256": value["content_sha256_before_self_field"],
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
