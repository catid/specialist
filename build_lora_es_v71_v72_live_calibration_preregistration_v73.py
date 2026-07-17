#!/usr/bin/env python3
"""Seal the V71 exact-audit plus V72 host-state four-GPU calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import build_lora_es_mirrored_calibration_preregistration_v66d as v66d
import run_lora_es_mirrored_calibration_v66 as v66


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_live_calibration_v73.json"
).resolve()
CONTROL_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66d_lora_es_mirrored_crn_qwen36_calibration"
).resolve()
RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v73_lora_es_exact_host_qwen36_calibration"
).resolve()

CONTROL_FILES_V73 = {
    "actor_cuda_work_log": {
        "path": CONTROL_RUN / "actor_cuda_work_receipts_v66d.jsonl",
        "file_sha256": (
            "aa10617c347b7ce5449165580dd4eaa98bb5131cfde5fcf9cda1134b380390e0"
        ),
    },
    "gpu_log": {
        "path": CONTROL_RUN / "gpu_activity_v66d.jsonl",
        "file_sha256": (
            "a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475"
        ),
    },
    "population": {
        "path": CONTROL_RUN / "mirrored_population_v66d.json",
        "file_sha256": (
            "9d172d15f82a54c697b8b860ff3131733d59006f1e4b790b5b9b87ded679e9d4"
        ),
        "content_sha256": (
            "64fafe006fc6ffaf333021075452d6b27514dbed6f36188b60fe27b59045dc58"
        ),
    },
    "update": {
        "path": CONTROL_RUN / "pair_difference_update_v66d.json",
        "file_sha256": (
            "f958f90b26c5b2afa4a81b03a0ab91c12d9684c2ce236bbb658d674e7a5eeffd"
        ),
        "content_sha256": (
            "7d16e2be89fffb2686f8a04a96417d10852efd1fec5ca488c19be9f9eb025aef"
        ),
    },
    "report": {
        "path": CONTROL_RUN / "mirrored_calibration_report_v66d.json",
        "file_sha256": (
            "12a5e854856d28bd8439cf3ed004664317086f8d117ae08e78b59f857f6102bb"
        ),
        "content_sha256": (
            "87d1eca139ee0b766b15517c81459becd0369c9d5f7ffb78269fdfce977de684"
        ),
    },
}


def artifacts_v73():
    stem = "v73_lora_es_exact_host_qwen36_calibration"
    return {
        "attempt": str(RUN.parent / f".{stem}.attempt.json"),
        "run_directory": str(RUN),
        "gpu_log": str(RUN / "gpu_activity_v73.jsonl"),
        "actor_cuda_work_log": str(
            RUN / "actor_cuda_work_receipts_v73.jsonl"
        ),
        "host_process_samples": str(RUN / "host_process_samples_v73.jsonl"),
        "host_process_summary": str(RUN / "host_process_summary_v73.json"),
        "population": str(RUN / "mirrored_population_v73.json"),
        "update": str(RUN / "pair_difference_update_v73.json"),
        "audit_traffic": str(RUN / "exact_audit_traffic_v73.json"),
        "equivalence": str(RUN / "v66d_equivalence_v73.json"),
        "report": str(RUN / "mirrored_calibration_report_v73.json"),
        "failure": str(RUN / "failure_v73.json"),
    }


def accepted_v66d_control_v73():
    result = {}
    loaded = {}
    for name, specification in CONTROL_FILES_V73.items():
        path = Path(specification["path"]).resolve()
        if (
            not path.is_file()
            or v66.file_sha256_v66(path) != specification["file_sha256"]
        ):
            raise RuntimeError(f"accepted V66d {name} artifact changed")
        receipt = {
            "path": str(path),
            "file_sha256": specification["file_sha256"],
        }
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="ascii"))
            content_sha = value.get("content_sha256_before_self_field")
            if content_sha != specification.get("content_sha256"):
                raise RuntimeError(f"accepted V66d {name} content changed")
            compact = {
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }
            if v66.mirrored.canonical_sha256_v66(compact) != content_sha:
                raise RuntimeError(f"accepted V66d {name} self hash changed")
            receipt["content_sha256"] = content_sha
            loaded[name] = value
        result[name] = receipt
    population = loaded["population"]
    update = loaded["update"]
    report = loaded["report"]
    actor_rows = [
        json.loads(line)
        for line in Path(
            CONTROL_FILES_V73["actor_cuda_work_log"]["path"]
        ).read_text(encoding="ascii").splitlines()
        if line
    ]
    if (
        population.get("schema") != "v66-mirrored-qwen36-population-evidence"
        or len(population.get("signed_rewards", [])) != 16
        or len(population.get("materializations", [])) != 16
        or len(population.get("restorations", [])) != 16
        or update.get("schema")
        != "v66-nonzero-qwen36-pair-difference-update-receipt"
        or update.get("nonzero_pair_differences") != 8
        or update.get("master_committed") is not False
        or update.get("all_four_abort_receipts_exact") is not True
        or report.get("schema")
        != "v66d-mirrored-crn-qwen36-calibration-report"
        or report.get("status")
        != "complete_nonzero_train_only_no_commit_actor_attributed"
        or report.get("protected_dev_ood_or_holdout_opened") is not False
        or report.get("checkpoint_snapshot_or_promotion_performed") is not False
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or len(actor_rows) != 16
        or len({row.get("work_id") for row in actor_rows}) != 16
    ):
        raise RuntimeError("accepted V66d calibration semantics changed")
    result["semantic_anchor"] = {
        "plan_sha256": population["plan_sha256"],
        "signed_reward_sha256": population["signed_reward_sha256"],
        "candidate_master_sha256": update["candidate_master_sha256"],
        "candidate_runtime_values_sha256": update[
            "candidate_runtime_values_sha256"
        ],
        "coefficient_sha256": update["coefficient_sha256"],
        "manifest_sha256": update["manifest_sha256"],
        "final_master_sha256": report["final_master_sha256"],
        "final_runtime_values_sha256": report[
            "final_runtime_values_sha256"
        ],
        "signed_candidate_count": 16,
        "candidate_count_per_actor": 4,
        "actor_work_id_sha256": v66.mirrored.canonical_sha256_v66(
            [row["work_id"] for row in actor_rows]
        ),
    }
    return result


def build_preregistration_v73():
    result = v66d.build_preregistration_v66d()
    result.pop("content_sha256_before_self_field", None)
    result["schema"] = (
        "lora-es-v71-v72-qwen36-live-calibration-preregistration-v73"
    )
    result["status"] = (
        "sealed_cpu_only_before_v73_train_model_ray_gpu_or_protected_access"
    )
    result["purpose"] = (
        "Dependency-ordered live confirmation of V71 exact-audit traffic and "
        "V72 one-master host ownership against the immutable accepted V66d "
        "candidate, reward, update, actor/GPU, and cleanup control."
    )
    result["accepted_v66d_control"] = accepted_v66d_control_v73()
    result["artifacts"] = artifacts_v73()
    result["fixed_recipe"]["integration_v73"] = {
        "dependency_order": [
            "accepted_v66d_control_identity",
            "four_exclusive_gpu_preflight",
            "v72_install_and_one_bank_receipt",
            "v71_candidate_exact_audits_before_reward_use",
            "rank_local_population_reward_acceptance",
            "rank_local_update_acceptance_and_v72_execute",
            "candidate_update_equivalence_before_abort",
            "exact_four_actor_abort_to_original_master",
            "audit_traffic_and_quiescent_one_bank_receipts",
            "host_telemetry_flush_before_actor_cleanup",
            "four_gpu_cleanup_idle_certificate",
        ],
        "candidate_reward_update_equivalence": "exact",
        "commit_or_checkpoint_authorized": False,
        "population_acceptance_tokens_are_rank_local": True,
        "update_acceptance_tokens_are_rank_local": True,
        "unknown_or_partial_rpc": "exact_abort_or_terminal_poison",
    }
    result["fixed_recipe"]["host_telemetry_v73"] = {
        "background_sample_interval_seconds": 0.5,
        "proc_status_fields": [
            "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem",
            "VmLck", "VmPin", "Threads",
        ],
        "process_stat_fields": ["minor_faults", "major_faults"],
        "numa_maps_at_phase_ack_and_named_boundaries": True,
        "numa_page_nodes_recorded": True,
        "actor_rank_pid_physical_gpu_binding_required": True,
        "host_monitor_stops_and_fsyncs_before_actor_cleanup": True,
    }
    result["runtime"].update({
        "worker_extension": (
            "eggroll_es_worker_lora_v72."
            "LoRAAdapterStateWorkerExtensionV72"
        ),
        "required_worker_endpoints": [
            "install_adapter_state_v41a",
            "capture_adapter_reference_v41a",
            "active_lora_slot_certificate_v66",
            "materialize_mirrored_adapter_v66",
            "begin_actor_gpu_work_v66d",
            "end_actor_gpu_work_v66d",
            "restore_mirrored_adapter_v66",
            "candidate_audit_matrix_v71",
            "accept_population_rewards_v71",
            "prepare_sharded_adapter_update_v71",
            "execute_sharded_adapter_update_v41a",
            "abort_mirrored_update_if_present_v66",
            "audit_traffic_receipt_v71",
            "host_state_residency_v72",
            "mirrored_adapter_state_certificate_v66",
        ],
    })
    result["acceptance"].update({
        "candidate_identities_equal_accepted_v66d": True,
        "signed_rewards_bit_exact_to_accepted_v66d": True,
        "update_coefficients_and_candidate_equal_accepted_v66d": True,
        "four_candidate_audits_per_actor_before_reward_acceptance": True,
        "population_and_update_acceptance_tokens_rank_local": True,
        "audit_transfer_bytes_and_calls_recorded_per_actor": True,
        "host_rss_hwm_locked_numa_and_faults_recorded_per_actor": True,
        "phase_and_worker_operation_times_recorded": True,
        "post_install_and_post_abort_unique_host_bank_count": 1,
        "executed_update_unique_host_bank_count": 2,
        "all_actor_cuda_and_gpu_wave_receipts_required": True,
        "cleanup_and_final_four_gpu_idle_required": True,
    })
    result["beads"] = [
        "specialist-0j5.19", "specialist-0j5.21", "specialist-0j5.20",
    ]
    bindings = result["implementation_bindings"]
    bindings["accepted_v66d_builder"] = bindings.pop("builder")
    bindings["accepted_v66d_runner"] = bindings.pop("runner")
    paths = {
        "builder": Path(__file__).resolve(),
        "runner": ROOT / "run_lora_es_v71_v72_live_calibration_v73.py",
        "v71_worker": ROOT / "eggroll_es_worker_lora_v71.py",
        "v71_contract": ROOT / "eggroll_es_audit_contract_v71.py",
        "v72_worker": ROOT / "eggroll_es_worker_lora_v72.py",
        "v72_contract": ROOT / "eggroll_es_host_state_contract_v72.py",
        "v66d_runner": ROOT / "run_lora_es_mirrored_calibration_v66d.py",
        "v66d_worker": ROOT / "eggroll_es_worker_lora_v66d.py",
        "gpu_telemetry": ROOT / "eggroll_es_gpu_telemetry_v66.py",
    }
    bindings.update({
        key: {
            "path": str(path.resolve()),
            "file_sha256": v66.file_sha256_v66(path),
        }
        for key, path in paths.items()
    })
    result["content_sha256_before_self_field"] = (
        v66.mirrored.canonical_sha256_v66(result)
    )
    return result


def write_preregistration_v73(path=OUTPUT):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    value = build_preregistration_v73()
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    value = build_preregistration_v73()
    if args.check:
        if (
            not output.is_file()
            or json.loads(output.read_text(encoding="ascii")) != value
        ):
            raise RuntimeError("v73 preregistration is absent or stale")
    else:
        write_preregistration_v73(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": (
            v66.file_sha256_v66(output) if output.is_file() else None
        ),
        "content_sha256": value["content_sha256_before_self_field"],
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
