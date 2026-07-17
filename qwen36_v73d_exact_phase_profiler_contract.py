#!/usr/bin/env python3
"""Immutable runtime-only contract surface for the V73D systems trace.

This module deliberately does not import the preregistration builder.  The two
runtime roots depend only on this surface and the generated, self-hashed JSON
contract, which avoids a circular closure receipt when the builder later binds
the runtime AST closure.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73d_exact_phase_profiler.json"
).resolve()
V73B_PREREGISTRATION = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_same_live_calibration_v73b.json"
).resolve()
V73_PREREGISTRATION = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_live_calibration_v73.json"
).resolve()
V73_PREREGISTRATION_FILE_SHA256 = (
    "320b038f07b615622cab0a2a5a9aec86aa06d7649e794201794f382d8ab3783e"
)
V73_PREREGISTRATION_CONTENT_SHA256 = (
    "8bea32f21d33970f1b234cfe59ebaa3eb60fcd47faca1aff1913d81f5dcfe08c"
)
V73B_POSTRUN = (
    ROOT
    / "experiments/eggroll_es_hpo/"
    "qwen36_v73b_lora_calibration_postrun_20260717.json"
).resolve()
V73B_RUN = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    "v73b_lora_es_same_live_qwen36_calibration"
).resolve()

BASH = Path("/usr/bin/bash")
BASHRC = (Path.home() / ".bashrc").resolve()
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
ENV = Path("/usr/bin/env")
NSYS = Path("/usr/local/cuda/bin/nsys")
REQUIRED_PYTHON = (ROOT / "es-at-scale/.venv/bin/python").absolute()
TARGET = (
    ROOT / "run_lora_es_v71_v72_profile_calibration_v73d.py"
).resolve()
LAUNCHER = (ROOT / "run_qwen36_v73d_exact_phase_profiler.py").resolve()
GUARD_DIRECTORY = (ROOT / "v73d_sitecustomize").resolve()
GUARD = (GUARD_DIRECTORY / "v73d_path_open_guard.py").resolve()
SITECUSTOMIZE = (GUARD_DIRECTORY / "sitecustomize.py").resolve()
WORKER = (ROOT / "eggroll_es_worker_lora_v73d.py").resolve()
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_v73d.LoRAAdapterStateWorkerExtensionV73D"
)
ACTOR_BOOTSTRAP_ENV = "SPECIALIST_V73D_ACTOR_BOOTSTRAP"
ACTOR_GUARD_SHA_ENV = "SPECIALIST_V73D_ACTOR_GUARD_SHA256"
BOUNDARY_REGISTRY = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "quarantine_boundary_registry_v3.json"
).resolve()
BOUNDARY_REGISTRY_FILE_SHA256 = (
    "3d8ef097a1419e03f4b735e6f8d30e5a876b0a8e86c4b0f1ac100114cb7daf5d"
)
BOUNDARY_REGISTRY_CONTENT_SHA256 = (
    "5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7"
)
HISTORICAL_REFERENCE_MODULE_IDENTITY_SET_SHA256 = (
    "4d5779609c025911357f7b7576d357d409fb8460fc0b6d5151ff8d729b30444c"
)

PREREG_FILE_PLACEHOLDER = "<PREREGISTRATION_FILE_SHA256>"
PREREG_CONTENT_PLACEHOLDER = "<PREREGISTRATION_CONTENT_SHA256>"
PHASE_DOMAIN = "eggroll_es_v73d_phase"
PHASES = (
    "setup",
    "activate_v434_lora_slot_all_actors",
    "install_canonical_v434_master_all_actors",
    "mirrored_wave_0_materialize_all_actors",
    "mirrored_wave_0_generation_all_actors",
    "wave_0_finalize_restore_all_actors",
    "mirrored_wave_1_materialize_all_actors",
    "mirrored_wave_1_generation_all_actors",
    "wave_1_finalize_restore_all_actors",
    "mirrored_wave_2_materialize_all_actors",
    "mirrored_wave_2_generation_all_actors",
    "wave_2_finalize_restore_all_actors",
    "mirrored_wave_3_materialize_all_actors",
    "mirrored_wave_3_generation_all_actors",
    "wave_3_finalize_restore_all_actors",
    "candidate_audit_matrix_all_actors",
    "population_reward_acceptance_all_actors",
    "pair_difference_update_accept_prepare_all_actors",
    "pair_difference_update_execute_all_actors",
    "pair_difference_update_abort_all_actors",
    "post_abort_final_audit_all_actors",
    "cleanup_all_actors",
)

BASHRC_EXEC_SCRIPT = (
    'source "$HOME/.bashrc" >/dev/null 2>&1 && exec "$@"'
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {"path": str(path), "file_sha256": file_sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _version(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return completed.stdout.strip()


def gpu_identity_v73d() -> list[dict[str, Any]]:
    output = _version([
        str(NVIDIA_SMI),
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 3, "V73D GPU identity query shape changed")
        rows.append({
            "physical_gpu_id": int(fields[0]),
            "uuid": fields[1],
            "pci_bus_id": fields[2],
        })
    _require(
        [row["physical_gpu_id"] for row in rows] == [0, 1, 2, 3]
        and len({row["uuid"] for row in rows}) == 4
        and len({row["pci_bus_id"] for row in rows}) == 4,
        "V73D requires exactly four uniquely attributed physical GPUs",
    )
    return rows


def gpu_topology_v73d() -> dict[str, Any]:
    command = [str(NVIDIA_SMI), "topo", "-m"]
    raw = _version(command)
    normalized = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    lines = normalized.splitlines()
    matrix_lines = lines[:5]
    _require(len(matrix_lines) == 5, "V73D GPU topology matrix changed")
    rows = []
    for gpu, line in enumerate(matrix_lines[1:]):
        fields = line.split("\t")
        _require(
            fields[0] == f"GPU{gpu}" and len(fields) >= 7,
            "V73D GPU topology row changed",
        )
        links = [field.strip() for field in fields[1:5]]
        rows.append({
            "physical_gpu_id": gpu,
            "pair_labels_gpu0_to_gpu3": links,
            "cpu_affinity": fields[5].strip(),
            "numa_affinity": fields[6].strip(),
            "gpu_numa_id": fields[-1].strip(),
        })
    off_diagonal = [
        label
        for row in rows
        for peer, label in enumerate(row["pair_labels_gpu0_to_gpu3"])
        if peer != row["physical_gpu_id"]
    ]
    _require(
        all(label == "NODE" for label in off_diagonal)
        and all(row["numa_affinity"] == "0" for row in rows)
        and not any(
            label.startswith("NV")
            for row in rows
            for label in row["pair_labels_gpu0_to_gpu3"]
        ),
        "V73D expected NODE/no-NVLink/NUMA-0 topology changed",
    )
    p2p = {}
    for mode, expected in (("r", "OK"), ("w", "OK"), ("n", "NS")):
        p2p_command = [str(NVIDIA_SMI), "topo", "-p2p", mode]
        p2p_raw = _version(p2p_command)
        p2p_normalized = re.sub(r"\x1b\[[0-9;]*m", "", p2p_raw)
        p2p_lines = p2p_normalized.splitlines()[:5]
        _require(len(p2p_lines) == 5, "V73D P2P topology matrix changed")
        p2p_rows = []
        for gpu, line in enumerate(p2p_lines[1:]):
            fields = [field.strip() for field in line.strip().split("\t")]
            _require(
                fields[0] == f"GPU{gpu}" and len(fields) >= 5,
                "V73D P2P topology row changed",
            )
            p2p_rows.append({
                "physical_gpu_id": gpu,
                "pair_status_gpu0_to_gpu3": fields[1:5],
            })
        off_diagonal_status = [
            label
            for row in p2p_rows
            for peer, label in enumerate(row["pair_status_gpu0_to_gpu3"])
            if peer != row["physical_gpu_id"]
        ]
        _require(
            all(label == expected for label in off_diagonal_status),
            f"V73D P2P {mode} capability matrix changed",
        )
        p2p[mode] = {
            "command": p2p_command,
            "normalized_stdout": p2p_normalized,
            "normalized_stdout_sha256": hashlib.sha256(
                p2p_normalized.encode("utf-8")
            ).hexdigest(),
            "rows": p2p_rows,
            "all_cross_gpu_pairs": expected,
        }
    return {
        "command": command,
        "normalized_stdout": normalized,
        "normalized_stdout_sha256": hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
        "rows": rows,
        "all_off_diagonal_gpu_pairs_report_node": True,
        "nvlink_pair_label_present": False,
        "all_numa_affinity_zero": True,
        "p2p_capability_matrices": p2p,
        "all_cross_gpu_peer_read_status_ok": True,
        "all_cross_gpu_peer_write_status_ok": True,
        "all_cross_gpu_nvlink_status_not_supported": True,
        "capability_matrix_used_as_actual_path_evidence": False,
        "actual_nccl_transport_or_path_inferred_from_node": False,
    }


def arm_artifacts_v73d(arm: str) -> dict[str, str]:
    if arm not in {"timeline", "hbm_metrics"}:
        raise ValueError(f"unsupported V73D profiler arm: {arm}")
    stem = f"v73d_{arm}_lora_es_same_live_qwen36_exact_phase"
    run = (ROOT / "experiments/eggroll_es_hpo/runs" / stem).resolve()
    profile = (ROOT / "experiments/eggroll_es_hpo/profiles" / stem).resolve()
    trace_base = profile / stem
    return {
        "application_attempt": str(run.parent / f".{stem}.attempt.json"),
        "run_directory": str(run),
        "gpu_log": str(run / "gpu_activity_v73d.jsonl"),
        "actor_cuda_work_log": str(run / "actor_cuda_work_receipts_v73d.jsonl"),
        "host_process_samples": str(run / "host_process_samples_v73d.jsonl"),
        "host_process_summary": str(run / "host_process_summary_v73d.json"),
        "population": str(run / "mirrored_population_v73d.json"),
        "update": str(run / "pair_difference_update_v73d.json"),
        "audit_traffic": str(run / "exact_audit_traffic_v73d.json"),
        "equivalence": str(run / "same_live_equivalence_v73d.json"),
        "phase_receipt": str(run / "exact_phase_ranges_v73d.json"),
        "path_guard_receipt": str(run / "systems_only_path_guard_v73d.json"),
        "report": str(run / "mirrored_calibration_report_v73d.json"),
        "failure": str(run / "failure_v73d.json"),
        "profile_directory": str(profile),
        "profile_attempt": str(profile.parent / f".{stem}.attempt.json"),
        "profile_failure": str(profile / "profile_failure_v73d.json"),
        "nsys_report": str(trace_base.with_suffix(".nsys-rep")),
        "sqlite_export": str(trace_base.with_suffix(".sqlite")),
        "profile_analysis": str(profile / "profile_analysis_v73d.json"),
        "nccl_debug_pattern": str(profile / "nccl_debug.*.log"),
    }


def _target_template(arm: str) -> list[str]:
    return [
        str(REQUIRED_PYTHON),
        str(TARGET),
        "--preregistration",
        str(OUTPUT),
        "--preregistration-sha256",
        PREREG_FILE_PLACEHOLDER,
        "--preregistration-content-sha256",
        PREREG_CONTENT_PLACEHOLDER,
        "--arm",
        arm,
        "--execute",
    ]


def command_template_v73d(arm: str) -> list[str]:
    artifacts = arm_artifacts_v73d(arm)
    profiler_command = [
        str(NSYS),
        "profile",
        "--trace=cuda,nvtx,nccl",
        "--nccl-trace=none",
        "--cuda-trace-scope=process-tree",
        "--cuda-memory-usage=true",
        "--cuda-event-trace=false",
        "--cuda-graph-trace=graph",
        "--sample=none",
        "--cpuctxsw=none",
        "--backtrace=none",
        "--python-sampling=false",
        "--python-backtrace=none",
        "--pytorch=none",
        "--discard-environment=true",
        "--force-overwrite=false",
        "--export=sqlite",
        "--stats=false",
        "--show-output=false",
        "--wait=all",
    ]
    if arm == "hbm_metrics":
        profiler_command.extend([
            "--gpu-metrics-devices=all",
            "--gpu-metrics-set=gb20x-top",
            "--gpu-metrics-frequency=10000",
        ])
    profiler_command.extend([
        f"--output={Path(artifacts['nsys_report']).with_suffix('')}",
        str(ENV),
        f"PYTHONPATH={GUARD_DIRECTORY}:{ROOT}",
        "SPECIALIST_V73D_SYSTEMS_ONLY_GUARD=1",
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL",
        (
            "NCCL_DEBUG_FILE="
            f"{Path(artifacts['profile_directory']) / 'nccl_debug.%h.%p.log'}"
        ),
        *_target_template(arm),
    ])
    return [
        str(BASH),
        "--noprofile",
        "--norc",
        "-i",
        "-c",
        BASHRC_EXEC_SCRIPT,
        "v73d-nsys-launch",
        *profiler_command,
    ]


def expand_command_v73d(
    arm: str,
    preregistration_file_sha256: str,
    preregistration_content_sha256: str,
) -> list[str]:
    if len(preregistration_file_sha256) != 64:
        raise ValueError("V73D preregistration file hash must be SHA-256")
    if len(preregistration_content_sha256) != 64:
        raise ValueError("V73D preregistration content hash must be SHA-256")
    replacements = {
        PREREG_FILE_PLACEHOLDER: preregistration_file_sha256,
        PREREG_CONTENT_PLACEHOLDER: preregistration_content_sha256,
    }
    command = [replacements.get(item, item) for item in command_template_v73d(arm)]
    _require(
        PREREG_FILE_PLACEHOLDER not in command
        and PREREG_CONTENT_PLACEHOLDER not in command,
        "V73D command template did not expand",
    )
    return command


def validate_generated_preregistration_v73d(
    value: Mapping[str, Any],
) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256(body),
        "V73D preregistration self hash changed",
    )
    _require(
        value.get("schema")
        == "qwen36-v73d-exact-phase-profiler-preregistration-v1"
        and value.get("status")
        == "sealed_cpu_only_before_v73d_model_ray_gpu_or_protected_access"
        and value.get("phase_contract", {}).get("exact_order") == list(PHASES)
        and value.get("phase_contract", {}).get("nvtx_domain") == PHASE_DOMAIN
        and value.get("arms", {}).get("timeline", {}).get("artifacts")
        == arm_artifacts_v73d("timeline")
        and value.get("arms", {}).get("hbm_metrics", {}).get("artifacts")
        == arm_artifacts_v73d("hbm_metrics")
        and value.get("arms", {}).get("timeline", {}).get("command_template")
        == command_template_v73d("timeline")
        and value.get("arms", {}).get("hbm_metrics", {}).get(
            "command_template_after_future_reseal"
        )
        == command_template_v73d("hbm_metrics")
        and value.get("exact_lora_collective_geometry", {}).get(
            "canonical_fp32_tensor_count"
        )
        == 70
        and value.get("exact_lora_collective_geometry", {}).get(
            "canonical_fp32_element_count"
        )
        == 4_528_128
        and value.get("exact_lora_collective_geometry", {}).get(
            "logical_payload_bytes_per_rank"
        )
        == 18_112_512
        and value.get("no_semantic_evaluation_contract", {}).get(
            "semantic_quality_selection_or_hpo"
        )
        is False
        and value.get("launch_policy", {}).get(
            "hbm_arm_can_run_under_current_contract"
        )
        is False,
        "V73D generated runtime contract changed",
    )
    bindings = value.get("implementation_bindings", {})
    boundary = value.get("systems_only_quarantine_boundary", {})
    bootstrap = value.get("ray_actor_guard_bootstrap", {})
    predecessor = value.get("immutable_v73c_attempt_1_predecessor", {})
    _require(
        bindings.get("runtime_contract")
        == source_identity(Path(__file__).resolve())
        and bindings.get("target_runner") == source_identity(TARGET)
        and bindings.get("profile_launcher_and_analyzer")
        == source_identity(LAUNCHER)
        and bindings.get("systems_only_path_guard") == source_identity(GUARD)
        and bindings.get("systems_only_sitecustomize")
        == source_identity(SITECUSTOMIZE)
        and bindings.get("systems_only_worker") == source_identity(WORKER)
        and boundary.get("systems_trace_only") is True
        and boundary.get("quality_hpo_or_promotion_authorized") is False
        and boundary.get("lineage_rehabilitation_authorized") is False
        and boundary.get("stage_a_opaque_historical_reference_count", 0) > 0
        and boundary.get("stage_a_exact_path_guard_bound") is True
        and boundary.get("stage_a_synthetic_denial_tests_passed") is True
        and boundary.get("stage_b_required_postrun_successful_protected_opens")
        == 0
        and boundary.get("stage_b_required_postrun_successful_protected_resolves")
        == 0
        and boundary.get("stage_b_required_postrun_successful_protected_metadata")
        == 0
        and boundary.get(
            "stage_b_required_postrun_successful_protected_enumerations"
        )
        == 0,
        "V73D runtime closure or systems-only quarantine boundary changed",
    )
    _require(
        bootstrap.get("controller_mechanism") == "controller_sitecustomize"
        and bootstrap.get("actor_mechanism")
        == "ray_actor_worker_extension_pre_parent_import"
        and bootstrap.get("actor_bootstrap_env_name") == ACTOR_BOOTSTRAP_ENV
        and bootstrap.get("actor_guard_sha_env_name") == ACTOR_GUARD_SHA_ENV
        and bootstrap.get("actor_guard_file_sha256") == file_sha256(GUARD)
        and bootstrap.get("boundary_registry", {}).get("file_sha256")
        == BOUNDARY_REGISTRY_FILE_SHA256
        and bootstrap.get("boundary_registry", {}).get("content_sha256")
        == BOUNDARY_REGISTRY_CONTENT_SHA256
        and bootstrap.get("ray_job_env_injected_before_ray_init") is True
        and bootstrap.get("actor_parent_import_before_guard")
        == "fail_closed"
        and bootstrap.get(
            "guarded_historical_reference_module_identity_count"
        )
        == 3
        and bootstrap.get(
            "guarded_historical_reference_module_identity_set_sha256"
        )
        == HISTORICAL_REFERENCE_MODULE_IDENTITY_SET_SHA256
        and predecessor.get("status")
        == "immutable_failed_closed_actor_sitecustomize_assumption"
        and predecessor.get("profile_attempt", {}).get("file_sha256")
        == "724b60b6ce33c85e129e5000e1753d8ebd25615f7ede97a17888533397785baf"
        and predecessor.get("profile_failure", {}).get("file_sha256")
        == "ae76cdcf38a71ac7e27035a38d48867a91ed68248c91ef4535ce25b0d79a0c17"
        and predecessor.get("run_attempt", {}).get("file_sha256")
        == "f73fd75b1a9780b2b8a46e7efebf3f514e7b9fb7723df8be2c4ca78c2db9e6b5"
        and predecessor.get("run_failure", {}).get("file_sha256")
        == "5708f456e7736944b7304c5a17486b82c15c5a451a51679734ff00e86cdefef7"
        and predecessor.get("nsys_report", {}).get("file_sha256")
        == "dbdce8049120ed3ffc61834497dd5f52546d07a6d53bc54c9b0c123acfeaf53e"
        and predecessor.get("sqlite_export", {}).get("file_sha256")
        == "73afe6d1f68f0c906224947781a236681ce4799d60a32da9cc7075dd66d4dc7d"
        and predecessor.get("run_failure", {}).get("content_sha256")
        == "a5eb95f7d987112dec0a71683d668d0369a56d7a7838e66c986bf6132cafed4d"
        and predecessor.get("profile_failure", {}).get("content_sha256")
        == "fceaf41c3d8e9a09b20e86d554b3cb8b59d97488f7cd8c083e8ec7048013422a",
        "V73D actor bootstrap or V73C predecessor binding changed",
    )
