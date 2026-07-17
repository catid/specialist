#!/usr/bin/env python3
"""Seal the prospective, data-free V73E exact-phase profiler contract.

This builder performs no model, dataset, Ray, CUDA, or GPU work.  It binds an
unprivileged Nsight Systems timeline arm that is runnable on the current host
and a separate GB20x DRAM-metric arm that is deliberately blocked by the
driver's profiling-permission setting.  Changing that setting does not make a
sealed contract silently runnable; it requires a new prospective amendment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import qwen36_v73e_exact_phase_profiler_contract as runtime_contract


ROOT = Path(__file__).resolve().parent
BASH = Path("/usr/bin/bash")
BASHRC = (Path.home() / ".bashrc").resolve()
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
ENV = runtime_contract.ENV
BASHRC_EXEC_SCRIPT = (
    'source "$HOME/.bashrc" >/dev/null 2>&1 && exec "$@"'
)
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73e_exact_phase_profiler.json"
).resolve()
EVIDENCE = (
    ROOT
    / "experiments/eggroll_es_hpo/"
    "qwen36_v73e_exact_phase_profiler_cpu_evidence_20260717.md"
).resolve()
SYSTEMS_ONLY_CLOSURE = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73e_systems_only_closure.json"
).resolve()
SYSTEMS_ONLY_CLOSURE_AUDITOR = (
    ROOT / "audit_v73e_systems_only_import_graph_v2.py"
).resolve()

V73B_PREREGISTRATION = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_same_live_calibration_v73b.json"
).resolve()
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
V73D_PREREGISTRATION = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73d_exact_phase_profiler.json"
).resolve()
V73D_PROFILE_ATTEMPT = (
    ROOT
    / "experiments/eggroll_es_hpo/profiles/"
    ".v73d_timeline_lora_es_same_live_qwen36_exact_phase.attempt.json"
).resolve()
V73D_PROFILE_DIR = (
    ROOT
    / "experiments/eggroll_es_hpo/profiles/"
    "v73d_timeline_lora_es_same_live_qwen36_exact_phase"
).resolve()
V73D_RUN_ATTEMPT = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    ".v73d_timeline_lora_es_same_live_qwen36_exact_phase.attempt.json"
).resolve()
V73D_RUN_DIR = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    "v73d_timeline_lora_es_same_live_qwen36_exact_phase"
).resolve()
BOUNDARY_REGISTRY_BUILDER = (
    ROOT / "build_quarantine_boundary_registry_v3.py"
).resolve()

NSYS = Path("/usr/local/cuda/bin/nsys")
NCU = Path("/usr/local/cuda/bin/ncu")
NSYS_NATIVE = Path(
    "/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys"
)
NCU_NATIVE = Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")
GB20X_TOP = Path(
    "/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/"
    "GpuMetrics/gb20x-top.config"
)
REQUIRED_PYTHON = (ROOT / "es-at-scale/.venv/bin/python").absolute()
NVTX_INIT = (
    ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/nvtx/__init__.py"
).resolve()
NVTX_NATIVE = (
    ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/nvtx/_lib/"
    "lib.cpython-312-x86_64-linux-gnu.so"
).resolve()
NVTOOLS = (
    ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib/"
    "libnvToolsExt.so.1"
).resolve()
NCCL_METADATA = (
    ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/"
    "nvidia_nccl_cu12-2.27.3.dist-info/METADATA"
).resolve()
NCCL_LIBRARY = (
    ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/"
    "libnccl.so.2"
).resolve()

TARGET = (
    ROOT / "run_lora_es_v71_v72_profile_calibration_v73e.py"
).resolve()
LAUNCHER = (ROOT / "run_qwen36_v73e_exact_phase_profiler.py").resolve()

PREREG_FILE_PLACEHOLDER = "<PREREGISTRATION_FILE_SHA256>"
PREREG_CONTENT_PLACEHOLDER = "<PREREGISTRATION_CONTENT_SHA256>"
PHASE_DOMAIN = "eggroll_es_v73e_phase"
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

V73B_INPUT_HASHES = {
    V73B_PREREGISTRATION: (
        "9c5ce43c36e08e038ee33e86380ab6c287ae1b4bcba4c80def623daeb00f7ed9"
    ),
    V73B_POSTRUN: (
        "d65be30ec769e8a18ce75a3ffc0aab5624cec35f909732e8082a4836c63140c4"
    ),
    V73B_RUN / "mirrored_calibration_report_v73b.json": (
        "ba1b0a76dd0a9955b5e3f779d1ef440037b12b4689b3b1ab640d1ee1a4cff44a"
    ),
    V73B_RUN / "actor_cuda_work_receipts_v73b.jsonl": (
        "30df82d21b28c7d5c94ede785c69c10896bad9d07db52ccf065e3031181d7013"
    ),
    V73B_RUN / "exact_audit_traffic_v73b.json": (
        "388fb6f544254c94e0c0ae11956932757834894103784f9e43d6b76e1bb3cb20"
    ),
}
V73D_ATTEMPT_1_HASHES = {
    V73D_PREREGISTRATION: (
        "d1810d51ecc49615d4067c4b8b151fa9154cb708a6658c02a497210238768c0a"
    ),
    V73D_PROFILE_ATTEMPT: (
        "82937bea1a3701e0125e8f21286c1a6ee8b70d379fa13d5c69d6ce7d75ae2c67"
    ),
    V73D_PROFILE_DIR / "profile_failure_v73d.json": (
        "025a38adce724d931a644dd090f6eb5b347f0d1ebd7bcc500a149af67f22bd50"
    ),
    V73D_PROFILE_DIR
    / "v73d_timeline_lora_es_same_live_qwen36_exact_phase.nsys-rep": (
        "94fd7fc159fd263142557042d8821fb8ef1df18265826298bb19bd0837cd5cf4"
    ),
    V73D_PROFILE_DIR
    / "v73d_timeline_lora_es_same_live_qwen36_exact_phase.sqlite": (
        "3d3dc120216a81fc1ec1ed59286df116f5535deda3d6be1d4f05e466d28f2ed7"
    ),
    V73D_RUN_ATTEMPT: (
        "4eb3993f06db7fb1f8bd4f4eddf4b84b0930b06780110caab845971027a51413"
    ),
    V73D_RUN_DIR / "failure_v73d.json": (
        "08c0a3e2ca3832486e6b94bfd4ad0529c8420d3537574c7f2dda6ad4a8ed3886"
    ),
    V73D_RUN_DIR / "exact_phase_ranges_v73d.json": (
        "12b510a7ef84e91c8d84bf6f27709eef10f802d36bfc6ed65c8a377fd315b618"
    ),
}


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


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


def _target_package_version(distribution: str) -> str:
    return _version([
        str(REQUIRED_PYTHON),
        "-c",
        (
            "import importlib.metadata;"
            f"print(importlib.metadata.version({distribution!r}))"
        ),
    ])


def _source(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and not path.is_symlink(), f"source absent: {path}")
    return {"path": str(path), "file_sha256": file_sha256(path)}


def _closure_path_identity(relative: str, *, schema: str) -> str:
    return canonical_sha256({"schema": schema, "value": relative})


def _load_systems_only_closure_v73e() -> dict[str, Any]:
    _require(
        SYSTEMS_ONLY_CLOSURE.is_file()
        and not SYSTEMS_ONLY_CLOSURE.is_symlink(),
        "V73E external systems-only closure receipt absent",
    )
    value = json.loads(SYSTEMS_ONLY_CLOSURE.read_text(encoding="ascii"))
    _require(isinstance(value, dict), "V73E closure receipt must be an object")
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    graph = value.get("import_graph", {})
    authority = value.get("authority", {})
    static = value.get("static_historical_references", {})
    synthetic = value.get("synthetic_runtime_guard_tests", {})
    expected_roots = {
        TARGET.name: file_sha256(TARGET),
        LAUNCHER.name: file_sha256(LAUNCHER),
    }
    expected_authority = {
        _closure_path_identity(
            path.name, schema="repository-root-module-path-v1"
        ): file_sha256(path)
        for path in (
            TARGET,
            LAUNCHER,
            Path(runtime_contract.__file__).resolve(),
        )
    }
    observed_authority = {
        str(row.get("module_path_identity_sha256")): row.get("file_sha256")
        for row in graph.get("authority_guard_source_bindings", ())
        if isinstance(row, dict)
    }
    expected_test_identity = _closure_path_identity(
        "test_qwen36_v73e_exact_phase_profiler.py",
        schema="repository-root-test-path-v1",
    )
    test_bindings = {
        str(row.get("path_identity_sha256")): row.get("file_sha256")
        for row in synthetic.get("test_source_bindings", ())
        if isinstance(row, dict)
    }
    quarantined_count = graph.get("quarantined_boundary_reference_count")
    semantic_count = graph.get("semantic_evaluation_path_reference_count")
    opaque_count = static.get("opaque_reference_binding_count")
    auditor = _source(SYSTEMS_ONLY_CLOSURE_AUDITOR)
    _require(
        isinstance(claimed, str)
        and claimed == canonical_sha256(body)
        and value.get("schema") == "qwen36-v73e-systems-only-closure-v1"
        and value.get("status")
        == "v73e_runtime_closure_systems_only_no_quality_authority"
        and graph.get("root_module_file_sha256") == expected_roots
        and graph.get("mutable_prereg_builder_reachable") is False
        and graph.get("authority_guard_module_count") == 3
        and observed_authority == expected_authority
        and graph.get("true_quality_or_promotion_authority_rule_count") == 0
        and graph.get("required_false_authority_rule_count") == 4
        and isinstance(graph.get("local_closure_modules"), int)
        and graph.get("local_closure_modules") >= 3
        and isinstance(graph.get("local_closure_content_sha256"), str)
        and len(graph["local_closure_content_sha256"]) == 64
        and isinstance(quarantined_count, int)
        and isinstance(semantic_count, int)
        and isinstance(opaque_count, int)
        and opaque_count == quarantined_count + semantic_count
        and opaque_count > 0
        and static.get("opaque_only") is True
        and static.get("does_not_authorize_path_resolution_or_open") is True
        and static.get("actual_open_denial_requires_distinct_postrun_receipt")
        is True
        and synthetic.get("schema")
        == "v73e-synthetic-runtime-guard-test-receipt-v1"
        and synthetic.get("passed") is True
        and synthetic.get("exit_code") == 0
        and synthetic.get("stdout_or_stderr_persisted") is False
        and test_bindings.get(expected_test_identity)
        == file_sha256(ROOT / "test_qwen36_v73e_exact_phase_profiler.py")
        and authority.get("systems_trace_only") is True
        and authority.get(
            "semantic_evaluation_or_quarantined_boundary_resolved"
        )
        is False
        and authority.get("quality_selection_hpo_or_promotion_authorized")
        is False
        and authority.get("historical_quarantined_lineage_rehabilitated")
        is False
        and value.get("implementation", {}).get("builder_file_sha256")
        == auditor["file_sha256"],
        "V73E external systems-only closure receipt changed",
    )
    return {
        "receipt": {
            **_source(SYSTEMS_ONLY_CLOSURE),
            "content_sha256": claimed,
            "auditor": auditor,
        },
        "opaque_historical_reference_count": opaque_count,
        "local_closure_module_count": graph["local_closure_modules"],
        "local_closure_content_sha256": graph[
            "local_closure_content_sha256"
        ],
        "synthetic_selected_test_node_count": synthetic[
            "selected_test_node_count"
        ],
    }


def _self_hashed_reference(
    path: Path, expected_content_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"V73D JSON object required: {path}")
    compact = dict(value)
    claimed = compact.pop("content_sha256_before_self_field", None)
    _require(
        claimed == expected_content_sha256
        and canonical_sha256(compact) == claimed,
        f"V73D attempt-1 self hash changed: {path}",
    )
    return {
        **_source(path),
        "content_sha256": claimed,
    }, value


def _validate_v73d_attempt_1_predecessor() -> dict[str, Any]:
    for path, expected in V73D_ATTEMPT_1_HASHES.items():
        _require(path.is_file(), f"V73D attempt-1 artifact absent: {path}")
        _require(
            file_sha256(path) == expected,
            f"V73D attempt-1 artifact changed: {path}",
        )
    prereg_ref, prereg = _self_hashed_reference(
        V73D_PREREGISTRATION,
        "512287c5a438cd1a22b099f61fe55e777de86a71b52555b49a47959425deb740",
    )
    profile_attempt_ref, profile_attempt = _self_hashed_reference(
        V73D_PROFILE_ATTEMPT,
        "02b0f154ff9e6c8df1b710e57adad14e935f9a2ddd8706d11d5622c60cd65cf7",
    )
    profile_failure_ref, profile_failure = _self_hashed_reference(
        V73D_PROFILE_DIR / "profile_failure_v73d.json",
        "0335438e9ac89680f9e78800fcfdcd0a1cb31a661f9255e935bd79add8f18d07",
    )
    run_attempt_ref, run_attempt = _self_hashed_reference(
        V73D_RUN_ATTEMPT,
        "4d46115a40c544a07a68fe3b027b3c99ef8ee0f172df31dfec4d920c83898e21",
    )
    run_failure_ref, run_failure = _self_hashed_reference(
        V73D_RUN_DIR / "failure_v73d.json",
        "d57e87c1a589bbd344d334aeb37b242357261dc18a602cb2cefd1be1af9de30f",
    )
    phase_ref, phase = _self_hashed_reference(
        V73D_RUN_DIR / "exact_phase_ranges_v73d.json",
        "5259c316491c692dbb3f854ee690f40397828fed617731ad13fd615d8893aba1",
    )
    _require(
        prereg.get("schema")
        == "qwen36-v73d-exact-phase-profiler-preregistration-v1"
        and profile_attempt.get("status")
        == "prelaunch_accepted_launching_fresh_no_commit_profile"
        and profile_attempt.get("four_gpu_idle_preflight", {}).get("passed")
        is True
        and profile_attempt.get("four_gpu_idle_preflight", {}).get(
            "no_compute_processes"
        )
        is True
        and profile_failure.get("status") == "target_or_profiler_failed_closed"
        and profile_failure.get("returncode") == 1
        and run_attempt.get("status")
        == "launching_train_only_exact_equivalence_no_commit"
        and run_failure.get("type") == "PermissionError"
        and run_failure.get("message")
        == "V73D systems-only protected open denied"
        and run_failure.get("ray_actor_guard_bootstrap") is None
        and run_failure.get("partial_actor_cuda_work_log", {}).get("rows") == 0
        and run_failure.get("partial_host_process_log", {}).get("rows") == 0
        and run_failure.get("gpu_time_accounting", {}).get(
            "promotion_charged_gpu_seconds"
        ) == 0
        and run_failure.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is True
        and run_failure.get("protected_dev_ood_or_holdout_opened") is False
        and run_failure.get("checkpoint_snapshot_or_promotion_performed")
        is False
        and phase.get("complete") is False
        and phase.get("observed_phase_order") == ["setup"]
        and phase.get("contains_prompts_questions_answers_or_outputs") is False,
        "V73D attempt-1 failure classification changed",
    )
    return {
        "status": (
            "immutable_failed_closed_legacy_adapter_quarantine_"
            "incompatibility"
        ),
        "v73d_sealed_source_commits": [
            "85d72354133f9ffeb76414a7c69b489bb5dfde12",
            "91296fbc6c5d15695bfb9c7aad3d45dda8d52bb8",
            "25769debd285793d4bce95f070f3512f9ac39ff5",
            "23d8ee104ca27527cf270b915eea74b2680adcaf",
        ],
        "preregistration": prereg_ref,
        "profile_attempt": profile_attempt_ref,
        "profile_failure": profile_failure_ref,
        "run_attempt": run_attempt_ref,
        "run_failure": run_failure_ref,
        "partial_phase_receipt": phase_ref,
        "nsys_report": _source(
            V73D_PROFILE_DIR
            / "v73d_timeline_lora_es_same_live_qwen36_exact_phase.nsys-rep"
        ),
        "sqlite_export": _source(
            V73D_PROFILE_DIR
            / "v73d_timeline_lora_es_same_live_qwen36_exact_phase.sqlite"
        ),
        "failure_cause": (
            "quarantined_legacy_adapter_contract_opened_during_setup"
        ),
        "ray_actor_guard_bootstrap_reached": False,
        "model_load_reached": False,
        "gpu_work_reached": False,
        "partial_actor_cuda_receipt_count": 0,
        "final_all_four_gpus_idle": True,
        "promotion_charged_gpu_seconds": 0,
        "protected_or_semantic_data_opened": False,
        "quality_or_semantic_authority": False,
        "artifacts_must_not_be_deleted_modified_or_reused": True,
    }


def _validate_quarantine_boundary_registry_v3() -> dict[str, Any]:
    path = runtime_contract.BOUNDARY_REGISTRY
    _require(
        path.is_file()
        and not path.is_symlink()
        and file_sha256(path)
        == runtime_contract.BOUNDARY_REGISTRY_FILE_SHA256,
        "V73E quarantine boundary registry file changed",
    )
    value = json.loads(path.read_text(encoding="ascii"))
    compact = dict(value)
    claimed = compact.pop("content_sha256_before_self_field", None)
    policy = value.get("ancestor_denial_policy", {})
    _require(
        claimed == runtime_contract.BOUNDARY_REGISTRY_CONTENT_SHA256
        and canonical_sha256(compact) == claimed
        and value.get("schema")
        == "specialist-content-free-quarantine-boundary-registry-v3"
        and value.get("status")
        == "active_fail_closed_content_free_quarantine_boundary"
        and value.get("exact_path_identity_count") == 3
        and value.get("prefix_identity_count") == 2
        and value.get("ancestor_denial_policy_sha256")
        == "3758e272396ebc0cbed5a933469665530275c9eabe99f1c0d6210464e7d1a48c"
        and policy.get("lexical_deny_before_resolution_stat_hash_or_open")
        is True
        and policy.get("lexically_allowed_resolution_rechecked_before_metadata_or_open")
        is True
        and policy.get("resolved_target_outside_repository_root_denied") is True
        and value.get("content_minimization", {}).get(
            "plaintext_boundary_paths_persisted"
        )
        is False
        and value.get("implementation", {}).get("builder_file_sha256")
        == file_sha256(BOUNDARY_REGISTRY_BUILDER),
        "V73E quarantine boundary registry content changed",
    )
    return {
        **_source(path),
        "content_sha256": claimed,
        "builder": _source(BOUNDARY_REGISTRY_BUILDER),
        "exact_path_identity_count": value["exact_path_identity_count"],
        "exact_path_identity_set_sha256": canonical_sha256(
            value["exact_path_identity_sha256"]
        ),
        "prefix_identity_count": value["prefix_identity_count"],
        "prefix_identity_set_sha256": canonical_sha256(
            value["prefix_identity_sha256"]
        ),
        "ancestor_denial_policy_sha256": value[
            "ancestor_denial_policy_sha256"
        ],
        "plaintext_boundary_paths_persisted": False,
    }


def _executable_invocation(path: Path) -> dict[str, Any]:
    path = Path(path).absolute()
    resolved = path.resolve()
    _require(path.is_file(), f"executable absent: {path}")
    return {
        "invocation_path": str(path),
        "resolved_file": _source(resolved),
    }


def _rm_profiling_admin_only(
    path: Path = Path("/proc/driver/nvidia/params"),
) -> int:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    selected = [line for line in lines if line.startswith("RmProfilingAdminOnly:")]
    _require(len(selected) == 1, "NVIDIA profiling permission field changed")
    raw = selected[0].split(":", 1)[1].strip()
    _require(raw in {"0", "1"}, "NVIDIA profiling permission value changed")
    return int(raw)


def gpu_identity_v73e() -> list[dict[str, Any]]:
    output = _version([
        str(NVIDIA_SMI),
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 3, "V73E GPU identity query shape changed")
        rows.append({
            "physical_gpu_id": int(fields[0]),
            "uuid": fields[1],
            "pci_bus_id": fields[2],
        })
    _require(
        [row["physical_gpu_id"] for row in rows] == [0, 1, 2, 3]
        and len({row["uuid"] for row in rows}) == 4
        and len({row["pci_bus_id"] for row in rows}) == 4,
        "V73E requires exactly four uniquely attributed physical GPUs",
    )
    return rows


def gpu_topology_v73e() -> dict[str, Any]:
    command = [str(NVIDIA_SMI), "topo", "-m"]
    raw = _version(command)
    normalized = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    lines = normalized.splitlines()
    matrix_lines = lines[:5]
    _require(len(matrix_lines) == 5, "V73E GPU topology matrix changed")
    rows = []
    for gpu, line in enumerate(matrix_lines[1:]):
        fields = line.split("\t")
        _require(
            fields[0] == f"GPU{gpu}" and len(fields) >= 7,
            "V73E GPU topology row changed",
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
        "V73E expected NODE/no-NVLink/NUMA-0 topology changed",
    )
    p2p = {}
    for mode, expected in (("r", "OK"), ("w", "OK"), ("n", "NS")):
        p2p_command = [str(NVIDIA_SMI), "topo", "-p2p", mode]
        p2p_raw = _version(p2p_command)
        p2p_normalized = re.sub(r"\x1b\[[0-9;]*m", "", p2p_raw)
        p2p_lines = p2p_normalized.splitlines()[:5]
        _require(len(p2p_lines) == 5, "V73E P2P topology matrix changed")
        p2p_rows = []
        for gpu, line in enumerate(p2p_lines[1:]):
            fields = [field.strip() for field in line.strip().split("\t")]
            _require(
                fields[0] == f"GPU{gpu}" and len(fields) >= 5,
                "V73E P2P topology row changed",
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
            f"V73E P2P {mode} capability matrix changed",
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


def arm_artifacts_v73e(arm: str) -> dict[str, str]:
    if arm not in {"timeline", "hbm_metrics"}:
        raise ValueError(f"unsupported V73E profiler arm: {arm}")
    stem = f"v73e_{arm}_lora_es_content_free_qwen36_exact_phase"
    run = (
        ROOT / "experiments/eggroll_es_hpo/runs" / stem
    ).resolve()
    profile = (
        ROOT / "experiments/eggroll_es_hpo/profiles" / stem
    ).resolve()
    trace_base = profile / stem
    return {
        "application_attempt": str(run.parent / f".{stem}.attempt.json"),
        "run_directory": str(run),
        "gpu_log": str(run / "gpu_activity_v73e.jsonl"),
        "actor_cuda_work_log": str(
            run / "actor_cuda_work_receipts_v73e.jsonl"
        ),
        "host_process_samples": str(run / "host_process_samples_v73e.jsonl"),
        "host_process_summary": str(run / "host_process_summary_v73e.json"),
        "population": str(run / "mirrored_population_v73e.json"),
        "update": str(run / "pair_difference_update_v73e.json"),
        "audit_traffic": str(run / "exact_audit_traffic_v73e.json"),
        "equivalence": str(run / "content_free_systems_consistency_v73e.json"),
        "phase_receipt": str(run / "exact_phase_ranges_v73e.json"),
        "path_guard_receipt": str(run / "systems_only_path_guard_v73e.json"),
        "report": str(run / "mirrored_calibration_report_v73e.json"),
        "failure": str(run / "failure_v73e.json"),
        "profile_directory": str(profile),
        "profile_attempt": str(profile.parent / f".{stem}.attempt.json"),
        "profile_failure": str(profile / "profile_failure_v73e.json"),
        "nsys_report": str(trace_base.with_suffix(".nsys-rep")),
        "sqlite_export": str(trace_base.with_suffix(".sqlite")),
        "profile_analysis": str(profile / "profile_analysis_v73e.json"),
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


def command_template_v73e(arm: str) -> list[str]:
    artifacts = arm_artifacts_v73e(arm)
    profiler_command = [
        str(NSYS),
        "profile",
        "--trace=cuda,nvtx,nccl",
        # Installed NCCL is 2.27.3, below the advanced-trace floor.  Explicit
        # ``none`` selects the supported legacy NCCL NVTX trace rather than
        # silently requesting an incompatible profiler plugin.
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
        *_target_template(arm),
    ])
    return [
        str(BASH),
        "--noprofile",
        "--norc",
        "-i",
        "-c",
        BASHRC_EXEC_SCRIPT,
        "v73e-nsys-launch",
        *profiler_command,
    ]


def expand_command_v73e(
    arm: str,
    preregistration_file_sha256: str,
    preregistration_content_sha256: str,
) -> list[str]:
    if len(preregistration_file_sha256) != 64:
        raise ValueError("V73E preregistration file hash must be SHA-256")
    if len(preregistration_content_sha256) != 64:
        raise ValueError("V73E preregistration content hash must be SHA-256")
    replacements = {
        PREREG_FILE_PLACEHOLDER: preregistration_file_sha256,
        PREREG_CONTENT_PLACEHOLDER: preregistration_content_sha256,
    }
    command = [replacements.get(item, item) for item in command_template_v73e(arm)]
    _require(
        PREREG_FILE_PLACEHOLDER not in command
        and PREREG_CONTENT_PLACEHOLDER not in command,
        "V73E command template did not expand",
    )
    return command


# Runtime command/path identity lives outside this builder so the launch-time
# AST closure can be sealed without recursively including the artifact that
# later binds that closure receipt.
arm_artifacts_v73e = runtime_contract.arm_artifacts_v73e
command_template_v73e = runtime_contract.command_template_v73e
expand_command_v73e = runtime_contract.expand_command_v73e
gpu_identity_v73e = runtime_contract.gpu_identity_v73e
gpu_topology_v73e = runtime_contract.gpu_topology_v73e


def _validate_v73b_control() -> dict[str, Any]:
    for path, expected in V73B_INPUT_HASHES.items():
        _require(path.is_file(), f"V73B control absent: {path}")
        _require(file_sha256(path) == expected, f"V73B control changed: {path}")
    postrun = json.loads(V73B_POSTRUN.read_text(encoding="ascii"))
    traffic = json.loads(
        (V73B_RUN / "exact_audit_traffic_v73b.json").read_text(
            encoding="ascii"
        )
    )
    v41a_source = (ROOT / "eggroll_es_worker_lora_v41a.py").read_text(
        encoding="ascii"
    )
    v72_source = (ROOT / "eggroll_es_host_state_contract_v72.py").read_text(
        encoding="ascii"
    )
    _require(
        postrun.get("passed") is True
        and postrun.get("status") == "accepted_same_live_v71_v72_lora_gate"
        and postrun.get("content_sha256_before_self_field")
        == "21689f75ecaaf583aedde50ad293ce3a9b5644009d62c2bc4624637280a651e7"
        and postrun.get("safety", {}).get("master_committed") is False
        and postrun.get("safety", {}).get(
            "protected_dev_ood_or_holdout_opened"
        )
        is False,
        "accepted V73B postrun semantics changed",
    )
    _require(
        traffic.get("known_code_path_device_transfer_outside_worker_counter_per_actor", {}).get(
            "update_reduced_fp32_delta_d2h_bytes"
        )
        == 18_112_512
        and "EXPECTED_MASTER_ELEMENTS_V41A = 4_528_128" in v41a_source
        and "MASTER_ELEMENTS_V72 = 4_528_128" in v72_source,
        "canonical V41A/V72 LoRA geometry changed",
    )
    return {
        "preregistration": _source(V73B_PREREGISTRATION),
        "postrun": {
            **_source(V73B_POSTRUN),
            "content_sha256": postrun["content_sha256_before_self_field"],
        },
        "run_report": _source(
            V73B_RUN / "mirrored_calibration_report_v73b.json"
        ),
        "actor_work_log": _source(
            V73B_RUN / "actor_cuda_work_receipts_v73b.jsonl"
        ),
        "audit_traffic": _source(
            V73B_RUN / "exact_audit_traffic_v73b.json"
        ),
        "wall_runtime_seconds": postrun["timing"]["end_to_end"]["wall"][
            "v73b_wall_runtime_seconds"
        ],
        "charged_gpu_seconds": postrun["timing"]["end_to_end"][
            "charged_gpu"
        ]["v73b_charged_gpu_seconds"],
        "unprofiled_timing_control_only": True,
    }


def toolchain_receipt_v73e() -> dict[str, Any]:
    required = (
        NSYS,
        NCU,
        BASH,
        BASHRC,
        NVIDIA_SMI,
        ENV,
        NSYS_NATIVE,
        NCU_NATIVE,
        GB20X_TOP,
        REQUIRED_PYTHON,
        NVTX_INIT,
        NVTX_NATIVE,
        NVTOOLS,
        NCCL_METADATA,
        NCCL_LIBRARY,
    )
    for path in required:
        _require(path.is_file(), f"V73E profiler tool absent: {path}")
    nsys_version = _version([str(NSYS), "--version"])
    ncu_version = _version([str(NCU), "--version"])
    _require(
        nsys_version
        == "NVIDIA Nsight Systems version 2026.1.3.425-261338342291v0",
        "Nsight Systems version changed",
    )
    _require(
        "Version 2026.2.1.0 (build 38283040) (public-release)" in ncu_version,
        "Nsight Compute version changed",
    )
    nvtx_version = _target_package_version("nvtx")
    _require(nvtx_version == "0.2.15", "Python NVTX version changed")
    nccl_version = _target_package_version("nvidia-nccl-cu12")
    _require(nccl_version == "2.27.3", "NCCL version changed")
    return {
        "launch_shell": {
            "bash": _source(BASH),
            "bashrc": _source(BASHRC),
            "fixed_exec_script": BASHRC_EXEC_SCRIPT,
            "interactive_flag_required_for_bashrc_noninteractive_guard": True,
            "subprocess_shell_flag": False,
            "environment_values_persisted": False,
        },
        "nvidia_smi": _source(NVIDIA_SMI),
        "environment_exec": _source(ENV),
        "nsight_systems": {
            **_source(NSYS),
            "native": _source(NSYS_NATIVE),
            "version_stdout": nsys_version,
        },
        "nsight_compute": {
            **_source(NCU),
            "native": _source(NCU_NATIVE),
            "version_stdout_sha256": hashlib.sha256(
                ncu_version.encode("utf-8")
            ).hexdigest(),
            "version_line": "Version 2026.2.1.0 (build 38283040) (public-release)",
        },
        "target_python": _executable_invocation(REQUIRED_PYTHON),
        "python_nvtx": {
            "version": nvtx_version,
            "package_init": _source(NVTX_INIT),
            "native_extension": _source(NVTX_NATIVE),
            "nvtools_library": _source(NVTOOLS),
            "does_not_initialize_cuda_in_controller": True,
        },
        "nccl": {
            "version": nccl_version,
            "package_metadata": _source(NCCL_METADATA),
            "library": _source(NCCL_LIBRARY),
            "advanced_trace_requested": False,
            "reason": (
                "installed_2.27.3_is_below_advanced_trace_2.28_floor_"
                "and_2.27.4_limited_support_floor"
            ),
            "selected_trace": "legacy_NCCL_NVTX_domain",
        },
        "gb20x_top_metric_set": {
            **_source(GB20X_TOP),
            "required_tool_defined_metrics": [
                "VRAM Total Bandwidth",
                "VRAM Read Bandwidth",
                "VRAM Write Bandwidth",
            ],
            "values_are_tool_defined_samples_not_exact_integrated_bytes": True,
        },
    }


def build_preregistration_v73e() -> dict[str, Any]:
    permission = _rm_profiling_admin_only()
    _require(
        permission == 1,
        "V73E sealed blocked HBM arm requires RmProfilingAdminOnly=1",
    )
    control = _validate_v73b_control()
    predecessor = _validate_v73d_attempt_1_predecessor()
    boundary_registry = _validate_quarantine_boundary_registry_v3()
    closure = _load_systems_only_closure_v73e()
    tools = toolchain_receipt_v73e()
    gpu_identity = gpu_identity_v73e()
    gpu_topology = gpu_topology_v73e()
    timeline_artifacts = arm_artifacts_v73e("timeline")
    hbm_artifacts = arm_artifacts_v73e("hbm_metrics")
    implementation = {
        "builder": _source(Path(__file__).resolve()),
        "runtime_contract": _source(Path(runtime_contract.__file__).resolve()),
        "target_runner": _source(TARGET),
        "profile_launcher_and_analyzer": _source(LAUNCHER),
        "systems_only_path_guard": _source(runtime_contract.GUARD),
        "systems_only_sitecustomize": _source(runtime_contract.SITECUSTOMIZE),
        "systems_only_worker": _source(runtime_contract.WORKER),
        "quarantine_boundary_registry_builder": _source(
            BOUNDARY_REGISTRY_BUILDER
        ),
        "v73b_builder": _source(
            ROOT / "build_lora_es_v71_v72_same_live_preregistration_v73b.py"
        ),
        "v73b_runner": _source(
            ROOT / "run_lora_es_v71_v72_same_live_calibration_v73b.py"
        ),
        "v73_adapter": _source(
            ROOT / "run_lora_es_v71_v72_live_calibration_v73.py"
        ),
        "v71_worker": _source(ROOT / "eggroll_es_worker_lora_v71.py"),
        "v72_worker": _source(ROOT / "eggroll_es_worker_lora_v72.py"),
        "v41a_canonical_lora_geometry": _source(
            ROOT / "eggroll_es_worker_lora_v41a.py"
        ),
        "v72_host_state_geometry": _source(
            ROOT / "eggroll_es_host_state_contract_v72.py"
        ),
    }
    staged_files = {
        "weights": _source(runtime_contract.STAGED_ADAPTER_WEIGHTS),
        "config": _source(runtime_contract.STAGED_ADAPTER_CONFIG),
        "manifest": _source(runtime_contract.STAGED_ADAPTER_MANIFEST),
        "transform_implementation": _source(
            runtime_contract.STAGED_TRANSFORM_IMPLEMENTATION
        ),
    }
    _require(
        staged_files["weights"]["file_sha256"]
        == runtime_contract.STAGED_ADAPTER_WEIGHTS_SHA256
        and staged_files["config"]["file_sha256"]
        == runtime_contract.STAGED_ADAPTER_CONFIG_SHA256
        and staged_files["manifest"]["file_sha256"]
        == runtime_contract.STAGED_ADAPTER_MANIFEST_FILE_SHA256
        and staged_files["transform_implementation"]["file_sha256"]
        == runtime_contract.STAGED_TRANSFORM_IMPLEMENTATION_SHA256,
        "V73E staged adapter identity changed",
    )
    panel = runtime_contract.content_free_token_panel_v73e()
    suffix_reward = runtime_contract.content_free_suffix_reward_config_v73e()
    _require(
        panel.get("content_sha256_before_self_field")
        == runtime_contract.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        and canonical_sha256(suffix_reward)
        == runtime_contract.CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256,
        "V73E deterministic content-free input contract changed",
    )
    result = {
        "schema": "qwen36-v73e-exact-phase-profiler-preregistration-v1",
        "status": (
            "sealed_cpu_only_before_v73e_model_ray_gpu_or_protected_access"
        ),
        "purpose": (
            "Supersede the immutable V73D setup failure by replacing both the "
            "quarantined legacy adapter and train bundle with a staged exact-"
            "inverse adapter and deterministic content-free token-ID panel. "
            "Retain V73B only as mechanics, timing, and aggregate-count context, "
            "install the V3 quarantine guard inside every prestarted Ray actor "
            "before V72, and measure post-V71/V72 "
            "CUDA kernels, memcpy, NCCL, and allocation events without using "
            "profiled timing as throughput evidence. Keep GB20x DRAM counters "
            "as a separately blocked arm rather than inferring HBM traffic."
        ),
        "bead": "specialist-nen.35",
        "related_beads": [
            "specialist-0j5.32",
            "specialist-0j5.36",
            "specialist-nen.33",
            "specialist-nen.34",
        ],
        "immutable_v73d_attempt_1_predecessor": predecessor,
        "accepted_unprofiled_control": control,
        "toolchain": tools,
        "physical_gpu_identity": gpu_identity,
        "physical_gpu_topology": gpu_topology,
        "implementation_bindings": implementation,
        "staged_only_adapter_contract": {
            "status": "sealed_staged_only_exact_inverse_required",
            "files": staged_files,
            "source_tensor_namespace": runtime_contract.STAGED_TARGET_PREFIX,
            "canonical_tensor_namespace": (
                runtime_contract.CANONICAL_SOURCE_PREFIX
            ),
            "tensor_count": runtime_contract.STAGED_TENSOR_COUNT,
            "element_count": runtime_contract.STAGED_ELEMENT_COUNT,
            "tensor_bytes": runtime_contract.STAGED_TENSOR_BYTES,
            "exact_inverse_key_mapping_sha256": (
                runtime_contract.INVERSE_KEY_MAPPING_SHA256
            ),
            "staged_transform_identity_sha256": (
                runtime_contract.STAGED_TRANSFORM_IDENTITY_SHA256
            ),
            "canonical_master_sha256": (
                runtime_contract.CANONICAL_MASTER_SHA256
            ),
            "canonical_runtime_values_sha256": (
                runtime_contract.CANONICAL_RUNTIME_VALUES_SHA256
            ),
            "quarantined_legacy_adapter_open_stat_hash_or_resolve": False,
        },
        "content_free_systems_workload": {
            "status": "deterministic_numeric_token_ids_no_semantic_authority",
            "token_panel_content_sha256": (
                runtime_contract.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
            ),
            "request_block_sha256": (
                runtime_contract.CONTENT_FREE_REQUEST_BLOCK_SHA256
            ),
            "raw_token_inventory_sha256": (
                runtime_contract.CONTENT_FREE_RAW_TOKEN_INVENTORY_SHA256
            ),
            "length_manifest_sha256": (
                runtime_contract.CONTENT_FREE_LENGTH_MANIFEST_SHA256
            ),
            "suffix_reward_config_sha256": (
                runtime_contract.CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256
            ),
            "evaluation_contract_sha256": (
                runtime_contract.CONTENT_FREE_EVALUATION_CONTRACT_SHA256
            ),
            "mirrored_plan_sha256": runtime_contract.CONTENT_FREE_PLAN_SHA256,
            "request_count": runtime_contract.CONTENT_FREE_TOKEN_PANEL_ROWS,
            "unscored_prefix_tokens_per_candidate": (
                runtime_contract.CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE
            ),
            "scored_suffix_tokens_per_candidate": (
                runtime_contract.CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
            ),
            "full_input_tokens_per_candidate": (
                runtime_contract.CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE
            ),
            "generated_tokens_per_candidate": (
                runtime_contract.CONTENT_FREE_GENERATED_TOKENS_PER_CANDIDATE
            ),
            "direction_seeds": list(
                runtime_contract.CONTENT_FREE_DIRECTION_SEEDS
            ),
            "sigma": runtime_contract.CONTENT_FREE_SIGMA,
            "learning_rate": runtime_contract.CONTENT_FREE_LEARNING_RATE,
            "historical_v73b_workload_control_or_semantic_authority": False,
            "qa_dev_ood_holdout_or_other_semantic_dataset_used": False,
        },
        "ray_actor_guard_bootstrap": {
            "controller_mechanism": "controller_sitecustomize",
            "actor_mechanism": (
                "ray_actor_worker_extension_pre_parent_import"
            ),
            "mechanisms_must_be_distinct": True,
            "actor_bootstrap_env_name": runtime_contract.ACTOR_BOOTSTRAP_ENV,
            "actor_guard_sha_env_name": runtime_contract.ACTOR_GUARD_SHA_ENV,
            "actor_guard_file_sha256": file_sha256(runtime_contract.GUARD),
            "ray_job_env_injected_before_ray_init": True,
            "ray_job_env_merges_with_existing_actor_specific_runtime_env": True,
            "actor_parent_import_before_guard": "fail_closed",
            "guarded_historical_reference_module_identity_count": 3,
            "guarded_historical_reference_module_identity_set_sha256": (
                runtime_contract.HISTORICAL_REFERENCE_MODULE_IDENTITY_SET_SHA256
            ),
            "controller_pid_marker_prevents_actor_sitecustomize_reuse": True,
            "boundary_registry": boundary_registry,
            "successful_protected_open_resolve_metadata_or_enumeration": 0,
        },
        "systems_only_quarantine_boundary": {
            "status": "historical_v1_descendant_systems_only_nonpromotable",
            "systems_trace_only": True,
            "quality_hpo_or_promotion_authorized": False,
            "lineage_rehabilitation_authorized": False,
            "v66_quality_authority_rehabilitated": False,
            "stage_a_opaque_historical_reference_count": closure[
                "opaque_historical_reference_count"
            ],
            "stage_a_opaque_historical_reference_count_is_nonzero": True,
            "stage_a_local_closure_module_count": closure[
                "local_closure_module_count"
            ],
            "stage_a_local_closure_content_sha256": closure[
                "local_closure_content_sha256"
            ],
            "stage_a_exact_path_guard_bound": True,
            "stage_a_synthetic_denial_tests_passed": True,
            "stage_a_synthetic_selected_test_node_count": closure[
                "synthetic_selected_test_node_count"
            ],
            "stage_b_required_postrun_successful_protected_opens": 0,
            "stage_b_required_postrun_successful_protected_resolves": 0,
            "stage_b_required_postrun_successful_protected_metadata": 0,
            "stage_b_required_postrun_successful_protected_enumerations": 0,
            "stage_b_required_postrun_quality_hpo_or_promotion": False,
            "external_runtime_ast_closure_receipt": closure["receipt"],
        },
        "exact_lora_collective_geometry": {
            "canonical_fp32_tensor_count": 70,
            "canonical_fp32_element_count": 4_528_128,
            "logical_payload_bytes_per_rank": 18_112_512,
            "logical_payload_bytes_all_four_ranks": 72_450_048,
            "nominal_ring_schedule_bytes_per_rank": 27_168_768,
            "nominal_ring_schedule_is_not_measured_physical_link_bytes": True,
            "legacy_full_weight_23_tensor_geometry_used": False,
            "v71_sampled_pcie_rx_attributed_to_this_collective": False,
        },
        "phase_contract": {
            "nvtx_domain": PHASE_DOMAIN,
            "exact_order": list(PHASES),
            "one_complete_nonoverlapping_range_per_phase": True,
            "same_controller_pid_and_thread_required": True,
            "cuda_event_must_be_wholly_contained_in_one_phase": True,
            "event_crossing_or_outside_phase_ranges": "fail_closed",
            "phase_timestamps_are_not_interchanged_with_worker_rpc_times": True,
            "final_abort_ends_before_post_abort_audit": True,
            "cleanup_has_its_own_range": True,
        },
        "arms": {
            "timeline": {
                "status": "prospectively_sealed_runnable_after_preregistration_commit",
                "launch_authorized_by_this_file_after_identity_checks": True,
                "requires_admin_or_privilege_change": False,
                "command_template": command_template_v73e("timeline"),
                "command_template_sha256": canonical_sha256(
                    command_template_v73e("timeline")
                ),
                "artifacts": timeline_artifacts,
                "measurements": {
                    "cuda_kernel_count_and_duration_by_exact_phase_and_gpu": True,
                    "cuda_memcpy_bytes_count_and_duration_by_exact_phase_and_gpu": True,
                    "nccl_api_duration_by_exact_phase": True,
                    "nccl_gpu_duration_separately_projected": False,
                    "nccl_trace_mode": "legacy_NVTX_API_ranges",
                    "nccl_gpu_work_is_inclusive_in_cuda_kernel_timeline": True,
                    "nccl_api_ranges_grouped_by_exact_pid_text_and_phase": True,
                    "seventy_tensor_call_count_is_reported_not_assumed": True,
                    "explicit_nccl_named_kernel_duration_reported_if_available": True,
                    "pack_flatten_or_materialize_named_kernel_duration_reported_if_available": True,
                    "missing_symbol_or_range_classification": "unavailable_not_inferred",
                    "actual_link_path_or_link_only_time_measured": False,
                    "logical_collective_payload_bytes_from_exact_v72_receipt": True,
                    "physical_nccl_link_bytes_inferred_from_algorithm": False,
                    "cuda_allocation_events_and_peak_by_exact_phase_and_gpu": True,
                    "hbm_or_dram_bandwidth": "not_collected_in_this_arm",
                },
            },
            "hbm_metrics": {
                "status": "blocked_before_launch_by_driver_profiling_permission",
                "launch_authorized_by_this_file_after_identity_checks": False,
                "requires_admin_or_privilege_change": True,
                "observed_driver_parameter": {
                    "path": "/proc/driver/nvidia/params",
                    "field": "RmProfilingAdminOnly",
                    "value": permission,
                },
                "observed_tool_failure_class": "ERR_NVGPUCTRPERM",
                "command_template_after_future_reseal": command_template_v73e(
                    "hbm_metrics"
                ),
                "command_template_sha256": canonical_sha256(
                    command_template_v73e("hbm_metrics")
                ),
                "artifacts": hbm_artifacts,
                "unblock_protocol": [
                    "administrator_changes_profiling_permission_outside_this_agent",
                    "rerun_read_only_metric_availability_probe",
                    "create_new_additive_preregistration_with_exact_metric_set",
                    "retain_separate_unprofiled_V73B_timing_control",
                    "launch_only_after_new_preregistration_is_committed",
                ],
                "current_launcher_behavior": (
                    "reject_before_directory_creation_subprocess_model_ray_or_gpu_work"
                ),
                "utilization_percent_must_not_be_converted_to_hbm_bytes": True,
            },
        },
        "no_semantic_evaluation_contract": {
            "content_free_numeric_panel_and_suffix_reward": True,
            "historical_v73b_panel_reward_or_semantic_authority": False,
            "safe_aggregate_request_prompt_generated_and_suffix_counts_matched": True,
            "semantic_quality_selection_or_hpo": False,
            "protected_dev_ood_or_holdout_opened": False,
            "raw_prompts_questions_answers_or_outputs_persisted": False,
            "target_stdout_and_stderr_redirected_to_devnull_before_workload": True,
            "process_stream_table_must_be_empty": True,
            "trace_payload": (
                "phase_labels_kernel_symbols_cuda_api_memcpy_sizes_allocation_"
                "events_and_nccl_metadata_only"
            ),
            "discard_environment_from_nsys_report": True,
            "os_runtime_file_trace_disabled": True,
            "cpu_and_python_sampling_disabled": True,
            "checkpoint_snapshot_commit_or_promotion_authorized": False,
            "historical_reward_values": "no_authority_not_loaded",
            "current_run_dual_compiler_and_four_actor_consistency_required": True,
        },
        "postrun_acceptance": {
            "all_preregistered_output_paths_fresh_regular_and_non_symlink": True,
            "all_four_physical_gpus_and_exact_actor_pids_in_trace": True,
            "all_16_work_receipts_and_four_waves_exact": True,
            "content_free_current_run_dual_compiler_whole_mapping_exact": True,
            "four_actor_candidate_and_runtime_identity_consensus": True,
            "exact_abort_no_commit_no_protected_access_and_final_idle": True,
            "timeline_sqlite_export_and_nsys_report_both_required": True,
            "missing_required_table_column_phase_or_gpu": "fail_closed",
            "profile_overhead_never_used_for_throughput_promotion": True,
            "top_three_residual_bottlenecks": (
                "rank_only_from_measured_phase_duration_bytes_and_supported_metrics"
            ),
        },
        "overhead_policy": {
            "unprofiled_control": "immutable_accepted_V73B",
            "profiled_wall_or_phase_time": "diagnostic_only",
            "profiled_vs_unprofiled_timing_speedup_claim_authorized": False,
            "profiler_overhead_reported_separately": True,
            "no_counterfactual_subtraction": True,
        },
        "launch_policy": {
            "subprocess_shell_flag": False,
            "fixed_bash_exec_wrapper": True,
            "source_bashrc_before_nsys_exec": True,
            "expanded_argv_sha256_attested_to_target": True,
            "prelaunch_four_gpu_identity_idle_and_no_compute_process_gate": True,
            "prelaunch_gate_recorded_before_nsys_process_creation": True,
            "force_overwrite": False,
            "fresh_output_paths": True,
            "timeline_arm_can_run_only_after_this_artifact_is_sealed": True,
            "hbm_arm_can_run_under_current_contract": False,
            "host_privilege_modification_authorized": False,
        },
    }
    return _self_hash(result)


def render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"


def render_evidence_v73e(
    value: Mapping[str, Any], output: Path, payload: str
) -> str:
    preregistration_file_sha256 = hashlib.sha256(
        payload.encode("ascii")
    ).hexdigest()
    preregistration_content_sha256 = value[
        "content_sha256_before_self_field"
    ]
    launch_argv = [
        str(REQUIRED_PYTHON),
        str(LAUNCHER),
        "--preregistration",
        str(output),
        "--preregistration-sha256",
        preregistration_file_sha256,
        "--preregistration-content-sha256",
        preregistration_content_sha256,
        "--arm",
        "timeline",
        "--execute",
    ]
    expanded_nsys = runtime_contract.expand_command_v73e(
        "timeline",
        preregistration_file_sha256,
        preregistration_content_sha256,
    )
    geometry = value["exact_lora_collective_geometry"]
    topology = value["physical_gpu_topology"]
    boundary = value["systems_only_quarantine_boundary"]
    bootstrap = value["ray_actor_guard_bootstrap"]
    predecessor = value["immutable_v73d_attempt_1_predecessor"]
    timeline = value["arms"]["timeline"]
    hbm = value["arms"]["hbm_metrics"]
    return "\n".join([
        "# Qwen3.6 V73E exact-phase profiler: CPU-only sealing evidence",
        "",
        "Status: prospectively sealed; no V73E model, Ray, CUDA, GPU, "
        "dataset, semantic evaluation, HPO, checkpoint, or promotion run was "
        "performed while producing this evidence.",
        "",
        "## Immutable identities",
        "",
        f"- Preregistration file SHA-256: `{preregistration_file_sha256}`",
        f"- Preregistration content SHA-256: `{preregistration_content_sha256}`",
        "- External systems-only closure content SHA-256: "
        f"`{boundary['external_runtime_ast_closure_receipt']['content_sha256']}`",
        "- Runtime closure module count: "
        f"{boundary['stage_a_local_closure_module_count']}",
        "- Opaque historical reference bindings: "
        f"{boundary['stage_a_opaque_historical_reference_count']} (nonzero; "
        "does not authorize resolution or access)",
        "",
        "## Additive repair and actor bootstrap",
        "",
        "- V73D attempt 1 remains immutable and failed closed during setup "
        "because its inherited adapter verifier tried to open a quarantined "
        "legacy adapter before Ray, model loading, or GPU work.",
        "- Immutable V73D run/profile failure file SHA-256: "
        f"`{predecessor['run_failure']['file_sha256']}` / "
        f"`{predecessor['profile_failure']['file_sha256']}`.",
        "- V73E controller mechanism: "
        f"`{bootstrap['controller_mechanism']}`; actor mechanism: "
        f"`{bootstrap['actor_mechanism']}`. They are separately attributed.",
        "- The Ray job runtime environment injects the exact guard hash before "
        "`ray.init`, while actor-specific runtime environments remain merged.",
        "- The actor extension refuses V72/V71/V41A or the three historical "
        "reference modules if any was imported before its guard install.",
        "- Boundary registry content SHA-256: "
        f"`{bootstrap['boundary_registry']['content_sha256']}`; required "
        "successful protected open/resolve/metadata/enumeration operations: 0.",
        "",
        "## GPU-time accounting",
        "",
        "- Reserved wall GPU-seconds are reported independently from directly "
        "measured model-resident and useful GPU-seconds.",
        "- The legacy allocation/residency estimate is diagnostic and is not "
        "accepted as either direct residency or useful compute evidence.",
        "- Profiled event time is not relabeled as unprofiled useful time; "
        "promotion-charged GPU-seconds remain 0.",
        "",
        "## Measurement contract",
        "",
        f"- Exact NVTX phase count: {len(PHASES)} in domain `{PHASE_DOMAIN}`.",
        "- Timeline arm: unprivileged CUDA/NVTX/legacy-NCCL/kernel/memcpy/"
        "allocation trace, launch-authorized only after fresh-path, exact "
        "toolchain, four-GPU identity, idle, and no-compute-process gates.",
        f"- Timeline command-template SHA-256: `{timeline['command_template_sha256']}`",
        "- HBM/DRAM arm: blocked before directory creation or subprocess; "
        f"`RmProfilingAdminOnly={hbm['observed_driver_parameter']['value']}` "
        f"and `{hbm['observed_tool_failure_class']}`.",
        "- No bandwidth bytes are inferred from utilization percentages; no "
        "NCCL transport/path is inferred from topology capability.",
        "",
        "## Exact LoRA geometry and host topology",
        "",
        f"- FP32 tensors/elements: {geometry['canonical_fp32_tensor_count']} / "
        f"{geometry['canonical_fp32_element_count']:,}.",
        "- Logical collective payload per rank: "
        f"{geometry['logical_payload_bytes_per_rank']:,} B; nominal ring "
        f"schedule {geometry['nominal_ring_schedule_bytes_per_rank']:,} B/rank "
        "(not measured physical-link bytes).",
        "- Four physical GPUs: all cross-pairs report NODE, peer read/write "
        "capability OK, NVLink capability NS, and NUMA affinity 0.",
        "- Topology matrix SHA-256: "
        f"`{topology['normalized_stdout_sha256']}`.",
        "",
        "## Exact future timeline launch (not executed)",
        "",
        "```bash",
        shlex.join(launch_argv),
        "```",
        "",
        "The launcher re-hashes the sealed `.bashrc` and 11 other timeline "
        "runtime inputs before any output directory or profiler subprocess. "
        "Its exact inner Nsight command is:",
        "",
        "```bash",
        shlex.join(expanded_nsys),
        "```",
        "",
        "The HBM arm remains unauthorized. An administrator permission change "
        "would require a new additive preregistration; it does not silently "
        "enable this contract.",
        "",
    ])


def _atomic_write(path: Path, payload: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    evidence = Path(args.evidence).resolve()
    _require(output == OUTPUT, "V73E preregistration path must remain canonical")
    _require(evidence == EVIDENCE, "V73E evidence path must remain canonical")
    value = build_preregistration_v73e()
    payload = render_json(value)
    evidence_payload = render_evidence_v73e(value, output, payload)
    if args.check:
        _require(output.is_file(), f"V73E preregistration absent: {output}")
        _require(
            output.read_text(encoding="ascii") == payload,
            f"V73E preregistration stale: {output}",
        )
        _require(evidence.is_file(), f"V73E evidence absent: {evidence}")
        _require(
            evidence.read_text(encoding="ascii") == evidence_payload,
            f"V73E evidence stale: {evidence}",
        )
    else:
        if output.exists() or evidence.exists():
            raise FileExistsError(output if output.exists() else evidence)
        _atomic_write(output, payload)
        _atomic_write(evidence, evidence_payload)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256(output) if output.is_file() else None,
        "content_sha256": value["content_sha256_before_self_field"],
        "evidence": str(evidence),
        "evidence_file_sha256": (
            file_sha256(evidence) if evidence.is_file() else None
        ),
        "timeline_launch_sealed": True,
        "hbm_arm_blocked": True,
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
