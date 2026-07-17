#!/usr/bin/env python3
"""Immutable runtime-only contract surface for the V73E systems trace.

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
    "qwen36_v73e_exact_phase_profiler.json"
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
    ROOT / "run_lora_es_v71_v72_profile_calibration_v73e.py"
).resolve()
LAUNCHER = (ROOT / "run_qwen36_v73e_exact_phase_profiler.py").resolve()
GUARD_DIRECTORY = (ROOT / "v73e_sitecustomize").resolve()
GUARD = (GUARD_DIRECTORY / "v73e_path_open_guard.py").resolve()
SITECUSTOMIZE = (GUARD_DIRECTORY / "sitecustomize.py").resolve()
WORKER = (ROOT / "eggroll_es_worker_lora_v73e.py").resolve()
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_v73e.LoRAAdapterStateWorkerExtensionV73E"
)
ACTOR_BOOTSTRAP_ENV = "SPECIALIST_V73E_ACTOR_BOOTSTRAP"
ACTOR_GUARD_SHA_ENV = "SPECIALIST_V73E_ACTOR_GUARD_SHA256"
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
STAGED_ADAPTER_DIRECTORY = (
    ROOT
    / "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d"
).resolve()
STAGED_ADAPTER_WEIGHTS = (
    STAGED_ADAPTER_DIRECTORY / "adapter_model.safetensors"
).resolve()
STAGED_ADAPTER_CONFIG = (
    STAGED_ADAPTER_DIRECTORY / "adapter_config.json"
).resolve()
STAGED_ADAPTER_MANIFEST = (
    STAGED_ADAPTER_DIRECTORY / "stage_manifest_v44a.json"
).resolve()
STAGED_TRANSFORM_IMPLEMENTATION = (
    ROOT / "stage_candidate_adapters_vllm_v44a.py"
).resolve()
STAGED_ADAPTER_WEIGHTS_SHA256 = (
    "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
)
STAGED_ADAPTER_CONFIG_SHA256 = (
    "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
)
STAGED_ADAPTER_MANIFEST_FILE_SHA256 = (
    "e30ba44563b5db56f4a487b26f4e2310fd3755b15f8db69d9400facd8baa3813"
)
STAGED_ADAPTER_MANIFEST_CONTENT_SHA256 = (
    "ea328ada018e1c0d182d329d2a9cb81f8f0375aef93738f3b9c0a00f63c82da3"
)
STAGED_TRANSFORM_IMPLEMENTATION_SHA256 = (
    "8559cb3a2c2870bed7e5ef99ee9ed177929e4c2d8680b61cccc73aad26a2f691"
)
STAGED_TARGET_PREFIX = "base_model.model.model.language_model.layers."
CANONICAL_SOURCE_PREFIX = "base_model.model.model.layers."
STAGED_TENSOR_COUNT = 70
STAGED_ELEMENT_COUNT = 4_528_128
STAGED_TENSOR_BYTES = 18_112_512
STAGED_TRANSFORM_IDENTITY_SHA256 = (
    "f210bf05e7fe38481d0a7d9c641a7f902e575521b50e98bdc021bf11b49cb1c8"
)
STAGED_ORDERED_VALUE_SEQUENCE_SHA256 = (
    "26daf52fac11a584891f745e9682c4409ff4aee3119814f0a083a91a192bdf45"
)
STAGED_TARGET_KEY_INVENTORY_SHA256 = (
    "f3f7988e8dfe2c88bf0be49160a2c0cd6beaa69a30ba3e5bfff75a5cab1084a9"
)
CANONICAL_SOURCE_KEY_INVENTORY_SHA256 = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)
INVERSE_KEY_MAPPING_SHA256 = (
    "94fc09b82d8486e96ab0cce2310f20fbf3aa21444d372b5c2ac21367d4dbd872"
)
CANONICAL_MASTER_SHA256 = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)
CANONICAL_RUNTIME_VALUES_SHA256 = (
    "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
)
HISTORICAL_PROTECTED_SOURCE_WEIGHTS_SHA256 = (
    "0d6efd4d5be626f41cdd711843f799b9b3c09e9ecf6a7a8e6e9aeeff09f6dc5b"
)
CONTENT_FREE_TOKEN_PANEL_SEED = (
    "specialist-v73e-content-free-token-id-panel-v1"
)
CONTENT_FREE_TOKEN_PANEL_ROWS = 64
CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE = 3_174
CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE = 1_649
CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE = 4_823
CONTENT_FREE_GENERATED_TOKENS_PER_CANDIDATE = 64
CONTENT_FREE_TOKEN_ID_MINIMUM = 1_000
CONTENT_FREE_TOKEN_ID_SPAN = 65_536
CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256 = (
    "0a6d2203b154f03fa5d563023cb36a847a17356253dd9031a122ad38f29b2874"
)
CONTENT_FREE_REQUEST_BLOCK_SHA256 = (
    "a981483938de2078bf84b497ca3982a544b9647c69fa4fbb38209cb3052f354c"
)
CONTENT_FREE_RAW_TOKEN_INVENTORY_SHA256 = (
    "d13a79bd17f879ee5973344579c6a7334c82ee244ea067879139a408c71f7cd4"
)
CONTENT_FREE_LENGTH_MANIFEST_SHA256 = (
    "6d13ea367936376333a6dd16cf39df6f93c1f1d059958cac3822c69bf0aaaedf"
)
CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256 = (
    "e1b09444be8893454e0c5e1ddbe9dc1c026cc1cb1604da6e40d3bb8c56924887"
)
CONTENT_FREE_DIRECTION_SEEDS = (
    140002291,
    1028842752,
    480373990,
    1037026679,
    759861149,
    227761095,
    428721957,
    150663570,
)
CONTENT_FREE_SIGMA = 0.0006
CONTENT_FREE_LEARNING_RATE = 0.00015
CONTENT_FREE_TP1_TUNED_TABLE_CONTENT_SHA256 = (
    "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
)
CONTENT_FREE_EVALUATION_CONTRACT_SHA256 = (
    "134d8aecc28cb52befff5dc8a9ec5bd447742207bd251fe28a1fa41608eae58d"
)
CONTENT_FREE_PLAN_SHA256 = (
    "98bcd2d0a2c0e2f532b96751e5acabf496627cc3a87c1fd7f12a60adcb3c38c0"
)

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


def content_free_token_panel_v73e() -> dict[str, Any]:
    """Build the deterministic, text-free systems-only teacher-force panel."""
    records = []
    for request_index in range(CONTENT_FREE_TOKEN_PANEL_ROWS):
        prefix_token_count = 49 + int(
            ((request_index * 17 + 11) % 64) < 38
        )
        answer_token_count = 25 + int(
            ((request_index * 29 + 7) % 64) < 49
        )
        answer_token_start = prefix_token_count
        total_token_count = prefix_token_count + answer_token_count
        token_ids = []
        for position in range(total_token_count):
            payload = (
                f"{CONTENT_FREE_TOKEN_PANEL_SEED}|"
                f"request={request_index:02d}|position={position:03d}"
            ).encode("ascii")
            value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            token_ids.append(
                CONTENT_FREE_TOKEN_ID_MINIMUM
                + value % CONTENT_FREE_TOKEN_ID_SPAN
            )
        records.append({
            "request_index": request_index,
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": canonical_sha256(token_ids),
            "unscored_prefix_token_count": prefix_token_count,
            "answer_token_start": answer_token_start,
            "answer_token_count": answer_token_count,
            "total_token_count": total_token_count,
            "text_question_answer_or_semantic_label_present": False,
        })
    body = {
        "schema": "qwen36-v73e-content-free-token-id-panel-v1",
        "status": "deterministic_systems_only_no_semantic_authority",
        "generator": (
            "sha256(seed|request=NN|position=NNN)-u64be-mod-span"
        ),
        "seed": CONTENT_FREE_TOKEN_PANEL_SEED,
        "request_count": CONTENT_FREE_TOKEN_PANEL_ROWS,
        "unscored_prefix_tokens_per_candidate": (
            CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE
        ),
        "total_tokens_per_candidate": CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE,
        "answer_tokens_per_candidate": (
            CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
        ),
        "token_id_minimum": CONTENT_FREE_TOKEN_ID_MINIMUM,
        "token_id_exclusive_maximum": (
            CONTENT_FREE_TOKEN_ID_MINIMUM + CONTENT_FREE_TOKEN_ID_SPAN
        ),
        "historical_safe_aggregate_answer_tokens_per_candidate": 1_649,
        "historical_safe_aggregate_full_input_tokens_per_candidate": 4_823,
        "generated_tokens_per_candidate": (
            CONTENT_FREE_GENERATED_TOKENS_PER_CANDIDATE
        ),
        "synthetic_per_request_unscored_prefix_length_distribution": {
            "50_tokens": 38,
            "49_tokens": 26,
        },
        "synthetic_per_request_answer_length_distribution": {
            "26_tokens": 49,
            "25_tokens": 15,
        },
        "synthetic_per_request_total_length_distribution": {
            "76_tokens": 28,
            "75_tokens": 31,
            "74_tokens": 5,
        },
        "historical_per_request_token_lengths_reused": False,
        "qa_dev_or_other_semantic_dataset_used": False,
        "historical_same_live_semantic_authority_inherited": False,
        "records": records,
    }
    body["content_sha256_before_self_field"] = canonical_sha256(body)
    return body


def content_free_suffix_reward_config_v73e() -> dict[str, Any]:
    return {
        "schema": "qwen36-v73e-content-free-suffix-logprob-config-v1",
        "objective": "mean_of_per_request_mean_scored_suffix_token_logprob",
        "request_count": CONTENT_FREE_TOKEN_PANEL_ROWS,
        "scored_suffix_tokens_per_candidate": (
            CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
        ),
        "full_input_tokens_per_candidate": (
            CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE
        ),
        "eos_scored": False,
        "token_ids_are_deterministic_numeric_systems_inputs": True,
        "question_answer_gold_or_semantic_label_present": False,
        "historical_reward_identity_or_quality_authority": False,
    }


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
    run = (ROOT / "experiments/eggroll_es_hpo/runs" / stem).resolve()
    profile = (ROOT / "experiments/eggroll_es_hpo/profiles" / stem).resolve()
    trace_base = profile / stem
    return {
        "application_attempt": str(run.parent / f".{stem}.attempt.json"),
        "run_directory": str(run),
        "gpu_log": str(run / "gpu_activity_v73e.jsonl"),
        "actor_cuda_work_log": str(run / "actor_cuda_work_receipts_v73e.jsonl"),
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
        "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD=1",
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


def validate_generated_preregistration_v73e(
    value: Mapping[str, Any],
) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256(body),
        "V73E preregistration self hash changed",
    )
    _require(
        value.get("schema")
        == "qwen36-v73e-exact-phase-profiler-preregistration-v1"
        and value.get("status")
        == "sealed_cpu_only_before_v73e_model_ray_gpu_or_protected_access"
        and value.get("phase_contract", {}).get("exact_order") == list(PHASES)
        and value.get("phase_contract", {}).get("nvtx_domain") == PHASE_DOMAIN
        and value.get("arms", {}).get("timeline", {}).get("artifacts")
        == arm_artifacts_v73e("timeline")
        and value.get("arms", {}).get("hbm_metrics", {}).get("artifacts")
        == arm_artifacts_v73e("hbm_metrics")
        and value.get("arms", {}).get("timeline", {}).get("command_template")
        == command_template_v73e("timeline")
        and value.get("arms", {}).get("hbm_metrics", {}).get(
            "command_template_after_future_reseal"
        )
        == command_template_v73e("hbm_metrics")
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
        and value.get("no_semantic_evaluation_contract", {}).get(
            "content_free_numeric_panel_and_suffix_reward"
        )
        is True
        and value.get("no_semantic_evaluation_contract", {}).get(
            "historical_v73b_panel_reward_or_semantic_authority"
        )
        is False
        and value.get("launch_policy", {}).get(
            "hbm_arm_can_run_under_current_contract"
        )
        is False,
        "V73E generated runtime contract changed",
    )
    bindings = value.get("implementation_bindings", {})
    boundary = value.get("systems_only_quarantine_boundary", {})
    bootstrap = value.get("ray_actor_guard_bootstrap", {})
    predecessor = value.get("immutable_v73d_attempt_1_predecessor", {})
    staged = value.get("staged_only_adapter_contract", {})
    workload = value.get("content_free_systems_workload", {})
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
        "V73E runtime closure or systems-only quarantine boundary changed",
    )
    _require(
        staged.get("status") == "sealed_staged_only_exact_inverse_required"
        and staged.get("files", {}).get("weights")
        == source_identity(STAGED_ADAPTER_WEIGHTS)
        and staged.get("files", {}).get("config")
        == source_identity(STAGED_ADAPTER_CONFIG)
        and staged.get("files", {}).get("manifest")
        == source_identity(STAGED_ADAPTER_MANIFEST)
        and staged.get("files", {}).get("transform_implementation")
        == source_identity(STAGED_TRANSFORM_IMPLEMENTATION)
        and staged.get("source_tensor_namespace") == STAGED_TARGET_PREFIX
        and staged.get("canonical_tensor_namespace") == CANONICAL_SOURCE_PREFIX
        and staged.get("tensor_count") == STAGED_TENSOR_COUNT
        and staged.get("element_count") == STAGED_ELEMENT_COUNT
        and staged.get("tensor_bytes") == STAGED_TENSOR_BYTES
        and staged.get("exact_inverse_key_mapping_sha256")
        == INVERSE_KEY_MAPPING_SHA256
        and staged.get("staged_transform_identity_sha256")
        == STAGED_TRANSFORM_IDENTITY_SHA256
        and staged.get("canonical_master_sha256") == CANONICAL_MASTER_SHA256
        and staged.get("canonical_runtime_values_sha256")
        == CANONICAL_RUNTIME_VALUES_SHA256
        and staged.get("quarantined_legacy_adapter_open_stat_hash_or_resolve")
        is False,
        "V73E staged-only adapter contract changed",
    )
    _require(
        workload.get("status")
        == "deterministic_numeric_token_ids_no_semantic_authority"
        and workload.get("token_panel_content_sha256")
        == CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        and workload.get("request_block_sha256")
        == CONTENT_FREE_REQUEST_BLOCK_SHA256
        and workload.get("raw_token_inventory_sha256")
        == CONTENT_FREE_RAW_TOKEN_INVENTORY_SHA256
        and workload.get("length_manifest_sha256")
        == CONTENT_FREE_LENGTH_MANIFEST_SHA256
        and workload.get("suffix_reward_config_sha256")
        == CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256
        and workload.get("evaluation_contract_sha256")
        == CONTENT_FREE_EVALUATION_CONTRACT_SHA256
        and workload.get("mirrored_plan_sha256") == CONTENT_FREE_PLAN_SHA256
        and workload.get("request_count") == CONTENT_FREE_TOKEN_PANEL_ROWS
        and workload.get("unscored_prefix_tokens_per_candidate")
        == CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE
        and workload.get("scored_suffix_tokens_per_candidate")
        == CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
        and workload.get("full_input_tokens_per_candidate")
        == CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE
        and workload.get("generated_tokens_per_candidate")
        == CONTENT_FREE_GENERATED_TOKENS_PER_CANDIDATE
        and workload.get("direction_seeds") == list(CONTENT_FREE_DIRECTION_SEEDS)
        and workload.get("sigma") == CONTENT_FREE_SIGMA
        and workload.get("learning_rate") == CONTENT_FREE_LEARNING_RATE
        and workload.get("historical_v73b_workload_control_or_semantic_authority")
        is False
        and workload.get("qa_dev_ood_holdout_or_other_semantic_dataset_used")
        is False,
        "V73E content-free workload contract changed",
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
        == "immutable_failed_closed_legacy_adapter_quarantine_incompatibility"
        and predecessor.get("profile_attempt", {}).get("file_sha256")
        == "82937bea1a3701e0125e8f21286c1a6ee8b70d379fa13d5c69d6ce7d75ae2c67"
        and predecessor.get("profile_failure", {}).get("file_sha256")
        == "025a38adce724d931a644dd090f6eb5b347f0d1ebd7bcc500a149af67f22bd50"
        and predecessor.get("run_attempt", {}).get("file_sha256")
        == "4eb3993f06db7fb1f8bd4f4eddf4b84b0930b06780110caab845971027a51413"
        and predecessor.get("run_failure", {}).get("file_sha256")
        == "08c0a3e2ca3832486e6b94bfd4ad0529c8420d3537574c7f2dda6ad4a8ed3886"
        and predecessor.get("nsys_report", {}).get("file_sha256")
        == "94fd7fc159fd263142557042d8821fb8ef1df18265826298bb19bd0837cd5cf4"
        and predecessor.get("sqlite_export", {}).get("file_sha256")
        == "3d3dc120216a81fc1ec1ed59286df116f5535deda3d6be1d4f05e466d28f2ed7"
        and predecessor.get("run_failure", {}).get("content_sha256")
        == "d57e87c1a589bbd344d334aeb37b242357261dc18a602cb2cefd1be1af9de30f"
        and predecessor.get("profile_failure", {}).get("content_sha256")
        == "0335438e9ac89680f9e78800fcfdcd0a1cb31a661f9255e935bd79add8f18d07",
        "V73E actor bootstrap or V73D predecessor binding changed",
    )
