#!/usr/bin/env python3
"""Build the CPU-only sealed V66 Qwen3.6 mirrored calibration contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66.json"
).resolve()
RECIPE_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
TRAIN_DATASET = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/train_v434_fold3_v49d.jsonl"
).resolve()
TRAIN_PANEL = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v52_v434_train_generation_panel.json"
).resolve()
TRAIN_MEMBERSHIP = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v52_v434_train_row_conflict_unit_membership.json"
).resolve()
SOURCE_ADAPTER = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/"
    "v434_equal_r32_seed17_init20260715041/final"
).resolve()
STAGED_ADAPTER = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d"
).resolve()
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
MODEL_SEAL = (
    ROOT / "experiments/eggroll_es_hpo/S6_V23A_INSERTION_MODEL_SEAL.json"
).resolve()
TUNED_TABLE = (
    ROOT / "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c/"
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition.json"
).resolve()
REQUIRED_PYTHON = (ROOT / "es-at-scale/.venv/bin/python").absolute()

DIRECTION_SEEDS_V66 = (
    140002291,
    1028842752,
    480373990,
    1037026679,
    759861149,
    227761095,
    428721957,
    150663570,
)

EXPECTED_FILES = {
    TRAIN_DATASET: "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a",
    TRAIN_PANEL: "dd2c857b75617351d64cfce29f5a8e5d79ce9da212e4db50d22f2de3795c70a1",
    TRAIN_MEMBERSHIP: "e9b073369966e21912a0bda86da501ab0975646df2a7d80bf5675c3dfec8c121",
    SOURCE_ADAPTER / "adapter_model.safetensors": (
        "0d6efd4d5be626f41cdd711843f799b9b3c09e9ecf6a7a8e6e9aeeff09f6dc5b"
    ),
    SOURCE_ADAPTER / "adapter_config.json": (
        "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
    ),
    STAGED_ADAPTER / "adapter_model.safetensors": (
        "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
    ),
    STAGED_ADAPTER / "adapter_config.json": (
        "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
    ),
    STAGED_ADAPTER / "stage_manifest_v44a.json": (
        "e30ba44563b5db56f4a487b26f4e2310fd3755b15f8db69d9400facd8baa3813"
    ),
    MODEL / "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    MODEL / "model.safetensors.index.json": (
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
    ),
    MODEL_SEAL: "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440",
}


def canonical_sha256_v66(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v66(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recipe_contract_v66() -> dict:
    value = json.loads(RECIPE_CONTRACT.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    content_sha = value.get("content_sha256_before_self_field")
    if (
        value.get("schema") != "specialist-recipe-evaluation-compute-contract-v1"
        or value.get("status")
        != "sealed_before_recipe_hpo_protected_holdout_unopened_by_hpo"
        or canonical_sha256_v66(compact) != content_sha
        or value.get("disjointness", {}).get("passed") is not True
        or value.get("roles", {}).get("train", {}).get("file_sha256")
        != EXPECTED_FILES[TRAIN_DATASET]
        or value.get("roles", {}).get("protected_holdout", {}).get(
            "access_authorized_by_this_contract"
        ) is not False
    ):
        raise RuntimeError("v66 global recipe evaluation contract changed")
    return {
        "path": str(RECIPE_CONTRACT),
        "file_sha256": file_sha256_v66(RECIPE_CONTRACT),
        "content_sha256": content_sha,
        "disjointness_passed": True,
        "protected_holdout_access_authorized": False,
    }


def _artifacts_v66() -> dict:
    run = (
        ROOT / "experiments/eggroll_es_hpo/runs/"
        "v66_lora_es_mirrored_crn_qwen36_calibration"
    ).resolve()
    return {
        "attempt": str(
            run.parent
            / ".v66_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
        ),
        "run_directory": str(run),
        "gpu_log": str(run / "gpu_activity_v66.jsonl"),
        "population": str(run / "mirrored_population_v66.json"),
        "update": str(run / "pair_difference_update_v66.json"),
        "report": str(run / "mirrored_calibration_report_v66.json"),
        "failure": str(run / "failure_v66.json"),
    }


def _implementation_bindings_v66() -> dict:
    paths = {
        "builder": Path(__file__).resolve(),
        "runner": ROOT / "run_lora_es_mirrored_calibration_v66.py",
        "mirrored_protocol": ROOT / "eggroll_es_mirrored_v66.py",
        "mirrored_worker": ROOT / "eggroll_es_worker_lora_v66.py",
        "canonical_lora_worker": ROOT / "eggroll_es_worker_lora_v41a.py",
        "distributed_update_worker": ROOT / "eggroll_es_worker_v3.py",
        "runtime_v52": ROOT / "run_lora_es_nested_population_v52.py",
        "runtime_v48b": ROOT / "run_lora_es_generation_boundary_v48b.py",
        "runtime_v40a": ROOT / "run_lora_topology_probe_v40a.py",
        "runtime_v64_model_seal": (
            ROOT / "run_lora_es_v59_vs_v434_robust_confirmation_v64.py"
        ),
        "dense_reward": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
    }
    return {
        key: {"path": str(path.resolve()), "file_sha256": file_sha256_v66(path)}
        for key, path in paths.items()
    }


def build_preregistration_v66() -> dict:
    for path, expected in EXPECTED_FILES.items():
        observed = file_sha256_v66(path)
        if observed != expected:
            raise RuntimeError(
                f"v66 sealed input changed: {path}: {observed} != {expected}"
            )
    tuned_file_sha = file_sha256_v66(TUNED_TABLE)
    recipe_contract = _recipe_contract_v66()
    result = {
        "schema": "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66",
        "status": "sealed_before_v66_train_semantics_model_ray_or_gpu_access",
        "purpose": (
            "Calibrate mirrored pair-difference LoRA ES on Qwen3.6 using only "
            "a registered train panel; execute and exactly abort one nonzero "
            "candidate without checkpointing or promotion."
        ),
        "authorization": {
            "gpu_launch": True,
            "train_panel_semantic_access": True,
            "nonzero_update_execute_then_exact_abort": True,
            "dev_access": False,
            "ood_access": False,
            "protected_holdout_access": False,
            "checkpoint_snapshot": False,
            "candidate_commit": False,
            "promotion": False,
        },
        "global_evaluation_compute_contract": recipe_contract,
        "fixed_recipe": {
            "model": str(MODEL),
            "model_label": "Qwen3.6-35B-A3B",
            "precision": "bfloat16",
            "engines": 4,
            "tensor_parallel_size": 1,
            "worker_extension": (
                "eggroll_es_worker_lora_v66."
                "LoRAAdapterStateWorkerExtensionV66"
            ),
            "direction_seeds": list(DIRECTION_SEEDS_V66),
            "direction_count": 8,
            "signed_population_size": 16,
            "sigma": 0.0006,
            "learning_rate": 0.00015,
            "train_rows_per_candidate": 64,
            "train_conflict_units_per_candidate": 64,
            "evaluation_seed": 2_026_071_702,
            "decode": {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1,
                "prompt_logprobs": 1,
                "detokenize": False,
            },
            "reward": (
                "uniform mean of per-example mean gold-answer token logprob"
            ),
            "candidate_action": "execute_uncommitted_then_exact_abort",
        },
        "common_random_number_contract": {
            "both_signs_share_exact_tokenized_prompt_block": True,
            "both_signs_share_decode_parameters_and_seed": True,
            "both_signs_share_judge_and_aggregation": True,
            "pairs_are_concurrent_within_complete_four_actor_waves": True,
            "signs_rotate_between_physical_ranks": True,
        },
        "train_only_input": {
            "dataset": {
                "path": str(TRAIN_DATASET),
                "file_sha256": EXPECTED_FILES[TRAIN_DATASET],
                "rows": 448,
            },
            "panel": {
                "path": str(TRAIN_PANEL),
                "file_sha256": EXPECTED_FILES[TRAIN_PANEL],
                "content_sha256": (
                    "cdfa9d10669171d5d814b55df1f674a89dfa557c5376b45c8d0073e5d1acaec7"
                ),
                "selected_rows": 64,
                "selected_conflict_units": 64,
            },
            "membership": {
                "path": str(TRAIN_MEMBERSHIP),
                "file_sha256": EXPECTED_FILES[TRAIN_MEMBERSHIP],
                "content_sha256": (
                    "a8870fdce8fbf631b3d3472fd03690f6987590ee6e8758dc8fdcb4556dcc9096"
                ),
            },
            "protected_dev_ood_or_holdout_paths": [],
        },
        "adapter": {
            "source": str(SOURCE_ADAPTER),
            "staged": str(STAGED_ADAPTER),
            "canonical_fp32_master_sha256": (
                "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
            ),
            "runtime_bf16_values_sha256": (
                "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
            ),
            "source_weights_sha256": EXPECTED_FILES[
                SOURCE_ADAPTER / "adapter_model.safetensors"
            ],
            "source_config_sha256": EXPECTED_FILES[
                SOURCE_ADAPTER / "adapter_config.json"
            ],
            "staged_weights_sha256": EXPECTED_FILES[
                STAGED_ADAPTER / "adapter_model.safetensors"
            ],
        },
        "runtime": {
            "required_python": str(REQUIRED_PYTHON),
            "exclusive_physical_gpus": [0, 1, 2, 3],
            "four_tp1_engines": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.82,
            "max_model_len": 2048,
            "tuned_table": str(TUNED_TABLE),
            "tuned_table_file_sha256": tuned_file_sha,
            "tuned_table_content_sha256": (
                "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
            ),
            "base_model_seal": {
                "path": str(MODEL_SEAL),
                "file_sha256": EXPECTED_FILES[MODEL_SEAL],
                "content_sha256": (
                    "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
                ),
                "all_model_files_verified_before_and_after": True,
            },
        },
        "acceptance": {
            "nonzero_pair_difference_count_minimum": 1,
            "nonzero_coefficient_l2_required": True,
            "candidate_master_must_differ": True,
            "candidate_bf16_runtime_must_differ": True,
            "all_four_replicas_candidate_consensus": True,
            "all_four_physical_gpus_positive_in_every_wave": True,
            "all_wave_restores_exact": True,
            "update_exactly_aborted_without_commit": True,
            "final_master_and_runtime_equal_initial": True,
            "final_gpu_idle": True,
        },
        "artifacts": _artifacts_v66(),
        "implementation_bindings": _implementation_bindings_v66(),
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v66(result)
    return result


def write_preregistration_v66(path: Path = OUTPUT) -> dict:
    path = Path(path).resolve()
    value = build_preregistration_v66()
    payload = (json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration_v66()
    output = Path(args.output).resolve()
    if args.check:
        if not output.is_file() or json.loads(
            output.read_text(encoding="utf-8")
        ) != value:
            raise RuntimeError("v66 preregistration is absent or stale")
    else:
        if output.exists():
            raise FileExistsError(output)
        write_preregistration_v66(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v66(output) if output.is_file() else None,
        "content_sha256": value["content_sha256_before_self_field"],
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
