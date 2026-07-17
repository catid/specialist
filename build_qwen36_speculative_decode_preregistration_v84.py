#!/usr/bin/env python3
"""Build/check the prospective Qwen3.6 speculative-decode V84 contract.

This builder is deliberately CPU/source-only.  It reads pinned configuration,
metadata, and installed source files; it never imports torch or vLLM, opens a
dataset, or launches a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VLLM_ROOT = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
VLLM_METADATA = (
    ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/vllm-0.25.0.dist-info/METADATA"
)
V78_PATH = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_text_only_checkpoint_v78.json"
)
BF16_ROOT = ROOT / "models/Qwen3.6-35B-A3B"
FP8_ROOT = ROOT / "models/Qwen3.6-35B-A3B-FP8"
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_exact_speculative_decode_v84.json"
)
REPORT = (
    ROOT
    / "experiments/eggroll_es_hpo/"
    "qwen36_exact_speculative_decode_v84_cpu_evidence_20260717.md"
)
SCHEMA = "qwen36-exact-speculative-decode-preregistration-v84"

PINNED_FILES = {
    "vllm/config/speculative.py": (
        VLLM_ROOT / "config/speculative.py",
        "3f1abd1ca3042fba239e7bf98b08f645f3e950c16ab510fbc99a49c5c507721f",
    ),
    "vllm/model_executor/models/qwen3_5_mtp.py": (
        VLLM_ROOT / "model_executor/models/qwen3_5_mtp.py",
        "9b36e5bfcee4faf8d04319e069032c3c4a01c4aaf49f86108eca788038c0c7fd",
    ),
    "vllm/v1/spec_decode/ngram_proposer.py": (
        VLLM_ROOT / "v1/spec_decode/ngram_proposer.py",
        "5051e1bca645fe4befec0a6340f0dfc169fbcbe070ac1da508b72f8ee9c91736",
    ),
    "vllm/lora/punica_wrapper/punica_gpu.py": (
        VLLM_ROOT / "lora/punica_wrapper/punica_gpu.py",
        "d0c8ce69191d733d479e50399bfc86d6b89a55915dda4dd0cf32b8921e32bf9d",
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        VLLM_ROOT / "v1/worker/gpu_model_runner.py",
        "6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e",
    ),
    "vllm-0.25.0.dist-info/METADATA": (
        VLLM_METADATA,
        "8a317380a72e2a4e58f80188aacf0dca83c5ac2e1995063964b7eac27b7bab24",
    ),
    "models/Qwen3.6-35B-A3B/config.json": (
        BF16_ROOT / "config.json",
        "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    ),
    "models/Qwen3.6-35B-A3B/model.safetensors.index.json": (
        BF16_ROOT / "model.safetensors.index.json",
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    ),
    "models/Qwen3.6-35B-A3B-FP8/config.json": (
        FP8_ROOT / "config.json",
        "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845",
    ),
    "models/Qwen3.6-35B-A3B-FP8/model.safetensors.index.json": (
        FP8_ROOT / "model.safetensors.index.json",
        "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6",
    ),
    "qwen36_text_only_checkpoint_v78.json": (
        V78_PATH,
        "6666f6e485d4ae12796d1731af34131a717e1d76497cc49a028281628204ec4e",
    ),
}


def canonical_sha256_v84(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v84(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_json_v84(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _require_v84(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json_v84(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require_v84(isinstance(value, dict), f"expected object: {path}")
    return value


def _verify_source_v84() -> dict:
    hashes = {}
    for label, (path, expected) in PINNED_FILES.items():
        actual = file_sha256_v84(path)
        _require_v84(actual == expected, f"pinned source changed: {label}")
        hashes[label] = actual

    metadata = VLLM_METADATA.read_text(encoding="utf-8")
    _require_v84("Version: 0.25.0" in metadata, "vLLM version changed")

    speculative = (VLLM_ROOT / "config/speculative.py").read_text(encoding="utf-8")
    qwen_mtp = (
        VLLM_ROOT / "model_executor/models/qwen3_5_mtp.py"
    ).read_text(encoding="utf-8")
    ngram = (
        VLLM_ROOT / "v1/spec_decode/ngram_proposer.py"
    ).read_text(encoding="utf-8")
    punica = (
        VLLM_ROOT / "lora/punica_wrapper/punica_gpu.py"
    ).read_text(encoding="utf-8")
    runner = (VLLM_ROOT / "v1/worker/gpu_model_runner.py").read_text(
        encoding="utf-8"
    )

    required_tokens = {
        "speculative.py": (
            '"qwen3_5_mtp"',
            '"Qwen3_5MoeMTP"',
            'self.method = "ngram"',
            'self.method = "mtp"',
            "multiple times of forward on same MTP layer",
        ),
        "qwen3_5_mtp.py": (
            "class Qwen3_5MTP",
            "class Qwen3_5MoeMTP",
            "mamba_cache_mode == \"all\"",
            "--mamba-cache-mode=align",
        ),
        "ngram_proposer.py": (
            "class NgramProposer",
            "No model to load.",
            "valid_ngram_draft",
        ),
        "punica_gpu.py": (
            "num_speculative_decoding_tokens + 1",
            "max_num_samples",
        ),
        "gpu_model_runner.py": (
            "NgramProposer",
            "scheduled_spec_decode_tokens",
            "set_active_loras",
        ),
    }
    bodies = {
        "speculative.py": speculative,
        "qwen3_5_mtp.py": qwen_mtp,
        "ngram_proposer.py": ngram,
        "punica_gpu.py": punica,
        "gpu_model_runner.py": runner,
    }
    for filename, tokens in required_tokens.items():
        for token in tokens:
            _require_v84(token in bodies[filename], f"source token changed: {filename}")

    return {
        "vllm_version": "0.25.0",
        "file_sha256": hashes,
        "qwen3_5_moe_mtp_class_present": True,
        "target_checkpoint_mtp_autodetection_present": True,
        "model_free_cpu_ngram_proposer_present": True,
        "punica_lora_allocates_for_speculative_sample_rows": True,
        "source_presence_is_not_live_compatibility_or_speed_evidence": True,
    }


def _verify_checkpoint_v84() -> dict:
    v78 = _load_json_v84(V78_PATH)
    v78_body = copy.deepcopy(v78)
    claimed = v78_body.pop("content_sha256_before_self_field")
    _require_v84(
        claimed == canonical_sha256_v84(v78_body), "V78 canonical hash changed"
    )
    _require_v84(
        claimed == "7164f1ba8f581f5e6253ee89aead3058366526fbee24913501c2aa9aec6ec121",
        "V78 identity changed",
    )

    bf16_config = _load_json_v84(BF16_ROOT / "config.json")
    fp8_config = _load_json_v84(FP8_ROOT / "config.json")
    for label, config in (("bf16", bf16_config), ("fp8", fp8_config)):
        _require_v84(config.get("model_type") == "qwen3_5_moe", f"{label} model")
        text = config.get("text_config", {})
        _require_v84(text.get("model_type") == "qwen3_5_moe_text", f"{label} text")
        _require_v84(text.get("mtp_num_hidden_layers") == 1, f"{label} MTP layers")
        _require_v84(text.get("num_hidden_layers") == 40, f"{label} target layers")

    categories = v78["checkpoints"]
    bf16_mtp = categories["bf16"]["categories"]["mtp"]
    fp8_mtp = categories["fp8_serialized"]["categories"]["mtp"]
    _require_v84(
        bf16_mtp["tensor_count"] == 19
        and bf16_mtp["logical_bytes"] == 1_689_281_536,
        "BF16 MTP inventory changed",
    )
    _require_v84(
        fp8_mtp["tensor_count"] == 1560
        and fp8_mtp["logical_bytes"] == 853_668_480,
        "FP8 MTP inventory changed",
    )
    fp8_mtp_file = FP8_ROOT / "mtp.safetensors"
    _require_v84(
        fp8_mtp_file.is_file() and fp8_mtp_file.stat().st_size == 853_860_608,
        "FP8 MTP file changed",
    )
    live = v78["cpu_source_conclusion"]
    _require_v84(
        live["observed_fp8_mtp_named_parameter_count_per_actor"] == 0,
        "baseline MTP residency changed",
    )

    return {
        "target_architecture": "Qwen3_5MoeForConditionalGeneration",
        "target_layer_count": 40,
        "checkpoint_native_mtp_layer_count": 1,
        "baseline_live_mtp_named_parameter_count_per_actor": 0,
        "bf16_mtp": {
            "tensor_count": 19,
            "logical_bytes": 1_689_281_536,
            "manifest_sha256": bf16_mtp["manifest_sha256"],
        },
        "serialized_fp8_mtp": {
            "tensor_count_including_quantization_metadata": 1560,
            "logical_bytes": 853_668_480,
            "physical_file_bytes": 853_860_608,
            "manifest_sha256": fp8_mtp["manifest_sha256"],
        },
        "incremental_live_vram_must_be_measured_not_assumed_from_checkpoint_bytes": True,
    }


def build_preregistration_v84() -> dict:
    source = _verify_source_v84()
    checkpoint = _verify_checkpoint_v84()
    value = {
        "schema": SCHEMA,
        "status": "source_cpu_preregistered_live_launch_not_authorized",
        "bead": "specialist-0j5.38",
        "model_family": "Qwen3.6-35B-A3B",
        "hypothesis": (
            "For multi-token ES generation, accepted draft tokens can reduce target "
            "forward passes and frozen-weight reads enough to improve rollout throughput."
        ),
        "installed_support_evidence": source,
        "checkpoint_mtp_inventory": checkpoint,
        "applicability_gate": {
            "v73b_v73d_max_tokens": 1,
            "v73b_v73d_requests_per_candidate": 64,
            "v73b_v73d_cannot_measure_multistep_acceptance": True,
            "production_hpo_cli_default_max_tokens": 32,
            "close_not_applicable_if_final_training_uses_one_token_teacher_forced_scoring": True,
        },
        "prospective_systems_arms": [
            {
                "arm_id": "target_only_control",
                "speculative_config": None,
                "num_speculative_tokens": 0,
                "incremental_draft_checkpoint_bytes": 0,
            },
            {
                "arm_id": "cpu_ngram_k4_n2to5",
                "speculative_config": {
                    "method": "ngram",
                    "num_speculative_tokens": 4,
                    "prompt_lookup_min": 2,
                    "prompt_lookup_max": 5,
                },
                "incremental_draft_checkpoint_bytes": 0,
                "neural_draft_model_loaded": False,
            },
            {
                "arm_id": "checkpoint_mtp_k1",
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                },
                "mamba_cache_mode": "align",
                "checkpoint_native_mtp_layer_replays_per_step": 1,
            },
            {
                "arm_id": "checkpoint_mtp_k3",
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 3,
                },
                "mamba_cache_mode": "align",
                "checkpoint_native_mtp_layer_replays_per_step": 3,
                "source_warns_acceptance_may_fall": True,
            },
        ],
        "explicitly_rejected_in_first_screen": [
            {
                "arm": "external_neural_draft_model",
                "reason": "adds unbounded model residency before native and model-free arms are measured",
            },
            {
                "arm": "ngram_gpu",
                "reason": "CPU ngram establishes acceptance before adding a GPU proposer and new residency",
            },
            {
                "arm": "semantic_or_protected_data_during_systems_screen",
                "reason": "systems acceptance uses a deterministic synthetic token-ID panel only",
            },
        ],
        "common_systems_workload": {
            "source": "deterministic_content_free_token_id_panel",
            "prompt_count": 64,
            "max_tokens": 32,
            "min_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "detokenize": False,
            "same_prompt_tokens_and_candidate_state_for_every_arm": True,
            "no_prompt_answer_or_semantic_output_persisted": True,
            "persist_only_token_hashes_and_aggregate_systems_metrics": True,
        },
        "candidate_isolation": {
            "exact_lora_candidate_identity_before_every_request": True,
            "new_request_ids_after_every_candidate_transition": True,
            "prefix_cache_disabled_or_reset_after_every_in_place_adapter_change": True,
            "draft_and_target_kv_or_mamba_state_never_cross_candidate_identity": True,
            "stale_adapter_or_cache_detection_fails_closed": True,
            "silent_non_speculative_fallback_fails_closed": True,
            "exact_master_restore_and_terminal_poison_semantics_unchanged": True,
        },
        "measurement": {
            "counterbalanced_pairs_per_arm_min": 3,
            "all_physical_gpu_ids_exact": [0, 1, 2, 3],
            "accepted_draft_tokens_and_acceptance_fraction": True,
            "target_forward_pass_count": True,
            "target_forwards_per_accepted_output_token": True,
            "steady_and_end_to_end_tokens_per_second": True,
            "rollouts_per_gpu_second": True,
            "incremental_named_parameter_and_allocator_vram_bytes": True,
            "kv_and_mamba_cache_capacity_displacement_bytes": True,
            "nsys_kernel_and_target_phase_timeline": True,
            "exact_hbm_bytes_require_privileged_counter_or_equivalent_primary_measurement": True,
            "utilization_percent_is_not_hbm_bytes": True,
            "profiler_run_cannot_establish_throughput": True,
            "unprofiled_timing_control_required": True,
            "cleanup_idle_and_no_foreign_pid_required": True,
        },
        "systems_selection": {
            "output_token_hashes_must_match_target_only_under_greedy_contract": True,
            "point_end_to_end_rollouts_per_gpu_second_must_exceed_control": True,
            "accepted_draft_tokens_must_be_nonzero": True,
            "target_forward_passes_per_output_token_must_fall": True,
            "no_pareto_gain_closes_task_not_applicable": True,
            "systems_winner_is_not_quality_or_training_promotion": True,
        },
        "later_quality_gate": {
            "source_disjoint_multi_item_qa_dev_required": True,
            "one_shot_protected_ood_only_after_recipe_freeze": True,
            "exact_same_decode_limits_and_stop_policy": True,
            "validation_and_ood_noninferiority_required": True,
            "promotion_default": False,
        },
        "dependencies": {
            "live_launch_waits_for": [
                "specialist-0j5.15",
                "specialist-0j5.17",
                "specialist-0j5.32",
            ],
            "v73d_only_validates_profiler_and_isolation_mechanics": True,
            "own_32_token_systems_contract_required_before_gpu_launch": True,
        },
        "authority": {
            "cpu_source_and_metadata_inspection_only": True,
            "torch_vllm_or_ray_imported_by_builder": False,
            "gpu_or_model_launch_performed": False,
            "dataset_prompt_protected_or_ood_rows_opened": False,
            "model_adapter_checkpoint_or_site_package_modified": False,
            "live_launch_authorized": False,
            "hpo_quality_or_promotion_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v84(value)
    return value


def render_report_v84(value: dict) -> str:
    inv = value["checkpoint_mtp_inventory"]
    lines = [
        "# Qwen3.6 exact speculative-decode V84 CPU evidence",
        "",
        f"- Contract: `{value['content_sha256_before_self_field']}`",
        "- Installed vLLM 0.25.0 explicitly supports `Qwen3_5MoeMTP` and a model-free CPU n-gram proposer.",
        f"- The checkpoint declares {inv['checkpoint_native_mtp_layer_count']} MTP layer; baseline live actors load zero MTP named parameters.",
        f"- Enabling MTP may add up to {inv['bf16_mtp']['logical_bytes']:,} BF16 or {inv['serialized_fp8_mtp']['logical_bytes']:,} serialized-FP8 logical bytes before allocator/cache effects.",
        "- V73D is a one-token workload, so it cannot measure multi-step speculative acceptance; the future systems screen is an additive synthetic 32-token contract.",
        "- No GPU/model/data/protected access or training/promotion occurred. Live work remains unauthorized.",
        "",
    ]
    return "\n".join(lines)


def check_artifacts_v84() -> dict:
    expected = build_preregistration_v84()
    _require_v84(_load_json_v84(OUTPUT) == expected, "V84 JSON artifact is stale")
    _require_v84(
        REPORT.read_text(encoding="utf-8") == render_report_v84(expected),
        "V84 report artifact is stale",
    )
    return expected


def write_artifacts_v84() -> dict:
    value = build_preregistration_v84()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_json_v84(value), encoding="ascii")
    REPORT.write_text(render_report_v84(value), encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _require_v84(not (args.write and args.check), "choose --write or --check")
    if args.write:
        value = write_artifacts_v84()
    elif args.check:
        value = check_artifacts_v84()
    else:
        value = build_preregistration_v84()
    print(value["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
