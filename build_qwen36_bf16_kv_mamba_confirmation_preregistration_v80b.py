#!/usr/bin/env python3
"""Build the additive, prospective V80B r2/r3 confirmation contract.

This is deliberately a CPU/file-only builder.  V80 r1 was already observed
before this contract was written, so r1 is sealed as historical evidence and
cannot be used to change the V80 runtime, thresholds, or promotion policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_bf16_kv_mamba_confirmation_v80b.json"
)
PARENT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_bf16_kv_mamba_capacity_matched_v80.json"
)
R1 = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v80_bf16_kv_mamba_capacity_0479_r1"
)
RUN_ROOT = ROOT / "experiments/eggroll_es_hpo/runs"

SCHEMA = "v80b-qwen36-bf16-kv-mamba-two-run-confirmation-preregistration"
PARENT_FILE_SHA256 = (
    "be7de6c8ac1d59feef0ec0d0e289be85612644efbe72fdb3847fa34d5dd50aad"
)
PARENT_CONTENT_SHA256 = (
    "7527ed6fe0154a79ecc0de46b00af4601b0e3deaac184f2af094fba15740149a"
)
R1_BUNDLE_SHA256 = (
    "73adc7ebe416d6065808cf918415d077989b1a064b47ef5132422b32da118e47"
)

SOURCE_SHA256 = {
    "probe_vllm_quantized_adapter_switch_v73.py": (
        "43661c32cd8d06deef6d8e2f0d83d889b00f554748b94c3345e2b2052cac66a9"
    ),
    "probe_vllm_fp8_attested_v76.py": (
        "a23d43ee5b6b334fdc58b93e0ce7e7d3fcf72ea4047549f3a0f4d5b715a3fc70"
    ),
    "probe_vllm_bf16_kv_mamba_bf16_v78c.py": (
        "761857944064a0b21ff528971d3f497e4e67865679fa51a30d385cab65835dcb"
    ),
    "probe_vllm_bf16_kv_mamba_capacity_v80.py": (
        "3679bfb1d7f1995701b8b96c100f41ac05fc8f3338977e5760f52aa1ef8009fd"
    ),
    "monitor_qwen36_fp8_kv_capacity_v79.py": (
        "6035eb32f90815ed2a2d8734d9e9072123b8ecc70d74449a2710e94b673ed3df"
    ),
    "launch_qwen36_bf16_kv_mamba_capacity_v80.sh": (
        "3a335c0e44a9b8130adec8c0e11f52d2246264aa0a47f5b173cbdb35093fc7ad"
    ),
}

CONFIRMATIONS = (
    "v80_bf16_kv_mamba_capacity_0479_r2",
    "v80_bf16_kv_mamba_capacity_0479_r3",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256_v80b(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v80b(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_parent_v80b() -> dict[str, Any]:
    _require(PARENT.is_file() and not PARENT.is_symlink(), "V80 parent missing")
    _require(
        file_sha256_v80b(PARENT) == PARENT_FILE_SHA256,
        "V80 parent file identity changed",
    )
    value = json.loads(PARENT.read_text(encoding="ascii"))
    _require(isinstance(value, dict), "V80 parent must be a JSON object")
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        value.get("schema")
        == "v80-qwen36-bf16-kv-mamba-capacity-matched-preregistration"
        and claimed == PARENT_CONTENT_SHA256
        and canonical_sha256_v80b(body) == claimed,
        "V80 parent content identity changed",
    )
    _require(
        value.get("status") == "cpu_preregistered_live_launch_not_performed"
        and value.get("authority", {}).get("scored_or_training_authority")
        is False,
        "V80 parent authority changed",
    )
    return value


def _inventory_v80b(directory: Path) -> dict[str, Any]:
    _require(
        directory.is_dir() and not directory.is_symlink(),
        f"sealed run missing: {directory}",
    )
    paths = sorted(directory.iterdir())
    _require(
        len(paths) == 11
        and all(path.is_file() and not path.is_symlink() for path in paths),
        "V80 r1 artifact cardinality/type changed",
    )
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256_v80b(path),
        }
        for path in paths
    ]
    bundle = canonical_sha256_v80b(rows)
    _require(bundle == R1_BUNDLE_SHA256, "V80 r1 sealed bundle changed")
    return {
        "file_count": len(rows),
        "bundle_sha256": bundle,
        "files": rows,
    }


def _observed_r1_v80b() -> dict[str, Any]:
    inventory = _inventory_v80b(R1)
    walls = []
    capacities = []
    process_group_flags = []
    shutdown_flags = []
    for gpu in range(4):
        path = R1 / f"gpu_{gpu}.json"
        receipt = json.loads(path.read_text(encoding="ascii"))
        _require(isinstance(receipt, dict), f"V80 r1 actor is not JSON: gpu{gpu}")
        body = copy.deepcopy(receipt)
        claimed = body.pop("content_sha256_before_self_field", None)
        _require(
            claimed == canonical_sha256_v80b(body)
            and receipt.get("schema")
            == "v80-qwen36-bf16-kv-mamba-capacity-matched-preflight"
            and receipt.get("actor_label") == f"gpu-{gpu}",
            f"V80 r1 actor identity changed: gpu{gpu}",
        )
        _require(
            receipt.get("preregistration_v80")
            == {
                "content_sha256": PARENT_CONTENT_SHA256,
                "file_sha256": PARENT_FILE_SHA256,
            },
            f"V80 r1 parent identity changed: gpu{gpu}",
        )
        runtime = receipt.get("runtime", {})
        cache = receipt.get("resolved_hybrid_cache_certificate", {})
        _require(
            runtime.get("gpu_memory_utilization") == 0.479
            and runtime.get("mamba_ssm_cache_dtype") == "bfloat16"
            and runtime.get("resolved_quantization") == "fp8"
            and cache.get("gpu_memory_utilization") == 0.479
            and cache.get("cache_dtype") == "auto"
            and cache.get("mamba_ssm_cache_dtype") == "bfloat16",
            f"V80 r1 runtime changed: gpu{gpu}",
        )
        walls.append(receipt["wall_runtime_seconds_excluding_model_load_and_cleanup"])
        capacities.append(cache["kv_cache_size_tokens"])
        process_group_flags.append(receipt.get("torch_process_group_destroyed"))
        shutdown_flags.append(receipt.get("engine_shutdown_completed"))
    _require(
        capacities == [162_669] * 4
        and shutdown_flags == [True] * 4
        and process_group_flags == [False] * 4,
        "V80 r1 observed capacity/cleanup receipt changed",
    )
    return {
        "run": R1.name,
        "artifact_inventory": inventory,
        "observed_before_v80b_preregistration": True,
        "observed_capacity_tokens_per_actor": capacities,
        "observed_actor_runtime_seconds": walls,
        "observed_actor_runtime_seconds_median": statistics.median(walls),
        "engine_shutdown_completed_per_actor": shutdown_flags,
        "torch_process_group_destroyed_receipt_value_per_actor": (
            process_group_flags
        ),
        "literal_parent_torch_process_group_gate_requires_true": True,
        "r1_may_justify_replication_only": True,
        "r1_may_not_change_runtime_thresholds_or_promotion": True,
    }


def _sources_v80b(parent: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for relative, expected in SOURCE_SHA256.items():
        path = ROOT / relative
        _require(path.is_file() and not path.is_symlink(), f"source missing: {relative}")
        actual = file_sha256_v80b(path)
        _require(actual == expected, f"sealed source changed: {relative}")
        rows.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": actual}
        )
    parent_rows = parent.get("sealed_evidence", {}).get("sources", {}).get("files")
    _require(
        isinstance(parent_rows, list)
        and all(any(row["path"] == item["path"] and row["sha256"] == item["sha256"] for row in rows) for item in parent_rows),
        "V80 transitive source ancestry changed",
    )
    return {"files": rows, "bundle_sha256": canonical_sha256_v80b(rows)}


def _command_v80b(run: str) -> str:
    return (
        "RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/"
        f"{run} bash /home/catid/specialist/"
        "launch_qwen36_bf16_kv_mamba_capacity_v80.sh"
    )


def build_preregistration_v80b(
    *, require_future_directories_absent: bool = False
) -> dict[str, Any]:
    parent = _load_parent_v80b()
    observed = _observed_r1_v80b()
    sources = _sources_v80b(parent)
    absent = [not (RUN_ROOT / run).exists() for run in CONFIRMATIONS]
    if require_future_directories_absent:
        _require(all(absent), "V80B confirmation requires fresh r2/r3 directories")

    parent_runtime = copy.deepcopy(parent["selected_runtime"])
    parent_gates = copy.deepcopy(parent["live_acceptance"])
    value = {
        "schema": SCHEMA,
        "bead": "specialist-0j5.24",
        "status": "prospective_exact_two_confirmations_preregistered_not_launched_by_builder",
        "authority": {
            "cpu_file_inspection_only": True,
            "dataset_prompt_generated_text_or_protected_data_opened": False,
            "gpu_or_model_launch_performed_by_builder": False,
            "model_adapter_or_training_update_performed": False,
            "checkpoint_config_or_runtime_promotion_performed": False,
            "scored_training_checkpoint_or_layout_promotion_authorized": False,
        },
        "parent_v80": {
            "path": str(PARENT.relative_to(ROOT)),
            "file_sha256": PARENT_FILE_SHA256,
            "content_sha256": PARENT_CONTENT_SHA256,
            "selected_runtime_retained_exactly": parent_runtime,
            "selected_runtime_canonical_sha256": canonical_sha256_v80b(
                parent_runtime
            ),
            "live_acceptance_retained_exactly": parent_gates,
            "live_acceptance_canonical_sha256": canonical_sha256_v80b(
                parent_gates
            ),
        },
        "sealed_executable_sources": sources,
        "post_observation_disclosure": observed,
        "prospective_integrity": {
            "r1_was_observed_before_this_preregistration": True,
            "r1_used_only_to_request_independent_confirmation": True,
            "r1_not_used_to_change_any_parent_runtime_field": True,
            "r1_not_used_to_change_any_parent_gate_or_threshold": True,
            "r1_not_used_to_authorize_semantic_ood_scoring_or_promotion": True,
            "parent_runtime_copied_without_edits": True,
            "parent_live_acceptance_copied_without_edits": True,
            "exactly_two_future_runs": True,
            "future_run_names_exact": list(CONFIRMATIONS),
            "future_run_directories_absent_at_preregistration_build": True,
            "future_directory_absence_required_when_generated": True,
            "threshold_tuning_after_r1_forbidden": True,
            "promotion_from_r1_r2_or_r3_forbidden": True,
        },
        "confirmatory_runs": [
            {
                "ordinal": index + 2,
                "run": run,
                "exact_command": _command_v80b(run),
                "fresh_run_directory_required": True,
                "source_runtime_and_parent_gates_unchanged": True,
                "independent_parent_gate_evaluation_required": True,
            }
            for index, run in enumerate(CONFIRMATIONS)
        ],
        "confirmation_analysis": {
            "exactly_r2_and_r3_no_additional_runs_under_v80b": True,
            "evaluate_each_run_independently_against_unchanged_parent_gates": True,
            "report_all_failures_without_exclusion_or_replacement": True,
            "report_three_run_runtime_and_capacity_descriptively": True,
            "parent_thresholds_must_not_be_reestimated_from_r1_r2_r3": True,
            "literal_torch_process_group_destroyed_true_gate_retained": True,
            "semantic_and_protected_ood_gates_remain_pending": True,
            "promotion_default": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v80b(value)
    return value


def render_json_v80b(value: dict[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v80b(
        require_future_directories_absent=not args.check
    )
    payload = render_json_v80b(value)
    if args.check:
        _require(OUTPUT.is_file(), "V80B preregistration missing")
        _require(
            OUTPUT.read_text(encoding="ascii") == payload,
            "V80B preregistration stale",
        )
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(payload, encoding="ascii")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "content_sha256": value["content_sha256_before_self_field"],
                "commands": [row["exact_command"] for row in value["confirmatory_runs"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
