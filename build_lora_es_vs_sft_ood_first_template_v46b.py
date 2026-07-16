#!/usr/bin/env python3
"""Build the unlaunchable V46B evaluation template without protected access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_lora_es_vs_sft_ood_first_template_v46b as runtime
import run_matched_lora_candidate_eval_v44a as core


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = runtime.DEFAULT_PREREGISTRATION


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def _parent(path: Path, file_sha256: str, content_sha256: str) -> dict:
    if core.file_sha256(path) != file_sha256:
        raise RuntimeError(f"V46B parent file identity changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha256
        or core.canonical_sha256(compact) != content_sha256
    ):
        raise RuntimeError(f"V46B parent content identity changed: {path}")
    return value


def build_v46b() -> dict:
    v43i = _parent(
        runtime.V43I_PREREGISTRATION,
        runtime.V43I_PREREGISTRATION_FILE_SHA256,
        runtime.V43I_PREREGISTRATION_CONTENT_SHA256,
    )
    v45d = _parent(
        runtime.V45D_PREREGISTRATION,
        runtime.V45D_PREREGISTRATION_FILE_SHA256,
        runtime.V45D_PREREGISTRATION_CONTENT_SHA256,
    )
    if (
        v43i.get("gpu_launch_authorized") is not True
        or v43i.get("sealed_holdout_opened") is not False
        or v45d.get("heldout_or_holdout_access_authorized") is not False
        or v45d.get("selection_protocol_v45a", {}).get(
            "every_candidate_ood_gated_before_shadow_ranking"
        ) is not True
        or v45d.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
    ):
        raise RuntimeError("V46B parent firewall/protocol commitment changed")
    value = {
        "schema": "lora-es-v43i-vs-sft-ood-first-eval-template-v46b",
        "status": "blocked_pending_v43i_and_v45d_exact_candidate_seals",
        "created_at_utc": "2026-07-15T23:59:00+00:00",
        "gpu_launch_authorized": False,
        "evaluation_launch_authorized": False,
        "heldout_or_holdout_access_authorized": False,
        "protected_semantics_inspected_during_v46b_revision": False,
        "purpose": (
            "Freeze a balanced four-GPU comparison of the eventual V43I LoRA-ES "
            "snapshot and the fixed V45D SFT boundary winner, with OOD eligibility "
            "decided before any shadow ranking."
        ),
        "parents": {
            "v43i_train_preregistration_only_not_result": {
                "path": str(runtime.V43I_PREREGISTRATION),
                "file_sha256": runtime.V43I_PREREGISTRATION_FILE_SHA256,
                "content_sha256": runtime.V43I_PREREGISTRATION_CONTENT_SHA256,
                "snapshot_identity_available": False,
            },
            "v45d_eval_preregistration_only_not_result": {
                "path": str(runtime.V45D_PREREGISTRATION),
                "file_sha256": runtime.V45D_PREREGISTRATION_FILE_SHA256,
                "content_sha256": runtime.V45D_PREREGISTRATION_CONTENT_SHA256,
                "winner_identity_available_to_template": False,
            },
        },
        "arms": list(runtime.ARMS_V46B),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V46B),
        "candidate_replica_arms": list(runtime.CANDIDATE_REPLICA_ARMS_V46B),
        "candidate_families": {
            family: list(replicas)
            for family, replicas in runtime.CANDIDATE_FAMILIES_V46B.items()
        },
        "candidate_artifact_seals": runtime.pending_candidate_seals_v46b(),
        "launch_blockers": [
            "V43I completed train report file/content hashes are absent",
            "V43I exact snapshot weights/config/canonical identity hashes are absent",
            "V43I staged vLLM adapter hashes are absent",
            "V45D aggregate report file/content hashes and fixed selected arm are absent",
            "V45D selected source and staged adapter hashes are absent",
            "a new post-result runtime/preregistration revision has not been sealed",
        ],
        "future_sealing_policy": {
            "mutate_this_template_in_place": False,
            "require_new_content_addressed_revision": True,
            "verify_v43i_report_status_and_candidate_gate_passed": True,
            "verify_v43i_snapshot_readback_identity": True,
            "verify_v45d_report_is_aggregate_only_and_holdout_false": True,
            "verify_v45d_selected_arm_was_ood_eligible_before_shadow_ranking": True,
            "stage_both_candidates_byte_exactly_before_final_preregistration": True,
            "bind_final_runtime_builder_tests_and_every_candidate_artifact": True,
        },
        "single_access_inputs": core.PROTECTED_INPUTS_V44A,
        "protected_access_firewall": {
            "implementation": "SingleSemanticAccessV44A plus V44C source-faithful preflight",
            "semantic_read_count_per_input": 1,
            "protected_input_labels": [
                "shadow", "split_manifest", "ood_qa", "ood_prose",
            ],
            "protected_files_hashed_or_opened_by_template_builder": False,
            "protected_files_hashed_or_opened_by_dry_run": False,
            "holdout_or_heldout_path_tokens_forbidden": True,
            "raw_local_mode": "0600",
            "raw_content_in_aggregate": False,
        },
        "runtime": {
            "engine_count": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "two_full_fixed_waves": [
                [{"arm": arm, "engine_index": engine} for arm, engine in wave]
                for wave in runtime.arm_wave_plan_v46b()
            ],
            "all_four_gpus_receive_one_identical_batch_per_wave": True,
            "four_exact_base_duplicates_required_on_every_surface": True,
            "two_exact_inference_replicas_required_per_candidate_family": True,
            "identical_prompts_sampling_params_and_seed_for_every_arm": True,
        },
        "ood_first_selection_protocol": {
            "candidate_families": list(runtime.FAMILY_TIE_ORDER_V46B),
            "eligibility_surfaces": [
                "OOD QA mean reward point non-degradation",
                "OOD QA exact-count point non-degradation",
                "OOD prose point non-degradation",
                "OOD prose paired-document bootstrap LCB non-degradation",
                "no shadow protocol/leak counter increase",
            ],
            "per_family_ood_eligibility_finalized_before_shadow_ranking": True,
            "ineligible_family_shadow_metrics_cannot_affect_selection": True,
            "shadow_ranking": (
                "lexicographic generated equal-unit mean, exact count, nonzero "
                "count, teacher-forced equal-unit answer logprob"
            ),
            "exact_tie_order": list(runtime.FAMILY_TIE_ORDER_V46B),
            "fallback_when_no_candidate_is_eligible": "base_a",
            "selected_candidate_must_also_improve_over_base_shadow": True,
        },
        "cpu_preflight_expected": v45d["cpu_preflight_expected"],
        "implementation_bindings": runtime.implementation_bindings_v46b(),
        "fresh_artifacts_reserved_but_not_creatable_by_template": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "raw_local": str(runtime.RAW),
            "gpu_log": str(runtime.GPU_LOG),
            "aggregate_report": str(runtime.REPORT),
        },
    }
    return core.self_hashed(value)


def main(argv: list[str] | None = None) -> int:
    output = Path(parser().parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v46b()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"],
        "launch_authorized": False,
        "protected_semantics_inspected": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
