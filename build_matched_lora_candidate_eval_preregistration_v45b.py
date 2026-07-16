#!/usr/bin/env python3
"""Seal V45B from V45A commitments without reopening protected semantics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45b as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_hpo_v43g_ood_eligible_eval_v45b.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def build() -> dict:
    prior = runtime.prior_preregistration_v45b()
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    definitions = [
        item for item in prior["candidate_definitions"]
        if item["arm"] in runtime.CANDIDATE_ARMS_V45B
    ]
    definitions.append({
        "arm": "lora_es_v43g",
        "training_family": "robust centered rank-one LoRA-ES",
        "completed_steps": 1,
        "source_weights_sha256": runtime.v43g_stage.EXPECTED["weights"],
        "canonical_master_sha256": runtime.v43g_stage.EXPECTED[
            "master_sha256"
        ],
        "source_report_sha256": runtime.v43g_stage.EXPECTED["report"],
        "staged_directory": str(runtime.v43g_stage.OUTPUT),
        "adapter_id": runtime.ADAPTER_IDS_V45B["lora_es_v43g"],
    })
    value.update({
        "schema": "matched-lora-v43g-ood-eligible-eval-preregistration-v45b",
        "status": "preregistered_before_fresh_v43g_extended_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(runtime.ARMS_V45B),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V45B),
        "padding_base_arms": ["base_d", "base_e", "base_f"],
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V45B),
        "candidate_definitions": definitions,
        "implementation_bindings": runtime.implementation_bindings_v45b(),
        "staged_adapters": runtime.staged_adapter_bindings_v45b(),
        "extends_preregistration": {
            "path": str(runtime.V45A_PREREG),
            "file_sha256": runtime.V45A_PREREG_FILE_SHA256,
            "content_sha256": runtime.V45A_PREREG_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
        },
        "v45a_aggregate_observed_before_v45b_preregistration": (
            runtime.prior_result_v45b()
        ),
        "protected_semantics_inspected_during_v45b_revision": False,
        "heldout_or_holdout_access_authorized": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "four_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v45b()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "six_base_replicates_exact_equivalence_required": True,
        "three_padding_base_replicates_excluded_from_candidate_ranking": True,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.CANDIDATE_ARMS_V45B),
        "six_base_duplicate_exact_equivalence_required": True,
        "padding_base_arms_excluded_from_eligibility_and_ranking": [
            "base_d", "base_e", "base_f"
        ],
    })
    value["selection_protocol_v45a"] = dict(value["selection_protocol_v45a"])
    value["selection_protocol_v45a"].update({
        "candidate_set": list(runtime.CANDIDATE_ARMS_V45B),
        "v43d_and_v43g_both_independently_ood_gated": True,
        "padding_base_arms_affect_candidate_eligibility_or_ranking": False,
    })
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    output = Path(parser().parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "cpu_preflight_content_sha256": value[
            "cpu_preflight_expected"
        ]["content_sha256_before_self_field"],
        "protected_semantics_inspected_during_revision": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
