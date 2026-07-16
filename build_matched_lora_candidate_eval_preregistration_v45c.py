#!/usr/bin/env python3
"""Seal V45C from V45B commitments without protected-data access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45c as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_hpo_v42h_v43g_ood_eligible_eval_v45c.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def build() -> dict:
    prior = runtime.prior_preregistration_v45c()
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    definitions = list(prior["candidate_definitions"])
    definitions.append({
        "arm": "sft_v42h", "training_family": "matched SFT",
        "learning_rate": 6e-5, "completed_steps": 48,
        "source_artifact_prefix": "final",
        "source_weights_sha256": runtime.v42h_stage.EXPECTED["weights"],
        "source_report_sha256": runtime.v42h_stage.EXPECTED["report"],
        "staged_directory": str(runtime.v42h_stage.OUTPUT),
        "adapter_id": runtime.ADAPTER_IDS_V45C["sft_v42h"],
    })
    value.update({
        "schema": "matched-lora-v42h-v43g-ood-eligible-preregistration-v45c",
        "status": "preregistered_before_fresh_v42h_extended_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(runtime.ARMS_V45C),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V45C),
        "padding_base_arms": list(runtime.PADDING_BASE_ARMS_V45C),
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V45C),
        "candidate_definitions": definitions,
        "implementation_bindings": runtime.implementation_bindings_v45c(),
        "staged_adapters": runtime.staged_adapter_bindings_v45c(),
        "extends_preregistration": {
            "path": str(runtime.V45B_PREREG),
            "file_sha256": runtime.V45B_PREREG_FILE_SHA256,
            "content_sha256": runtime.V45B_PREREG_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
        },
        "v45b_aggregate_observed_before_v45c_preregistration": (
            runtime.prior_result_v45c()
        ),
        "protected_semantics_inspected_during_v45c_revision": False,
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
        "five_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v45c()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "nine_base_replicates_exact_equivalence_required": True,
        "six_padding_base_replicates_excluded_from_candidate_ranking": True,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.CANDIDATE_ARMS_V45C),
        "nine_base_duplicate_exact_equivalence_required": True,
        "padding_base_arms_excluded_from_eligibility_and_ranking": list(
            runtime.PADDING_BASE_ARMS_V45C
        ),
    })
    value["selection_protocol_v45a"] = dict(value["selection_protocol_v45a"])
    value["selection_protocol_v45a"].update({
        "candidate_set": list(runtime.CANDIDATE_ARMS_V45C),
        "v42h_independently_ood_gated_before_shadow_ranking": True,
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
        "path": str(output), "file_sha256": core.file_sha256(output),
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
