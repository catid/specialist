#!/usr/bin/env python3
"""Bind repeat-stable V42G evidence and the V45D V42I boundary result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core


ROOT = Path(__file__).resolve().parent
PRIOR = (
    ROOT / "experiments/eval_reports/"
    "v45_sft_v42g_repetition_evidence_v46a.json"
).resolve()
V45D = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_boundary_ood_eligible_eval_v45d.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/eval_reports/"
    "sft_boundary_resolution_stability_evidence_v46a.json"
).resolve()
PRIOR_FILE_SHA256 = (
    "c4e2351d22889f4f8ab606c2ba4270c12bdf0edfd0f6639755bb58b684393340"
)
PRIOR_CONTENT_SHA256 = (
    "c49ab2394c0f0a4a55c7ad7d5c5a748a1107b831aa12c314d4ba02d04b195f46"
)
V45D_FILE_SHA256 = (
    "a38ef3b9e947f2b6a6328f9eea679d7031a515ce59c8d0804144ae6d587985be"
)
V45D_CONTENT_SHA256 = (
    "f39a12fc33e55c957b540033220eb1ff82d0e1459e962ee30f591024d0b6a9a8"
)


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def build() -> dict:
    if core.file_sha256(PRIOR) != PRIOR_FILE_SHA256:
        raise RuntimeError("V46A prior repetition evidence file changed")
    prior = json.loads(PRIOR.read_text())
    if (
        prior.get("content_sha256_before_self_field") != PRIOR_CONTENT_SHA256
        or _compact_sha(prior) != PRIOR_CONTENT_SHA256
        or prior.get("consistency", {}).get("sft_v42g_selected_count") != 3
        or prior.get("consistency", {}).get(
            "sft_v42g_ood_eligible_count"
        ) != 3
        or prior.get("consistency", {}).get("strict_final_gate_pass_count") != 3
        or prior.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("V46A prior repetition evidence content changed")

    if core.file_sha256(V45D) != V45D_FILE_SHA256:
        raise RuntimeError("V46A V45D aggregate file changed")
    v45d = json.loads(V45D.read_text())
    table = v45d.get("selection", {}).get("per_arm_gate_table", {})
    base = v45d.get("base_duplicate_equivalence", {})
    staged = v45d.get("staged_adapters", {}).get("sft_v42i", {})
    if (
        v45d.get("content_sha256_before_self_field") != V45D_CONTENT_SHA256
        or _compact_sha(v45d) != V45D_CONTENT_SHA256
        or v45d.get("status") != "complete_aggregate_only_no_heldout_access"
        or v45d.get("selection", {}).get("selected_arm") != "sft_v42i"
        or table.get("sft_v42i", {}).get("eligible") is not True
        or table.get("sft_v42g", {}).get("eligible") is not True
        or v45d.get("final_gate", {}).get("passed") is not True
        or base.get("all_splits") is not True
        or not all(base.get(split) is True for split in (
            "shadow", "ood_qa", "ood_prose"
        ))
        or v45d.get("gpu_activity", {}).get(
            "all_four_attributed_positive_each_phase"
        ) is not True
        or v45d.get("heldout_or_holdout_opened") is not False
        or staged.get("weights_file_sha256")
        != "79207dd2c0b46aaef4af5933aaac9fbbaf837db91241ab9d352e652b5c53afad"
        or staged.get("manifest_file_sha256")
        != "f3bc58058032fd3cc00176ddda9de861d2cc3a989b8ce2e7499e15f1adb6d0c5"
        or staged.get("manifest_content_sha256")
        != "040f13fbddcb16b15d8771f736cedac1a89a5b987805c442da8e03445cc1c838"
        or staged.get("transformed_identity_sha256")
        != "d185cbe52414054759188334fd38b96dbda601957bc8931256fd1e3c0fe71041"
        or staged.get("tensor_bytes_preserved_exactly") is not True
    ):
        raise RuntimeError("V46A V45D boundary evidence changed")

    shadow = v45d["shadow"]["metrics"]
    value = {
        "schema": "sft-boundary-resolution-stability-evidence-v46a",
        "status": "complete_aggregate_only_holdout_unopened",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_repetition_evidence": {
            "path": str(PRIOR),
            "file_sha256": PRIOR_FILE_SHA256,
            "content_sha256": PRIOR_CONTENT_SHA256,
            "finding": (
                "V42G selected, OOD-eligible, and strict-gate-passing in 3/3 "
                "repetitions despite documented numerical spread"
            ),
        },
        "v45d_boundary_report": {
            "path": str(V45D),
            "file_sha256": V45D_FILE_SHA256,
            "content_sha256": V45D_CONTENT_SHA256,
            "selected_arm": "sft_v42i",
            "selected_arm_ood_eligible": True,
            "strict_final_gate_passed": True,
            "all_four_base_duplicates_exact_all_splits": True,
            "all_four_gpus_positive_each_phase": True,
            "heldout_or_holdout_opened": False,
        },
        "current_sft_control": {
            "arm": "sft_v42i",
            "learning_rate": 5.5e-5,
            "completed_steps": 48,
            "source_report_file_sha256": (
                "3076ff21d7d7910cc9ae33f1c00c69b10d8e72c6c8366bb1029ceca17812cee6"
            ),
            "source_report_content_sha256": (
                "16d8898b6b81da33a6968c254e2d5c5684dd6a284ee0874b9f762bfc140b4341"
            ),
            "source_weights_sha256": (
                "9e83783c20dfb5eec91b7217d885270efed8aec216c80374444dcbc55fd7dab8"
            ),
            "source_config_sha256": (
                "0e8060efd40772233390f3f97ace489e473b2bc76572e7566b83afe3dd83cc51"
            ),
            "staged_weights_sha256": staged["weights_file_sha256"],
            "staged_config_sha256": staged["adapter_config_file_sha256"],
            "stage_manifest_file_sha256": staged["manifest_file_sha256"],
            "stage_manifest_content_sha256": staged[
                "manifest_content_sha256"
            ],
            "transformed_identity_sha256": staged[
                "transformed_identity_sha256"
            ],
            "all_tensor_bytes_preserved_exactly": True,
        },
        "boundary_comparison": {
            "sft_v42i_shadow_generated_equal_unit_mean_reward": shadow[
                "sft_v42i"
            ]["generated_equal_unit_mean_reward"],
            "sft_v42g_shadow_generated_equal_unit_mean_reward": shadow[
                "sft_v42g"
            ]["generated_equal_unit_mean_reward"],
            "sft_v42i_minus_sft_v42g_shadow_reward": (
                shadow["sft_v42i"]["generated_equal_unit_mean_reward"]
                - shadow["sft_v42g"]["generated_equal_unit_mean_reward"]
            ),
            "sft_v42i_ood_qa_gate": table["sft_v42i"]["ood_qa"],
            "sft_v42i_ood_prose_gate": table["sft_v42i"]["ood_prose"],
            "sft_v42g_remained_eligible": True,
        },
        "replication_scope": {
            "v42g_has_three_repetitions": True,
            "v42i_has_one_boundary_evaluation": True,
            "v42i_repeated_stability_claimed": False,
            "v43i_ood_first_resolution_required_before_holdout_launch": True,
        },
        "protected_semantics_accessed_while_building": False,
        "heldout_or_holdout_opened": False,
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output), "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "current_sft_control": "sft_v42i",
        "v42i_repeated_stability_claimed": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
