#!/usr/bin/env python3
"""Aggregate three sealed V45 repetitions without protected-data access."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eval_reports/"
    "v45_sft_v42g_repetition_evidence_v46a.json"
).resolve()
REPORTS = {
    "v45a": {
        "path": ROOT / "experiments/eval_reports/"
        "matched_lora_hpo_earlystop_ood_eligible_eval_v45a.json",
        "file_sha256": "64f0edab64c66aa208a03759f759a3fe62d3bc886e6fce55180abe7b57cb919d",
        "content_sha256": "419c5855dbbaf7365d259a22d588e58ce593d340efb502b1c2682067d9b57c7b",
    },
    "v45b": {
        "path": ROOT / "experiments/eval_reports/"
        "matched_lora_hpo_v43g_ood_eligible_eval_v45b.json",
        "file_sha256": "cd24ac92b97efa2940594c1929b1b02f611c50812830ecff2b13b02a5a15dbf1",
        "content_sha256": "779e69731374a5a1c4300ecce27b8aced82e21dfa1e5f81f16bdcde05a65941e",
    },
    "v45c": {
        "path": ROOT / "experiments/eval_reports/"
        "matched_lora_hpo_v42h_v43g_ood_eligible_eval_v45c.json",
        "file_sha256": "2081b319a777a5c9bbe5129cd123b7661b50c9915594f277ca7676486fa09daf",
        "content_sha256": "3a1737796e442ae99a464ce128aa2721ef8c91516ba84ac97f13489e53dcadb0",
    },
}
SOURCE_REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42g_matched_init_equal_unit_fold3_v412_lr5e5/runtime_report_v42g.json"
).resolve()
SOURCE_WEIGHTS = (
    ROOT / "experiments/sft_controls/"
    "v42g_matched_init_equal_unit_fold3_v412_lr5e5/"
    "middle_late_r32_seed17_init20260715041/final/adapter_model.safetensors"
).resolve()
SOURCE_CONFIG = SOURCE_WEIGHTS.with_name("adapter_config.json")
STAGED = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42g_sft_qwen35_vllm_namespace_v45a"
).resolve()
STAGE_MANIFEST = STAGED / "stage_manifest_v44a.json"
FIXED_CANDIDATE = {
    "arm": "sft_v42g",
    "training_family": "matched SFT",
    "learning_rate": 5e-5,
    "completed_steps": 48,
    "source_report_file_sha256": "82633c37a17e597961ca8a69d2a1a07218dbd6d0df53d9ebb75217ba54549430",
    "source_report_content_sha256": "5c9f0d80061b0cd6f95ac401802fe96fdff42e585c94737a536a65186bd66986",
    "source_weights_sha256": "b5820bd3a01605bbd7c488900bdeb27dc2dbaf5073e9712edfbff0f7402f3bd0",
    "source_config_sha256": "99548642c32678b1a69f0e2c1151c96f10b87b821e6588feefebd353684f2178",
    "staged_weights_sha256": "0ae7050347b885cdbde042ec3ef0cfbdbc9d4040e432fab2a85536282c6d7f16",
    "staged_config_sha256": "99548642c32678b1a69f0e2c1151c96f10b87b821e6588feefebd353684f2178",
    "stage_manifest_file_sha256": "dae592e2e8322bb1bd72e3b68146d843082e8c689617e5659c0acd1869b9e373",
    "stage_manifest_content_sha256": "dd7eb90c411726a53ae6bc6c3fbe68a94107a02b99e6bc403e4fd76da8bcc533",
    "transformed_identity_sha256": "598727b73b094d7f55647accf009cf9b315766f9bf45182e5be4e228debc04a2",
}


def _compact_sha(value: dict) -> str:
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    return core.canonical_sha256(compact)


def _load_report(label: str, spec: dict) -> dict:
    path = Path(spec["path"]).resolve()
    if core.file_sha256(path) != spec["file_sha256"]:
        raise RuntimeError(f"V46A {label} aggregate file changed")
    value = json.loads(path.read_text())
    selection = value.get("selection", {})
    gate = selection.get("per_arm_gate_table", {}).get("sft_v42g", {})
    base = value.get("base_duplicate_equivalence", {})
    if (
        value.get("content_sha256_before_self_field")
        != spec["content_sha256"]
        or _compact_sha(value) != spec["content_sha256"]
        or value.get("status")
        != "complete_aggregate_only_no_heldout_access"
        or value.get("heldout_or_holdout_opened") is not False
        or selection.get("selected_arm") != "sft_v42g"
        or gate.get("eligible") is not True
        or gate.get("ood_qa", {}).get("passed") is not True
        or gate.get("ood_prose", {}).get("passed") is not True
        or gate.get("no_protocol_or_leak_counter_increase") is not True
        or value.get("final_gate", {}).get("passed") is not True
        or base.get("all_splits") is not True
        or not all(base.get(split) is True for split in (
            "shadow", "ood_qa", "ood_prose"
        ))
    ):
        raise RuntimeError(f"V46A {label} aggregate evidence changed")
    return value


def _validate_candidate() -> dict:
    expected_files = {
        SOURCE_REPORT: FIXED_CANDIDATE["source_report_file_sha256"],
        SOURCE_WEIGHTS: FIXED_CANDIDATE["source_weights_sha256"],
        SOURCE_CONFIG: FIXED_CANDIDATE["source_config_sha256"],
        STAGED / "adapter_model.safetensors": FIXED_CANDIDATE[
            "staged_weights_sha256"
        ],
        STAGED / "adapter_config.json": FIXED_CANDIDATE[
            "staged_config_sha256"
        ],
        STAGE_MANIFEST: FIXED_CANDIDATE["stage_manifest_file_sha256"],
    }
    for path, expected in expected_files.items():
        if core.file_sha256(path) != expected:
            raise RuntimeError(f"V46A fixed candidate artifact changed: {path}")
    report = json.loads(SOURCE_REPORT.read_text())
    manifest = json.loads(STAGE_MANIFEST.read_text())
    if (
        _compact_sha(report)
        != FIXED_CANDIDATE["source_report_content_sha256"]
        or report.get("content_sha256_before_self_field")
        != FIXED_CANDIDATE["source_report_content_sha256"]
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("status")
        != "complete_train_only_lr5e5_state_sealed_shadow_unopened"
        or _compact_sha(manifest)
        != FIXED_CANDIDATE["stage_manifest_content_sha256"]
        or manifest.get("content_sha256_before_self_field")
        != FIXED_CANDIDATE["stage_manifest_content_sha256"]
        or manifest.get("transformed_identity", {}).get("sha256")
        != FIXED_CANDIDATE["transformed_identity_sha256"]
        or manifest.get("transformed_identity", {}).get(
            "all_tensor_bytes_preserved_exactly"
        ) is not True
        or manifest.get("source", {}).get("weights_file_sha256")
        != FIXED_CANDIDATE["source_weights_sha256"]
        or manifest.get("artifact", {}).get("weights_file_sha256")
        != FIXED_CANDIDATE["staged_weights_sha256"]
    ):
        raise RuntimeError("V46A fixed V42G identity changed")
    return {
        **FIXED_CANDIDATE,
        "source_report_path": str(SOURCE_REPORT),
        "source_weights_path": str(SOURCE_WEIGHTS),
        "staged_directory": str(STAGED),
        "stage_manifest_path": str(STAGE_MANIFEST),
        "tensor_count": manifest["transformed_identity"]["tensor_count"],
        "elements": manifest["transformed_identity"]["elements"],
        "all_tensor_bytes_preserved_exactly": True,
    }


METRICS = {
    "shadow_generated_equal_unit_mean_reward": (
        "shadow", "metrics", "sft_v42g", "generated_equal_unit_mean_reward"
    ),
    "shadow_generated_row_mean_reward": (
        "shadow", "metrics", "sft_v42g", "generated_row_mean_reward"
    ),
    "shadow_generated_exact_count": (
        "shadow", "metrics", "sft_v42g", "generated_exact_count"
    ),
    "shadow_generated_nonzero_count": (
        "shadow", "metrics", "sft_v42g", "generated_nonzero_count"
    ),
    "shadow_teacher_forced_equal_unit_mean_answer_logprob": (
        "shadow", "metrics", "sft_v42g",
        "teacher_forced_equal_unit_mean_answer_logprob"
    ),
    "ood_qa_generated_equal_unit_mean_reward": (
        "ood_qa", "sft_v42g", "generated_equal_unit_mean_reward"
    ),
    "ood_qa_generated_exact_count": (
        "ood_qa", "sft_v42g", "generated_exact_count"
    ),
    "ood_qa_generated_nonzero_count": (
        "ood_qa", "sft_v42g", "generated_nonzero_count"
    ),
    "ood_qa_teacher_forced_equal_unit_mean_answer_logprob": (
        "ood_qa", "sft_v42g",
        "teacher_forced_equal_unit_mean_answer_logprob"
    ),
    "ood_prose_mean_token_logprob": (
        "ood_prose", "sft_v42g", "mean_token_logprob"
    ),
    "ood_qa_mean_reward_delta": (
        "selection", "per_arm_gate_table", "sft_v42g", "ood_qa",
        "mean_reward_delta"
    ),
    "ood_qa_exact_count_delta": (
        "selection", "per_arm_gate_table", "sft_v42g", "ood_qa",
        "exact_count_delta"
    ),
    "ood_prose_delta": (
        "selection", "per_arm_gate_table", "sft_v42g", "ood_prose",
        "delta"
    ),
}


def _at(value: dict, path: tuple[str, ...]):
    for key in path:
        value = value[key]
    return value


def _spread(values_by_run: dict[str, int | float]) -> dict:
    values = [float(value) for value in values_by_run.values()]
    low, high = min(values), max(values)
    return {
        "values_by_run": values_by_run,
        "mean": math.fsum(values) / len(values),
        "min": low,
        "max": high,
        "range": high - low,
    }


def build() -> dict:
    reports = {
        label: _load_report(label, spec) for label, spec in REPORTS.items()
    }
    evidence = {
        "schema": "v45-sft-v42g-repetition-evidence-v46a",
        "status": "complete_aggregate_only_three_of_three_stable",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "bind three independent aggregate evaluations and quantify the "
            "documented generation numerical spread before any holdout access"
        ),
        "report_bindings": {
            label: {
                "path": str(Path(spec["path"]).resolve()),
                "file_sha256": spec["file_sha256"],
                "content_sha256": spec["content_sha256"],
                "selected_arm": "sft_v42g",
                "sft_v42g_ood_eligible": True,
                "strict_final_gate_passed": True,
                "base_duplicate_equivalence_all_splits": True,
                "heldout_or_holdout_opened": False,
            }
            for label, spec in REPORTS.items()
        },
        "fixed_candidate_identity": _validate_candidate(),
        "consistency": {
            "repetition_count": 3,
            "sft_v42g_selected_count": 3,
            "sft_v42g_ood_eligible_count": 3,
            "strict_final_gate_pass_count": 3,
            "base_duplicate_equivalence_count": 3,
            "selected_in_all_repetitions": True,
            "ood_eligible_in_all_repetitions": True,
            "strict_gate_passed_in_all_repetitions": True,
            "base_duplicates_exact_in_all_repetitions": True,
        },
        "metric_spread": {
            metric: _spread({
                label: _at(report, path) for label, report in reports.items()
            })
            for metric, path in METRICS.items()
        },
        "interpretation": {
            "generation_is_numerically_bitwise_identical": False,
            "selection_and_gate_decisions_are_identical": True,
            "conclusion": (
                "V42G remains selected, OOD-eligible, and strict-gate-safe in "
                "3/3 repetitions despite small generated-score/count and "
                "teacher-forced numerical spread"
            ),
        },
        "protected_semantics_accessed_while_building": False,
        "heldout_or_holdout_opened": False,
        "raw_questions_answers_or_generations_read": False,
    }
    evidence["content_sha256_before_self_field"] = core.canonical_sha256(evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    evidence = build()
    core.atomic_json(output, evidence)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": evidence["content_sha256_before_self_field"],
        "selected_eligible_strict_gate": "3/3",
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
