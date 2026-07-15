#!/usr/bin/env python3
"""Build compact train-only V36C ES-versus-SFT runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path

import run_sft_train_only_control_v36a as runtime


ROOT = Path(__file__).resolve().parent
SFT_REPORT = (
    ROOT / "experiments/sft_controls/v36b_v412/runtime_report_v36b.json"
).resolve()
ES_RUNTIME_REPORT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab/"
    "v27c_tuned_runtime_ab_report_v28a.json"
).resolve()
ES_RECIPE_ATTEMPT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".snapshot794_layer_v13b_document_balanced_five_panel_alpha_zero_"
    "runtime_forwarded_resident_sign_basis20260714.launch_attempt.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/sft_controls/v36b_v412/"
    "es_sft_runtime_comparison_v36c.json"
).resolve()

EXPECTED = {
    "sft_report_file": "00c9790935f970d9b14fd8335c737a079b48125562c38655e6ecdcdfe9f1bbc9",
    "sft_report_content": "739bc80b6536197d29a855ef3723934888b3d9c0ef38354ee5c6968f1e9bd16d",
    "es_runtime_file": "94f2bf79633d4fedac6918e505985933623f213c4143a0d374748456b5a02f0a",
    "es_runtime_content": "892205cb9d807e8a47be6b74e55cb69d8167aabd2565bb963e11c4ff463a73cd",
    "es_recipe_file": "00513c2e32d839c895a2cac5d4d00717be88b89bccfc9f841265c8cff9be6b6c",
    "es_recipe_content": "188c972e97d531d9ace9e3ae3ca17dee943e449795c6731552b8039cf04285c4",
}


def _load(path: Path, file_hash: str, content_hash: str) -> dict:
    if runtime.file_sha256(path) != file_hash:
        raise RuntimeError(f"runtime-comparison file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256_before_self_field") != content_hash:
        raise RuntimeError(f"runtime-comparison content changed: {path}")
    if content_hash != runtime.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }):
        raise RuntimeError(f"runtime-comparison self-hash failed: {path}")
    return value


def build():
    sft = _load(
        SFT_REPORT, EXPECTED["sft_report_file"], EXPECTED["sft_report_content"]
    )
    es_runtime = _load(
        ES_RUNTIME_REPORT,
        EXPECTED["es_runtime_file"],
        EXPECTED["es_runtime_content"],
    )
    es_recipe = _load(
        ES_RECIPE_ATTEMPT,
        EXPECTED["es_recipe_file"],
        EXPECTED["es_recipe_content"],
    )
    population = es_recipe["recipe"]["population_size"]
    signs = len(es_recipe["recipe"]["signs"])
    panel_rows = sum(
        panel["rows"] for panel in es_recipe["recipe"]["panels"].values()
    )
    engines = es_recipe["recipe"]["hardware"]["engine_count"]
    calls_per_gpu = population * signs // engines
    es_sequence_presentations = population * signs * panel_rows
    tokens_per_call = es_runtime["summary"]["performance"][
        "processed_tokens_per_engine_call"
    ]
    tuned_cells = [
        cell for cell in es_runtime["summary"]["performance"]["cells"].values()
        if cell["arm"] == "v27c_tuned"
    ]
    slowest_tuned_tokens_per_second = min(
        cell["median_processed_tokens_per_second"] for cell in tuned_cells
    )
    es_tokens_per_gpu = calls_per_gpu * tokens_per_call
    es_token_presentations = es_tokens_per_gpu * engines
    es_scoring_critical_path_seconds = (
        es_tokens_per_gpu / slowest_tuned_tokens_per_second
    )

    encoding = sft["observed_encoding_audit"]["value"]
    sft_nominal_rows = encoding["train_rows"] * 3
    sft_padded_rows = 532 * 3
    sft_nominal_tokens = (
        encoding["train_prompt_tokens"] + encoding["train_answer_tokens"]
    ) * 3
    sft_train_runtime = sft["trainer_metrics"]["train_runtime"]
    sft_wall_runtime = sft["wall_runtime_seconds"]
    sft_trainable = sft["observed_trainable_inventory"]["value"]["elements"]
    es_dense_elements = 142_999_552
    es_peak_mib = 86_973_087_744 / (1024 * 1024)
    sft_peak_mib = max(
        item["peak_memory_used_mib"]
        for item in sft["gpu_activity"]["by_gpu"].values()
    )

    comparison = {
        "es_sequence_presentations": es_sequence_presentations,
        "es_token_presentations": es_token_presentations,
        "es_calls_per_gpu": calls_per_gpu,
        "es_scoring_critical_path_seconds": es_scoring_critical_path_seconds,
        "es_scoring_timing_scope": (
            "optimistic resident inference critical path; excludes perturbation "
            "install/restore, reduction/update, model load, compile, and warmup"
        ),
        "sft_nominal_sequence_presentations": sft_nominal_rows,
        "sft_padded_sequence_presentations": sft_padded_rows,
        "sft_nominal_token_presentations": sft_nominal_tokens,
        "sft_optimizer_steps": 57,
        "sft_train_runtime_seconds": sft_train_runtime,
        "sft_wall_runtime_seconds": sft_wall_runtime,
        "es_over_sft_padded_sequence_ratio": (
            es_sequence_presentations / sft_padded_rows
        ),
        "es_over_sft_nominal_token_ratio": (
            es_token_presentations / sft_nominal_tokens
        ),
        "sft_train_runtime_over_es_scoring_path": (
            sft_train_runtime / es_scoring_critical_path_seconds
        ),
        "sft_wall_runtime_over_es_scoring_path": (
            sft_wall_runtime / es_scoring_critical_path_seconds
        ),
        "es_dense_over_sft_lora_element_ratio": es_dense_elements / sft_trainable,
        "es_peak_memory_mib": es_peak_mib,
        "sft_peak_memory_mib": sft_peak_mib,
        "sft_over_es_peak_memory_ratio": sft_peak_mib / es_peak_mib,
    }
    result = {
        "schema": "specialist-train-only-es-sft-runtime-comparison-v36c",
        "status": "complete_train_only_runtime_comparison",
        "contains_validation_ood_or_holdout_content": False,
        "sources": {
            "sft_report": {
                "path": str(SFT_REPORT),
                "file_sha256": EXPECTED["sft_report_file"],
                "content_sha256": EXPECTED["sft_report_content"],
            },
            "es_runtime_report": {
                "path": str(ES_RUNTIME_REPORT),
                "file_sha256": EXPECTED["es_runtime_file"],
                "content_sha256": EXPECTED["es_runtime_content"],
            },
            "es_recipe_attempt": {
                "path": str(ES_RECIPE_ATTEMPT),
                "file_sha256": EXPECTED["es_recipe_file"],
                "content_sha256": EXPECTED["es_recipe_content"],
            },
        },
        "comparison": comparison,
        "interpretation": {
            "supported": (
                "The sparse-LoRA control produced 57 first-order steps while "
                "using 8.29x fewer nominal token presentations and 11.23x fewer "
                "padded sequence presentations than one population-32 ES update."
            ),
            "not_supported": (
                "The ES 18.5-second value is not an end-to-end update measurement, "
                "and train loss does not establish validation or OOD quality."
            ),
            "next_gate": (
                "document/conflict-unit-disjoint train shadow comparison under "
                "equal wall-clock budgets before any sealed evaluation"
            ),
        },
    }
    result["content_sha256_before_self_field"] = runtime.canonical_sha256(result)
    return result


def main():
    result = build()
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
