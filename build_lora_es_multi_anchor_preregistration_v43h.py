#!/usr/bin/env python3
"""Seal the CPU/static multi-anchor LoRA-ES V43H design contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_general_qa_proxy_anchor_v43h as qa_builder
import eggroll_es_multi_anchor_v43h as projection


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_multi_anchor_static_v43h.json"
).resolve()
V44C_AGGREGATE = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44c.json"
).resolve()
V43G_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_robust_v43g.json"
).resolve()
V45A_AGGREGATE = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_hpo_earlystop_ood_eligible_eval_v45a.json"
).resolve()
PROSE_ANCHOR = qa_builder.PARENT_PROSE
PROSE_REPORT = qa_builder.PARENT_REPORT
QA_PROXY = qa_builder.DEFAULT_OUTPUT
QA_PROXY_REPORT = qa_builder.DEFAULT_REPORT
EXPECTED_V44C_FILE_SHA256 = "3b3a55b9b208fd563825f09c36f855d74f05cad2bf9201467fb454a08105b076"
EXPECTED_V43G_FILE_SHA256 = "b923bd72ed9d09936200b7aae0f6c25671f901a41d26e062e18371eab845e3a6"
EXPECTED_V45A_FILE_SHA256 = "64f0edab64c66aa208a03759f759a3fe62d3bc886e6fce55180abe7b57cb919d"
EXPECTED_QA_PROXY_SHA256 = "d250980c5b88308452aba4ee8d3e43090b1444c250547c2515c838593f2f391f"
EXPECTED_QA_PROXY_REPORT_SHA256 = "b176f4ea0d1d4b466c83b68c243bfb4a9e315df8574d9ba2c606333fbfc4ee8f"
ANCHOR_PANEL_SIZE_V43H = 32
ANCHOR_PANEL_SEED_V43H = "v43h-fixed-multi-anchor-panel-20260715"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return projection.canonical_sha256_v43h(value)


def _self_hashed(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if content != canonical_sha256(compact):
        raise RuntimeError(f"v43h parent self hash changed: {path}")
    return value


def _v44c_numeric_evidence() -> dict:
    if file_sha256(V44C_AGGREGATE) != EXPECTED_V44C_FILE_SHA256:
        raise RuntimeError("v43h V44C aggregate identity changed")
    value = _self_hashed(V44C_AGGREGATE)
    if (
        value.get("status") != "complete_aggregate_only_no_heldout_access"
        or value.get("heldout_or_holdout_opened") is not False
        or value.get("raw_questions_answers_or_generations_persisted_in_aggregate")
        is not False
        or value.get("selection", {}).get("selected_arm") != "sft_v42b"
        or value.get("final_gate", {}).get("passed") is not False
    ):
        raise RuntimeError("v43h V44C aggregate-only evidence changed")
    return {
        "schema": value["schema"],
        "content_sha256": value["content_sha256_before_self_field"],
        "selected_arm": value["selection"]["selected_arm"],
        "shadow_improvement": value["final_gate"]["shadow_improvement"],
        "ood_qa_mean_reward_delta": value["ood_qa_gate"]["mean_reward_delta"],
        "ood_qa_exact_count_delta": value["ood_qa_gate"]["exact_count_delta"],
        "ood_qa_gate_passed": value["ood_qa_gate"]["passed"],
        "ood_prose_point_delta": value["ood_prose_gate"]["delta"],
        "ood_prose_paired_95_ci": value["ood_prose_gate"][
            "paired_document_bootstrap_95_ci"
        ],
        "ood_prose_gate_passed": value["ood_prose_gate"]["passed"],
        "raw_semantics_persisted": False,
        "heldout_or_holdout_opened": False,
    }


def _v45a_numeric_evidence() -> dict:
    if file_sha256(V45A_AGGREGATE) != EXPECTED_V45A_FILE_SHA256:
        raise RuntimeError("v43h V45A aggregate identity changed")
    value = _self_hashed(V45A_AGGREGATE)
    if (
        value.get("status") != "complete_aggregate_only_no_heldout_access"
        or value.get("heldout_or_holdout_opened") is not False
        or value.get("raw_questions_answers_or_generations_persisted_in_aggregate")
        is not False
        or value.get("selection", {}).get("selected_arm") != "sft_v42g"
        or value.get("final_gate", {}).get("passed") is not True
    ):
        raise RuntimeError("v43h V45A aggregate-only evidence changed")
    base_shadow = value["shadow"]["metrics"]["base_a"]
    selected_shadow = value["shadow"]["metrics"]["sft_v42g"]
    return {
        "schema": value["schema"],
        "content_sha256": value["content_sha256_before_self_field"],
        "selected_arm": "sft_v42g",
        "shadow_equal_unit_base": base_shadow["generated_equal_unit_mean_reward"],
        "shadow_equal_unit_selected": selected_shadow[
            "generated_equal_unit_mean_reward"
        ],
        "ood_qa_mean_reward_delta": value["ood_qa_gate"]["mean_reward_delta"],
        "ood_qa_exact_count_delta": value["ood_qa_gate"]["exact_count_delta"],
        "ood_prose_point_delta": value["ood_prose_gate"]["delta"],
        "ood_prose_paired_95_ci": value["ood_prose_gate"][
            "paired_document_bootstrap_95_ci"
        ],
        "final_gate_passed": True,
        "raw_semantics_persisted": False,
        "heldout_or_holdout_opened": False,
    }


def _anchor_panel() -> dict:
    if (
        file_sha256(QA_PROXY) != EXPECTED_QA_PROXY_SHA256
        or file_sha256(QA_PROXY_REPORT) != EXPECTED_QA_PROXY_REPORT_SHA256
    ):
        raise RuntimeError("v43h QA proxy identity changed")
    report = _self_hashed(QA_PROXY_REPORT)
    if (
        report.get("output_sha256") != EXPECTED_QA_PROXY_SHA256
        or report.get("rows") != 128
        or report.get("direct_benchmark_source", {}).get("opened") is not False
        or report.get("direct_benchmark_source", {}).get(
            "authorized_for_qa_semantics"
        ) is not False
    ):
        raise RuntimeError("v43h QA proxy firewall receipt changed")
    items = [
        json.loads(line) for line in QA_PROXY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ordered = sorted(items, key=lambda item: hashlib.sha256(
        (ANCHOR_PANEL_SEED_V43H + "\0" + item["item_id"]).encode("utf-8")
    ).digest())
    selected = ordered[:ANCHOR_PANEL_SIZE_V43H]
    identities = [item["item_id"] for item in selected]
    documents = [item["document_id"] for item in selected]
    if len(set(identities)) != 32 or len(set(documents)) != 32:
        raise RuntimeError("v43h fixed anchor panel is not document unique")
    return {
        "selection": "keyed SHA-256 order without semantic selection",
        "seed": ANCHOR_PANEL_SEED_V43H,
        "items": len(selected),
        "item_identity_sha256": canonical_sha256(identities),
        "document_identity_sha256": canonical_sha256(documents),
        "full_candidate_acceptance_items": len(items),
    }


def build_v43h() -> dict:
    if file_sha256(V43G_PREREGISTRATION) != EXPECTED_V43G_FILE_SHA256:
        raise RuntimeError("v43h V43G preregistration identity changed")
    v43g = _self_hashed(V43G_PREREGISTRATION)
    if (
        v43g.get("sealed_holdout_opened") is not False
        or v43g.get("recipe", {}).get("sigma") != 0.0006
        or v43g.get("recipe", {}).get("alpha") != 0.00015
        or v43g.get("recipe", {}).get("signed_replicates_per_direction") != 4
    ):
        raise RuntimeError("v43h inherited V43G train-only recipe changed")
    if (
        file_sha256(PROSE_ANCHOR) != qa_builder.PARENT_PROSE_SHA256
        or file_sha256(PROSE_REPORT) != qa_builder.PARENT_REPORT_SHA256
    ):
        raise RuntimeError("v43h approved prose anchor identity changed")
    qa_report = _self_hashed(QA_PROXY_REPORT)
    evidence = _v44c_numeric_evidence()
    v45a_evidence = _v45a_numeric_evidence()
    if (
        evidence["ood_qa_mean_reward_delta"] != -0.4081396422205246
        or evidence["ood_qa_exact_count_delta"] != -11
        or evidence["ood_prose_paired_95_ci"]
        != [-0.0018113212382170451, 0.007535356011299148]
    ):
        raise RuntimeError("v43h frozen V44C motivation changed")
    value = {
        "schema": "matched-lora-es-multi-anchor-preregistration-v43h",
        "status": "cpu_static_design_preregistered_runtime_not_implemented",
        "created_at_utc": "2026-07-15T23:10:00+00:00",
        "gpu_launch_authorized": False,
        "model_or_gpu_opened_by_builder": False,
        "sealed_holdout_opened": False,
        "shadow_dev_eval_ood_or_benchmark_semantics_authorized": False,
        "purpose": (
            "Preserve train-only general prose and general-knowledge instruction "
            "behavior while optimizing the specialist objective, in response to "
            "V44C aggregate evidence that in-domain selection can hide OOD-QA loss."
        ),
        "parents": {
            "v43g_train_only_preregistration": {
                "path": str(V43G_PREREGISTRATION),
                "file_sha256": EXPECTED_V43G_FILE_SHA256,
                "content_sha256": v43g["content_sha256_before_self_field"],
            },
            "v44c_aggregate_only_evidence": {
                "path": str(V44C_AGGREGATE),
                "file_sha256": EXPECTED_V44C_FILE_SHA256,
                "numeric_evidence": evidence,
            },
            "v45a_aggregate_only_counterevidence": {
                "path": str(V45A_AGGREGATE),
                "file_sha256": EXPECTED_V45A_FILE_SHA256,
                "numeric_evidence": v45a_evidence,
                "interpretation": (
                    "conservative SFT/early stopping can avoid the V44C collapse; "
                    "V43H anchors specifically defend ES against the observed "
                    "high-update overfit mode, not against all fine-tuning"
                ),
            },
            "approved_prose_anchor": {
                "path": str(PROSE_ANCHOR),
                "file_sha256": qa_builder.PARENT_PROSE_SHA256,
                "report_path": str(PROSE_REPORT),
                "report_file_sha256": qa_builder.PARENT_REPORT_SHA256,
            },
            "general_qa_proxy_anchor": {
                "path": str(QA_PROXY),
                "file_sha256": EXPECTED_QA_PROXY_SHA256,
                "report_path": str(QA_PROXY_REPORT),
                "report_file_sha256": EXPECTED_QA_PROXY_REPORT_SHA256,
                "report_content_sha256": qa_report[
                    "content_sha256_before_self_field"
                ],
            },
        },
        "source_firewall": {
            "direct_hotpotqa_benchmark_qa_authorized": False,
            "direct_hotpotqa_benchmark_opened": False,
            "direct_hotpotqa_path": str(qa_builder.DIRECT_BENCHMARK_SOURCE),
            "qa_proxy_semantic_source": str(PROSE_ANCHOR),
            "qa_proxy_parent_collision_receipt_inherited": True,
            "protected_semantics_used_for_anchor_construction": False,
        },
        "prose_anchor_audit": {
            "rows": 128,
            "unique_documents": 128,
            "unique_texts": 128,
            "word_count_min_median_max": [81, 131.5, 242],
            "snippet_join_double_period_rows": 119,
            "use": "preservation constraint, not quality-target corpus expansion",
        },
        "recipe": {
            "population_size": 8,
            "signed_replicates_per_direction": 4,
            "sigma": 0.0006,
            "alpha": 0.00015,
            "domain_fitness": "V43G equal-conflict-unit teacher-forced answer logprob",
            "per_objective_shaping": (
                "median-of-four signed scores then centered ranks over 16 signs; "
                "coefficient=u_plus-u_minus"
            ),
            "required_gradient_anchors": ["prose_lm", "qa_answer_logprob"],
            "qa_generation_role": (
                "full-anchor executed-candidate non-degradation gate; not a required "
                "gradient because greedy generation may have zero population spread"
            ),
            "projection": {
                "method": "simultaneous Euclidean active-set projection",
                "constraints": "coefficient dot each anchor >= 0",
                "sequential_projection_forbidden": True,
                "zero_spread_required_anchor": "skip update",
                "trust_region_max_norm_ratio": (
                    projection.TRUST_REGION_NORM_RATIO_V43H
                ),
                "restandardize_after_projection": False,
            },
            "population_anchor_panel": _anchor_panel(),
            "prose_scoring": (
                "teacher-forced selected-token mean logprob on approved prose text"
            ),
            "qa_logprob_scoring": (
                "teacher-forced answer-token mean logprob under fixed instruction chat"
            ),
            "qa_generation_scoring": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
                "metrics": ["normalized_token_f1", "exact_count", "nonzero_count"],
            },
        },
        "uncommitted_candidate_gate": {
            "protocol": (
                "prepare and execute exact FP32 sharded candidate; score before commit; "
                "commit only on all gates, otherwise abort to exact master"
            ),
            "full_anchor_documents": 128,
            "all_four_actor_states_exact": True,
            "prepopulation_numeric_calibration": (
                "fit per-metric fixed-panel inference margin before population; margin "
                "must remain under the historical V43F catastrophic ceiling"
            ),
            "required": [
                "specialist train equal-unit point delta positive",
                "prose logprob paired delta inside prefit noninferiority margin",
                "QA answer logprob paired delta inside prefit noninferiority margin",
                "QA generation F1/exact/nonzero counts do not degrade beyond prefit jitter",
                "abort restores exact canonical master on any failure",
            ],
        },
        "required_future_runtime_gates": {
            "all_four_gpus_attributed_positive_activity": True,
            "all_v43g_exact_state_and_restore_gates_preserved": True,
            "all_required_anchor_halfspaces_satisfied": True,
            "coefficient_norm_at_most_half_domain_norm": True,
            "full_train_only_multi_anchor_candidate_gate_passed_before_commit": True,
            "no_eval_ood_heldout_or_benchmark_semantic_access": True,
            "runtime_and_worker_hashes_sealed_before_launch": True,
        },
        "implementation_bindings": {
            "legacy_single_anchor": file_sha256(ROOT / "eggroll_es_anchor.py"),
            "multi_anchor_projection": file_sha256(
                ROOT / "eggroll_es_multi_anchor_v43h.py"
            ),
            "qa_proxy_builder": file_sha256(
                ROOT / "build_general_qa_proxy_anchor_v43h.py"
            ),
            "projection_tests": file_sha256(
                ROOT / "test_eggroll_es_multi_anchor_v43h.py"
            ),
        },
        "launch_blocker": (
            "A GPU runtime integrating fused anchor scoring, precommit candidate "
            "gating, exact abort, and sealed implementation hashes is not yet built."
        ),
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    value = build_v43h()
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if args.check:
        if output.read_bytes() != raw:
            raise RuntimeError("v43h static preregistration changed")
        return 0
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(raw)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
