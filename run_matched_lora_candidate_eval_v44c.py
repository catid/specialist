#!/usr/bin/env python3
"""Fresh V44C retry with source-faithful CPU parsing before model creation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import qa_quality
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44b as env_retry


ROOT = Path(__file__).resolve().parent
FAILED_ATTEMPT_V44B = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v44b_matched_lora_sft_es_fold3_ood_eval_retry_env.attempt.json"
).resolve()
FAILED_RUN_V44B = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v44b_matched_lora_sft_es_fold3_ood_eval_retry_env"
).resolve()
FAILED_FAILURE_V44B = FAILED_RUN_V44B / "failure_v44a.json"
FAILED_GPU_LOG_V44B = FAILED_RUN_V44B / "gpu_activity_v44b.jsonl"

EXPERIMENT = "v44c_matched_lora_sft_es_fold3_ood_eval_parser_preflight"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v44c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v44c.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44c.json"
).resolve()

FAILED_EXPECTED_V44B = {
    "attempt_file_sha256": (
        "4886a7d3130557e5eec9f25a4014ad19a43a0c84a9fc92cf5c385d808c020553"
    ),
    "attempt_content_sha256": (
        "259fc25e3cbe806a955929dee6361560a84c066e08ace59a9dc218de4a414d98"
    ),
    "failure_file_sha256": (
        "d929e8b77fbc10340723aba09405dc3d64fc7efc746576738b7d1081d988e7f9"
    ),
    "failure_content_sha256": (
        "e3459fee5848e2695ef4971c4cec27ceec1d9cb8148228416f7865677239e530"
    ),
    "gpu_log_file_sha256": (
        "6df2d223c408fa3cedbb0f092767f4c1d73027268e6465a0d27f101eca949eb9"
    ),
}


def failed_launch_provenance_v44c() -> dict:
    attempt = env_retry._read_self_hashed_nonprotected_v44b(
        FAILED_ATTEMPT_V44B,
        FAILED_EXPECTED_V44B["attempt_file_sha256"],
        FAILED_EXPECTED_V44B["attempt_content_sha256"],
    )
    failure = env_retry._read_self_hashed_nonprotected_v44b(
        FAILED_FAILURE_V44B,
        FAILED_EXPECTED_V44B["failure_file_sha256"],
        FAILED_EXPECTED_V44B["failure_content_sha256"],
    )
    if core.file_sha256(FAILED_GPU_LOG_V44B) != FAILED_EXPECTED_V44B[
        "gpu_log_file_sha256"
    ]:
        raise RuntimeError("V44C V44B GPU log changed")
    if (
        attempt.get("heldout_or_holdout_opened") is not False
        or failure.get("type") != "ValueError"
        or failure.get("message")
        != "text field is not a supported QA serialization"
        or failure.get("heldout_or_holdout_opened") is not False
        or 'firewall.jsonl("ood_qa")' not in failure.get("traceback", "")
        or (FAILED_RUN_V44B / "raw_items_v44b.json").exists()
        or env_retry.REPORT.exists()
    ):
        raise RuntimeError("V44C V44B parser-failure provenance changed")
    return {
        **FAILED_EXPECTED_V44B,
        "attempt": str(FAILED_ATTEMPT_V44B),
        "failure": str(FAILED_FAILURE_V44B),
        "gpu_log": str(FAILED_GPU_LOG_V44B),
        "failure_after_model_creation": True,
        "known_protected_semantic_access_order": [
            "split_manifest", "shadow", "ood_qa"
        ],
        "ood_prose_known_unread_by_control_flow": True,
        "heldout_or_holdout_opened": False,
        "raw_item_artifact_persisted": False,
        "aggregate_report_persisted": False,
        "shadow_metrics_persisted_or_observed": False,
    }


def clean_explicit_pair_v44c(question, answer) -> tuple[str, str]:
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("V44C explicit QA fields must both be strings")
    pair = qa_quality._clean_pair(question, answer)
    if pair is None:
        raise ValueError("V44C explicit QA fields are empty or contain protocol syntax")
    return pair


def qa_pair_from_record_v44c(item: dict) -> tuple[str, str]:
    """Parse only explicit fields or an existing strict supported serialization."""
    if not isinstance(item, dict):
        raise ValueError("V44C QA record must be an object")
    has_question = "question" in item
    has_answer = "answer" in item
    if has_question != has_answer:
        raise ValueError("V44C QA record has only one explicit field")
    explicit = (
        clean_explicit_pair_v44c(item["question"], item["answer"])
        if has_question else None
    )
    has_text = "text" in item
    if has_text and not isinstance(item["text"], str):
        raise ValueError("V44C QA text serialization must be a string")
    parsed = qa_quality.parse_qa(item["text"]) if has_text else None
    if has_text and parsed is None:
        raise ValueError("V44C text field is not a supported strict QA serialization")
    if explicit is not None and parsed is not None:
        if tuple(map(qa_quality.normalize_text, explicit)) != tuple(
            map(qa_quality.normalize_text, parsed)
        ):
            raise ValueError("V44C explicit QA fields disagree with serialized text")
        return explicit
    if explicit is not None:
        # The permitted OOD QA source is explicit-field JSONL with no synthetic
        # text serialization.  Preserve those source fields verbatim (trim only).
        return explicit
    if parsed is not None:
        return parsed
    raise ValueError("V44C QA record has neither explicit fields nor serialized QA")


def schema_receipt_v44c(rows: list[dict], label: str) -> dict:
    keysets = Counter(tuple(sorted(row)) for row in rows)
    types = Counter(
        (key, type(value).__name__)
        for row in rows for key, value in row.items()
    )
    item_ids = [row.get("item_id") for row in rows if "item_id" in row]
    if item_ids and (
        len(item_ids) != len(rows)
        or not all(isinstance(value, str) and value.strip() for value in item_ids)
        or len(set(item_ids)) != len(item_ids)
    ):
        raise ValueError(f"V44C {label} item_id inventory changed")
    return {
        "label": label,
        "rows": len(rows),
        "keyset_counts": [
            {"keys": list(keys), "count": count}
            for keys, count in sorted(keysets.items())
        ],
        "field_type_counts": [
            {"field": key, "type": kind, "count": count}
            for (key, kind), count in sorted(types.items())
        ],
        "item_id_inventory_sha256": (
            core.canonical_sha256(item_ids) if item_ids else None
        ),
    }


def shadow_bundle_v44c(rows: list[dict], manifest: dict) -> tuple[dict, dict, dict]:
    if len(rows) != 83:
        raise RuntimeError("V44C shadow row count changed")
    fold = manifest.get("folds", [None] * 4)[3]
    intersections = fold.get("train_dev_edge_identity_intersections", {})
    if (
        not isinstance(intersections, dict) or not intersections
        or any(intersections.values())
        or fold.get("train", {}).get("rows") != 448
        or fold.get("shadow_dev", {}).get("rows") != 83
    ):
        raise RuntimeError("V44C shadow/train document-disjoint proof changed")
    commitments = {}
    for unit in manifest.get("content_free_unit_commitments", []):
        if unit.get("fold") == 3:
            for row_sha in unit["row_sha256"]:
                commitments[row_sha] = (
                    unit["unit_identity_sha256"], unit["row_count"]
                )
    row_hashes = [core.v39a.shadow.row_sha256(row) for row in rows]
    if set(row_hashes) != set(commitments) or len(commitments) != 83:
        raise RuntimeError("V44C shadow content-free commitments changed")
    pairs = [qa_pair_from_record_v44c(row) for row in rows]
    weights = [1.0 / (51 * commitments[row_sha][1]) for row_sha in row_hashes]
    if (
        len({unit for unit, _ in commitments.values()}) != 51
        or not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-15, rel_tol=0)
    ):
        raise RuntimeError("V44C shadow conflict-unit weights changed")
    bundle = {
        "questions": [pair[0] for pair in pairs],
        "answers": [pair[1] for pair in pairs],
        "item_sha256": row_hashes,
        "weights": weights,
        "rows": 83,
        "units": 51,
    }
    proof = {
        "document_disjoint_from_fold3_train": True,
        "train_rows": 448,
        "shadow_rows": 83,
        "shadow_conflict_units": 51,
        "edge_identity_intersections": intersections,
        "split_manifest_file_sha256": core.PROTECTED_INPUTS_V44A[
            "split_manifest"
        ]["file_sha256"],
    }
    receipt = schema_receipt_v44c(rows, "shadow")
    receipt["parsed_pair_identity_sha256"] = core.canonical_sha256(pairs)
    receipt["bundle_numeric_identity_sha256"] = core.canonical_sha256({
        "item_sha256": row_hashes, "weights": weights
    })
    return bundle, proof, receipt


def ood_qa_bundle_v44c(rows: list[dict]) -> tuple[dict, dict]:
    if len(rows) != 24:
        raise RuntimeError("V44C OOD QA row count changed")
    pairs = [qa_pair_from_record_v44c(row) for row in rows]
    identities = [
        core.canonical_sha256({"question": q, "answer": a}) for q, a in pairs
    ]
    if len(set(identities)) != len(rows):
        raise RuntimeError("V44C OOD QA identities repeated")
    bundle = {
        "questions": [pair[0] for pair in pairs],
        "answers": [pair[1] for pair in pairs],
        "item_sha256": identities,
        "weights": [1.0 / len(rows)] * len(rows),
        "rows": len(rows), "units": len(rows),
    }
    receipt = schema_receipt_v44c(rows, "ood_qa")
    receipt.update({
        "serialization": "explicit_question_answer_fields_no_text_field",
        "parsed_pair_identity_sha256": core.canonical_sha256(pairs),
        "evaluation_identity_sha256": core.canonical_sha256(identities),
    })
    return bundle, receipt


def ood_prose_preflight_v44c(rows: list[dict]) -> dict:
    if len(rows) != 16:
        raise RuntimeError("V44C OOD prose row count changed")
    for row in rows:
        if (
            not isinstance(row.get("text"), str) or not row["text"].strip()
            or not isinstance(row.get("item_id"), str)
            or not row["item_id"].strip()
            or qa_quality.has_protocol_tokens(row["text"])
        ):
            raise ValueError("V44C OOD prose source fields changed")
    receipt = schema_receipt_v44c(rows, "ood_prose")
    receipt["source_text_identity_sha256"] = core.canonical_sha256([
        hashlib.sha256(row["text"].encode()).hexdigest() for row in rows
    ])
    return receipt


def run_cpu_preflight_v44c(firewall: core.SingleSemanticAccessV44A) -> dict:
    manifest = firewall.json("split_manifest")
    shadow_rows = firewall.jsonl("shadow")
    ood_qa_rows = firewall.jsonl("ood_qa")
    ood_prose_rows = firewall.jsonl("ood_prose")
    shadow_bundle, proof, shadow_receipt = shadow_bundle_v44c(
        shadow_rows, manifest
    )
    ood_bundle, qa_receipt = ood_qa_bundle_v44c(ood_qa_rows)
    prose_receipt = ood_prose_preflight_v44c(ood_prose_rows)
    aggregate = {
        "schema": "source-faithful-protected-parser-preflight-v44c",
        "status": "complete_before_model_creation",
        "parser_runtime_sha256": core.file_sha256(Path(__file__).resolve()),
        "protected_file_sha256": {
            label: item["file_sha256"] for label, item in firewall.receipts.items()
        },
        "semantic_access_counts": {
            label: item["semantic_read_count"]
            for label, item in firewall.receipts.items()
        },
        "schema_receipts": [shadow_receipt, qa_receipt, prose_receipt],
        "split_manifest_schema_identity_sha256": core.canonical_sha256({
            "keys": sorted(manifest),
            "fold_count": len(manifest.get("folds", [])),
        }),
        "all_four_authorized_inputs_parsed_once": True,
        "holdout_or_heldout_opened": False,
    }
    aggregate["content_sha256_before_self_field"] = core.canonical_sha256(
        aggregate
    )
    return {
        "status": "complete_before_model_creation",
        "shadow_bundle": shadow_bundle,
        "document_disjointness": proof,
        "ood_qa_bundle": ood_bundle,
        "ood_prose_rows": ood_prose_rows,
        "aggregate_receipt": aggregate,
    }


def protected_preflight_v44c(firewall, prereg) -> dict:
    result = run_cpu_preflight_v44c(firewall)
    if result["aggregate_receipt"] != prereg.get("cpu_preflight_expected"):
        raise RuntimeError("V44C CPU parser preflight identity changed")
    return result


def offline_authorized_schema_audit_v44c() -> dict:
    """Authorized schema audit used to freeze the post-V44B parser repair."""
    firewall = core.SingleSemanticAccessV44A(core.PROTECTED_INPUTS_V44A)
    return run_cpu_preflight_v44c(firewall)["aggregate_receipt"]


def implementation_bindings_v44c() -> dict:
    return {
        "retry_runtime": core.file_sha256(Path(__file__).resolve()),
        "core": core.implementation_bindings_v44a(),
        "environment_runtime": core.file_sha256(Path(env_retry.__file__).resolve()),
        "worker_extension": core.file_sha256(
            ROOT / "eggroll_es_worker_lora_topology_v40a.py"
        ),
        "environment": env_retry.environment_bindings_v44b(),
        "failed_launch": failed_launch_provenance_v44c(),
    }


def load_preregistration_v44c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V44C preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256(compact)
        or value.get("schema")
        != "matched-lora-candidate-eval-preregistration-v44c"
        or value.get("status")
        != "preregistered_after_schema_audit_before_fresh_parser_retry"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("schema_only_authorized_inputs_opened_before_preregistration")
        is not True
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("staged_adapters") != core.staged_adapter_bindings_v44a()
        or value.get("implementation_bindings") != implementation_bindings_v44c()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V44C preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    # Dry-run verifies commitments only; the protected files are not reopened.
    return value


def main(argv: list[str] | None = None) -> int:
    env_retry.environment_bindings_v44b()
    saved = {
        "EXPERIMENT": core.EXPERIMENT, "RUN_DIR": core.RUN_DIR,
        "ATTEMPT": core.ATTEMPT, "RAW": core.RAW,
        "GPU_LOG": core.GPU_LOG, "REPORT": core.REPORT,
        "load": core.load_preregistration,
        "preflight": core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A,
    }
    core.EXPERIMENT = EXPERIMENT
    core.RUN_DIR = RUN_DIR
    core.ATTEMPT = ATTEMPT
    core.RAW = RAW
    core.GPU_LOG = GPU_LOG
    core.REPORT = REPORT
    core.load_preregistration = load_preregistration_v44c
    core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = protected_preflight_v44c
    try:
        return core.main(argv)
    finally:
        core.EXPERIMENT = saved["EXPERIMENT"]
        core.RUN_DIR = saved["RUN_DIR"]
        core.ATTEMPT = saved["ATTEMPT"]
        core.RAW = saved["RAW"]
        core.GPU_LOG = saved["GPU_LOG"]
        core.REPORT = saved["REPORT"]
        core.load_preregistration = saved["load"]
        core.PRE_MODEL_PROTECTED_PREFLIGHT_V44A = saved["preflight"]


if __name__ == "__main__":
    raise SystemExit(main())
