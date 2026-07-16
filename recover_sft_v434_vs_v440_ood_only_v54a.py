#!/usr/bin/env python3
"""Recover V54A OOD aggregates CPU-only from immutable run receipts."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import recover_sft_v434_sampling_midpoint_ood_only_v49d as base


ROOT = Path(__file__).resolve().parent
SOURCE_EXPERIMENT = "v54a_v434_equal_vs_v440_equal_replicated_ood_only"
SOURCE_RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / SOURCE_EXPERIMENT
).resolve()
SOURCE_ATTEMPT = (
    SOURCE_RUN_DIR.parent / f".{SOURCE_EXPERIMENT}.attempt.json"
).resolve()
SOURCE_FAILURE = (SOURCE_RUN_DIR / "failure_v49d.json").resolve()
SOURCE_RAW = (SOURCE_RUN_DIR / "raw_items_v54a.json").resolve()
SOURCE_GPU_LOG = (SOURCE_RUN_DIR / "gpu_activity_v54a.jsonl").resolve()
SOURCE_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_v440_equal_replicated_ood_only_v54a.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_v440_equal_replicated_ood_only_recovery_v54a.json"
).resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v434_equal_vs_v440_equal_replicated_ood_only_recovery_v54a.json"
).resolve()
BASE_ARMS = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS = {
    "v434_equal": ("v434_equal_a", "v434_equal_b"),
    "v440_equal": ("v440_equal_a", "v440_equal_b"),
}
CANDIDATE_ARMS = tuple(
    arm for replicas in LOGICAL_REPLICAS.values() for arm in replicas
)
ARMS = BASE_ARMS + CANDIDATE_ARMS
PHASES = ("ood_qa", "ood_prose")
ROW_COUNTS = {"ood_qa": 24, "ood_prose": 16}
WAVES = (
    tuple((arm, index) for index, arm in enumerate(BASE_ARMS)),
    tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS)),
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_715
SOURCE_EXPECTED = {
    "preregistration_file": "4e7c54eba5d6e0e2b45ca3dd8a85cc97527473d3cc6bc37e02ae18e97b5ffd95",
    "preregistration_content": "bbb482cd8ded82db337baa2b9ebfddca56240ab126ad06b96d6b5fc65b0ce0ae",
    "attempt_file": "47e038648640dd25222fb93405b1f5bae892bd3d4eaa62eede1f6ce830ffe150",
    "attempt_content": "6e896d4ed0deba99ee9cb9747167b04cbc5d28c7bce0b11f8c4cbb9c2805ee64",
    "failure_file": "21a80dc18efa83b00db57d0273429949373144908c6cd68f0960fe054a296aee",
    "failure_content": "4d457de320b582844287cfefaf2395ae4fae2fa8e36a0c141ae0e0d15038eff1",
    "raw_file": "032614a9a5f58c5b7fed93243ce10822e71a2e358c89427eb16740ea74a655f9",
    "gpu_log_file": "199c6607e5e9a2e8c2242aa73862c57788f658b39916d4e70b7ec0878de9de8a",
}

canonical_sha256 = base.canonical_sha256
file_sha256 = base.file_sha256
self_hashed = base.self_hashed
atomic_json = base.atomic_json
_load_json = base._load_json
_validate_self_hash = base._validate_self_hash


@contextmanager
def _patched_base_contract():
    names = {
        "BASE_ARMS": BASE_ARMS,
        "LOGICAL_REPLICAS": LOGICAL_REPLICAS,
        "CANDIDATE_ARMS": CANDIDATE_ARMS,
        "ARMS": ARMS,
        "PHASES": PHASES,
        "ROW_COUNTS": ROW_COUNTS,
        "WAVES": WAVES,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    }
    saved = {name: getattr(base, name) for name in names}
    for name, value in names.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def receipt_inventory_v54a(raw: dict) -> dict:
    with _patched_base_contract():
        value = base.receipt_inventory_v49d(raw)
    value["schema"] = "v54a-offline-raw-receipt-inventory"
    return value


def gpu_receipt_v54a(path: Path) -> dict:
    with _patched_base_contract():
        value = base.gpu_receipt_v49d(path)
    value["schema"] = "v54a-offline-ood-only-gpu-receipt"
    return value


def implementation_bindings_v54a() -> dict:
    paths = {
        "recovery_runtime": Path(__file__).resolve(),
        "recovery_builder": (
            ROOT / "build_sft_v434_vs_v440_ood_recovery_preregistration_v54a.py"
        ),
        "recovery_tests": ROOT / "test_sft_v434_vs_v440_ood_recovery_v54a.py",
        "source_runtime": (
            ROOT / "run_sft_v434_vs_v440_replicated_ood_only_v54a.py"
        ),
        "metric_runtime": Path(base.metric_runtime.__file__).resolve(),
        "paired_qa_runtime": Path(base.paired_qa.__file__).resolve(),
        "reward_runtime": Path(base.reward_runtime.__file__).resolve(),
        "recovery_parent_v49d": Path(base.__file__).resolve(),
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _gate_table_v54a(qa: dict, prose_details: dict, raw: dict):
    table = {}
    with _patched_base_contract():
        for logical, replicas in LOGICAL_REPLICAS.items():
            gates = [
                base._replica_gate(qa, prose_details, raw, arm)
                for arm in replicas
            ]
            table[logical] = {
                "replicas": list(replicas),
                "replica_gates": gates,
                "both_replicas_independently_ood_eligible": all(
                    gate["eligible"] for gate in gates
                ),
            }
    v434 = LOGICAL_REPLICAS["v434_equal"]
    v440 = LOGICAL_REPLICAS["v440_equal"]

    def mean_qa(arms, field):
        return sum(float(qa[arm][field]) for arm in arms) / 2.0

    def mean_prose(arms):
        return sum(
            float(prose_details[arm]["mean_token_logprob"])
            for arm in arms
        ) / 2.0

    direct = {
        "comparison": "mean(v440_equal replicas)-mean(v434_equal replicas)",
        "v440_minus_v434_mean_reward": (
            mean_qa(v440, "generated_equal_unit_mean_reward")
            - mean_qa(v434, "generated_equal_unit_mean_reward")
        ),
        "v440_minus_v434_mean_exact_count": (
            mean_qa(v440, "generated_exact_count")
            - mean_qa(v434, "generated_exact_count")
        ),
        "v440_minus_v434_mean_prose_token_logprob": (
            mean_prose(v440) - mean_prose(v434)
        ),
        "paired_bootstrap_ci_role": "informational_not_a_direct_gate",
    }
    direct["reward_nonnegative"] = (
        direct["v440_minus_v434_mean_reward"] >= 0.0
    )
    direct["exact_nonnegative"] = (
        direct["v440_minus_v434_mean_exact_count"] >= 0.0
    )
    direct["prose_nonregression"] = (
        direct["v440_minus_v434_mean_prose_token_logprob"] >= 0.0
    )
    direct["all_direct_point_gates_passed"] = (
        direct["reward_nonnegative"]
        and direct["exact_nonnegative"]
        and direct["prose_nonregression"]
    )
    return table, direct


def load_preregistration_v54a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V54A recovery preregistration file changed")
    value = _load_json(path)
    content = _validate_self_hash(value, "V54A recovery preregistration")
    if (
        content != args.preregistration_content_sha256
        or value.get("schema")
        != "v54a-ood-only-offline-recovery-preregistration"
        or value.get("status") != "sealed_before_offline_recovery"
        or value.get("generation_or_gpu_access_authorized") is not False
        or value.get("protected_semantic_input_access_authorized") is not False
        or value.get("source_artifacts", {}).get("raw", {}).get("path")
        != str(SOURCE_RAW)
        or value.get("implementation_bindings")
        != implementation_bindings_v54a()
    ):
        raise RuntimeError("V54A recovery preregistration content changed")
    return value


def _validate_source_artifact(path: Path, expected: dict, label: str) -> dict:
    if (
        str(path) != expected.get("path")
        or file_sha256(path) != expected.get("file_sha256")
    ):
        raise RuntimeError(f"V54A sealed {label} artifact changed")
    value = _load_json(path)
    content = _validate_self_hash(value, label)
    if content != expected.get("content_sha256"):
        raise RuntimeError(f"V54A sealed {label} content changed")
    return value


def recover_v54a(prereg: dict) -> dict:
    source = prereg["source_artifacts"]
    source_prereg = _validate_source_artifact(
        SOURCE_PREREGISTRATION,
        source["source_preregistration"],
        "source preregistration",
    )
    attempt = _validate_source_artifact(
        SOURCE_ATTEMPT, source["attempt"], "source attempt"
    )
    failure = _validate_source_artifact(
        SOURCE_FAILURE, source["failure"], "source failure"
    )
    expected_waves = [
        [{"arm": arm, "engine_index": engine} for arm, engine in wave]
        for wave in WAVES
    ]
    if (
        source_prereg.get("schema")
        != "sft-v434-equal-vs-v440-equal-replicated-ood-only-v54a"
        or source_prereg.get("runtime", {}).get("two_full_fixed_waves")
        != expected_waves
        or source_prereg.get("shadow_access_authorized") is not False
        or source_prereg.get("heldout_or_holdout_access_authorized") is not False
        or attempt.get("protected_semantic_access_count") != 0
        or attempt.get("shadow_opened") is not False
        or attempt.get("heldout_or_holdout_opened") is not False
        or failure.get("protected_semantic_access_count") != 2
        or failure.get("protected_semantic_access_labels")
        != ["ood_prose", "ood_qa"]
        or failure.get("shadow_opened") is not False
        or failure.get("heldout_or_holdout_opened") is not False
        or failure.get("message") != "v39a GPU 0 inactive in shadow"
    ):
        raise RuntimeError("V54A source seal scope/access receipt changed")
    if (
        file_sha256(SOURCE_RAW) != source["raw"]["file_sha256"]
        or oct(SOURCE_RAW.stat().st_mode & 0o777) != "0o600"
    ):
        raise RuntimeError("V54A sealed raw receipt changed")
    raw = _load_json(SOURCE_RAW)
    inventory = receipt_inventory_v54a(raw)
    if inventory != prereg["expected_receipt_inventory"]:
        raise RuntimeError("V54A raw receipt inventory/wave hashes changed")
    receipts = raw["single_access_receipts"]
    if (
        set(receipts) != set(PHASES)
        or any(
            receipts[label].get("semantic_read_count") != 1
            for label in PHASES
        )
        or any(receipts[label].get("rows") != ROW_COUNTS[label] for label in PHASES)
        or {
            label: {
                "path": receipts[label].get("path"),
                "file_sha256": receipts[label].get("file_sha256"),
            }
            for label in PHASES
        } != source_prereg.get("single_access_inputs")
    ):
        raise RuntimeError("V54A semantic access receipts changed")
    gpu = gpu_receipt_v54a(SOURCE_GPU_LOG)
    if gpu != prereg["expected_gpu_receipt"]:
        raise RuntimeError("V54A GPU receipt changed")
    with _patched_base_contract():
        qa = {arm: base._qa_aggregate(raw["ood_qa"][arm]) for arm in ARMS}
        prose_details = {
            arm: base._prose_detail(raw["ood_prose"][arm]) for arm in ARMS
        }
        prose = {
            arm: base._prose_aggregate(prose_details[arm]) for arm in ARMS
        }
        base_equivalence = {
            "ood_qa": base._assert_exact_bases(qa, "ood_qa"),
            "ood_prose": base._assert_exact_bases(prose, "ood_prose"),
        }
    gates, direct = _gate_table_v54a(qa, prose_details, raw)
    return self_hashed({
        "schema": "v54a-v434-equal-v440-equal-ood-only-offline-recovery",
        "status": "complete_from_sealed_receipts_shadow_and_holdout_unopened",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_preregistration": {
            "path": str(Path(prereg["artifact_paths"]["preregistration"])),
            "file_sha256": file_sha256(
                Path(prereg["artifact_paths"]["preregistration"])
            ),
            "content_sha256": prereg["content_sha256_before_self_field"],
        },
        "source_generation": {
            "experiment": SOURCE_EXPERIMENT,
            "status": "both_waves_complete_then_out_of_scope_telemetry_failure",
            "failure": source["failure"],
            "raw_local": {**source["raw"], "mode": "0600", "git_eligible": False},
            "gpu_log": source["gpu_log"],
        },
        "receipt_inventory": inventory,
        "base_duplicate_equivalence": base_equivalence,
        "ood_qa": qa,
        "ood_prose": prose,
        "per_logical_candidate_gate_table": gates,
        "direct_v440_vs_v434_point_gates": direct,
        "gpu_activity": gpu,
        "access_proof": {
            "source_protected_semantic_access_count": 2,
            "source_protected_semantic_access_labels": ["ood_prose", "ood_qa"],
            "source_each_authorized_semantic_input_read_exactly_once": True,
            "offline_recovery_protected_semantic_input_reads": 0,
            "offline_recovery_model_or_generation_access": False,
            "offline_recovery_gpu_access": False,
            "raw_receipt_labels_exact": ["ood_qa", "ood_prose"],
            "gpu_phase_labels_exact": ["ood_qa", "ood_prose"],
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
        },
        "recovery_scope": {
            "metrics_and_gates_recomputed_from_sealed_raw_receipt_only": True,
            "raw_receipt_whole_file_sha256_validated_before_parse": True,
            "inventory_and_two_wave_receipt_hashes_validated": True,
            "gpu_receipt_whole_file_sha256_validated": True,
            "selection_or_promotion_authorized": False,
        },
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", default=str(DEFAULT_PREREGISTRATION))
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v54a(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "source_raw_file_sha256": prereg["source_artifacts"]["raw"]["file_sha256"],
            "source_gpu_log_file_sha256": prereg["source_artifacts"]["gpu_log"]["file_sha256"],
            "protected_semantic_input_reads": 0,
            "model_or_generation_accessed": False,
            "gpu_accessed": False,
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if REPORT.exists():
        raise FileExistsError(REPORT)
    report = recover_v54a(prereg)
    atomic_json(REPORT, report)
    print(json.dumps({
        "report": str(REPORT),
        "report_file_sha256": file_sha256(REPORT),
        "report_content_sha256": report["content_sha256_before_self_field"],
        "protected_semantic_input_reads": 0,
        "model_or_generation_accessed": False,
        "gpu_accessed": False,
        "shadow_opened": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
