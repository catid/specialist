#!/usr/bin/env python3
"""Freeze the train-only V38A population-32 nonzero ES arm."""

from __future__ import annotations

import json
from pathlib import Path

import run_eggroll_es_equal_unit_v38a as runtime
import run_sft_train_only_control_v36a as hashing
import train_eggroll_es_equal_unit_v38a as trainer


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "equal_unit_fold3_nonzero_v38a.json"
).resolve()
EXPECTED_BINDINGS = {
    "runtime": "ffe71974b4935c6943ac03dffc048754bce4fb39388e8f5b23e04521ac9851d6",
    "trainer": "a76dc19b0c1c9d5b57bebd66544662553935dfbc23f72a47e172f4366cba5852",
    "worker": "0b5a2a86a0aa51f34e38ddad653d34bfdbb7779e409fd6e47a4d9f3dcb3fec50",
    "dataset": "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a",
    "split_manifest": "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d",
    "layer_plan": "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df",
    "model_config": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "model_index": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    "trainer_v13": "1a8a4145a85c183bb6121914357b7e6bce916b4f76a0693887ac41fa3a8c4c6e",
    "trainer_v11c": "c663b62f9d7990a2c59d8b46ad6258209b590bb29aa48946755a7d263a3d0799",
    "worker_v11c": "d75951483058de340185fc81f6ed050deeac1551107c0357349a1f311cdb2c22",
    "trainer_v4": "9b771f5f5578cc233bea9b92ee48dbad8bbf7f363fae44d4dadde5e926f6cd65",
    "worker_v4": "876033b6b2ac8a869f0f82656c7e49434b7ba25789c1ddf776ac433659f72f59",
    "worker_v3": "636698c7ea8e8155ea08298e428b5c40c01141ff10db59290d74831db360e5b1",
}


def observed_bindings() -> dict:
    paths = {
        "runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
        "trainer": ROOT / "train_eggroll_es_equal_unit_v38a.py",
        "worker": ROOT / "eggroll_es_worker_v38a.py",
        "dataset": runtime.DATASET,
        "split_manifest": runtime.SPLIT_MANIFEST,
        "layer_plan": runtime.LAYER_PLAN,
        "model_config": runtime.MODEL / "config.json",
        "model_index": runtime.MODEL / "model.safetensors.index.json",
        "trainer_v13": ROOT / "train_eggroll_es_specialist_anchor_v13.py",
        "trainer_v11c": ROOT / "train_eggroll_es_specialist_anchor_v11c.py",
        "worker_v11c": ROOT / "eggroll_es_worker_v11c.py",
        "trainer_v4": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
        "worker_v4": ROOT / "eggroll_es_worker_v4.py",
        "worker_v3": ROOT / "eggroll_es_worker_v3.py",
    }
    return {key: hashing.file_sha256(path) for key, path in paths.items()}


def build() -> dict:
    observed = observed_bindings()
    if observed != EXPECTED_BINDINGS:
        raise RuntimeError("v38a implementation bindings changed")
    train_bundle = trainer.load_equal_unit_train_bundle(
        runtime.DATASET, observed["dataset"], runtime.SPLIT_MANIFEST,
        observed["split_manifest"],
    )
    if (
        train_bundle["content_sha256_before_self_field"]
        != "ba0a951f153b60fb5729e0169a4d780277d9d860cbacb1f95fc7433afc875e19"
        or train_bundle["weight_identity_sha256"]
        != "a0fab310d38e55709a414c552e83fd80c09d805a94626b7180f21036d2f44e4e"
    ):
        raise RuntimeError("v38a train-bundle identity changed")
    result = {
        "schema": "eggroll-es-equal-unit-preregistration-v38a",
        "status": "preregistered_not_yet_run",
        "experiment_name": runtime.EXPERIMENT,
        "shadow_dev_external_eval_ood_or_holdout_opened": False,
        "prelaunch_integrity_revision": {
            "prior_preregistration_file_sha256": (
                "20430e4c264a01522a53bfb21b63cdf2e9375f2928da77f87c02021318f7275e"
            ),
            "prior_attempt_launched": False,
            "change": (
                "reopen and rehash the serialized selected snapshot against the "
                "live final selected identity; replace detached Ray reservations "
                "with driver-scoped groups and require all four exact GCS records "
                "to reach REMOVED before shutdown"
            ),
        },
        "implementation_bindings": observed,
        "artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
            "selected_runtime_snapshot": str(runtime.SNAPSHOT),
        },
        "recipe": {
            "model": str(runtime.MODEL),
            "world_size": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "population_size": 32,
            "perturbation_mode": "antithetic_plus_minus",
            "signed_directions": 64,
            "sigma": 0.0003,
            "alpha": 0.00015,
            "alpha_search": False,
            "update_count": 1,
            "seeds": list(trainer.SEEDS),
            "seed_sha256": trainer.canonical_sha256(trainer.SEEDS),
            "perturbation_basis_sha256": (
                trainer.anchor_v13.PERTURBATION_BASIS_SHA256_V13
            ),
            "dataset_rows": 448,
            "conflict_units": 208,
            "train_bundle_content_sha256": train_bundle[
                "content_sha256_before_self_field"
            ],
            "weight_identity_sha256": train_bundle[
                "weight_identity_sha256"
            ],
            "objective": (
                "teacher-forced mean answer-token logprob per row, mean rows "
                "within each conservative conflict unit, then mean 208 units"
            ),
            "prompt": "exact specialist_template(question)+raw_answer",
            "eos_appended_or_scored": False,
            "max_total_tokens": 1024,
            "prompt_tokens_per_signed_direction": 22_340,
            "answer_tokens_per_signed_direction": 11_420,
            "token_presentations": 2_160_640,
            "sequence_presentations": 28_672,
            "central_response": "0.5 * (plus - minus)",
            "coefficient_standardization": (
                "population z-score with epsilon 1e-8"
            ),
            "layer_plan_sha256": (
                "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
            ),
            "layers": [20, 21, 22, 23],
            "checkpoint_units": 35,
            "packed_runtime_tensors": 23,
            "selected_elements": 142_999_552,
            "selected_bytes": 285_999_104,
            "checkpoint_to_runtime_mapping_sha256": (
                "7ebd15cb9c04cfe8dab67009dc2af5c0054131a11c15a6c8e83f277ffef4585c"
            ),
        },
        "required_runtime_gates": {
            "exclusive_idle_preflight": True,
            "all_four_worker_pids_unique_and_physically_mapped": True,
            "all_four_gpus_attributed_positive_activity": True,
            "pre_and_post_population_base_probe_exact": True,
            "exact_restore_and_population_boundary_audit": True,
            "coefficient_spread_nonzero": True,
            "two_phase_four_rank_fp32_update": True,
            "target_alpha_exact": 0.00015,
            "selected_final_identity_differs_from_base": True,
            "unselected_identity_equals_origin": True,
            "selected_snapshot_rank_zero_only_and_content_addressed": True,
            "selected_snapshot_reopened_identity_equals_live_final": True,
            "driver_scoped_non_detached_placement_groups": True,
            "all_four_exact_placement_groups_gcs_removed": True,
            "all_four_compute_process_lists_empty_after_ray_shutdown": True,
        },
        "decision_firewall": {
            "authorized": "train-only state seal and runtime comparison",
            "forbidden": [
                "shadow-dev access", "external validation access", "OOD access",
                "holdout access", "quality claim", "model promotion",
            ],
            "next_gate": (
                "after state seal, one preregistered common base/SFT/ES fold-3 "
                "shadow-dev evaluation"
            ),
        },
    }
    result["content_sha256_before_self_field"] = hashing.canonical_sha256(result)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate-prelaunch-integrity-revision", action="store_true")
    arguments = parser.parse_args()
    value = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if not arguments.migrate_prelaunch_integrity_revision:
            raise RuntimeError("v38a preregistration already exists")
        if (
            hashing.file_sha256(OUTPUT)
            != "20430e4c264a01522a53bfb21b63cdf2e9375f2928da77f87c02021318f7275e"
            or runtime.ATTEMPT.exists()
            or runtime.RUN_DIR.exists()
        ):
            raise RuntimeError("v38a prelaunch preregistration cannot migrate")
    hashing.atomic_write_json(OUTPUT, value)
    print(OUTPUT)
    print(value["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
