#!/usr/bin/env python3
"""CPU-only sealed V48B evidence and V48D backtracking scale plans."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import eggroll_es_worker_v3 as worker_v3
import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_generation_boundary_v48b as v48b


TARGET_NORM_RATIOS_V48D = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
SOURCE_NORM_RATIO_V48D = 0.5
RESTORED_MASTER_SHA256_V48D = (
    "dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192"
)
RESTORED_RUNTIME_SHA256_V48D = (
    "8ba98f0a9fad3c6faba57ba2b20f72507baaf9ece45bcb3e4430dbf3ab61a482"
)
EXPECTED_FILES_V48D = {
    "population_reliability": (
        "6b620ef5febfd2c6af9f10c75541caa66c6e7a52b83de83ab144e9b595d5fdd8",
        "d909c5f7c013529025f16f9000bebf18dc5060957af89b2ab4480ca4ba7c0a59",
    ),
    "candidate_gate": (
        "6c731524d25496953581e5e21abbd9acfd21b600fbe34a8589f83c53fe6c9e5c",
        "c31b4b2f9a6fb948a28321b4fd95a5ead68176502e4948209d9de2a2051a4ae2",
    ),
    "exact_abort": (
        "01d16a0a5312fec17494d5595f54902754f494ace0375087042cdd3fab545cd7",
        "3b65d4c166965544706c4aedabb9ce3c4ed21e5ad327ec46c8fbcd8ee73b2898",
    ),
    "failure": (
        "4db42d8f2d80cd7924c783d239fae00c038ffa12806899be6f2e840ef47a91e5",
        "bff971476d1ece3527755d1c8e9f7a17324425481be50821f7b39903e30f45df",
    ),
    "numeric_calibration": (
        "d15d166b6bc56a536ddcce1ee7112928d7b5f687f2ca7a1b72b161bc26f25373",
        "00c98fe457f67c08a7bad9a108068cbb1b2c037512b9da92ad116d8ab46fe504",
    ),
    "anchor_calibration": (
        "f8ef1fa291d7253b6802e0bcd125d173161e6e52e69d03848866a262ea55d0d0",
        "b9c83372c2847aaccb60833cf96b6a9fcb2fc3969334537838d170c7302c02a0",
    ),
}
EXPECTED_SUBSET_FILE_SHA256_V48D = (
    "e1c3601d1328ee2a1345423632e423628857147042755b8ac15e21a2506a0060"
)
EXPECTED_SUBSET_CONTENT_SHA256_V48D = (
    "99fa38820a55d3ba7af0a10b5ae26238ffff754fded001a6cc248319cfcc23ca"
)
EXPECTED_PREREG_FILE_SHA256_V48D = (
    "34e19fe84ff061b98a8627f07daab59f5cbb8c718668fc479454114fef67c3d0"
)
EXPECTED_PREREG_CONTENT_SHA256_V48D = (
    "4d5e17a07551377f0ef39c3dfa306fda68b9b669eeafd3c78ebbfff894072d1e"
)
REQUEST_ORDER_SHA256_V48D = (
    "8823e00d9600dc1ce56b2f5f5024a184d0b46ada034523ea2acbe6c7cd920dc7"
)


def file_sha256_v48d(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _sealed(path: Path, expected: tuple[str, str]) -> dict:
    if file_sha256_v48d(path) != expected[0]:
        raise RuntimeError(f"v48d parent file identity changed: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        value.get("content_sha256_before_self_field") != expected[1]
        or v48b.v43i.v40a.canonical_sha256(_compact(value)) != expected[1]
    ):
        raise RuntimeError(f"v48d parent content identity changed: {path}")
    return value


def scale_plans_v48d(
    source_coefficients: list[float], projection_diagnostics: dict,
) -> list[dict]:
    if (
        len(source_coefficients) != 8
        or not math.isclose(
            float(projection_diagnostics.get("trust_region_norm_ratio", -1.0)),
            SOURCE_NORM_RATIO_V48D, rel_tol=0.0, abs_tol=1e-12,
        )
        or not math.isclose(
            float(projection_diagnostics.get("update_norm_ratio", -1.0)),
            SOURCE_NORM_RATIO_V48D, rel_tol=0.0, abs_tol=1e-12,
        )
        or projection_diagnostics.get("all_anchor_halfspaces_satisfied")
        is not True
    ):
        raise ValueError("v48d requires the exact V48B 0.5-ratio projection")
    source = [float(value) for value in source_coefficients]
    if not all(math.isfinite(value) for value in source):
        raise ValueError("v48d source projection is non-finite")
    source_norm = math.sqrt(math.fsum(value * value for value in source))
    unconstrained = float(projection_diagnostics["unconstrained_domain_norm"])
    if not math.isclose(
        source_norm / unconstrained, SOURCE_NORM_RATIO_V48D,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError("v48d source coefficient norm changed")
    result = []
    for target in TARGET_NORM_RATIOS_V48D:
        multiplier = target / SOURCE_NORM_RATIO_V48D
        coefficients = [value * multiplier for value in source]
        actual = math.sqrt(math.fsum(value * value for value in coefficients))
        ratio = actual / unconstrained
        if not math.isclose(ratio, target, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v48d scaled coefficient norm changed")
        result.append({
            "target_norm_ratio": target,
            "v48b_coefficient_multiplier": multiplier,
            "coefficients": coefficients,
            "coefficient_sha256": worker_v3.coefficient_sha256_v3(
                v48b.v43i.SEEDS, coefficients,
            ),
            "actual_norm_ratio": ratio,
            "all_four_projection_halfspaces_preserved_by_positive_scaling": True,
        })
    return result


def largest_passing_scale_v48d(results: list[dict]) -> float | None:
    """Validate descending evaluation and exact abort before continuation."""
    if not isinstance(results, list) or len(results) > len(TARGET_NORM_RATIOS_V48D):
        raise ValueError("v48d scale result coverage changed")
    for index, result in enumerate(results):
        target = TARGET_NORM_RATIOS_V48D[index]
        if result.get("target_norm_ratio") != target:
            raise ValueError("v48d scale evaluation order changed")
        if result.get("gate_passed") is True:
            if index != len(results) - 1:
                raise ValueError("v48d evaluated a smaller scale after a pass")
            return target
        if result.get("exact_abort_readback_passed") is not True:
            raise ValueError("v48d rejected scale was not exactly aborted")
    return None


def load_v48b_evidence_v48d(
    paths: dict[str, Path], subset_path: Path, preregistration_path: Path,
) -> dict:
    if set(paths) != set(EXPECTED_FILES_V48D):
        raise ValueError("v48d V48B evidence inventory changed")
    values = {
        label: _sealed(Path(paths[label]), EXPECTED_FILES_V48D[label])
        for label in EXPECTED_FILES_V48D
    }
    subset = _sealed(Path(subset_path), (
        EXPECTED_SUBSET_FILE_SHA256_V48D,
        EXPECTED_SUBSET_CONTENT_SHA256_V48D,
    ))
    prereg = _sealed(Path(preregistration_path), (
        EXPECTED_PREREG_FILE_SHA256_V48D,
        EXPECTED_PREREG_CONTENT_SHA256_V48D,
    ))
    reliability = values["population_reliability"]
    population = reliability.get("population", {})
    projection = population.get("projection", {})
    diagnostics = projection.get("diagnostics", {})
    candidate = values["candidate_gate"]
    checks = candidate.get("gate", {}).get("checks", {})
    abort = values["exact_abort"]
    failure = values["failure"]
    anchor_calibration = values["anchor_calibration"]
    common = population.get("common_random_plan", {})
    coefficients = population.get("coefficients")
    if (
        reliability.get("schema")
        != "matched-lora-es-population-reliability-v43i"
        or reliability.get("status") != "complete_before_update_decision"
        or reliability.get("reliability_gate", {}).get("passed") is not True
        or population.get("all_exact_restores_passed") is not True
        or population.get("restoration_certificate_count") != 64
        or population.get("fused_requests_per_actor_state") != 608
        or common.get("signed_actor_state_receipts") != 64
        or common.get("all_use_identical_selected_items_order_and_sampling")
        is not True
        or common.get("request_order_sha256") != REQUEST_ORDER_SHA256_V48D
        or population.get("direct_generation_boundary_objective", {}).get(
            "fragile_generated_f1_is_primary_not_halfspace_only"
        ) is not True
        or diagnostics.get("decision") != "project_and_trust_region"
        or diagnostics.get("all_anchor_halfspaces_satisfied") is not True
        or diagnostics.get("anchor_order") != [
            "domain", "fragile_generation_f1", "prose_lm",
            "qa_answer_logprob",
        ]
        or not isinstance(coefficients, list) or len(coefficients) != 8
        or candidate.get("schema")
        != "matched-lora-es-uncommitted-candidate-gate-v43i"
        or candidate.get("gate", {}).get("schema")
        != "uncommitted-generation-boundary-candidate-gate-v48b"
        or candidate.get("gate", {}).get("passed") is not False
        or checks.get("fragile_generation_f1_noninferiority") is not False
        or checks.get("prose_lm_noninferiority") is not False
        or checks.get("qa_generation_f1_noninferiority") is not False
        or abort.get("schema") != "controller-exact-abort-readback-v43i"
        or abort.get("status") != "candidate_rejected_and_exactly_restored"
        or abort.get("all_four_ranks_exact") is not True
        or abort.get("restored_master_identity", {}).get("sha256")
        != RESTORED_MASTER_SHA256_V48D
        or abort.get("restored_runtime_values_sha256")
        != RESTORED_RUNTIME_SHA256_V48D
        or failure.get("schema") != "matched-lora-es-failure-v43i"
        or failure.get("transaction_accepted_before_failure") is not False
        or failure.get("persisted_phase_artifacts", {}).get("snapshot") is not None
        or "failed preservation gate and was exactly restored"
        not in failure.get("message", "")
        or anchor_calibration.get("calibrated_margins", {}).get("passed")
        is not True
        or subset.get("selected_rows") != 64
        or subset.get("selected_conflict_units") != 64
        or subset.get("request_order_sha256") != REQUEST_ORDER_SHA256_V48D
        or prereg.get("gpu_launch_authorized") is not True
        or prereg.get("protected_semantic_access_authorized") is not False
        or prereg.get("shadow_ood_holdout_or_benchmark_authorized") is not False
        or prereg.get("recipe", {}).get("request_order_sha256")
        != REQUEST_ORDER_SHA256_V48D
    ):
        raise RuntimeError("v48d required V48B evidence changed")
    plans = scale_plans_v48d(coefficients, diagnostics)
    return {
        "schema": "sealed-v48b-generation-boundary-evidence-for-v48d",
        "file_and_content_sha256": {
            label: {"file_sha256": hashes[0], "content_sha256": hashes[1]}
            for label, hashes in EXPECTED_FILES_V48D.items()
        },
        "v48b_preregistration": {
            "file_sha256": EXPECTED_PREREG_FILE_SHA256_V48D,
            "content_sha256": EXPECTED_PREREG_CONTENT_SHA256_V48D,
        },
        "sealed_subset": {
            "file_sha256": EXPECTED_SUBSET_FILE_SHA256_V48D,
            "content_sha256": EXPECTED_SUBSET_CONTENT_SHA256_V48D,
            "request_order_sha256": REQUEST_ORDER_SHA256_V48D,
        },
        "v48b_projected_coefficients": [float(value) for value in coefficients],
        "v48b_projection_content_sha256": projection["content_sha256"],
        "v48b_population_content_sha256": reliability[
            "content_sha256_before_self_field"
        ],
        "v48b_candidate_gate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "restored_master_identity": abort["restored_master_identity"],
        "restored_runtime_values_sha256": RESTORED_RUNTIME_SHA256_V48D,
        "anchor_calibrated_margins": anchor_calibration[
            "calibrated_margins"
        ],
        "numeric_calibration_bootstrap_bounds": values[
            "numeric_calibration"
        ]["bootstrap"]["bounds"],
        "scale_plans": plans,
        "population_resampled": False,
        "projection_recomputed": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
