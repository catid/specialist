#!/usr/bin/env python3
"""CPU-only evidence and deterministic untried-scale planning for V43L."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import eggroll_es_worker_v3 as worker_v3
import lora_es_fused_anchor_runtime_v43i as fused


TARGET_NORM_RATIOS_V43L = (0.03125, 0.015625)
V43I_NORM_RATIO = 0.5
EXPECTED_FILES_V43L = {
    "population_reliability": (
        "7a1f04dc57aa42cab36ccb60d3382c798614bef9f29d2c84708dfc9178a330f6",
        "998cc1567ca96cd5a91c236e86347b72931ecc0f3cb525426cd83e2bc861797b",
    ),
    "candidate_gate": (
        "fc6ed9fc2908cd33975a21877e22269dfc1cdd4c5f89a7d14b51d5b0823bdc6a",
        "070d225fd279de5cb969c9c74d2200bf1978fbbf7e0ea2967be24688234f70b1",
    ),
    "exact_abort": (
        "83cc533e72100018a0498c4555c7f4f543bf78cf2c6bc2bdc248a6918ceea9b5",
        "039f6c8d122c1d06aa06d42671631a604d8599af9873cabb11c60b2c97371de8",
    ),
    "failure": (
        "5ed607e88edc27e944f4e9e3011a41f5537d88fdd6671bcbfbdc0d4bfc2368e6",
        "9f2b4016be93d84637e0c864c8a4c76fc1a08377ecae521414ad458774e118b7",
    ),
    "numeric_calibration": (
        "24a9cd3b42c39e29fed576e897bf2d6740faf9d170412385c6554c1019447d2f",
        "eb563952aa0a5d938f4db0a10fee2a5410792ceda8a282edf574b897c4214f7e",
    ),
    "anchor_calibration": (
        "4e1d3025ac1627e126f65921c22dea8a3671ab1e6afba89b470763e319d99b02",
        "4f7b6d85b4c96ded425e537d0931c98a8d9ac759bf3a594589338596bfd9230b",
    ),
}
RESTORED_MASTER_SHA256_V43L = (
    "dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192"
)
RESTORED_RUNTIME_SHA256_V43L = (
    "8ba98f0a9fad3c6faba57ba2b20f72507baaf9ece45bcb3e4430dbf3ab61a482"
)


def file_sha256_v43l(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256_v43l(value: object) -> str:
    return fused.canonical_sha256_v43i(value)


def _sealed(path: Path, expected: tuple[str, str]) -> dict:
    if file_sha256_v43l(path) != expected[0]:
        raise RuntimeError(f"v43l parent file identity changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != expected[1]
        or canonical_sha256_v43l(compact) != expected[1]
    ):
        raise RuntimeError(f"v43l parent content identity changed: {path}")
    return value


def load_v43i_evidence_v43l(paths: dict[str, Path]) -> dict:
    if set(paths) != set(EXPECTED_FILES_V43L):
        raise ValueError("v43l V43I evidence inventory changed")
    values = {
        label: _sealed(Path(paths[label]), EXPECTED_FILES_V43L[label])
        for label in EXPECTED_FILES_V43L
    }
    reliability = values["population_reliability"]
    candidate = values["candidate_gate"]
    abort = values["exact_abort"]
    failure = values["failure"]
    calibration = values["anchor_calibration"]
    population = reliability.get("population", {})
    projection = population.get("projection", {})
    coefficients = population.get("coefficients")
    if (
        reliability.get("schema")
        != "matched-lora-es-population-reliability-v43i"
        or reliability.get("reliability_gate", {}).get("passed") is not True
        or projection.get("diagnostics", {}).get("decision")
        != "project_and_trust_region"
        or projection.get("diagnostics", {}).get("trust_region_norm_ratio")
        != V43I_NORM_RATIO
        or projection.get("diagnostics", {}).get(
            "all_anchor_halfspaces_satisfied"
        ) is not True
        or not isinstance(coefficients, list)
        or len(coefficients) != 8
        or not all(math.isfinite(float(value)) for value in coefficients)
        or candidate.get("schema")
        != "matched-lora-es-uncommitted-candidate-gate-v43i"
        or candidate.get("gate", {}).get("passed") is not False
        or candidate.get("gate", {}).get("checks", {}).get(
            "domain_point_improvement"
        ) is not False
        or candidate.get("gate", {}).get("checks", {}).get(
            "qa_generation_f1_noninferiority"
        ) is not False
        or abort.get("schema") != "controller-exact-abort-readback-v43i"
        or abort.get("status") != "candidate_rejected_and_exactly_restored"
        or abort.get("all_four_ranks_exact") is not True
        or abort.get("restored_master_identity", {}).get("sha256")
        != RESTORED_MASTER_SHA256_V43L
        or abort.get("restored_runtime_values_sha256")
        != RESTORED_RUNTIME_SHA256_V43L
        or failure.get("schema") != "matched-lora-es-failure-v43i"
        or "failed preservation gate and was exactly restored"
        not in failure.get("message", "")
        or calibration.get("schema")
        != "matched-lora-es-anchor-calibration-v43i"
        or calibration.get("calibrated_margins", {}).get("passed") is not True
    ):
        raise RuntimeError("v43l required V43I negative/restoration evidence changed")
    plans = scale_plans_v43l(coefficients, projection["diagnostics"])
    return {
        "schema": "sealed-v43i-negative-evidence-for-backtracking-v43l",
        "file_and_content_sha256": {
            label: {"file_sha256": hashes[0], "content_sha256": hashes[1]}
            for label, hashes in EXPECTED_FILES_V43L.items()
        },
        "v43i_projected_coefficients": [float(value) for value in coefficients],
        "v43i_projection_content_sha256": projection["content_sha256"],
        "v43i_candidate_gate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "restored_master_identity": abort["restored_master_identity"],
        "restored_runtime_values_sha256": RESTORED_RUNTIME_SHA256_V43L,
        "anchor_calibrated_margins": calibration["calibrated_margins"],
        "numeric_calibration_bootstrap_bounds": values[
            "numeric_calibration"
        ]["bootstrap"]["bounds"],
        "scale_plans": plans,
        "protected_semantics_opened": False,
    }


def scale_plans_v43l(
    v43i_coefficients: list[float], projection_diagnostics: dict,
) -> list[dict]:
    if (
        len(v43i_coefficients) != 8
        or projection_diagnostics.get("trust_region_norm_ratio")
        != V43I_NORM_RATIO
        or projection_diagnostics.get("update_norm_ratio") != V43I_NORM_RATIO
    ):
        raise ValueError("v43l requires the exact V43I 0.5-ratio projection")
    source_norm = math.sqrt(math.fsum(float(x) ** 2 for x in v43i_coefficients))
    unconstrained = float(projection_diagnostics["unconstrained_domain_norm"])
    if not math.isclose(
        source_norm / unconstrained, V43I_NORM_RATIO,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError("v43l V43I coefficient norm no longer matches projection")
    plans = []
    for target in TARGET_NORM_RATIOS_V43L:
        multiplier = target / V43I_NORM_RATIO
        coefficients = [float(value) * multiplier for value in v43i_coefficients]
        actual = math.sqrt(math.fsum(value * value for value in coefficients))
        ratio = actual / unconstrained
        if not math.isclose(ratio, target, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v43l deterministic scaled norm changed")
        plans.append({
            "target_norm_ratio": target,
            "v43i_coefficient_multiplier": multiplier,
            "coefficients": coefficients,
            "coefficient_sha256": worker_v3.coefficient_sha256_v3(
                # The exact seeds are asserted by the runtime/preregistration.
                [
                    140002291, 1028842752, 480373990, 1037026679,
                    759861149, 227761095, 428721957, 150663570,
                ],
                coefficients,
            ),
            "actual_norm_ratio": ratio,
            "anchor_halfspaces_preserved_by_positive_homogeneous_scaling": True,
        })
    return plans


def selected_diagnostic_scale_v43l(results: list[dict]) -> float | None:
    """Validate the six-gate continuation rule and return an accepted ratio.

    The 0.015625 diagnostic is authorized only when 0.03125 fails the existing
    six train-only preservation checks.  A consensus failure after a six-gate
    pass is exactly aborted but does not authorize a smaller diagnostic.
    """
    if not isinstance(results, list) or len(results) > len(TARGET_NORM_RATIOS_V43L):
        raise ValueError("v43l scale result coverage changed")
    for index, result in enumerate(results):
        if result.get("target_norm_ratio") != TARGET_NORM_RATIOS_V43L[index]:
            raise ValueError("v43l scale evaluation order changed")
        six_gate_passed = result.get("six_train_only_gates_passed") is True
        consensus_passed = result.get("candidate_consensus_passed") is True
        accepted = six_gate_passed and consensus_passed
        if accepted:
            if index != len(results) - 1:
                raise ValueError("v43l evaluated a smaller scale after a passing scale")
            return TARGET_NORM_RATIOS_V43L[index]
        if result.get("exact_abort_readback_passed") is not True:
            raise ValueError("v43l failed scale was not exactly aborted")
        if six_gate_passed:
            if index != len(results) - 1:
                raise ValueError(
                    "v43l evaluated a smaller scale after six-gate pass"
                )
            return None
        if index == 1 and results[0].get(
            "six_train_only_gates_passed"
        ) is not False:
            raise ValueError("v43l 0.015625 lacked a 0.03125 six-gate failure")
    return None
