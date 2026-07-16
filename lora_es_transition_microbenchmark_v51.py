#!/usr/bin/env python3
"""CPU-only V51 design derived from sealed V48B/V50 train-only artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V48B_RUN = ROOT / "experiments/eggroll_es_hpo/runs/v48b_matched_lora_es_generation_boundary_pop8"
V50_RUN = ROOT / "experiments/eggroll_es_hpo/runs/v50_matched_lora_es_generation_boundary_pop8_cpu_pipeline"
V48B_POPULATION = V48B_RUN / "population_reliability_v48b.json"
V50_POPULATION = V50_RUN / "population_reliability_v50.json"
V48B_GPU_LOG = V48B_RUN / "gpu_activity_v48b.jsonl"
V50_GPU_LOG = V50_RUN / "gpu_activity_v50.jsonl"

EXPECTED_FILES_V51 = {
    "v48b_population": (
        "6b620ef5febfd2c6af9f10c75541caa66c6e7a52b83de83ab144e9b595d5fdd8",
        "d909c5f7c013529025f16f9000bebf18dc5060957af89b2ab4480ca4ba7c0a59",
    ),
    "v50_population": (
        "0226c703c14e672fb30b1d3ebb5b5414fc9c26e5c0f4c26dfff7d1be405164f4",
        "0df65209d45ae51385808d0776528193b8e3d013199af80bfbaf7e8d2a2afa63",
    ),
    "v48b_gpu_log": (
        "d947202d586128af38a3615313229ca94e75b978a92f4b105e8e220dca122add",
        None,
    ),
    "v50_gpu_log": (
        "cd2050101ede6c42f7cbc9e4111de782d6f8f5068c3abb3a7711d1d87b3a2e37",
        None,
    ),
}
EXPECTED_STATE_INVENTORY_SHA256_V51 = (
    "62c70af2382c17973c4bd8fed3f74d63658da9cc9d5a60ba87d7fd850e151b40"
)
EXPECTED_MASTER_SHA256_V51 = (
    "dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192"
)
EXPECTED_MASTER_RUNTIME_SHA256_V51 = (
    "8ba98f0a9fad3c6faba57ba2b20f72507baaf9ece45bcb3e4430dbf3ab61a482"
)
POPULATION_PHASE_V51 = "fused_complete_actor_block_population_pop8"


def file_sha256_v51(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v51(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _sealed_json_v51(path: Path, expected: tuple[str, str | None]) -> dict:
    if file_sha256_v51(path) != expected[0]:
        raise RuntimeError(f"v51 sealed file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        expected[1] is not None
        and (
            value.get("content_sha256_before_self_field") != expected[1]
            or canonical_sha256_v51(compact) != expected[1]
        )
    ):
        raise RuntimeError(f"v51 sealed JSON content changed: {path}")
    return value


def _state_inventory_from_population_v51(value: dict) -> list[dict]:
    population = value["population"]
    result = []
    state_index = 0
    for direction in range(8):
        for label, sign in (("plus", 1), ("minus", -1)):
            certificates = population["perturbation_certificates"][label][
                direction
            ]
            if len(certificates) != 4:
                raise RuntimeError("v51 perturbation certificate coverage changed")
            candidates = {
                item["candidate_identity"]["sha256"] for item in certificates
            }
            runtimes = {
                item["materialization"]["runtime_values_sha256"]
                for item in certificates
            }
            masters = {
                item["base_identity"]["sha256"] for item in certificates
            }
            seeds = {int(item["seed"]) for item in certificates}
            signs = {int(item["sign"]) for item in certificates}
            if (
                len(candidates) != 1 or len(runtimes) != 1
                or masters != {EXPECTED_MASTER_SHA256_V51}
                or len(seeds) != 1 or signs != {sign}
            ):
                raise RuntimeError("v51 state identity consensus changed")
            result.append({
                "state_index": state_index,
                "direction": direction,
                "label": label,
                "sign": sign,
                "seed": next(iter(seeds)),
                "candidate_identity_sha256": next(iter(candidates)),
                "runtime_values_sha256": next(iter(runtimes)),
            })
            state_index += 1
    if canonical_sha256_v51(result) != EXPECTED_STATE_INVENTORY_SHA256_V51:
        raise RuntimeError("v51 pinned state inventory changed")
    return result


def state_inventory_v51() -> list[dict]:
    v48b = _sealed_json_v51(
        V48B_POPULATION, EXPECTED_FILES_V51["v48b_population"],
    )
    v50 = _sealed_json_v51(
        V50_POPULATION, EXPECTED_FILES_V51["v50_population"],
    )
    first = _state_inventory_from_population_v51(v48b)
    second = _state_inventory_from_population_v51(v50)
    if first != second:
        raise RuntimeError("v51 V48B/V50 exact state identities differ")
    return first


def _gpu_phase_metrics_v51(path: Path, expected_file_sha: str) -> dict:
    if file_sha256_v51(path) != expected_file_sha:
        raise RuntimeError(f"v51 sealed GPU log changed: {path}")
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["phase"] == POPULATION_PHASE_V51:
                groups[row["sampled_at_utc"]].append(row)
    samples = []
    for stamp, rows in sorted(groups.items()):
        if (
            len(rows) != 4
            or sorted(row["gpu"] for row in rows) != list(range(4))
            or any(row["foreign_compute_pids"] for row in rows)
        ):
            raise RuntimeError("v51 historical GPU attribution changed")
        samples.append((
            datetime.fromisoformat(stamp),
            sum(row["utilization_percent"] >= 10 for row in rows),
        ))
    wall = (samples[-1][0] - samples[0][0]).total_seconds()
    all_four_seconds = none_seconds = 0.0
    idle_runs = []
    current_idle = 0.0
    for (left, count), (right, _next) in zip(samples, samples[1:]):
        elapsed = (right - left).total_seconds()
        if count == 4:
            all_four_seconds += elapsed
        if count == 0:
            none_seconds += elapsed
            current_idle += elapsed
        elif current_idle:
            idle_runs.append(current_idle)
            current_idle = 0.0
    if current_idle:
        idle_runs.append(current_idle)
    return {
        "samples": len(samples),
        "wall_seconds": wall,
        "all_four_ge10_seconds": all_four_seconds,
        "all_four_ge10_fraction": all_four_seconds / wall,
        "none_ge10_seconds": none_seconds,
        "none_ge10_fraction": none_seconds / wall,
        "synchronized_idle_gap_count": len(idle_runs),
        "synchronized_idle_gap_median_seconds": statistics.median(idle_runs),
        "synchronized_idle_gap_max_seconds": max(idle_runs),
        "all_idle_gaps_ge2_seconds": all(value >= 2.0 for value in idle_runs),
    }


def build_design_v51() -> dict:
    inventory = state_inventory_v51()
    v48b_metrics = _gpu_phase_metrics_v51(
        V48B_GPU_LOG, EXPECTED_FILES_V51["v48b_gpu_log"][0],
    )
    v50_metrics = _gpu_phase_metrics_v51(
        V50_GPU_LOG, EXPECTED_FILES_V51["v50_gpu_log"][0],
    )
    result = {
        "schema": "direct-pinned-master-transition-design-v51",
        "selected_transition": {
            "name": "single-slot direct pinned-master rematerialization",
            "states": 16,
            "runtime_slots": 1,
            "state_materializations_retained": 16,
            "intermediate_master_restores_eliminated": 15,
            "final_exact_master_restores_retained": 1,
            "candidate_derived_from": "immutable canonical FP32 master",
            "candidate_derived_from_prior_candidate": False,
            "bf16_algebraic_rollback_used": False,
            "extra_gpu_memory_required": False,
            "expected_candidate_inventory_sha256": (
                EXPECTED_STATE_INVENTORY_SHA256_V51
            ),
            "expected_states": inventory,
            "final_master_sha256": EXPECTED_MASTER_SHA256_V51,
            "final_runtime_values_sha256": (
                EXPECTED_MASTER_RUNTIME_SHA256_V51
            ),
        },
        "deferred_alternative": {
            "name": "two resident LoRA runtime slots",
            "status": "deferred_until_single_slot_transition_is_measured",
            "reason": (
                "changes the verified sole-slot topology and adds resident GPU "
                "state without being necessary to remove intermediate restores"
            ),
        },
        "required_timing": {
            "phases": [
                "materialize", "generate", "score", "restore", "drain",
            ],
            "states": 16,
            "actors_per_state": 4,
            "actor_phase_receipts": 16 * 4 * 5,
            "worker_monotonic_duration_required": [
                "materialize", "score", "final_restore",
            ],
            "controller_observed_duration_required": [
                "generate", "drain",
            ],
        },
        "historical_train_only_baseline": {
            "v48b": v48b_metrics,
            "v50": v50_metrics,
            "interpretation": (
                "descriptive timing baseline only; independent model generations "
                "are not a bit-exact population-quality comparator"
            ),
        },
        "source_artifacts": {
            "v48b_population": str(V48B_POPULATION),
            "v50_population": str(V50_POPULATION),
            "v48b_gpu_log": str(V48B_GPU_LOG),
            "v50_gpu_log": str(V50_GPU_LOG),
            "files": {
                key: list(value) for key, value in EXPECTED_FILES_V51.items()
            },
        },
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    result["content_sha256"] = canonical_sha256_v51(result)
    return result
