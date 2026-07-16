#!/usr/bin/env python3
"""Derive the sealed V51 speed/correctness decision from runtime receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_direct_pinned_master_transition_v51_retry1.json"
)
RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v51_direct_pinned_master_transition_microbenchmark_retry1"
)
REPORT = RUN / "transition_microbenchmark_report_v51.json"
TIMING = RUN / "per_state_timing_v51.json"
POPULATION = RUN / "population_v51.json"
GPU_LOG = RUN / "gpu_activity_v51.jsonl"
V50_GPU_LOG = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v50_matched_lora_es_generation_boundary_pop8_cpu_pipeline/"
    "gpu_activity_v50.jsonl"
)
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/audits/"
    "v51_direct_master_transition_vs_v50.json"
)

EXPECTED_FILES = {
    PREREG: "41b1c3d93cff777aef470427ce597f12e631699c9d8a26c536605155a4c475e0",
    REPORT: "a727990745f6d41e0eb36104a19ce1d92c1645c2bf01bfb7ce3b02e4bbe29b1f",
    TIMING: "a7c03ffdfba16d5ab277ea724a1fa71299cb587132c7269bbfc64f4a92b686c6",
    POPULATION: "ff4b23544ba641191e364c9dc000af1878f0972adddd5d4479ffa1b68b0de0ce",
    GPU_LOG: "229af9a053535f70b6bd4d00fd7447ea053bed810cc85006b336b7525b68d101",
    V50_GPU_LOG: "cd2050101ede6c42f7cbc9e4111de782d6f8f5068c3abb3a7711d1d87b3a2e37",
}
EXPECTED_CONTENT = {
    PREREG: "02dd64eb07694c6d659c3036b63037cf607b942f5e2bc98972d75c1c4b3abf79",
    REPORT: "42b31602793aa727dcfc5622fce4c0e1c1a094e2b8234f3f6f402f52db68ba37",
    TIMING: "7ec7b3486a7143f3d7d58f4f678add547f4062b649779aa466b8de2622b7593e",
    POPULATION: "a342aaaef035b17b228fdd9f04357b548d8363f16d16dfd10640605918c8d5ac",
}
MASTER_SHA = "dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192"
RUNTIME_SHA = "8ba98f0a9fad3c6faba57ba2b20f72507baaf9ece45bcb3e4430dbf3ab61a482"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def sealed_json(path: Path) -> dict:
    if file_sha256(path) != EXPECTED_FILES[path]:
        raise RuntimeError(f"sealed V51 audit input changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != EXPECTED_CONTENT[path]
        or canonical_sha256(compact) != EXPECTED_CONTENT[path]
    ):
        raise RuntimeError(f"sealed V51 JSON content changed: {path}")
    return value


def gpu_metrics(path: Path, phase: str) -> dict:
    if file_sha256(path) != EXPECTED_FILES[path]:
        raise RuntimeError(f"sealed GPU log changed: {path}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["phase"] == phase:
            groups[row["sampled_at_utc"]].append(row)
    samples = []
    for stamp, rows in sorted(groups.items()):
        if (
            len(rows) != 4
            or sorted(row["gpu"] for row in rows) != list(range(4))
            or any(row["foreign_compute_pids"] for row in rows)
        ):
            raise RuntimeError("V51 GPU attribution changed")
        samples.append((
            datetime.fromisoformat(stamp),
            sum(row["utilization_percent"] >= 10 for row in rows),
        ))
    wall = (samples[-1][0] - samples[0][0]).total_seconds()
    all_four = none = current_idle = 0.0
    gaps: list[float] = []
    for (left, count), (right, _next) in zip(samples, samples[1:]):
        elapsed = (right - left).total_seconds()
        if count == 4:
            all_four += elapsed
        if count == 0:
            none += elapsed
            current_idle += elapsed
        elif current_idle:
            gaps.append(current_idle)
            current_idle = 0.0
    if current_idle:
        gaps.append(current_idle)
    return {
        "samples": len(samples),
        "wall_seconds": wall,
        "all_four_ge10_seconds": all_four,
        "all_four_ge10_fraction": all_four / wall,
        "none_ge10_seconds": none,
        "none_ge10_fraction": none / wall,
        "synchronized_idle_gap_count": len(gaps),
        "synchronized_idle_gap_median_seconds": statistics.median(gaps),
        "synchronized_idle_gap_max_seconds": max(gaps),
    }


def build_audit() -> dict:
    prereg = sealed_json(PREREG)
    report = sealed_json(REPORT)
    timing = sealed_json(TIMING)
    population = sealed_json(POPULATION)
    expected_states = prereg["design"]["selected_transition"]["expected_states"]
    observed_states = [item["state"] for item in timing["states"]]
    coverage = timing["coverage"]
    body = population["population"]
    score_audits = report["score_audits"]
    final = report["final_state_certificates"]
    correctness = {
        "all_16_precommitted_states_exact": observed_states == expected_states,
        "all_320_timing_receipts": (
            coverage["states"] == 16
            and coverage["phase_actor_receipts"] == {
                phase: 64 for phase in (
                    "materialize", "generate", "score", "restore", "drain"
                )
            }
        ),
        "15_intermediate_restores_elided": (
            body["intermediate_master_restores_eliminated"] == 15
            and coverage["intermediate_restore_actor_receipts_elided"] == 60
        ),
        "four_actual_final_restores": (
            body["actual_final_restore_actor_receipts"] == 4
            and coverage["actual_final_restore_actor_receipts"] == 4
        ),
        "final_master_exact_all_four": (
            {item["current_identity"]["sha256"] for item in final}
            == {MASTER_SHA}
        ),
        "final_runtime_exact_all_four": (
            {item["materialization"]["runtime_values_sha256"] for item in final}
            == {RUNTIME_SHA}
        ),
        "base_score_exact_after_population": (
            score_audits["preinstall"]["consensus"]
            == score_audits["postinstall"]["consensus"]
            == score_audits["post_population"]["consensus"]
        ),
        "all_four_gpus_positive": report["gpu_activity"][
            "all_four_attributed_positive"
        ],
        "strict_cleanup": (
            report["cleanup"]["all_four_gcs_states_removed"]
            and report["final_gpu_idle"]["all_four_compute_process_lists_empty"]
        ),
        "no_update_or_snapshot": (
            report["optimizer_update_performed"] is False
            and report["snapshot"] is None
        ),
        "no_protected_access": (
            report["protected_semantics_opened"] is False
            and report["shadow_ood_holdout_or_benchmark_opened"] is False
        ),
    }
    v50 = gpu_metrics(V50_GPU_LOG, "fused_complete_actor_block_population_pop8")
    v51 = gpu_metrics(GPU_LOG, "direct_pinned_master_population_v51")
    wall_reduction = 1.0 - v51["wall_seconds"] / v50["wall_seconds"]
    gap_reduction = 1.0 - (
        v51["synchronized_idle_gap_median_seconds"]
        / v50["synchronized_idle_gap_median_seconds"]
    )
    speed = {
        "v50": v50,
        "v51": v51,
        "population_wall_reduction_fraction": wall_reduction,
        "median_idle_gap_reduction_fraction": gap_reduction,
        "population_wall_gate": {
            "minimum_reduction_fraction": 0.10,
            "passed": wall_reduction >= 0.10,
        },
        "median_idle_gap_gate": {
            "minimum_reduction_fraction": 0.20,
            "passed": gap_reduction >= 0.20,
        },
    }
    accepted = (
        all(correctness.values())
        and speed["population_wall_gate"]["passed"]
        and speed["median_idle_gap_gate"]["passed"]
    )
    result = {
        "schema": "direct-pinned-master-transition-audit-v51",
        "status": "accepted_for_future_lora_es_population_transitions"
        if accepted else "rejected",
        "correctness": correctness,
        "speed": speed,
        "decision": {
            "accepted": accepted,
            "scope": "population transition implementation only",
            "quality_or_checkpoint_promotion_authorized": False,
            "reason": (
                "The direct-master single-slot transition passed every exact "
                "state/restore/safety gate and both preregistered speed gates."
            ),
        },
        "sealed_inputs": {
            str(path.relative_to(ROOT)): {
                "file_sha256": digest,
                **({"content_sha256": EXPECTED_CONTENT[path]}
                   if path in EXPECTED_CONTENT else {}),
            }
            for path, digest in EXPECTED_FILES.items()
        },
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "accepted": value["decision"]["accepted"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
