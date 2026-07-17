#!/usr/bin/env python3
"""Seal the corrected V79 cleanup monitor before the V79B live replicate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_matched_v79.json"
)
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_cleanup_v79b.json"
)
PROBE = ROOT / "probe_vllm_fp8_kv_capacity_v79.py"
MONITOR = ROOT / "monitor_qwen36_fp8_kv_capacity_v79.py"
LAUNCHER = ROOT / "launch_qwen36_fp8_kv_capacity_v79.sh"

SCHEMA = "v79b-qwen36-fp8-kv-cleanup-preregistration"
PARENT_FILE_SHA256 = (
    "0e195c05fd72e36656ee6536d6656d932aac0028fbbc7983f688df9dc7b18753"
)
PARENT_CONTENT_SHA256 = (
    "6c73ac0f6bf4019cdf297546e4315dc99b68d9549a24c03f4eaa9c8ebb589023"
)
SOURCE_SHA256 = {
    "probe_vllm_fp8_kv_capacity_v79.py": (
        "6b72de1bd7d7878ba4183bae618108f8cd1cf997e33c7447ee2459700e15ff45"
    ),
    "monitor_qwen36_fp8_kv_capacity_v79.py": (
        "6035eb32f90815ed2a2d8734d9e9072123b8ecc70d74449a2710e94b673ed3df"
    ),
    "launch_qwen36_fp8_kv_capacity_v79.sh": (
        "4ca93e3a171787bb56613bf3648365ae96a355e28ec95f094b12d1982b6772df"
    ),
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent() -> dict:
    if file_sha256(PARENT) != PARENT_FILE_SHA256:
        raise RuntimeError("V79 parent preregistration file changed")
    value = json.loads(PARENT.read_text(encoding="ascii"))
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    if (
        claimed != PARENT_CONTENT_SHA256
        or canonical_sha256(body) != claimed
        or value.get("schema")
        != "v79-qwen36-fp8-kv-capacity-matched-preregistration"
        or value.get("authority", {}).get("scored_or_training_authority")
        is not False
    ):
        raise RuntimeError("V79 parent preregistration content changed")
    return value


def source_inventory() -> list[dict]:
    rows = []
    for path in (PROBE, MONITOR, LAUNCHER):
        actual = file_sha256(path)
        if actual != SOURCE_SHA256[path.name]:
            raise RuntimeError(f"V79B source changed: {path.name}")
        rows.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return rows


def build_v79b() -> dict:
    parent = load_parent()
    sources = source_inventory()
    value = {
        "schema": SCHEMA,
        "bead": "specialist-0j5.24",
        "status": "preregistered_before_v79b_live_cleanup_replicate",
        "authority": {
            "data_free_runtime_diagnostic_only": True,
            "dataset_or_protected_data_opened": False,
            "model_update_or_training_performed": False,
            "scored_training_checkpoint_or_promotion_authorized": False,
        },
        "parent_v79": {
            "path": str(PARENT.relative_to(ROOT)),
            "file_sha256": PARENT_FILE_SHA256,
            "content_sha256": PARENT_CONTENT_SHA256,
            "selected_runtime_retained_exactly": parent["selected_runtime"],
            "live_acceptance_retained_except_stronger_cleanup_observation": (
                parent["live_acceptance"]
            ),
        },
        "implementation_correction": {
            "model_or_workload_change": False,
            "probe_parent_receipt_fix": (
                "stop requiring gpu_memory_utilization in a V73 receipt that "
                "does not persist that field; engine kwargs and live resolved "
                "certificate remain independently fail-closed"
            ),
            "monitor_cleanup_fix": (
                "retain already ancestry-attributed engine PIDs after "
                "reparenting, then use an external unitless nvidia-smi "
                "snapshot for the cleanup memory gate because the in-process "
                "pynvml observer reports a 598 MiB self-overhead"
            ),
            "earlier_v79_runs_are_diagnostic_not_v79b_acceptance": True,
        },
        "sealed_sources": {
            "files": sources,
            "bundle_sha256": canonical_sha256(sources),
            "site_packages_modified": False,
        },
        "cleanup_acceptance": {
            "minimum_consecutive_batches": 3,
            "sample_interval_seconds": 0.5,
            "maximum_wait_seconds": 60.0,
            "all_actor_roots_dead": True,
            "compute_pids_exact": [],
            "gpu_utilization_percent_exact": 0,
            "memory_used_mib_max": 4,
            "memory_and_utilization_source": (
                "external_nvidia_smi_while_pynvml_telemetry_remains_live"
            ),
            "foreign_compute_pid_forbidden_during_measurement": True,
        },
        "launch": {
            "fresh_run_directory": (
                "experiments/eggroll_es_hpo/runs/"
                "v79b_fp8_kv_capacity_0485_r5_sealed_cleanup"
            ),
            "exact_command": (
                "RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/"
                "v79b_fp8_kv_capacity_0485_r5_sealed_cleanup bash "
                "/home/catid/specialist/launch_qwen36_fp8_kv_capacity_v79.sh"
            ),
            "launch_performed_by_builder": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def render(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(build_v79b())
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="ascii") != expected:
            raise RuntimeError("V79B preregistration is stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(expected, encoding="ascii")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "content_sha256": json.loads(expected)[
                    "content_sha256_before_self_field"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
