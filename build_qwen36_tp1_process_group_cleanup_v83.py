#!/usr/bin/env python3
"""Seal the additive TP1 process-group cleanup interpretation for V80.

The immutable V63 shutdown helper persisted a field named
``torch_process_group_destroyed``.  Its value is actually the result of
``dist.is_initialized()`` *before* shutdown; ``destroy_process_group`` is only
called when that value is true.  This CPU-only builder binds that exact source
and the three V80 external-cleanup traces instead of rewriting history.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "experiments/eggroll_es_hpo/runs"
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_tp1_process_group_cleanup_v83.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_tp1_process_group_cleanup_v83_20260717.md"
)
LEGACY_SOURCE = ROOT / "probe_vllm_two_adapter_switch_v63.py"

SCHEMA = "v83-qwen36-tp1-process-group-cleanup-amendment"
LEGACY_SOURCE_SHA256 = (
    "115774a63f54480fa4796f24f5b47a82fda1c2a761db4cf3ae0b6b83e85165d6"
)
RUN_BUNDLES = {
    "v80_bf16_kv_mamba_capacity_0479_r1": (
        "73adc7ebe416d6065808cf918415d077989b1a064b47ef5132422b32da118e47"
    ),
    "v80_bf16_kv_mamba_capacity_0479_r2": (
        "3ac1156ac629d8c27d70756c08fa98652c74351549e5fea09711473900b584bc"
    ),
    "v80_bf16_kv_mamba_capacity_0479_r3": (
        "cd5107654df8367284a5fc720c50c3007ce16513ae77b55e7c5ccc52af3c3d5f"
    ),
}
GPU_IDS = (0, 1, 2, 3)


def canonical_sha256_v83(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256_v83(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_explicit_cleanup_v83(value: Mapping[str, Any]) -> dict[str, bool]:
    required = {
        "process_group_initialized_before_shutdown",
        "destroy_process_group_attempted",
        "process_group_initialized_after_shutdown",
    }
    if set(value) != required or any(
        not isinstance(value[field], bool) for field in required
    ):
        raise ValueError("V83 process-group receipt is missing or ambiguous")
    before = value["process_group_initialized_before_shutdown"]
    attempted = value["destroy_process_group_attempted"]
    after = value["process_group_initialized_after_shutdown"]
    if before and (not attempted or after):
        raise RuntimeError("initialized process group was not destroyed")
    if not before and (attempted or after):
        raise RuntimeError("never-initialized TP1 process-group state changed")
    return {
        "process_group_initialized_before_shutdown": before,
        "destroy_process_group_attempted": attempted,
        "process_group_initialized_after_shutdown": after,
        "cleanup_semantics_passed": True,
    }


def _inventory(run: str) -> tuple[list[dict[str, Any]], str]:
    directory = RUN_ROOT / run
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    if len(paths) != 11:
        raise RuntimeError(f"V83 run inventory changed: {run}")
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256_v83(path),
        }
        for path in paths
    ]
    bundle = canonical_sha256_v83(rows)
    if bundle != RUN_BUNDLES[run]:
        raise RuntimeError(f"V83 sealed run bundle changed: {run}")
    return rows, bundle


def _validate_receipt(run: str, gpu: int) -> dict[str, Any]:
    path = RUN_ROOT / run / f"gpu_{gpu}.json"
    value = json.loads(path.read_text(encoding="ascii"))
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    if (
        value.get("schema")
        != "v80-qwen36-bf16-kv-mamba-capacity-matched-preflight"
        or canonical_sha256_v83(body) != claimed
        or value.get("actor_label") != f"gpu-{gpu}"
        or value.get("torch_process_group_destroyed") is not False
        or value.get("engine_shutdown_completed") is not True
    ):
        raise RuntimeError(f"V83 legacy receipt changed: {run}/gpu-{gpu}")
    return {
        "gpu": gpu,
        "receipt_file_sha256": file_sha256_v83(path),
        "receipt_content_sha256": claimed,
        "legacy_field_value": False,
        "source_bound_interpretation": {
            "process_group_initialized_before_shutdown": False,
            "destroy_process_group_attempted": False,
            "destroy_was_required": False,
        },
    }


def _validate_cleanup(run: str) -> dict[str, Any]:
    path = RUN_ROOT / run / "gpu_telemetry_v80.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    batches = sorted({row.get("batch_index") for row in rows})
    if len(batches) < 3:
        raise RuntimeError(f"V83 telemetry is incomplete: {run}")
    accepted = []
    for batch in batches[-3:]:
        selected = [row for row in rows if row.get("batch_index") == batch]
        if (
            len(selected) != 4
            or sorted(row.get("gpu") for row in selected) != list(GPU_IDS)
            or any(
                row.get("actor_root_alive") is not False
                or row.get("compute_pids") != []
                or row.get("foreign_compute_pids") != []
                or row.get("cleanup_nvidia_smi_memory_used_mib", 5) > 4
                or row.get("cleanup_nvidia_smi_gpu_utilization_percent") != 0
                for row in selected
            )
        ):
            raise RuntimeError(f"V83 external cleanup changed: {run}/{batch}")
        accepted.append(
            {
                "batch_index": batch,
                "gpus": list(GPU_IDS),
                "actor_roots_dead": True,
                "compute_pids": [],
                "foreign_compute_pids": [],
                "external_memory_used_mib_max": max(
                    row["cleanup_nvidia_smi_memory_used_mib"] for row in selected
                ),
                "external_gpu_utilization_percent_max": max(
                    row["cleanup_nvidia_smi_gpu_utilization_percent"]
                    for row in selected
                ),
            }
        )
    return {
        "telemetry_file_sha256": file_sha256_v83(path),
        "minimum_consecutive_idle_batches": 3,
        "accepted_final_batches": accepted,
    }


def build_v83() -> dict[str, Any]:
    if file_sha256_v83(LEGACY_SOURCE) != LEGACY_SOURCE_SHA256:
        raise RuntimeError("V83 legacy shutdown source changed")
    runs = {}
    for run in RUN_BUNDLES:
        inventory, bundle = _inventory(run)
        runs[run] = {
            "artifact_inventory": inventory,
            "bundle_sha256": bundle,
            "actor_receipts": [_validate_receipt(run, gpu) for gpu in GPU_IDS],
            "external_cleanup": _validate_cleanup(run),
        }
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "bead": "specialist-nen.30",
        "status": "additive_interpretation_of_immutable_completed_runs",
        "authority": {
            "cpu_artifact_analysis_only": True,
            "model_or_gpu_launched": False,
            "dataset_or_protected_content_opened": False,
            "checkpoint_or_layout_promotion_authorized": False,
        },
        "immutable_parent_result": {
            "literal_torch_process_group_destroyed_clause_passed": False,
            "result_rewritten": False,
            "reason": "the legacy field name does not match its source semantics",
        },
        "legacy_source_binding": {
            "path": str(LEGACY_SOURCE.relative_to(ROOT)),
            "file_sha256": LEGACY_SOURCE_SHA256,
            "field_name": "torch_process_group_destroyed",
            "actual_value_semantics": "dist_is_initialized_before_shutdown",
            "destroy_call_guard": "called_if_and_only_if_legacy_value_is_true",
            "all_v80_values": False,
        },
        "additive_v83_verdict": {
            "tp1_process_group_was_never_initialized_all_actors_all_runs": True,
            "destroy_was_not_required": True,
            "actor_and_descendant_process_exit_independently_proven": True,
            "three_consecutive_external_idle_batches_all_runs": True,
            "cleanup_semantics_passed": True,
            "all_other_v80_gates_reused_not_rewritten": True,
            "semantic_ood_and_promotion_gates_still_pending": True,
        },
        "future_receipt_contract": {
            "required_boolean_fields": [
                "process_group_initialized_before_shutdown",
                "destroy_process_group_attempted",
                "process_group_initialized_after_shutdown",
            ],
            "never_initialized_acceptance": {
                "process_group_initialized_before_shutdown": False,
                "destroy_process_group_attempted": False,
                "process_group_initialized_after_shutdown": False,
            },
            "initialized_acceptance": {
                "process_group_initialized_before_shutdown": True,
                "destroy_process_group_attempted": True,
                "process_group_initialized_after_shutdown": False,
            },
            "external_dead_pid_and_idle_cleanup_remains_required": True,
            "missing_or_ambiguous_fields_fail_closed": True,
        },
        "runs": runs,
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v83(value)
    return value


def render_v83(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def report_v83(value: Mapping[str, Any]) -> str:
    run_rows = "\n".join(
        f"| {run} | 4/4 false | 3 | 4 MiB | 0% |"
        for run in value["runs"]
    )
    return f"""# Qwen3.6 TP1 process-group cleanup amendment V83

The immutable V80 parent verdict remains recorded as failed: it literally
required `torch_process_group_destroyed=true`, while all actors persisted
`false`.  V83 does not rewrite that result.  It binds the exact legacy source
and records that the field actually meant `dist.is_initialized()` **before**
shutdown.  The helper calls `destroy_process_group()` only when that value is
true.  Therefore the four TP1 actors in each run had no process group to
destroy.

| Run | legacy value | consecutive external idle batches | max memory | max util |
|---|---:|---:|---:|---:|
{run_rows}

Every accepted cleanup batch also has dead actor roots, no descendant compute
PID, and no foreign compute PID.  The additive cleanup interpretation passes;
all semantic, OOD, and promotion gates remain pending.

Future receipts must separately persist initialization-before, destroy
attempt, and initialization-after.  Never-initialized TP1 is accepted only as
`false/false/false`; an initialized group is accepted only as
`true/true/false`.  External dead-process and idle-GPU cleanup remains
mandatory in both cases.

- Legacy source SHA-256: `{LEGACY_SOURCE_SHA256}`
- V83 content SHA-256: `{value['content_sha256_before_self_field']}`
- Dataset/protected/model/GPU access by builder: none
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_v83()
    expected = render_v83(value)
    report = report_v83(value)
    if args.check:
        if OUTPUT.read_text(encoding="ascii") != expected:
            raise RuntimeError("V83 artifact is stale")
        if REPORT.read_text(encoding="ascii") != report:
            raise RuntimeError("V83 report is stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(expected, encoding="ascii")
        REPORT.write_text(report, encoding="ascii")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "report": str(REPORT),
                "content_sha256": value["content_sha256_before_self_field"],
                "cleanup_semantics_passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
