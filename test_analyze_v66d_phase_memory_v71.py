from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

import analyze_v66d_phase_memory_v71 as analysis
import eggroll_es_gpu_telemetry_v66 as telemetry


def _completion():
    return {
        "report_schema": "v66d-mirrored-crn-qwen36-calibration-report",
        "report_status": "complete_nonzero_train_only_no_commit_actor_attributed",
        "master_committed": False,
        "all_four_abort_receipts_exact": True,
        "final_exact_master_restored": True,
        "checkpoint_count": 0,
        "protected_dev_ood_or_holdout_opened": False,
    }


def _fixture():
    # Exercise a non-identity actor/GPU permutation.
    rank_to_gpu = {0: 2, 1: 0, 2: 3, 3: 1}
    bindings = [
        {
            "actor_rank": rank,
            "worker_pid": 7001 + rank,
            "physical_gpu_id": rank_to_gpu[rank],
        }
        for rank in range(4)
    ]
    by_gpu = telemetry.binding_by_gpu_v66(bindings)
    by_rank = {item["actor_rank"]: item for item in bindings}
    contract_sha = "a" * 64
    assignments = []
    receipts = []
    for wave in range(4):
        for rank in range(4):
            assignment = {
                "wave_index": wave,
                "engine_rank": rank,
                "direction_seed": 1000 + wave * 4 + rank,
                "sign": 1 if rank % 2 == 0 else -1,
                "pair_id": hashlib.sha256(
                    f"pair:{wave}:{rank // 2}".encode("ascii")
                ).hexdigest(),
                "evaluation_contract_sha256": contract_sha,
            }
            assignments.append(assignment)
            binding = by_rank[rank]
            receipt = {
                "schema": telemetry.WORK_RECEIPT_SCHEMA_V66D,
                "work_id": telemetry.work_id_v66d(assignment),
                **assignment,
                "worker_pid": binding["worker_pid"],
                "physical_gpu_id": binding["physical_gpu_id"],
                "cuda_event": {
                    "backend": "torch.cuda.Event",
                    "start_recorded": True,
                    "end_recorded": True,
                    "end_synchronized": True,
                    "elapsed_ms": 10.0 + rank,
                    "worker_monotonic_elapsed_ns": 20_000_000 + rank,
                },
                "output_cardinality": {
                    "request_outputs": 64,
                    "samples": 64,
                    "generated_tokens": 64,
                    "prompt_tokens": 128,
                },
            }
            receipts.append(telemetry.seal_actor_work_receipt_v66d(receipt))

    phases = ["setup", *analysis.expected_phases_v71(4)]
    rows = []
    sequence = 0
    start = datetime(2026, 7, 17, tzinfo=timezone.utc)
    monotonic_ns = 1_000_000_000
    for epoch, phase in enumerate(phases):
        for tick in range(2):
            sampled = (start + timedelta(milliseconds=250 * (epoch * 2 + tick))).isoformat()
            for gpu in range(4):
                binding = by_gpu[gpu]
                pid = binding["worker_pid"]
                rows.append({
                    "schema": telemetry.SCHEMA_V66,
                    "sequence": sequence,
                    "sampled_at_utc": sampled,
                    "monotonic_ns": monotonic_ns,
                    "phase": phase,
                    "phase_epoch": epoch,
                    "gpu": gpu,
                    "actor_rank": binding["actor_rank"],
                    "expected_pid": pid,
                    "compute_pids": [pid],
                    "process_memory_mib": {str(pid): 80_000 + epoch},
                    "foreign_compute_pids": [],
                    "gpu_utilization_percent": 50 if "generation" in phase else 2,
                    "memory_utilization_percent": 20,
                    "memory_used_mib": 80_100 + epoch,
                    "memory_total_mib": 97_887,
                    "pcie_rx_kib_per_second": 1024 + epoch,
                    "pcie_tx_kib_per_second": 2048 + epoch,
                    "power_draw_mw": 100_000 + gpu,
                })
                sequence += 1
            monotonic_ns += 250_000_000
    return {
        "rows": rows,
        "bindings": bindings,
        "assignments": assignments,
        "receipts": receipts,
        "adapter_metadata": analysis.load_adapter_metadata_v71(),
        "completion_proof": _completion(),
    }


def _analyze(fixture):
    return analysis.analyze_records_v71(**fixture)


def _reseal(receipt):
    receipt.pop("receipt_sha256", None)
    return telemetry.seal_actor_work_receipt_v66d(receipt)


def test_preregistration_and_static_contract_are_exact():
    assert json.loads(analysis.PREREGISTRATION_V71.read_text()) == (
        analysis.build_preregistration_v71()
    )
    contract = analysis.verify_static_contract_v71()
    assert contract["files"] == analysis.EXPECTED_STATIC_SHA256


def test_synthetic_full_analysis_and_transfer_ledger():
    result = _analyze(_fixture())
    assert result["passed"] is True
    assert result["telemetry_rows"] == 144
    assert result["complete_four_gpu_batches"] == 36
    assert result["phase_count"] == 18
    assert result["actor_event_coverage"]["receipt_count"] == 16
    transfers = result["transfer_byte_lower_bounds"]
    assert transfers["successful_runtime_materialization_and_audit_calls"] == 60
    assert transfers["h2d_bytes_total_lower_bound"] == 880_361_472
    assert transfers["d2h_bytes_total_lower_bound"] == 18_703_319_040
    assert transfers["component_lower_bounds"]["base_layer_exact_hash_d2h"] == (
        17_159_946_240
    )


def test_finalized_v66d_artifacts_pass_with_bound_results():
    result = analysis.analyze_finalized_run_v71()
    assert result["telemetry_rows"] == 644
    assert result["complete_four_gpu_batches"] == 161
    assert result["actor_event_coverage"]["receipt_count"] == 16
    assert result["artifact_receipts"]["gpu_activity_v66d.jsonl"][
        "file_sha256"
    ] == "a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475"
    assert {
        item["peak_memory_used_mib"] for item in result["overall_per_gpu"].values()
    } == {84_138}
    assert {
        item["minimum_headroom_mib"] for item in result["overall_per_gpu"].values()
    } == {13_749}
    assert result["sampled_pcie_integrals"] == {
        "claim": (
            "left-rectangle estimates over adjacent samples within each phase; "
            "not exact byte counts or lower bounds"
        ),
        "rx_bytes_sum_gpus": 9_421_838_461,
        "tx_bytes_sum_gpus": 15_099_460_768,
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["rows"].pop(),
        lambda value: value["rows"][5].__setitem__("sequence", 4),
        lambda value: value["rows"][5].__setitem__("gpu", 0),
        lambda value: value["rows"][8].__setitem__("monotonic_ns", 1),
        lambda value: value["rows"][8].__setitem__("phase", "typo"),
        lambda value: value["rows"][8].__setitem__("phase_epoch", 99),
        lambda value: value["rows"][0].__setitem__("sampled_at_utc", "naive"),
        lambda value: value["rows"][0].__setitem__("gpu_utilization_percent", True),
        lambda value: value["rows"][0]["process_memory_mib"].__setitem__("7002", -1),
        lambda value: value["rows"][0].__setitem__("foreign_compute_pids", [999]),
        lambda value: value["rows"][0].__setitem__("extra", 1),
    ],
)
def test_adversarial_telemetry_fails_closed(mutator):
    fixture = _fixture()
    mutator(fixture)
    with pytest.raises(RuntimeError):
        _analyze(fixture)


def test_missing_whole_phase_fails_closed():
    fixture = _fixture()
    phase = "mirrored_wave_1_materialize_all_actors"
    fixture["rows"] = [row for row in fixture["rows"] if row["phase"] != phase]
    for sequence, row in enumerate(fixture["rows"]):
        row["sequence"] = sequence
    with pytest.raises(RuntimeError, match="phase order/coverage"):
        _analyze(fixture)


def test_partial_pcie_support_is_reported_not_fabricated():
    fixture = _fixture()
    phase = "activate_v434_lora_slot_all_actors"
    row = next(row for row in fixture["rows"] if row["phase"] == phase and row["gpu"] == 0)
    row["pcie_rx_kib_per_second"] = None
    result = _analyze(fixture)
    rx = result["phase_statistics"][phase]["per_gpu"]["0"]["pcie_rx_bytes"]
    assert rx["supported_samples"] == 1
    assert rx["sampled_left_rectangle_estimate"] is None


def test_missing_or_hash_drifted_actor_receipt_fails_closed():
    fixture = _fixture()
    fixture["receipts"].pop()
    with pytest.raises(RuntimeError):
        _analyze(fixture)
    fixture = _fixture()
    fixture["receipts"][0]["cuda_event"]["elapsed_ms"] += 1.0
    with pytest.raises(RuntimeError, match="receipt hash"):
        _analyze(fixture)


def test_actor_wall_phase_cardinality_and_binding_checks_fail_closed():
    for mutation in ("wall", "phase", "cardinality", "binding"):
        fixture = _fixture()
        receipt = copy.deepcopy(fixture["receipts"][0])
        if mutation == "wall":
            receipt["cuda_event"]["elapsed_ms"] = 100.0
            receipt["cuda_event"]["worker_monotonic_elapsed_ns"] = 10_000_000
        elif mutation == "phase":
            receipt["cuda_event"]["elapsed_ms"] = 600.0
            receipt["cuda_event"]["worker_monotonic_elapsed_ns"] = 700_000_000
        elif mutation == "cardinality":
            receipt["output_cardinality"]["generated_tokens"] = 63
        else:
            receipt["physical_gpu_id"] = (receipt["physical_gpu_id"] + 1) % 4
        fixture["receipts"][0] = _reseal(receipt)
        with pytest.raises(RuntimeError):
            _analyze(fixture)


def test_assignment_metadata_and_completion_drift_fail_closed():
    fixture = _fixture()
    fixture["assignments"].append(copy.deepcopy(fixture["assignments"][0]))
    with pytest.raises(RuntimeError):
        _analyze(fixture)
    fixture = _fixture()
    fixture["adapter_metadata"]["runtime_elements"] += 1
    with pytest.raises(RuntimeError, match="adapter transfer metadata"):
        _analyze(fixture)
    fixture = _fixture()
    fixture["completion_proof"]["checkpoint_count"] = 1
    with pytest.raises(RuntimeError, match="completion proof"):
        _analyze(fixture)
