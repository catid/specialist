import threading
import time
import hashlib

import pytest

import eggroll_es_gpu_telemetry_v66 as subject


BINDINGS = [
    {"actor_rank": rank, "worker_pid": 1000 + rank, "physical_gpu_id": gpu}
    for rank, gpu in enumerate((2, 0, 3, 1))
]


def row(phase, gpu, *, useful=True, resident=True, idle=False, sequence=0):
    binding = subject.binding_by_gpu_v66(BINDINGS)[gpu]
    pid = binding["worker_pid"]
    pids = [] if idle or not resident else [pid]
    return {
        "schema": subject.SCHEMA_V66,
        "sequence": sequence,
        "sampled_at_utc": "2026-07-17T00:00:00+00:00",
        "monotonic_ns": 1_000_000_000 + sequence * 250_000_000,
        "phase": phase,
        "phase_epoch": 1,
        "gpu": gpu,
        "actor_rank": binding["actor_rank"],
        "expected_pid": pid,
        "compute_pids": pids,
        "process_memory_mib": {str(pid): 82490} if pids else {},
        "foreign_compute_pids": [],
        "gpu_utilization_percent": 20 if useful else 0,
        "memory_utilization_percent": 12 if useful else 0,
        "memory_used_mib": 82494 if pids else 4,
        "memory_total_mib": 97887,
        "pcie_rx_kib_per_second": 1000 if useful else 0,
        "pcie_tx_kib_per_second": 2000 if useful else 0,
        "power_draw_mw": 300000 if useful else 50000,
    }


def contract():
    return [
        {"phase": "generation", "kind": "useful_gpu",
         "minimum_samples_per_gpu": 1},
        {"phase": "cpu_audit", "kind": "resident_cpu_bound",
         "minimum_samples_per_gpu": 1},
        {"phase": "cleanup", "kind": "cleanup_idle",
         "minimum_samples_per_gpu": 1},
    ]


def complete_rows():
    rows = []
    sequence = 0
    for phase, useful, idle in (
        ("generation", True, False),
        ("cpu_audit", False, False),
        ("cleanup", False, True),
    ):
        for gpu in range(4):
            rows.append(row(
                phase, gpu, useful=useful, idle=idle, sequence=sequence,
            ))
            sequence += 1
    return rows


def test_permuted_actor_bindings_and_phase_summary_pass():
    summary = subject.summarize_phases_v66(
        complete_rows(), BINDINGS, contract(),
    )
    assert summary["passed"] is True
    assert [
        item["physical_gpu_id"] for item in summary["bindings"]
    ] == [2, 0, 3, 1]
    assert summary["by_phase"]["generation"]["by_gpu"]["0"][
        "useful_samples"
    ] == 1
    assert summary["by_phase"]["cpu_audit"]["by_gpu"]["0"][
        "useful_samples"
    ] == 0


def test_missing_or_dead_actor_fails_closed():
    rows = complete_rows()
    for item in rows:
        if item["phase"] == "generation" and item["gpu"] == 2:
            item["compute_pids"] = []
            item["process_memory_mib"] = {}
    with pytest.raises(RuntimeError, match="lost its expected actor"):
        subject.summarize_phases_v66(rows, BINDINGS, contract())


def test_foreign_process_fails_closed():
    rows = complete_rows()
    rows[0]["compute_pids"].append(9999)
    rows[0]["process_memory_mib"]["9999"] = 1
    rows[0]["foreign_compute_pids"] = [9999]
    with pytest.raises(RuntimeError, match="PID attribution"):
        subject.summarize_phases_v66(rows, BINDINGS, contract())


def test_zero_activity_useful_phase_fails_closed():
    rows = complete_rows()
    for item in rows:
        if item["phase"] == "generation" and item["gpu"] == 1:
            item["gpu_utilization_percent"] = 0
            item["memory_utilization_percent"] = 0
            item["pcie_rx_kib_per_second"] = 0
            item["pcie_tx_kib_per_second"] = 0
    with pytest.raises(RuntimeError, match="lacked attributed useful work"):
        subject.summarize_phases_v66(rows, BINDINGS, contract())


def test_nonidle_cleanup_fails_closed():
    rows = complete_rows()
    cleanup = next(
        item for item in rows
        if item["phase"] == "cleanup" and item["gpu"] == 3
    )
    binding = subject.binding_by_gpu_v66(BINDINGS)[3]
    cleanup["compute_pids"] = [binding["worker_pid"]]
    cleanup["process_memory_mib"] = {str(binding["worker_pid"]): 10}
    cleanup["memory_used_mib"] = 20
    with pytest.raises(RuntimeError, match="did not become idle"):
        subject.summarize_phases_v66(rows, BINDINGS, contract())


def test_phase_transition_waits_for_monitor_acknowledgement():
    phase = subject.PhaseHandshakeV66()

    def acknowledge():
        while True:
            name, epoch = phase.snapshot()
            if name == "generation":
                phase.acknowledge(epoch, [0, 1, 2, 3])
                return
            time.sleep(0.001)

    worker = threading.Thread(target=acknowledge)
    worker.start()
    assert phase.transition("generation", timeout_seconds=1.0) == {
        "phase": "generation", "epoch": 1, "sampled": True,
    }
    worker.join(timeout=1.0)
    assert not worker.is_alive()


def test_phase_transition_timeout_fails_closed():
    phase = subject.PhaseHandshakeV66()
    with pytest.raises(TimeoutError, match="did not sample"):
        phase.transition("generation", timeout_seconds=0.01)


def assignments_v66d():
    result = []
    contract_sha = "c" * 64
    for wave in range(4):
        for rank in range(4):
            result.append({
                "wave_index": wave,
                "engine_rank": rank,
                "direction_seed": 1000 + wave * 4 + rank,
                "sign": 1 if (wave + rank) % 2 == 0 else -1,
                "pair_id": hashlib.sha256(
                    f"pair-{wave}-{rank}".encode()
                ).hexdigest(),
                "evaluation_contract_sha256": contract_sha,
            })
    return result


def receipt_v66d(assignment, *, elapsed_ms=2.5):
    binding = next(
        item for item in BINDINGS
        if item["actor_rank"] == assignment["engine_rank"]
    )
    value = {
        "schema": subject.WORK_RECEIPT_SCHEMA_V66D,
        "work_id": subject.work_id_v66d(assignment),
        **assignment,
        "worker_pid": binding["worker_pid"],
        "physical_gpu_id": binding["physical_gpu_id"],
        "cuda_event": {
            "backend": "torch.cuda.Event",
            "start_recorded": True,
            "end_recorded": True,
            "end_synchronized": True,
            "elapsed_ms": elapsed_ms,
            "worker_monotonic_elapsed_ns": 3_000_000,
        },
        "output_cardinality": {
            "request_outputs": 64,
            "samples": 64,
            "generated_tokens": 64,
            "prompt_tokens": 4096,
        },
    }
    return subject.seal_actor_work_receipt_v66d(value)


def generation_rows_v66d(*, useful=False):
    rows = []
    sequence = 0
    for wave in range(4):
        for gpu in range(4):
            item = row(
                f"mirrored_wave_{wave}_generation_all_actors",
                gpu,
                useful=useful,
                sequence=sequence,
            )
            item["phase_epoch"] = wave + 1
            item["monotonic_ns"] = 1_000_000_000 + wave * 250_000_000
            rows.append(item)
            sequence += 1
    return rows


def test_phase_cannot_end_after_partial_device_batch_before_monitor_tick():
    phase = subject.PhaseHandshakeV66()
    completed = threading.Event()

    def transition():
        phase.transition("short_generation", timeout_seconds=1.0)
        completed.set()

    worker = threading.Thread(target=transition)
    worker.start()
    while phase.snapshot()[0] != "short_generation":
        time.sleep(0.001)
    epoch = phase.snapshot()[1]
    with pytest.raises(RuntimeError, match="complete four-GPU batch"):
        phase.acknowledge(epoch, [0, 1, 2])
    time.sleep(0.01)
    assert completed.is_set() is False
    phase.acknowledge(epoch, [0, 1, 2, 3])
    worker.join(timeout=1.0)
    assert completed.is_set() is True


def test_short_zero_nvml_phase_passes_with_exact_actor_cuda_receipts():
    assignments = assignments_v66d()
    summary = subject.summarize_mirrored_waves_v66d(
        generation_rows_v66d(useful=False),
        BINDINGS,
        assignments,
        [receipt_v66d(item) for item in assignments],
    )
    assert summary[
        "sampling_handshake_acknowledged_all_four_rows_each_wave"
    ] is True
    assert summary["actor_cuda_event_receipt_for_every_candidate"] is True
    assert summary["by_wave"]["0"]["0"]["attribution_sources"] == [
        "actor_cuda_event_and_output_cardinality"
    ]


def test_resident_but_idle_actor_receipt_fails_closed():
    assignments = assignments_v66d()
    receipts = [receipt_v66d(item) for item in assignments]
    idle = dict(receipts[5])
    idle["output_cardinality"] = dict(idle["output_cardinality"])
    idle["output_cardinality"]["generated_tokens"] = 0
    receipts[5] = subject.seal_actor_work_receipt_v66d(idle)
    with pytest.raises(RuntimeError, match="generated token count"):
        subject.summarize_mirrored_waves_v66d(
            generation_rows_v66d(), BINDINGS, assignments, receipts,
        )


def test_missing_cuda_event_receipt_fails_closed():
    assignments = assignments_v66d()
    receipts = [receipt_v66d(item) for item in assignments][:-1]
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        subject.summarize_mirrored_waves_v66d(
            generation_rows_v66d(), BINDINGS, assignments, receipts,
        )


def test_wrong_receipt_pid_or_gpu_fails_closed():
    assignments = assignments_v66d()
    receipts = [receipt_v66d(item) for item in assignments]
    wrong = dict(receipts[0])
    wrong["worker_pid"] += 99
    receipts[0] = subject.seal_actor_work_receipt_v66d(wrong)
    with pytest.raises(RuntimeError, match="PID/GPU binding"):
        subject.summarize_mirrored_waves_v66d(
            generation_rows_v66d(), BINDINGS, assignments, receipts,
        )


def test_partial_four_actor_wave_is_rejected_before_receipt_validation():
    assignments = assignments_v66d()[:-1]
    with pytest.raises(RuntimeError, match="complete four-actor wave"):
        subject.summarize_mirrored_waves_v66d(
            generation_rows_v66d(),
            BINDINGS,
            assignments,
            [receipt_v66d(item) for item in assignments],
        )


def test_partial_monitor_batch_cannot_masquerade_as_handshake_ack():
    assignments = assignments_v66d()
    rows = generation_rows_v66d()
    changed = next(
        item for item in rows if item["phase_epoch"] == 1 and item["gpu"] == 3
    )
    changed["monotonic_ns"] += 1
    with pytest.raises(RuntimeError, match="complete four-GPU sample batch"):
        subject.summarize_mirrored_waves_v66d(
            rows,
            BINDINGS,
            assignments,
            [receipt_v66d(item) for item in assignments],
        )
