import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import monitor_qwen36_fp8_kv_capacity_v79 as subject


def test_actor_pid_map_requires_exact_unique_four(tmp_path: Path):
    path = tmp_path / "pids.csv"
    path.write_text("0,101\n1,102\n2,103\n3,104\n", encoding="ascii")
    assert subject.read_actor_pids_v79(path) == {
        0: 101,
        1: 102,
        2: 103,
        3: 104,
    }
    path.write_text("0,101\n1,101\n2,103\n3,104\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="four unique"):
        subject.read_actor_pids_v79(path)


def test_process_ancestry_attributes_current_process():
    assert subject.is_descendant_v79(os.getpid(), os.getpid()) is True
    assert subject.is_descendant_v79(os.getpid(), 999_999_999) is False


class _Nvml:
    NVML_PCIE_UTIL_RX_BYTES = 0
    NVML_PCIE_UTIL_TX_BYTES = 1

    class NVMLError(Exception):
        pass

    @staticmethod
    def nvmlDeviceGetPcieThroughput(_handle, counter):
        return 1000 + counter

    @staticmethod
    def nvmlDeviceGetComputeRunningProcesses(_handle):
        return []

    @staticmethod
    def nvmlDeviceGetUtilizationRates(_handle):
        return SimpleNamespace(gpu=12, memory=34)

    @staticmethod
    def nvmlDeviceGetMemoryInfo(_handle):
        return SimpleNamespace(used=4 * 1024 * 1024, total=97_887 * 1024 * 1024)

    @staticmethod
    def nvmlDeviceGetUUID(handle):
        return f"GPU-{handle}"

    @staticmethod
    def nvmlDeviceGetPowerUsage(_handle):
        return 50_000


def test_sample_binds_hbm_pcie_power_and_pid_fields():
    rows = subject.sample_batch_v79(
        _Nvml,
        {gpu: gpu for gpu in subject.GPU_IDS_V79},
        {gpu: 100 + gpu for gpu in subject.GPU_IDS_V79},
        2,
        require_pcie=True,
    )
    assert [row["gpu"] for row in rows] == [0, 1, 2, 3]
    assert [row["sequence"] for row in rows] == [8, 9, 10, 11]
    assert all(row["memory_utilization_percent"] == 34 for row in rows)
    assert all(row["pcie_rx_kib_per_second"] == 1000 for row in rows)
    assert all(row["pcie_tx_kib_per_second"] == 1001 for row in rows)
    assert all(row["hbm_bytes_per_second_inferred"] is False for row in rows)
    assert all(row["pcie_counters_supported"] is True for row in rows)


def test_pcie_unsupported_fails_closed():
    class Unsupported(_Nvml):
        nvmlDeviceGetPcieThroughput = None

    with pytest.raises(RuntimeError, match="unsupported"):
        subject.sample_batch_v79(
            Unsupported,
            {gpu: gpu for gpu in subject.GPU_IDS_V79},
            {gpu: 100 + gpu for gpu in subject.GPU_IDS_V79},
            0,
            require_pcie=True,
        )
