import copy

import pytest

import benchmark_eggroll_es_host_state_v72 as benchmark
import eggroll_es_host_state_contract_v72 as contract


def test_isolated_cpu_benchmark_proves_exact_bytes_and_lower_observed_rss():
    value = benchmark.run_benchmark_v72()
    certificate = value["validation"]
    assert certificate["passed"] is True
    assert certificate["exact_peak_tensor_bytes_saved"] \
        == 5 * contract.MASTER_BYTES_V72
    assert certificate["exact_copy_bytes_saved"] \
        == 25 * contract.MASTER_BYTES_V72
    assert certificate["observed_rss_delta_saved_bytes"] > 0
    assert value["gpu_visible_to_children"] is False
    assert value["gpu_launch_performed"] is False


def test_benchmark_validation_rejects_tampering():
    value = benchmark.run_benchmark_v72()
    value.pop("validation")
    tampered = copy.deepcopy(value)
    tampered["arms"]["proposed_peak"]["bank_count"] = 3
    with pytest.raises(RuntimeError, match="content hash changed"):
        benchmark.validate_benchmark_v72(tampered)
