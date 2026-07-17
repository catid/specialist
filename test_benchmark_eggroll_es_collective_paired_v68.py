import math

import pytest

import benchmark_eggroll_es_collective_paired_v68 as subject


def blocks(pair_count=4):
    result = []
    for pair_index, order in enumerate(subject.block_orders_v68(pair_count)):
        for layout in order:
            result.append({
                "pair_index": pair_index,
                "layout": layout,
                "iterations": 64,
                "seconds": 2.0 if layout == "native" else 2.2,
            })
    return result


def test_order_is_exactly_counterbalanced():
    value = subject.block_orders_v68(6)
    assert value == (
        ("native", "flat"), ("flat", "native"),
        ("native", "flat"), ("flat", "native"),
        ("native", "flat"), ("flat", "native"),
    )
    with pytest.raises(ValueError):
        subject.block_orders_v68(3)


def test_paired_summary_uses_with_process_ratios():
    value = subject.paired_summary_v68(blocks(), 4)
    assert value["native_faster_pair_count"] == 4
    assert value["flat_faster_pair_count"] == 0
    assert value["native_over_flat_geometric_speed"] == pytest.approx(1.1)
    assert value["native_over_flat_median_speed"] == pytest.approx(1.1)


def test_summary_rejects_order_count_and_nonfinite_timing():
    value = blocks()
    value[0], value[1] = value[1], value[0]
    with pytest.raises(RuntimeError, match="order"):
        subject.paired_summary_v68(value, 4)
    with pytest.raises(RuntimeError, match="count"):
        subject.paired_summary_v68(blocks()[:-1], 4)
    value = blocks()
    value[0]["seconds"] = math.nan
    with pytest.raises(RuntimeError, match="timing"):
        subject.paired_summary_v68(value, 4)


def test_geometric_mean_rejects_nonpositive_or_nonfinite_values():
    assert subject.geometric_mean_v68([1.0, 4.0]) == pytest.approx(2.0)
    for values in ([], [0.0], [math.inf], [math.nan]):
        with pytest.raises(ValueError):
            subject.geometric_mean_v68(values)
