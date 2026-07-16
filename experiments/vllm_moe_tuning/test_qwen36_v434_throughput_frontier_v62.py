#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


REPORT = Path(__file__).with_name("qwen36_v434_throughput_frontier_v62.json")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_frontier_report_is_self_hashed_cpu_only_and_launch_ineligible():
    value = _report()
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == _canonical_sha256(compact)
    assert value["scope"]["source_files"] == 56
    assert value["scope"]["gpu_or_model_accessed_while_building_report"] is False
    assert value["scope"]["active_v62a_run_read_or_modified"] is False
    assert not any(
        value["authorization"][key]
        for key in value["authorization"]
        if key.endswith("_authorized")
    )


def test_frontier_and_length_ratios_recompute_exactly():
    value = _report()
    crossover = value["concurrency_frontier"]["crossover_points_4096_requests"]
    best = max(crossover, key=lambda item: item["tokens_per_second"])
    assert best["max_num_seqs"] == 92
    assert best["tokens_per_second"] == 5853.401884016819
    for item in value["matched_length_interactions_seq88_vs_seq68"]:
        expected = (
            item["seq88_tokens_per_second"]
            / item["seq68_tokens_per_second"] - 1.0
        ) * 100.0
        assert math.isclose(item["seq88_vs_seq68_percent"], expected)


def test_lora_pairs_and_actor_variability_recompute_exactly():
    value = _report()
    for item in value["lora_vs_base_paired_results"]:
        ratio = item["lora_tokens_per_second"] / item["base_tokens_per_second"]
        assert math.isclose(item["lora_over_base_ratio"], ratio)
        assert math.isclose(item["lora_vs_base_percent"], (ratio - 1.0) * 100.0)
        assert ratio < 1.0
    repeats = value["actor_variability"]["base_frontier_seq68_four_gpu_repeats"]
    points = list(repeats["per_gpu_tokens_per_second"].values())
    assert math.isclose(repeats["mean_tokens_per_second"], statistics.mean(points))
    assert math.isclose(
        repeats["population_standard_deviation"],
        statistics.pstdev(points),
    )


def test_report_preserves_exact_runtime_identities_and_random_caveat():
    value = _report()
    identity = value["exact_runtime_identities"]
    assert identity["software"]["vllm_version"] == "0.25.0"
    assert identity["model"]["model_config_file_sha256"] == (
        "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
    )
    assert identity["v434_lora"]["adapter_weights_file_sha256"] == (
        "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
    )
    assert identity["v27c_moe_tuning"]["config_file_sha256"] == (
        "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
    )
    caveat = value["workload_and_interpretation_caveats"]
    assert caveat["random_seeds"].startswith("varied across runs")
    assert caveat["ordinary_warmup_requests"] == 4
    assert caveat["warm88_filename_warmup_requests"] == 88
    assert caveat["semantic_quality_claims_authorized"] is False
