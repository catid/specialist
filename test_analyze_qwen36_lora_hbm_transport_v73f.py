from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import analyze_qwen36_lora_hbm_transport_v73f as analysis


def _artifact(*, transport_iterations: int, materialize_iterations: int) -> dict:
    operations = {
        "pageable_sync_h2d": (transport_iterations, 4.0 * transport_iterations),
        "pinned_nonblocking_h2d": (transport_iterations, 2.0 * transport_iterations),
        "tensorwise_gpu_materialize": (
            materialize_iterations,
            9.0 * materialize_iterations,
        ),
        "flat_gpu_materialize": (
            materialize_iterations,
            3.0 * materialize_iterations,
        ),
    }
    ranks = []
    for rank in range(4):
        repeats = []
        for repeat in range(3):
            row = {
                "repeat": repeat,
                "arm_order": list(analysis.OPERATIONS),
            }
            for name, (iterations, elapsed) in operations.items():
                row[name] = {
                    "iterations": iterations,
                    "cuda_elapsed_ms": elapsed,
                    "wall_elapsed_ms": elapsed,
                }
            repeats.append(row)
        ranks.append(
            {
                "rank": rank,
                "physical_gpu_id": rank,
                "staged_adapter_weights_sha256": "a" * 64,
                "flat_and_tensorwise_output_exact": True,
                "peak_allocated_bytes": 100 + rank,
                "peak_reserved_bytes": 200 + rank,
                "repeats": repeats,
            }
        )
    value = {
        "schema": analysis.EXPECTED_SCHEMA,
        "status": analysis.EXPECTED_STATUS,
        "authority": copy.deepcopy(analysis.EXPECTED_AUTHORITY),
        "world_size": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "repeats": 3,
        "transport_iterations": transport_iterations,
        "materialize_iterations": materialize_iterations,
        "ranks": ranks,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256(value)
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="ascii")


def test_analyze_three_exact_four_gpu_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    paths = []
    for index, scale in enumerate((1, 2, 4), 1):
        path = tmp_path / f"replicate-{index}.json"
        _write(
            path,
            _artifact(
                transport_iterations=100 * scale,
                materialize_iterations=200 * scale,
            ),
        )
        paths.append(path)

    result = analysis.analyze(paths)

    assert result["status"] == analysis.EXPECTED_STATUS
    assert len(result["runs"]) == 3
    assert result["aggregate_paired_speedup"][
        "pinned_nonblocking_over_pageable_sync_h2d"
    ]["geometric_mean"] == pytest.approx(2.0)
    assert result["aggregate_paired_speedup"][
        "flat_over_tensorwise_gpu_materialize"
    ]["geometric_mean"] == pytest.approx(3.0)
    assert result["content_sha256_before_self_field"] == analysis.canonical_sha256(
        {key: value for key, value in result.items() if key != "content_sha256_before_self_field"}
    )


@pytest.mark.parametrize("mutation", ("authority", "output"))
def test_rejects_authority_or_output_tampering(tmp_path: Path, mutation: str) -> None:
    value = _artifact(transport_iterations=100, materialize_iterations=200)
    if mutation == "authority":
        value["authority"]["quality_hpo_or_recipe_promotion_authorized"] = True
    else:
        value["ranks"][2]["flat_and_tensorwise_output_exact"] = False
    value.pop("content_sha256_before_self_field")
    value["content_sha256_before_self_field"] = analysis.canonical_sha256(value)
    path = tmp_path / "tampered.json"
    _write(path, value)

    with pytest.raises(RuntimeError):
        analysis._load_and_validate(path)
