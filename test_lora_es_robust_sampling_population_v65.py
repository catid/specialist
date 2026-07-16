#!/usr/bin/env python3
"""CPU-only contracts for the V65 robust-sampling population."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import build_lora_es_robust_sampling_preregistration_v65 as builder
import eggroll_es_worker_robust_sampling_v65 as worker
import lora_es_robust_sampling_population_v65 as subject


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _metric_matrix(f1: float) -> list:
    value = np.zeros((64, 4, 2, 3), dtype=np.float64)
    value[..., 0] = f1
    value[..., 1] = float(f1 == 1.0)
    value[..., 2] = float(f1 > 0.0)
    return value.tolist()


def test_v65_ranking_panel_is_exact_hash_only_v61_projection():
    preview = subject.read_exact_self_hashed_v65(
        subject.PREVIEW_V61,
        subject.PREVIEW_V61_FILE_SHA256,
        subject.PREVIEW_V61_CONTENT_SHA256,
    )
    first = subject.build_ranking_panel_v65(preview)
    second = subject.build_ranking_panel_v65(copy.deepcopy(preview))
    assert first == second
    assert first["ranking_units"] == 64
    assert len(first["items"]) == 64
    assert first["untouched_partition_counts"] == {
        "train_holdback": 50,
        "exact_sentinel": 4,
        "unused_reserve": 90,
    }
    assert first["ranking_disjoint_from_every_untouched_partition"] is True
    assert first["question_answer_or_generation_text_persisted"] is False
    assert first["protected_semantics_opened"] is False
    assert all(set(item) == {
        "request_index", "row_sha256", "unit_identity_sha256", "stratum",
        "selection_priority_sha256",
    } for item in first["items"])
    assert all(
        item["request_index"] == index
        and len(item["row_sha256"]) == 64
        and len(item["unit_identity_sha256"]) == 64
        and len(item["selection_priority_sha256"]) == 64
        for index, item in enumerate(first["items"])
    )
    encoded = json.dumps(first, sort_keys=True).lower()
    for forbidden in (
        '"question"', '"answer"', '"text"', '"prompt"', '"generation"',
        '"base_mean_f1"', '"base_exact_actor_count"',
        '"base_nonzero_actor_count"',
    ):
        assert forbidden not in encoded
    assert first["content_sha256_before_self_field"] == (
        subject.self_content_sha256_v65(first)
    )


def test_v65_panel_fails_closed_on_partition_overlap_or_outcome_field():
    preview = subject.read_exact_self_hashed_v65(
        subject.PREVIEW_V61,
        subject.PREVIEW_V61_FILE_SHA256,
        subject.PREVIEW_V61_CONTENT_SHA256,
    )
    changed = copy.deepcopy(preview)
    changed["panels"]["untouched_holdback"][0]["unit_identity_sha256"] = (
        changed["panels"]["ranking"][0]["unit_identity_sha256"]
    )
    with pytest.raises(RuntimeError):
        subject.build_ranking_panel_v65(changed)

    changed = copy.deepcopy(preview)
    changed["panels"]["ranking"][0]["role_index"] = 1
    with pytest.raises(RuntimeError):
        subject.build_ranking_panel_v65(changed)


def test_v65_state_grid_repeats_each_v53_signed_state_exactly_twice():
    states = subject.state_derivations_v65()
    assert len(states) == subject.STATE_COUNT_V65 == 64
    assert [row["state_index"] for row in states] == list(range(64))
    assert {row["sigma"] for row in states} == {0.0048}
    assert {row["master_sha256"] for row in states} == {
        subject.design52.MASTER_SHA256_V52
    }
    assert [row["seed"] for row in states[::4]] == list(subject.SEEDS_V65)
    for direction in range(16):
        block = states[direction * 4:(direction + 1) * 4]
        assert [(row["label"], row["sign"], row["pass_index"])
                for row in block] == [
            ("plus", 1, 0), ("minus", -1, 0),
            ("minus", -1, 1), ("plus", 1, 1),
        ]
        assert [row["candidate_after_antithetic_peer"] for row in block] == [
            False, True, False, True,
        ]
    assert subject.GENERATION_COMPLETIONS_V65 == 16_384


def test_v65_population_analysis_is_paired_centered_and_update_free(
    monkeypatch,
):
    calls = []

    def fake_bootstrap(reference, candidate, *, bootstrap_indices):
        direction = len(calls)
        calls.append((
            reference.shape, candidate.shape, bootstrap_indices.shape,
        ))
        return {
            "robust_generation_fitness": float(direction + 1),
            "lower_confidence_bounds": {
                "stability_improvement": float(16 - direction),
            },
        }

    monkeypatch.setattr(
        subject, "paired_cluster_bootstrap_v65", fake_bootstrap,
    )
    monkeypatch.setattr(
        subject, "_split_pass_reliability_v65",
        lambda signed, robust, stability: {
            "schema": "synthetic-passing-discriminability",
            "passed": True,
            "checks": {"synthetic": True},
        },
    )
    signed = {
        "plus": [_metric_matrix(0.6) for _ in range(16)],
        "minus": [_metric_matrix(0.5) for _ in range(16)],
    }
    result = subject.analyze_signed_metrics_v65(signed)
    assert len(calls) == 16
    assert all(call == ((64, 4, 2, 3), (64, 4, 2, 3), (4096, 64))
               for call in calls)
    assert result["generation_completions"] == 16_384
    assert result["exact_used_for_population_ranking"] is False
    assert result["candidate_update_or_projection_performed"] is False
    assert result["train_holdback_or_exact_sentinel_opened"] is False
    assert result["protected_semantics_opened"] is False
    for key in (
        "robust_generation_unit_norm_centered_coefficients",
        "stability_lcb_unit_norm_centered_coefficients",
    ):
        coefficients = np.asarray(result[key])
        assert coefficients.shape == (16,)
        assert np.sum(coefficients) == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(coefficients) == pytest.approx(1.0)


@pytest.mark.parametrize("tamper", [
    "missing_direction", "bad_shape", "bad_exact", "exact_without_full_f1",
    "bad_nonzero",
])
def test_v65_population_analysis_fails_closed_on_malformed_metrics(
    monkeypatch, tamper,
):
    signed = {
        "plus": [_metric_matrix(0.6) for _ in range(16)],
        "minus": [_metric_matrix(0.5) for _ in range(16)],
    }
    if tamper == "missing_direction":
        signed["plus"].pop()
    elif tamper == "bad_shape":
        signed["plus"][0] = signed["plus"][0][:-1]
    elif tamper == "bad_exact":
        signed["plus"][0][0][0][0][1] = 0.5
    elif tamper == "exact_without_full_f1":
        signed["plus"][0][0][0][0][1] = 1.0
    elif tamper == "bad_nonzero":
        signed["plus"][0][0][0][0][2] = 0.0

    with pytest.raises((ValueError, RuntimeError)):
        subject.analyze_signed_metrics_v65(signed)


def test_v65_cluster_bootstrap_preserves_and_averages_all_eight_replicas():
    reference = np.zeros((64, 4, 2, 3), dtype=np.float64)
    reference[..., 0] = 0.1
    reference[..., 2] = 1.0
    candidate = reference.copy()
    replica_values = np.asarray([
        [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9],
    ])
    candidate[..., 0] = replica_values
    indices = np.tile(np.arange(64, dtype=np.int64), (4096, 1))
    result = subject.paired_cluster_bootstrap_v65(
        reference, candidate, bootstrap_indices=indices,
    )
    expected = float(np.mean(replica_values - 0.1))
    assert result["paired_replicas_per_unit_preserved_and_averaged"] == 8
    assert result["resampled_axis"] == "conflict_unit_only"
    assert result["point"]["f1_delta"] == pytest.approx(expected)
    assert result["lower_confidence_bounds"]["f1_delta"] == pytest.approx(
        expected
    )
    assert result["upper_confidence_bounds"]["f1_delta"] == pytest.approx(
        expected
    )


def test_v65_discriminability_gate_nulls_coefficients_below_noise():
    signed = {
        "plus": [_metric_matrix(0.6) for _ in range(16)],
        "minus": [_metric_matrix(0.5) for _ in range(16)],
    }
    result = subject.analyze_signed_metrics_v65(signed)
    gate = result["discriminability_gate"]
    assert gate["passed"] is False
    assert gate["checks"][
        "direction_spread_strictly_above_twice_v61c_null_halfwidth"
    ] is False
    assert gate["checks"][
        "stability_direction_spread_strictly_positive"
    ] is False
    assert result[
        "coefficients_actionable_for_later_preregistered_projection"
    ] is False
    assert result["robust_generation_unit_norm_centered_coefficients"] is None
    assert result["stability_lcb_unit_norm_centered_coefficients"] is None


def test_v65_exact_v53_state_identity_order_is_bound_and_unique():
    arm = subject.read_exact_self_hashed_v65(
        subject.V53_SIGMA_ARM,
        subject.V53_SIGMA_ARM_FILE_SHA256,
        subject.V53_SIGMA_ARM_CONTENT_SHA256,
    )
    identities = subject.expected_v53_state_identities_v65(arm)
    assert len(identities) == 32
    assert [(row["direction"], row["label"]) for row in identities] == [
        (direction, label)
        for direction in range(16) for label in ("plus", "minus")
    ]
    assert len({row["candidate_identity_sha256"] for row in identities}) == 32
    assert len({row["runtime_values_sha256"] for row in identities}) == 32
    assert subject.canonical_sha256_v65([
        row["candidate_identity_sha256"] for row in identities
    ]) == subject.V53_CANDIDATE_IDENTITY_INVENTORY_SHA256
    assert subject.canonical_sha256_v65([
        row["runtime_values_sha256"] for row in identities
    ]) == subject.V53_RUNTIME_IDENTITY_INVENTORY_SHA256


def test_v65_cpu_scorer_returns_only_hashes_and_numeric_metrics(monkeypatch):
    fake_fused = SimpleNamespace(
        _extract_answer=lambda text: text.removeprefix("answer:"),
        _tokens=lambda text: text.split(),
        _f1=lambda prediction, answer: float(prediction == answer),
    )
    fake_module = SimpleNamespace(
        v43i=SimpleNamespace(fused=fake_fused),
    )
    monkeypatch.setitem(
        sys.modules, "run_lora_es_generation_boundary_v48b", fake_module,
    )
    panel = [{
        "request_index": index,
        "row_sha256": _sha(f"row-{index}"),
        "unit_identity_sha256": _sha(f"unit-{index}"),
    } for index in range(64)]
    answers = [f"gold-{index}" for index in range(64)]
    batch = [SimpleNamespace(outputs=[SimpleNamespace(text=f"answer:gold-{index}")])
             for index in range(64)]
    rows = worker.score_generation_batch_v65(panel, answers, batch)
    assert len(rows) == 64
    assert all(set(row) == {
        "request_index", "row_sha256", "unit_identity_sha256",
        "prediction_sha256", "f1", "exact", "nonzero",
    } for row in rows)
    assert all(row["f1"] == row["exact"] == row["nonzero"] == 1
               for row in rows)
    encoded = json.dumps(rows, sort_keys=True)
    assert "gold-" not in encoded and "answer:" not in encoded


def _test_implementation_entries() -> dict:
    # The live runner is independently authored and intentionally need not
    # exist while these in-memory builder tests are first written.
    return {
        "population_design_v65": builder.ROOT / (
            "lora_es_robust_sampling_population_v65.py"
        ),
        "cpu_scoring_worker_v65": builder.ROOT / (
            "eggroll_es_worker_robust_sampling_v65.py"
        ),
        "preregistration_builder_v65": builder.ROOT / (
            "build_lora_es_robust_sampling_preregistration_v65.py"
        ),
        "base_model_byte_receipt_runtime_v64": builder.ROOT / (
            "run_lora_es_v59_vs_v434_robust_confirmation_v64.py"
        ),
        "transition_runtime_v51": builder.ROOT / (
            "run_lora_es_transition_microbenchmark_v51.py"
        ),
    }


def test_v65_builder_is_deterministic_hash_only_and_separately_authorizing(
    tmp_path,
):
    panel_path = tmp_path / "ranking-panel.json"
    first_panel, first = builder.build_v65(
        ranking_panel_output=panel_path,
        implementation_entry_paths=_test_implementation_entries(),
    )
    second_panel, second = builder.build_v65(
        ranking_panel_output=panel_path,
        implementation_entry_paths=_test_implementation_entries(),
    )
    assert first_panel == second_panel
    assert first == second
    assert "created_at_utc" not in first
    assert first["specific_v65_four_gpu_population_measurement_authorized"] is True
    assert first["prior_evidence_or_eligibility_alone_authorizes_launch"] is False
    authorization = first["authorization"]
    assert authorization["authority_origin"] == (
        "this_specific_v65_preregistration_only"
    )
    assert authorization["gpu_launch"] is True
    assert authorization["population_generation_measurement"] is True
    assert authorization["physical_gpu_ids"] == [0, 1, 2, 3]
    assert authorization["actors"] == 4
    for forbidden in (
        "projection", "optimizer_update", "candidate_snapshot", "promotion",
        "train_holdback", "exact_sentinel", "unused_reserve", "ood_shadow",
        "protected_semantics", "terminal_holdout",
    ):
        assert authorization[forbidden] is False
    assert first["ranking_panel"] == {
        "path": str(panel_path.resolve()),
        "file_sha256": builder.payload_sha256_v65(first_panel),
        "content_sha256": first_panel["content_sha256_before_self_field"],
        "units": 64,
        "request_order_sha256": first_panel["request_order_sha256"],
        "unit_order_sha256": first_panel["unit_order_sha256"],
        "hash_only": True,
        "question_answer_or_generation_text_persisted": False,
    }
    assert first["content_sha256_before_self_field"] == (
        subject.self_content_sha256_v65(first)
    )
    assert first["fixed_measurement_recipe"]["population_size"] == 16
    assert first["fixed_measurement_recipe"]["sigma"] == 0.0048
    assert first["fixed_measurement_recipe"]["unique_exact_v53_signed_states"] == 32
    assert first["fixed_measurement_recipe"]["state_occurrences"] == 64
    assert first["fixed_measurement_recipe"][
        "scored_generation_completions"
    ] == 16_384
    assert first["fixed_measurement_recipe"][
        "total_generation_completions"
    ] == 17_408
    assert first["fixed_measurement_recipe"]["unscored_master_warmup"] == {
        "periods": 4,
        "state": "exact unchanged V434 pinned master",
        "actors": 4,
        "ranking_requests_per_actor_period": 64,
        "discarded_generation_completions": 1_024,
        "occurs_before_every_signed_state": True,
        "exact_master_receipt_before_and_after_each_period": True,
        "raw_outputs_scored_or_persisted": False,
        "generation_metrics_computed_or_persisted": False,
        "adaptive_retry_drop_reorder_or_early_stop": False,
    }
    assert first["fixed_measurement_recipe"]["fixed_engine_controls"] == {
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "VLLM_BATCH_INVARIANT": False,
    }
    receipt = first["fixed_measurement_recipe"][
        "sanitized_live_engine_config_receipt"
    ]
    assert receipt["required_from_every_actor"] is True
    assert receipt["actors"] == 4
    assert receipt[
        "all_fields_must_exactly_equal_fixed_engine_controls"
    ] is True
    assert first["fixed_measurement_recipe"]["required_transition_rpc"] == (
        "transition_antithetic_from_pinned_master_v51"
    )
    assert first["fixed_measurement_recipe"][
        "intermediate_master_restore_between_occurrences"
    ] is False
    assert first["fixed_measurement_recipe"][
        "single_exact_master_restore_after_all_64_occurrences"
    ] is True
    assert first["population_analysis_contract"][
        "projection_or_update_performed"
    ] is False


def test_v65_builder_binds_exact_v61_v53_v62b_model_v434_and_dataset():
    panel, sources = builder.sealed_source_bindings_v65()
    implementation = builder.implementation_bindings_v65(
        _test_implementation_entries()
    )
    value = builder.build_preregistration_v65(
        panel, sources, implementation,
        ranking_panel_output=builder.RANKING_PANEL_OUTPUT,
    )
    assert sources["v61_hash_only_preview"]["file_sha256"] == (
        subject.PREVIEW_V61_FILE_SHA256
    )
    arm = sources["v53_exact_sigma_0p0048_arm"]
    assert arm["file_sha256"] == subject.V53_SIGMA_ARM_FILE_SHA256
    assert arm["content_sha256"] == subject.V53_SIGMA_ARM_CONTENT_SHA256
    assert arm["candidate_identity_inventory_sha256"] == (
        subject.V53_CANDIDATE_IDENTITY_INVENTORY_SHA256
    )
    assert arm["runtime_identity_inventory_sha256"] == (
        subject.V53_RUNTIME_IDENTITY_INVENTORY_SHA256
    )
    assert len(arm["ordered_state_identities"]) == 32
    assert arm["ordered_state_identities_sha256"] == (
        subject.canonical_sha256_v65(arm["ordered_state_identities"])
    )
    v61c = sources["v61c_eight_replica_null_calibration"]
    assert v61c["finalized"]["file_sha256"] == (
        subject.V61C_FINALIZED_FILE_SHA256
    )
    assert v61c["evidence"]["file_sha256"] == (
        builder.V61C_EVIDENCE_FILE_SHA256
    )
    assert v61c["within_unit_actor_pair_replicas_preserved_and_averaged"] == 8
    assert v61c["ranking_f1_null_halfwidth"] == (
        subject.V61C_RANKING_F1_NULL_HALFWIDTH
    )
    calibration = sources["v62b_finalized_calibration"]
    assert calibration["file_sha256"] == builder.V62B_FINALIZED_FILE_SHA256
    assert calibration["content_sha256"] == (
        builder.V62B_FINALIZED_CONTENT_SHA256
    )
    assert calibration["launch_or_update_authorized_by_source"] is False

    recipe = value["fixed_measurement_recipe"]
    assert recipe["v434_adapter"]["canonical_fp32_master_sha256"] == (
        subject.design52.MASTER_SHA256_V52
    )
    assert recipe["v434_adapter"]["bf16_runtime_values_sha256"] == (
        subject.design52.MASTER_RUNTIME_SHA256_V52
    )
    model = recipe["base_model"]
    assert model["all_top_level_files_fingerprint_sha256"] == (
        builder.BASE_MODEL_ALL_FILES_FINGERPRINT_SHA256
    )
    assert model["builder_read_model_directory_bytes"] is False
    dataset = recipe["dataset"]
    assert dataset["staged_semantic_rows"][
        "authorized_zero_based_line_interval"
    ] == [0, 63]
    assert dataset["staged_semantic_rows"][
        "authorized_exact_prefix_byte_count"
    ] == 136_848
    assert dataset["staged_semantic_rows"][
        "authorized_exact_prefix_sha256"
    ] == "8259894003268a2fafed6a9a66ce3e604d5eb76cdf19a1c1c759e5ffc5916c70"
    assert dataset["staged_semantic_rows"][
        "line_64_or_later_read_or_decoded_live"
    ] is False
    assert dataset["staged_semantic_rows"][
        "full_file_hash_verification_or_full_file_read_live"
    ] is False
    assert dataset["live_train_holdback_or_sentinel_rows_authorized"] is False

    analysis_contract = value["population_analysis_contract"]
    assert analysis_contract[
        "within_unit_actor_pass_replicas_preserved_and_averaged"
    ] == 8
    assert analysis_contract["single_replica_per_resampled_unit_sampling"] is False
    gate = analysis_contract["discriminability_gate"]
    assert gate["split_pass_spearman_minimum_inclusive"] == 0.5
    assert gate["split_pass_centered_cosine_minimum_inclusive"] == 0.5
    assert gate[
        "direction_population_standard_deviation_strictly_greater_than"
    ] == 2 * subject.V61C_RANKING_F1_NULL_HALFWIDTH
    assert "stability_direction_spread_strictly_positive" in gate[
        "required_check_keys"
    ]

    exposure = value["historical_adaptive_exposure"]
    assert exposure["manifest_sha256"] == (
        builder.ADAPTIVE_PANEL_OVERLAP_MANIFEST_SHA256_V65
    )
    assert exposure["globally_candidate_unexposed_claimed"] is False
    assert exposure["future_promotion_requires_panel_redesign"] is True
    assert exposure["v52_v53_panel_overlap_with_v61_partitions"] == {
        "ranking": {"conflict_units": 21, "selected_rows": 19},
        "holdback": {"conflict_units": 17, "selected_rows": 15},
        "exact_sentinel": {"conflict_units": 1, "selected_rows": 1},
        "unused_reserve": {"conflict_units": 25, "selected_rows": 17},
    }


def test_v65_implementation_binding_includes_complete_local_closure():
    bindings = builder.implementation_bindings_v65(
        _test_implementation_entries()
    )
    assert all(
        row["file_sha256"] == subject.file_sha256_v65(row["path"])
        for row in bindings.values()
    )
    relative_paths = {
        str(builder.Path(row["path"]).resolve().relative_to(builder.ROOT))
        for row in bindings.values()
    }
    assert {
        "lora_es_robust_sampling_population_v65.py",
        "eggroll_es_worker_robust_sampling_v65.py",
        "build_lora_es_robust_sampling_preregistration_v65.py",
        "run_lora_es_v59_vs_v434_robust_confirmation_v64.py",
        "run_lora_es_generation_boundary_v48b.py",
        "run_lora_es_transition_microbenchmark_v51.py",
    }.issubset(relative_paths)


def test_v65_builder_pair_write_is_exclusive_and_rolls_no_overwrite(tmp_path):
    first = tmp_path / "panel.json"
    second = tmp_path / "prereg.json"
    builder._exclusive_write_pair_v65(first, b"panel\n", second, b"prereg\n")
    assert first.read_bytes() == b"panel\n"
    assert second.read_bytes() == b"prereg\n"
    with pytest.raises(FileExistsError):
        builder._exclusive_write_pair_v65(first, b"changed", second, b"changed")
    assert first.read_bytes() == b"panel\n"
    assert second.read_bytes() == b"prereg\n"


def test_v65_builder_output_is_launchable_by_runtime_loader(tmp_path):
    panel_path = tmp_path / "ranking-panel.json"
    prereg_path = tmp_path / "preregistration.json"
    _panel, preregistration = builder.build_v65(
        ranking_panel_output=panel_path,
        preregistration_output=prereg_path,
    )
    prereg_path.write_bytes(builder.json_payload_v65(preregistration))
    runtime = importlib.import_module(
        "run_lora_es_robust_sampling_population_v65"
    )
    args = SimpleNamespace(
        preregistration=str(prereg_path),
        preregistration_sha256=subject.file_sha256_v65(prereg_path),
        preregistration_content_sha256=preregistration[
            "content_sha256_before_self_field"
        ],
    )
    assert runtime.load_preregistration_v65(args) == preregistration


def test_v65_builder_fails_closed_on_rehashed_authorization_tamper(tmp_path):
    panel, sources = builder.sealed_source_bindings_v65()
    implementation = builder.implementation_bindings_v65(
        _test_implementation_entries()
    )
    changed = copy.deepcopy(panel)
    changed["ranking_units"] = 63
    changed["content_sha256_before_self_field"] = (
        subject.self_content_sha256_v65(changed)
    )
    with pytest.raises(RuntimeError):
        builder.build_preregistration_v65(
            changed, sources, implementation,
            ranking_panel_output=tmp_path / "panel.json",
        )

    changed_sources = copy.deepcopy(sources)
    changed_sources["extra"] = {}
    with pytest.raises(RuntimeError):
        builder.build_preregistration_v65(
            panel, changed_sources, implementation,
            ranking_panel_output=tmp_path / "panel.json",
        )
