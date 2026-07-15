import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_eggroll_es_lagged_replay_calibration_v35a as runtime


def _foundation():
    preregistration = runtime.load_preregistration()
    manifest = runtime.load_manifest(preregistration)
    layer = runtime.load_layer_bundle(preregistration)
    implementation = runtime.implementation_identity()
    recipe = runtime.recipe_v35a(
        preregistration, manifest, layer, implementation
    )
    return preregistration, manifest, layer, implementation, recipe


def _metadata_and_values():
    preregistration = runtime.load_preregistration()
    metadata = []
    cursor = 0
    for panel in runtime.OPTIMIZATION_PANELS:
        for stratum in runtime.STRATA:
            count = preregistration["difficulty_calibration"][
                "tier_counts_per_panel"
            ][stratum]["panel_rows"]
            for _ in range(count):
                metadata.append({
                    "panel": panel,
                    "stratum": stratum,
                    "row_sha256": f"{cursor:064x}",
                })
                cursor += 1
    values = np.tile(np.arange(168, dtype=np.float64), (4, 1))
    return preregistration, metadata, values


def _dense_fixture(logprob=-0.5):
    items = []
    outputs = []
    for index in range(runtime.UNION_ROWS):
        items.append({
            "example_index": index,
            "prompt_sha256": f"{index:064x}",
            "answer_sha256": f"{index + 1:064x}",
            "prompt_token_count": 1,
            "answer_token_start": 1,
            "answer_token_count": 1,
            "prompt_token_ids": [10, 20 + index],
            "prompt_token_ids_sha256": runtime.canonical_sha256([10, 20 + index]),
            "eos_appended": False,
        })
        outputs.append(SimpleNamespace(
            prompt_token_ids=[10, 20 + index],
            prompt_logprobs=[None, {
                20 + index: SimpleNamespace(logprob=logprob)
            }],
        ))
    return items, outputs


def test_v35a_frozen_preregistration_protocol_manifest_and_source_bindings():
    preregistration = runtime.load_preregistration()
    manifest = runtime.load_manifest(preregistration)
    assert runtime.file_sha256(runtime.PREREG_PATH) == runtime.PREREG_FILE_SHA256
    assert runtime.file_sha256(runtime.PROTOCOL_PATH) == runtime.PROTOCOL_FILE_SHA256
    assert runtime.file_sha256(runtime.MANIFEST_PATH) == runtime.MANIFEST_FILE_SHA256
    assert manifest["source"]["jsonl_sha256"] == runtime.SOURCE_FILE_SHA256
    assert preregistration["content_sha256_before_self_field"] == (
        runtime.PREREG_CONTENT_SHA256
    )


def test_v35a_projects_exact_optimization_union_and_excludes_screens():
    preregistration = runtime.load_preregistration()
    manifest = runtime.load_manifest(preregistration)
    projected = runtime.optimization_metadata(manifest)
    assert len(projected) == 168
    assert {item["panel"] for item in projected} == set(
        runtime.OPTIMIZATION_PANELS
    )
    assert all(item["panel"] not in runtime.SCREEN_PANELS for item in projected)
    poisoned = copy.deepcopy(manifest)
    screen = next(p for p in poisoned["panels"] if p["name"] == "train_screen_0")
    screen["items"][0]["row_index"] = projected[0]["row_index"]
    with pytest.raises(RuntimeError, match="overlap"):
        runtime.optimization_metadata(poisoned)


def test_v35a_selective_jsonl_projection_does_not_decode_unselected_lines(tmp_path):
    metadata = []
    raw_lines = []
    for index in range(168):
        row = {
            "question": f"q{index}",
            "answer": f"a{index}",
            "fact_id": f"f{index}",
            "document_sha256": f"{index:064x}",
        }
        encoded = json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
        raw_lines.append(encoded)
        metadata.append({
            "row_index": index,
            "row_sha256": runtime.sampler_v13.row_sha256(row),
            "fact_id": row["fact_id"],
            "document_sha256": row["document_sha256"],
            "panel": runtime.OPTIMIZATION_PANELS[index // 56],
            "position": index % 56,
            "stratum": "resources_general",
        })
    raw_lines.extend([b"not-json-screen-line\n", b"\xffscreen-bytes\n"])
    source = tmp_path / "train_only.jsonl"
    source.write_bytes(b"".join(raw_lines))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = runtime.stream_exact_optimization_rows(
        source, metadata, expected_sha256=digest, expected_rows=170
    )
    assert len(rows) == 168
    assert rows[-1]["question"] == "q167"


def test_v35a_materialized_union_binds_each_selected_source_identity():
    preregistration = runtime.load_preregistration()
    manifest = runtime.load_manifest(preregistration)
    rows, audit = runtime.materialize_optimization_union(
        preregistration, manifest
    )
    assert len(rows) == 168
    assert audit["screen_source_lines_decoded"] is False
    assert audit["screen_rows_materialized_as_requests"] is False


def test_v35a_exact_gold_token_trace_commitment_detects_one_logprob_change():
    items, outputs = _dense_fixture()
    values, digest = runtime._score_union_outputs(items, outputs)
    assert values.shape == (168,)
    changed = copy.deepcopy(outputs)
    changed[73].prompt_logprobs[1][93].logprob = -0.5000001
    changed_values, changed_digest = runtime._score_union_outputs(items, changed)
    assert digest != changed_digest
    assert not np.array_equal(values, changed_values)


def test_v35a_phase_guards_require_exact_engines_and_A_B_C():
    values = np.ones((4, 168), dtype=np.float64)
    phase = (values, ["same"] * 4)
    phases = {name: copy.deepcopy(phase) for name in runtime.PHASES}
    audit = runtime.assert_cross_phase_exact(phases)
    assert audit["all_dense_results_exact_across_A_B_C"] is True
    changed = copy.deepcopy(phases)
    changed["C"][1][2] = "one-token-trace-changed"
    with pytest.raises(RuntimeError, match="not exact"):
        runtime.assert_cross_phase_exact(changed)
    changed = copy.deepcopy(phases)
    changed["B"][0][3, 1] = 2.0
    with pytest.raises(RuntimeError, match="not exact"):
        runtime.assert_cross_phase_exact(changed)


def test_v35a_phase_B_only_ranking_has_exact_review_pool_counts_and_ties():
    preregistration, metadata, values = _metadata_and_values()
    values[:, :] = 0.0
    provisional, _digest = runtime.build_provisional_pool(
        metadata, values, preregistration
    )
    assert len(provisional) == 87
    assert all(
        sum(item["panel"] == panel for item in provisional) == 29
        for panel in runtime.OPTIMIZATION_PANELS
    )
    per_stratum = {
        stratum: sum(
            item["panel"] == "optimization_0" and item["stratum"] == stratum
            for item in provisional
        )
        for stratum in runtime.STRATA
    }
    assert per_stratum == {
        "safety_consent": 5,
        "technique": 8,
        "equipment_material": 3,
        "resources_general": 13,
    }
    first_group = [
        item for item in provisional
        if item["panel"] == "optimization_0"
        and item["stratum"] == runtime.STRATA[0]
    ]
    assert [item["row_sha256"] for item in first_group] == sorted(
        item["row_sha256"] for item in first_group
    )


def test_v35a_ranking_rejects_any_cross_engine_difference():
    preregistration, metadata, values = _metadata_and_values()
    values[2, 17] += 1e-9
    with pytest.raises(RuntimeError, match="exact four-engine phase B"):
        runtime.build_provisional_pool(metadata, values, preregistration)


def test_v35a_manual_review_shuffle_is_deterministic_content_independent():
    preregistration, metadata, values = _metadata_and_values()
    provisional, _ = runtime.build_provisional_pool(
        metadata, values, preregistration
    )
    first = runtime.deterministic_manual_review_order(provisional)
    second = runtime.deterministic_manual_review_order(list(reversed(provisional)))
    assert first == second
    assert len(first) == len(set(first)) == 87


def test_v35a_content_free_schema_rejects_payloads_and_misplaced_row_ids():
    for key in (
        "question", "answer", "prompt_token_ids", "scores", "logprobs",
        "outputs", "pid", "memory_used", "manual_review_text",
    ):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            runtime.assert_content_free({key: "secret"})
    with pytest.raises(RuntimeError, match="provisional-pool schema"):
        runtime.assert_content_free({"row_sha256": "0" * 64})
    runtime.assert_content_free({
        "row_sha256": "0" * 64,
        "panel": "optimization_0",
        "stratum": "technique",
        "calibration_rank": 1,
    })


def test_v35a_recipe_and_budget_are_exact_and_close_authority():
    _prereg, _manifest, _layer, _implementation, recipe = _foundation()
    assert recipe["request_accounting"] == {
        "requests_per_engine_per_phase": 168,
        "engines": 4,
        "requests_per_phase": 672,
        "calibration_estimand_requests": 672,
        "integrity_repeat_requests": 1344,
        "physical_total_requests": 2016,
    }
    assert recipe["integrity_phases"]["calibration_estimand_phase"] == "B"
    assert recipe["screen_firewall"]["decoded_into_requests"] is False
    assert recipe["authority"]["manual_review_of_provisional_training_rows"] is True
    assert all(
        value is False
        for key, value in recipe["authority"].items()
        if key != "manual_review_of_provisional_training_rows"
    )


def test_v35a_dry_run_launches_no_GPU_or_runtime(capsys):
    result = runtime.main(["--v35a-dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output == result
    assert result["gpu_launched"] is False
    assert result["runtime_launched"] is False
    assert result["request_accounting"]["physical_total_requests"] == 2016
    assert result["provisional_candidate_count"] == 87
    assert result["final_hard_tier_capacity_after_manual_review"] == 39


def test_v35a_dry_run_exact_optional_hashes_fail_closed(capsys):
    first = runtime.main(["--v35a-dry-run"])
    capsys.readouterr()
    result = runtime.main([
        "--v35a-dry-run",
        "--expected-implementation-bundle-sha256",
        first["implementation_bundle_sha256"],
        "--expected-recipe-sha256", first["recipe_sha256"],
    ])
    assert result["gpu_launched"] is False
    capsys.readouterr()
    with pytest.raises(ValueError, match="implementation bundle"):
        runtime.main([
            "--v35a-dry-run",
            "--expected-implementation-bundle-sha256", "0" * 64,
        ])


def test_v35a_real_launch_requires_hashes_and_exact_source_commit():
    prereg, _manifest, _layer, implementation, recipe = _foundation()
    base = {
        "v35a_dry_run": False,
        "expected_implementation_bundle_sha256": None,
        "expected_recipe_sha256": None,
        "expected_source_commit": None,
    }
    with pytest.raises(ValueError, match="implementation bundle"):
        runtime.validate_runtime_args(
            argparse.Namespace(**base), prereg, implementation, recipe
        )
    base["expected_implementation_bundle_sha256"] = implementation["bundle_sha256"]
    with pytest.raises(ValueError, match="expected recipe"):
        runtime.validate_runtime_args(
            argparse.Namespace(**base), prereg, implementation, recipe
        )
    base["expected_recipe_sha256"] = recipe["content_sha256_before_self_field"]
    with pytest.raises(ValueError, match="source commit"):
        runtime.validate_runtime_args(
            argparse.Namespace(**base), prereg, implementation, recipe
        )


def test_v35a_forbids_nontrain_mutation_argv_and_backend_overrides(monkeypatch):
    for token in (
        "--validation=x", "--holdout=x", "--ood=x", "--checkpoint=x",
        "--update=x", "--promotion=x", "--replay-fraction=.1",
    ):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            runtime.assert_train_only_argv([token])
    prereg, _manifest, _layer, implementation, recipe = _foundation()
    args = argparse.Namespace(
        v35a_dry_run=True,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
        expected_source_commit=None,
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariant"):
        runtime.validate_runtime_args(args, prereg, implementation, recipe)


def test_v35a_allowlist_accepts_only_curator_and_named_background_paths():
    result = runtime.validate_worktree_status(
        "?? data/manual_reviews/context_merit_audit_v410/a.json\n"
        "?? experiments/dataset_probes/a.json\n"
        "?? experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl\n"
    )
    assert result["all_tracked_files_clean"] is True
    assert result["allowed_untracked_entry_count"] == 3
    for status in (
        " M run.py\n", "?? unknown.txt\n",
        "?? data/manual_reviews/context_merit_audit_v389/a.json\n",
    ):
        with pytest.raises(RuntimeError, match="committed-clean"):
            runtime.validate_worktree_status(status)


def test_v35a_partial_constructor_preserves_cleanup_handle(monkeypatch):
    class Broken:
        def __new__(cls):
            return object.__new__(cls)

        def __init__(self, **_kwargs):
            self._v35a_partial_engines = ["engine"]
            self._v35a_partial_groups = ["group"]
            raise RuntimeError("constructor failed")

    monkeypatch.setattr(runtime, "load_runtime_trainer", lambda *_args: Broken)
    with pytest.raises(RuntimeError) as captured:
        runtime.make_trainer_fail_closed(
            {"frozen_recipe": {"model": "unused"}}, {}
        )
    partial = getattr(captured.value, "_v35a_partial_trainer")
    assert partial._v35a_partial_engines == ["engine"]
    assert partial._v35a_partial_groups == ["group"]


def test_v35a_partial_cleanup_closes_process_pool_even_if_base_cleanup_fails(
    monkeypatch,
):
    events = []

    class Pool:
        def close(self):
            events.append("pool_close")

        def join(self):
            events.append("pool_join")

    partial = SimpleNamespace(
        _v35a_partial_engines=[],
        _v35a_partial_groups=[],
        mp_pool=Pool(),
    )

    def broken_close(_trainer):
        events.append("base_close")
        raise RuntimeError("partial base cleanup")

    monkeypatch.setattr(runtime.runtime_v23a.base, "close_trainer", broken_close)
    with pytest.raises(RuntimeError, match="partial base cleanup"):
        runtime.close_trainer_fail_closed(partial)
    assert events == ["base_close", "pool_close", "pool_join"]


def test_v35a_runtime_mixin_closes_update_eval_and_training_surfaces():
    instance = object.__new__(runtime.LaggedReplayCalibrationRuntimeMixinV35A)
    for method in (
        "configure_anchor", "train_step", "fit", "eval_step",
        "evaluate_handle", "apply_seed_coefficients",
    ):
        with pytest.raises(RuntimeError, match="closes"):
            getattr(instance, method)()


def test_v35a_fresh_paths_and_writes_are_exclusive(tmp_path, monkeypatch):
    target = tmp_path / "claim.json"
    monkeypatch.setattr(runtime, "ATTEMPT_PATH", target)
    value = {"schema": "claim", "content_sha256_before_self_field": "x"}
    runtime._write_attempt(value)
    with pytest.raises(RuntimeError, match="exclusive output already exists"):
        runtime._write_attempt(value)
