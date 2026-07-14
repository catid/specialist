import copy
import hashlib
import json
from pathlib import Path

import pytest

import build_eggroll_es_edge_split_plans_v6 as plan_builder
import eggroll_es_worker_v4 as worker_v4
import eggroll_es_worker_v6 as worker_v6
import run_eggroll_es_anchor_line_search_v6 as line_search_v6
import train_eggroll_es_specialist_anchor_v6 as anchor_v6


def load_bundle(plan_sha256):
    spec = anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6[plan_sha256]
    return anchor_v6.load_frozen_layer_plan_v6(
        spec["path"],
        expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=plan_sha256,
        expected_model_config_sha256=anchor_v6.MODEL_CONFIG_SHA256_V6,
    )


def smoke_args():
    return [
        "--v6-stage", "smoke",
        "--population-size", "4",
        "--batch-size", "8",
        "--seed", "42",
        "--target-alphas", "0,0.00000078125",
        "--anchor-items-per-step", "128",
        "--anchor-max-input-tokens", "512",
        "--min-anchor-cosine", "0.8",
        "--experiment-name", "snapshot794_layer_v6_front_smoke_seed42",
    ]


def test_four_plan_artifacts_are_deterministic_exact_capacity_matched():
    directory = anchor_v6.ROOT / "experiments/layer_plans"
    checked = plan_builder.check_directory_v6(directory)
    assert set(checked) == {"front", "middle_early", "middle_late", "back"}
    for plan_sha256, spec in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.items():
        bundle = load_bundle(plan_sha256)
        assert checked[spec["plan"]]["file_sha256"] == spec["file_sha256"]
        assert bundle["manifest"] == plan_builder.build_plan_v6(spec["plan"])
        assert bundle["edge_split_v6"] == {
            "schema": "eggroll-es-edge-split-plan-v6",
            "plan": spec["plan"],
            "layers": spec["layers"],
            "paired_control": spec["paired_control"],
            "source_unit_count": 35,
            "runtime_selected_parameter_count": 23,
            "selected_element_count": 142_999_552,
        }
        frozen = worker_v6.FROZEN_LAYER_PLANS_V6[plan_sha256]
        assert frozen["runtime_selected_parameter_count"] == 23
        assert frozen["selected_element_count"] == 142_999_552
        assert frozen["selected_byte_count"] == 285_999_104


def test_worker_reuses_v4_update_bytecode_without_mutating_v4_allowlist():
    before = copy.deepcopy(worker_v4.FROZEN_LAYER_PLANS_V4)
    assert worker_v6._install_layer_plan_v6.__code__ is (
        worker_v4.LayerRestrictedExactAuditWorkerExtensionV4
        .install_layer_plan_v4.__code__
    )
    assert worker_v6.validate_frozen_layer_plan_v6.__code__ is (
        worker_v4.validate_frozen_layer_plan_v4.__code__
    )
    middle_sha = next(
        sha for sha, spec in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.items()
        if spec["plan"] == "middle_early"
    )
    spec = anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6[middle_sha]
    plan, frozen = worker_v6.validate_frozen_layer_plan_v6(
        Path(spec["path"]).read_bytes(), spec["file_sha256"], middle_sha,
    )
    assert plan["layers"] == [16, 17, 18, 19]
    assert frozen["runtime_selected_parameter_count"] == 23
    assert worker_v4.FROZEN_LAYER_PLANS_V4 == before
    with pytest.raises(ValueError, match="not frozen"):
        worker_v6.validate_frozen_layer_plan_v6(
            Path(spec["path"]).read_bytes(), spec["file_sha256"], "0" * 64,
        )


def test_pairs_are_exact_symmetric_and_only_allowlisted_files_load(tmp_path):
    by_name = {
        spec["plan"]: spec
        for spec in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.values()
    }
    assert anchor_v6.EDGE_SPLIT_PAIR_V6 == {
        "front": "middle_early", "middle_early": "front",
        "back": "middle_late", "middle_late": "back",
    }
    for name, peer in anchor_v6.EDGE_SPLIT_PAIR_V6.items():
        assert by_name[name]["paired_control"] == peer
        assert by_name[peer]["paired_control"] == name
    plan_sha256, spec = next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.items()))
    copied = tmp_path / "copied.json"
    copied.write_bytes(Path(spec["path"]).read_bytes())
    with pytest.raises(ValueError, match="not a frozen v6 arm"):
        anchor_v6.load_frozen_layer_plan_v6(
            copied,
            expected_file_sha256=spec["file_sha256"],
            expected_plan_sha256=plan_sha256,
            expected_model_config_sha256=anchor_v6.MODEL_CONFIG_SHA256_V6,
        )
    with pytest.raises(ValueError, match="outside"):
        anchor_v6.load_frozen_layer_plan_v6(
            spec["path"],
            expected_file_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_plan_sha256="0" * 64,
            expected_model_config_sha256=anchor_v6.MODEL_CONFIG_SHA256_V6,
        )


def test_smoke_cli_is_exact_and_pilot_fails_without_four_smoke_gate():
    bundle = load_bundle(next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6)))
    execution, remaining = line_search_v6.validate_frozen_execution_cli_v6(
        smoke_args(), bundle,
    )
    assert execution["stage"] == "smoke"
    assert execution["population_size"] == 4
    assert execution["batch_size"] == 8
    assert execution["target_alphas"] == line_search_v6.SMOKE_TARGETS_V6
    assert "--v6-stage" not in remaining
    assert remaining[:8] == [
        "--population-size", "4", "--batch-size", "8",
        "--seed", "42", "--target-alphas", "0,0.00000078125",
    ]
    for mutation in (
        ["--population-size", "8"],
        ["--batch-size", "64"],
        ["--seed", "43"],
        ["--target-alphas", "0,0.0000015625"],
    ):
        args = smoke_args()
        flag = mutation[0]
        args[args.index(flag) + 1] = mutation[1]
        with pytest.raises(ValueError, match="frozen|smoke recipe"):
            line_search_v6.validate_frozen_execution_cli_v6(args, bundle)
    with pytest.raises(ValueError, match="smoke evidence"):
        line_search_v6.validate_frozen_execution_cli_v6([
            "--v6-stage", "pilot", "--population-size", "16",
            "--batch-size", "64", "--seed", "42",
            "--target-alphas", "0,0.00000078125,0.0000015625,"
            "0.000003125,0.00000625",
            "--anchor-items-per-step", "128", "--min-anchor-cosine", "0.8",
            "--experiment-name", "snapshot794_layer_v6_front_pilot_seed42",
        ], bundle)


@pytest.mark.parametrize("flag,value", [
    ("--train-dataset", "/tmp/copied-train"),
    ("--eval-dataset", "/tmp/copied-eval"),
    ("--reward-function-timeout", "999"),
    ("--output-directory", "/tmp/runs"),
    ("--logging", "wandb"),
    ("--wandb-project", "other"),
    ("--experiment-name", "mislabeled-smoke"),
])
def test_v6_runtime_namespace_is_exact(flag, value):
    bundle = load_bundle(next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6)))
    args = smoke_args()
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])
    with pytest.raises(ValueError, match="runtime path|run name"):
        line_search_v6.validate_frozen_execution_cli_v6(args, bundle)


def test_loaded_trainer_uses_v6_worker_and_v5_document_lcb_mro():
    bundle = load_bundle(next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6)))
    trainer_class = anchor_v6.load_trainer(bundle)
    names = [item.__name__ for item in trainer_class.__mro__]
    assert names[:3] == [
        "EdgeSplitDocumentLCBTrainerV6", "EdgeSplitContractMixinV6",
        "DocumentLCBAnchoredMixinV5",
    ]
    assert trainer_class.launch_engines.__globals__["WORKER_EXTENSION"] == (
        anchor_v6.WORKER_EXTENSION
    )
    assert anchor_v6.anchor_v4.WORKER_EXTENSION != anchor_v6.WORKER_EXTENSION


def minimal_v6_journal(plan_sha256):
    spec = anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6[plan_sha256]
    implementation = {
        key: anchor_v6.file_sha256(path)
        for key, path in line_search_v6.V6_IMPLEMENTATION_PATHS.items()
    }
    snapshot = {
        "schema": "eggroll-es-anchor-line-search-snapshot-v6",
        "implementation": implementation,
        "train": {}, "evaluations": {}, "anchor": {},
        "fixed_train_batch": {}, "distributed_update_v3": {},
        "frozen_layer_plan_v4": {}, "dense_gold_reward_v4": {},
        "distributed_update_v4": {}, "document_lcb_anchor_v5": {},
        "recipe": {
            "population_size": 4, "batch_size": 8, "seed": 42,
            "target_alphas": list(line_search_v6.SMOKE_TARGETS_V6),
        },
        "edge_split_v6": {
            "schema": "eggroll-es-edge-split-snapshot-v6",
            "family": "four_arm_capacity_matched_edge_split",
            "arm": spec["plan"],
            "paired_control": spec["paired_control"],
            "layers": spec["layers"],
            "source_unit_count": 35,
            "runtime_selected_parameter_count": 23,
            "selected_element_count": 142_999_552,
            "plan_file_sha256": spec["file_sha256"],
            "plan_sha256": plan_sha256,
            "stage": "smoke",
            "recipe": line_search_v6.frozen_recipe_v6(
                spec["plan"], "smoke",
                f"snapshot794_layer_v6_{spec['plan']}_smoke_seed42",
                None,
            ),
            "document_lcb_config_sha256": (
                anchor_v6.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
            ),
            "optimization_data": "train_and_anchor_only",
            "legacy_audit_scope_config_sha256": (
                line_search_v6.V6_LEGACY_AUDIT_SCOPE_CONFIG_SHA256
            ),
            "implementation_bundle_sha256": (
                line_search_v6.driver_v1.canonical_sha256(implementation)
            ),
        },
    }
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v6",
        "status": "complete", "in_progress": None,
        "snapshot": snapshot,
        "policy": {
            "alpha_order": "zero_then_strictly_increasing",
            "branching": False, "resume": False, "rollback": False,
            "selection_during_execution": False,
            "ood_qa_max_degradation": 0.0,
            "ood_prose_max_degradation": 0.0,
            "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
            "direct_alpha_confirmation_required": True,
            "frozen_layer_plan_required": True,
            "dense_gold_reward_required": True,
            "document_lcb_anchor_required": True,
            "optimization_data": "train_and_anchor_only",
            "edge_split_family_v6": "four_arm_capacity_matched",
            "pilot_requires_four_smokes_v6": True,
            "ood_validation_heldout_as_objective": False,
        },
        "targets": list(line_search_v6.SMOKE_TARGETS_V6),
        "states": [{}, {}],
        "coefficient_plan": {
            "coefficient_sha256": "a" * 64,
            "frozen_layer_plan_v4": {"plan_sha256": plan_sha256},
        },
        "trainer_configuration": {},
        "seeds": [],
    }
    journal["content_sha256_before_self_field"] = (
        line_search_v6.driver_v1.canonical_sha256(journal)
    )
    return journal


def test_v6_offline_wrapper_binds_arm_recipe_implementation_and_v5_audit(
    monkeypatch,
):
    plan_sha256 = next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6))
    journal = minimal_v6_journal(plan_sha256)
    seen = {}

    def inherited(value):
        seen["journal"] = value
        assert value["schema"] == "eggroll-es-anchor-alpha-line-search-v5"
        assert value["snapshot"]["schema"] == (
            "eggroll-es-anchor-line-search-snapshot-v5"
        )
        return {
            "seed": 42, "state_count": 2,
            "coefficient_sha256": "a" * 64,
            "robust_plan_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        line_search_v6.driver_v5, "validate_completed_journal_v5", inherited,
    )
    audit = line_search_v6.validate_completed_journal_v6(journal)
    assert audit["arm"] == "front"
    assert audit["paired_control"] == "middle_early"
    assert audit["stage"] == "smoke"
    assert audit["state_count"] == 2
    tampered = copy.deepcopy(journal)
    tampered["snapshot"]["edge_split_v6"]["selected_element_count"] += 1
    tampered["content_sha256_before_self_field"] = (
        line_search_v6.driver_v1.canonical_sha256({
            key: value for key, value in tampered.items()
            if key != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="snapshot contract"):
        line_search_v6.validate_completed_journal_v6(tampered)


def test_v5_compatibility_strips_only_the_nested_v6_snapshot_extension():
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v6",
        "snapshot": {
            "schema": "eggroll-es-anchor-line-search-snapshot-v6",
            "edge_split_v6": {
                "ood_validation_heldout_as_objective": False,
            },
        },
        "policy": {
            "edge_split_family_v6": "four_arm_capacity_matched",
            "pilot_requires_four_smokes_v6": True,
        },
        "edge_split_v6": {"unexpected_top_level_field": True},
        "content_sha256_before_self_field": "0" * 64,
    }
    compatible = line_search_v6._v5_compatibility_journal_v6(journal)
    assert "edge_split_v6" not in compatible["snapshot"]
    assert compatible["edge_split_v6"] == {
        "unexpected_top_level_field": True,
    }
    assert compatible["schema"] == "eggroll-es-anchor-alpha-line-search-v5"
    assert compatible["snapshot"]["schema"] == (
        "eggroll-es-anchor-line-search-snapshot-v5"
    )


def test_v6_snapshot_extension_survives_initial_v5_offline_sanitization():
    plan_sha256 = next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6))
    journal = minimal_v6_journal(plan_sha256)
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v5"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v5"
    )
    compatible = line_search_v6.driver_v5._v4_compatibility_journal_v5(
        journal,
    )
    line_search_v6.offline_audit._assert_no_heldout(compatible)
    assert compatible["snapshot"]["edge_split_v6"][
        "optimization_data"
    ] == "train_and_anchor_only"


def v6_legacy_binding(plan_sha256):
    frozen = worker_v6.FROZEN_LAYER_PLANS_V6[plan_sha256]
    return {
        "layer_plan_file_sha256": frozen["file_sha256"],
        "layer_plan_sha256": plan_sha256,
        "checkpoint_to_runtime_mapping_sha256": frozen[
            "checkpoint_to_runtime_mapping_sha256"
        ],
        "source_unit_count": 35,
        "runtime_selected_name_sha256": frozen[
            "runtime_selected_name_sha256"
        ],
        "selected_parameter_manifest_sha256": "1" * 64,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "unselected_origin_sha256": "2" * 64,
        "dense_reward_sha256": (
            line_search_v6.offline_audit.V4_DENSE_REWARD_SHA256
        ),
    }


def test_scoped_legacy_audit_accepts_exactly_all_four_v6_capacities():
    with line_search_v6.scoped_legacy_audit_v6():
        for plan_sha256, spec in (
            anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.items()
        ):
            layer = {
                "path": str(spec["path"]),
                "file_sha256": spec["file_sha256"],
                "plan_sha256": plan_sha256,
                "model_config_path": "/safe/Qwen3.6-35B-A3B/config.json",
                "model_config_sha256": anchor_v6.MODEL_CONFIG_SHA256_V6,
            }
            assert line_search_v6.offline_audit._validate_v4_layer_identity(
                layer, spec["plan"], require_runtime=False,
            )["plan_sha256"] == plan_sha256
            assert line_search_v6.offline_audit._validate_v4_bindings(
                v6_legacy_binding(plan_sha256), spec["plan"],
            )["runtime_selected_parameter_count"] == 23


def test_scoped_legacy_audit_rejects_old_v4_plan_and_restores_on_exit():
    original = line_search_v6._legacy_audit_globals_v6()
    old_plan_sha256 = next(iter(original["V4_FROZEN_LAYER_PLANS"]))
    with line_search_v6.scoped_legacy_audit_v6():
        with pytest.raises(
            line_search_v6.offline_audit.JournalValidationError,
            match="not frozen",
        ):
            line_search_v6.offline_audit._validate_v4_layer_identity({
                "path": "/safe/old-v4.json",
                "file_sha256": original["V4_FROZEN_LAYER_PLANS"][
                    old_plan_sha256
                ]["file_sha256"],
                "plan_sha256": old_plan_sha256,
                "model_config_path": "/safe/config.json",
                "model_config_sha256": anchor_v6.MODEL_CONFIG_SHA256_V6,
            }, "old v4", require_runtime=False)
    restored = line_search_v6._legacy_audit_globals_v6()
    assert all(restored[key] is original[key] for key in original)


def test_scoped_legacy_audit_restores_after_failure_and_rejects_reentry():
    original = line_search_v6._legacy_audit_globals_v6()
    with pytest.raises(ValueError, match="synthetic scope failure"):
        with line_search_v6.scoped_legacy_audit_v6():
            with pytest.raises(RuntimeError, match="concurrent or reentrant"):
                with line_search_v6.scoped_legacy_audit_v6():
                    pass
            raise ValueError("synthetic scope failure")
    restored = line_search_v6._legacy_audit_globals_v6()
    assert all(restored[key] is original[key] for key in original)


def test_scoped_legacy_audit_rejects_original_global_drift(monkeypatch):
    monkeypatch.setattr(
        line_search_v6.offline_audit, "V4_SOURCE_UNIT_COUNT", 71,
    )
    with pytest.raises(RuntimeError, match="globals drifted"):
        with line_search_v6.scoped_legacy_audit_v6():
            pass


def test_v6_validator_rejects_unexpected_top_level_and_heldout_data():
    plan_sha256 = next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6))
    for mutation in (
        lambda journal: journal.__setitem__("unexpected", True),
        lambda journal: journal["snapshot"]["edge_split_v6"].__setitem__(
            "heldout_source", "/forbidden/heldout.arrow",
        ),
    ):
        journal = minimal_v6_journal(plan_sha256)
        mutation(journal)
        journal["content_sha256_before_self_field"] = (
            line_search_v6.driver_v1.canonical_sha256({
                key: value for key, value in journal.items()
                if key != "content_sha256_before_self_field"
            })
        )
        with pytest.raises(RuntimeError, match="wrong schema|contract"):
            line_search_v6.validate_completed_journal_v6(journal)


def test_v6_binds_every_implementation_file_and_rejects_bundle_forgery(
    monkeypatch,
):
    plan_sha256 = next(iter(anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6))
    monkeypatch.setattr(
        line_search_v6.driver_v5, "validate_completed_journal_v5",
        lambda journal: {
            "seed": 42, "state_count": 2,
            "coefficient_sha256": "a" * 64,
            "robust_plan_sha256": "b" * 64,
        },
    )
    baseline = minimal_v6_journal(plan_sha256)
    assert set(baseline["snapshot"]["implementation"]) == set(
        line_search_v6.V6_IMPLEMENTATION_PATHS
    )
    assert line_search_v6.validate_completed_journal_v6(baseline)[
        "arm"
    ] == "front"
    for key in line_search_v6.V6_IMPLEMENTATION_PATHS:
        forged = copy.deepcopy(baseline)
        forged["snapshot"]["implementation"][key] = "0" * 64
        forged["snapshot"]["edge_split_v6"][
            "implementation_bundle_sha256"
        ] = line_search_v6.driver_v1.canonical_sha256(
            forged["snapshot"]["implementation"]
        )
        forged["content_sha256_before_self_field"] = (
            line_search_v6.driver_v1.canonical_sha256({
                name: value for name, value in forged.items()
                if name != "content_sha256_before_self_field"
            })
        )
        with pytest.raises(RuntimeError, match="implementation identity"):
            line_search_v6.validate_completed_journal_v6(forged)
    for mutation in ("missing", "extra"):
        forged = copy.deepcopy(baseline)
        if mutation == "missing":
            forged["snapshot"]["implementation"].pop(
                "distributed_trainer_v5",
            )
        else:
            forged["snapshot"]["implementation"]["unbound_source"] = "0" * 64
        forged["snapshot"]["edge_split_v6"][
            "implementation_bundle_sha256"
        ] = line_search_v6.driver_v1.canonical_sha256(
            forged["snapshot"]["implementation"]
        )
        forged["content_sha256_before_self_field"] = (
            line_search_v6.driver_v1.canonical_sha256({
                name: value for name, value in forged.items()
                if name != "content_sha256_before_self_field"
            })
        )
        with pytest.raises(RuntimeError, match="implementation identity"):
            line_search_v6.validate_completed_journal_v6(forged)


def test_smoke_gate_json_and_entries_are_exact_keyed(tmp_path):
    top_extra = tmp_path / "top-extra.json"
    top_extra.write_text(json.dumps({
        "schema": "eggroll-es-edge-split-smoke-gate-v6",
        "arms": [{}, {}, {}, {}],
        "unexpected": True,
    }))
    with pytest.raises(ValueError, match="schema changed"):
        line_search_v6._smoke_evidence_v6(top_extra)
    entry_extra = tmp_path / "entry-extra.json"
    entry_extra.write_text(json.dumps({
        "schema": "eggroll-es-edge-split-smoke-gate-v6",
        "arms": [{
            "arm": "front", "journal": "/safe/front.json",
            "content_sha256": "0" * 64, "unexpected": True,
        }, {}, {}, {}],
    }))
    with pytest.raises(ValueError, match="entry fields changed"):
        line_search_v6._smoke_evidence_v6(entry_extra)
