import hashlib
import json
from types import SimpleNamespace

import pytest

import run_eggroll_es_anchor_line_search_v4 as line_search_v4
import train_eggroll_es_specialist_anchor_v4 as anchor_v4
from eggroll_es_worker_v4 import update_manifest_v4 as worker_update_manifest_v4


class CharacterTokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class BoundaryChangingTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "prompt":
            return [1, 2]
        if text == "promptanswer":
            return [1, 9, 3]
        raise AssertionError(text)


class LongTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(100 if text == "p" else 1025))


def output_for(item, values):
    logprobs = [None] * len(item["prompt_token_ids"])
    start = item["answer_token_start"]
    assert len(values) == item["answer_token_count"]
    for offset, value in enumerate(values):
        position = start + offset
        token = item["prompt_token_ids"][position]
        logprobs[position] = {token: {"logprob": value}}
    return SimpleNamespace(
        prompt_token_ids=list(item["prompt_token_ids"]),
        prompt_logprobs=logprobs,
    )


def make_frozen_plan(tmp_path, unit_count=70):
    model_config = tmp_path / "config.json"
    model_config.write_text('{"text_config":{"num_hidden_layers":40}}\n')
    config_sha = hashlib.sha256(model_config.read_bytes()).hexdigest()
    units = [f"runtime.source.unit.{index}" for index in range(unit_count)]
    payload = {
        "schema": "qwen36-es-layer-plan-v1",
        "model_config": str(model_config),
        "model_config_sha256": config_sha,
        "plan": "front_back",
        "layers": [0, 1, 2, 3, 36, 37, 38, 39],
        "layer_types": {},
        "groups": ["dense"],
        "num_units": len(units),
        "units": units,
        "include_regex": "^(?:" + "|".join(units) + ")$",
    }
    plan_sha = anchor_v4.canonical_sha256(payload)
    manifest = {**payload, "plan_sha256": plan_sha}
    path = tmp_path / "layer-plan.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle = anchor_v4.load_frozen_layer_plan_v4(
        path,
        expected_file_sha256=file_sha,
        expected_plan_sha256=plan_sha,
        expected_model_config_sha256=config_sha,
    )
    anchor_v4.FROZEN_RUNTIME_EXPECTATIONS_V4[plan_sha] = {
        "source_unit_count": unit_count,
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "selected_byte_count": 571_998_208,
    }
    return bundle


def runtime_bindings(bundle):
    return {
        "layer_plan_file_sha256": bundle["file_sha256"],
        "layer_plan_sha256": bundle["plan_sha256"],
        "checkpoint_to_runtime_mapping_sha256": "1" * 64,
        "source_unit_count": bundle["manifest"]["num_units"],
        "runtime_selected_name_sha256": "4" * 64,
        "selected_parameter_manifest_sha256": "2" * 64,
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "unselected_origin_sha256": "3" * 64,
        "dense_reward_sha256": (
            anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        ),
    }


def install_reports(bundle):
    bindings = runtime_bindings(bundle)
    return [{
        "schema": "eggroll-es-layer-plan-installed-v4",
        "installed": True,
        "rank": rank,
        "world_size": 4,
        "reference_present_before_install": False,
        "reference_generation_before_install": 0,
        "selected_byte_count": 571_998_208,
        "initial_identity": {
            "selected": {
                "parameter_count": 46,
                "total_elements": 285_999_104,
                "total_bytes": 571_998_208,
            },
        },
        **bindings,
    } for rank in range(4)]


def test_gold_reward_requires_exact_prompt_prefix_boundary():
    with pytest.raises(ValueError, match="boundary mismatch"):
        anchor_v4.prepare_gold_answer_items_v4(
            BoundaryChangingTokenizer(), ["prompt"], ["answer"],
        )


def test_gold_reward_fails_instead_of_truncating_above_1024_tokens():
    with pytest.raises(ValueError, match="1024-token cap"):
        anchor_v4.prepare_gold_answer_items_v4(
            LongTokenizer(), ["p"], ["a"],
        )


def test_gold_reward_rejects_engine_side_truncation():
    items = anchor_v4.prepare_gold_answer_items_v4(
        CharacterTokenizer(), ["prompt:"], ["gold"],
    )
    output = output_for(items[0], [-1.0] * 4)
    output.prompt_token_ids = output.prompt_token_ids[:-1]
    with pytest.raises(ValueError, match="truncated or changed"):
        anchor_v4.score_gold_answer_outputs_v4(items, [output])


def test_gold_reward_means_tokens_per_example_then_examples_without_eos():
    items = anchor_v4.prepare_gold_answer_items_v4(
        CharacterTokenizer(), ["p:", "q:"], ["a", "bcd"],
    )
    result = anchor_v4.score_gold_answer_outputs_v4(items, [
        output_for(items[0], [-1.0]),
        output_for(items[1], [-2.0, -3.0, -4.0]),
    ])
    assert result["mean_example_mean_logprob"] == pytest.approx(-2.0)
    assert result["answer_token_count"] == 4
    assert all(item["eos_scored"] is False for item in result["examples"])
    assert all(item["eos_appended"] is False for item in items)
    assert result["reward_config_sha256"] == anchor_v4.canonical_sha256(
        anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
    )


def test_frozen_plan_rejects_file_plan_and_model_config_hash_drift(tmp_path):
    bundle = make_frozen_plan(tmp_path)
    with pytest.raises(ValueError, match="file SHA256 changed"):
        anchor_v4.load_frozen_layer_plan_v4(
            bundle["path"],
            expected_file_sha256="0" * 64,
            expected_plan_sha256=bundle["plan_sha256"],
            expected_model_config_sha256=bundle["model_config_sha256"],
        )
    with pytest.raises(ValueError, match="canonical SHA256 changed"):
        anchor_v4.load_frozen_layer_plan_v4(
            bundle["path"],
            expected_file_sha256=bundle["file_sha256"],
            expected_plan_sha256="0" * 64,
            expected_model_config_sha256=bundle["model_config_sha256"],
        )
    with pytest.raises(ValueError, match="model-config SHA256 changed"):
        anchor_v4.load_frozen_layer_plan_v4(
            bundle["path"],
            expected_file_sha256=bundle["file_sha256"],
            expected_plan_sha256=bundle["plan_sha256"],
            expected_model_config_sha256="0" * 64,
        )


def test_layer_plan_cli_requires_paired_path_and_all_expected_hashes(tmp_path):
    bundle = make_frozen_plan(tmp_path)
    argv = [
        "--layer-plan-json", bundle["path"],
        "--expected-layer-plan-file-sha256", bundle["file_sha256"],
        "--expected-layer-plan-sha256", bundle["plan_sha256"],
        "--expected-model-config-sha256", bundle["model_config_sha256"],
        "--experiment-name", "cpu-test",
    ]
    parsed, remaining = anchor_v4.parse_frozen_layer_plan_cli_v4(argv)
    assert parsed["plan_sha256"] == bundle["plan_sha256"]
    assert remaining == ["--experiment-name", "cpu-test"]
    with pytest.raises(ValueError, match="must be paired"):
        anchor_v4.parse_frozen_layer_plan_cli_v4(argv[:-4])


def test_controller_requires_unanimous_explicit_70_to_46_runtime_mapping(
    tmp_path,
):
    bundle = make_frozen_plan(tmp_path)
    reports = install_reports(bundle)
    result = anchor_v4.validate_layer_plan_installations_v4(reports, bundle)
    assert result["source_unit_count"] == 70
    assert result["runtime_selected_parameter_count"] == 46
    assert result["selected_element_count"] == 285_999_104
    reports[-1]["checkpoint_to_runtime_mapping_sha256"] = "5" * 64
    with pytest.raises(RuntimeError, match="different runtime layer mappings"):
        anchor_v4.validate_layer_plan_installations_v4(reports, bundle)


def test_controller_rejects_worker_plan_or_reward_identity_mismatch(tmp_path):
    bundle = make_frozen_plan(tmp_path)
    reports = install_reports(bundle)
    reports[2]["dense_reward_sha256"] = "5" * 64
    with pytest.raises(RuntimeError, match="different or late layer plan"):
        anchor_v4.validate_layer_plan_installations_v4(reports, bundle)


def test_controller_independently_enforces_frozen_runtime_counts_and_bytes(
    tmp_path,
):
    bundle = make_frozen_plan(tmp_path)
    reports = install_reports(bundle)
    for report in reports:
        report["runtime_selected_parameter_count"] = 45
    with pytest.raises(RuntimeError, match="mapping counts are invalid"):
        anchor_v4.validate_layer_plan_installations_v4(reports, bundle)

    reports = install_reports(bundle)
    for report in reports:
        report["selected_byte_count"] -= 2
    with pytest.raises(RuntimeError, match="mapping counts are invalid"):
        anchor_v4.validate_layer_plan_installations_v4(reports, bundle)


def test_layer_plan_is_installed_before_inherited_reference_capture(
    tmp_path, monkeypatch,
):
    bundle = make_frozen_plan(tmp_path)
    events = []

    def inherited_configure(self, *args, **kwargs):
        events.append("inherited_reference_capture")
        return "configured"

    monkeypatch.setattr(
        anchor_v4.anchor_v3.anchor_v2.ExactRestoredAnchoredStepMixin,
        "configure_anchor",
        inherited_configure,
    )

    class FakeCoordinator(anchor_v4.FrozenLayerDenseRewardMixinV4):
        def __init__(self):
            self.engines = [object()] * 4
            self.n_vllm_engines = 4
            self.n_gpu_per_vllm_engine = 1
            self.population_size = 8

        def _rpc_all_engines_v4(self, method, args):
            events.append(method)
            if method == "install_layer_plan_v4":
                assert isinstance(args[0], bytes)
                return install_reports(bundle)
            if method == "inspect_cached_distributed_update_state_v4":
                assert args == (4, "exact_reference")
                return [{"placeholder": True}] * 4
            raise AssertionError(method)

        def _validate_worker_states_v4(self, states, require_fresh):
            assert require_fresh is True
            return {
                "reference_generation": 1,
                "reference_identity": {"sha256": "reference"},
                "current_identity": {"sha256": "reference"},
            }

    trainer = FakeCoordinator()
    assert trainer.configure_anchor(
        {"rows": []}, frozen_layer_plan=bundle,
    ) == "configured"
    assert events == [
        "install_layer_plan_v4",
        "inherited_reference_capture",
        "inspect_cached_distributed_update_state_v4",
    ]


def test_v4_snapshot_inherits_v2_and_v3_implementation_identities(
    tmp_path, monkeypatch,
):
    bundle = make_frozen_plan(tmp_path)
    line_search_v4.set_active_layer_plan_bundle_v4(bundle)
    inherited = {
        "corrected_driver": "a" * 64,
        "exact_worker": "b" * 64,
        "distributed_driver_v3": "c" * 64,
        "distributed_trainer_v3": "d" * 64,
        "distributed_worker_v3": "e" * 64,
    }
    monkeypatch.setattr(
        line_search_v4.driver_v3,
        "build_snapshot",
        lambda *args, **kwargs: {
            **frozen_s6_snapshot_v4(),
            "implementation": dict(inherited),
        },
    )
    snapshot = line_search_v4.build_snapshot()
    assert snapshot["schema"] == "eggroll-es-anchor-line-search-snapshot-v4"
    for key, value in inherited.items():
        assert snapshot["implementation"][key] == value
    assert snapshot["implementation"]["distributed_driver_v4"]
    assert snapshot["implementation"]["distributed_trainer_v4"]
    assert snapshot["implementation"]["distributed_worker_v4"]
    assert snapshot["frozen_layer_plan_v4"]["plan_sha256"] == (
        bundle["plan_sha256"]
    )
    assert snapshot["dense_gold_reward_v4"]["reward_config_sha256"] == (
        anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
    )


def test_v4_wrapper_exposes_effective_adapter_api():
    assert line_search_v4.validate_effective_anchor_api() == (
        "coefficient_sha256",
        "load_anchor_prose",
        "load_trainer",
        "load_frozen_layer_plan_v4",
    )


def frozen_s6_snapshot_v4():
    split = lambda name: {
        "rows": line_search_v4.FROZEN_S6_V4[name]["rows"],
        "arrow_files": [{
            "path": f"/safe/{name}.arrow",
            "sha256": line_search_v4.FROZEN_S6_V4[name]["arrow_sha256"],
        }],
    }
    anchor = line_search_v4.FROZEN_S6_V4["anchor"]
    return {
        "train": split("train"),
        "evaluations": {
            "validation": split("validation"),
            "ood_qa": split("ood_qa"),
        },
        "anchor": {
            "rows": anchor["rows"],
            "sha256": anchor["sha256"],
            "report": {"sha256": anchor["report_sha256"]},
        },
    }


def test_v4_cli_freezes_eval_batch_model_sigma_engines_and_splits():
    contract = line_search_v4.validate_frozen_execution_cli_v4([])
    assert contract["mini_batch_size"] == 64
    assert contract["gpu_ids"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="evaluation batch size"):
        line_search_v4.validate_frozen_execution_cli_v4([
            "--mini-batch-size", "8",
        ])
    with pytest.raises(ValueError, match="four TP=1 engines"):
        line_search_v4.validate_frozen_execution_cli_v4([
            "--use-gpus", "0,1",
        ])


def test_v4_snapshot_and_alpha_zero_baseline_are_frozen_before_population():
    snapshot_value = frozen_s6_snapshot_v4()
    assert line_search_v4.validate_frozen_s6_snapshot_v4(snapshot_value)
    baseline = line_search_v4.FROZEN_S6_BASELINE_V4
    state = {
        "target_alpha": 0.0,
        "qa": {
            split: dict(baseline[split])
            for split in ("validation", "ood_qa")
        },
        "ood_prose": dict(baseline["ood_prose"]),
    }
    assert line_search_v4.validate_frozen_s6_baseline_v4(
        state, snapshot_value,
    )
    drifted = json.loads(json.dumps(state))
    drifted["qa"]["validation"]["mean_reward"] += 0.001
    with pytest.raises(RuntimeError, match="did not reproduce frozen S6"):
        line_search_v4.validate_frozen_s6_baseline_v4(
            drifted, snapshot_value,
        )


def test_v4_update_manifest_binds_mapping_and_dense_reward():
    bindings = {
        key: index for index, key in enumerate(
            anchor_v4.UPDATE_BINDING_KEYS_V4,
        )
    }
    manifest = anchor_v4.update_manifest_v4(
        coefficient_sha256="coef",
        population_size=8,
        world_size=4,
        reference_generation=1,
        plan_id="plan",
        update_sequence=1,
        previous_alpha=0.0,
        target_alpha=0.1,
        expected_base_sha256="base",
        bindings=bindings,
    )
    assert all(manifest[key] == value for key, value in bindings.items())
    assert manifest["dense_reward_sha256"] == bindings["dense_reward_sha256"]
    assert manifest["checkpoint_to_runtime_mapping_sha256"] == (
        bindings["checkpoint_to_runtime_mapping_sha256"]
    )


def test_controller_and_worker_reconstruct_identical_v4_manifest():
    bindings = runtime_bindings({
        "file_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "manifest": {"num_units": 70},
    })
    common = {
        "coefficient_sha256": "c" * 64,
        "population_size": 8,
        "world_size": 4,
        "reference_generation": 2,
        "plan_id": "plan-a",
        "update_sequence": 1,
        "previous_alpha": 0.0,
        "target_alpha": 0.1,
        "expected_base_sha256": "d" * 64,
    }
    controller = anchor_v4.update_manifest_v4(**common, bindings=bindings)
    worker = worker_update_manifest_v4(**common, **bindings)
    assert controller == worker
    assert anchor_v4.canonical_sha256(controller) == (
        anchor_v4.canonical_sha256(worker)
    )


def zero_application_v4_plan(bundle):
    seeds = [11, 22, 33, 44]
    coefficients = [0.5, -0.5, 1.0, -1.0]
    coefficient_sha = anchor_v4.coefficient_sha256(seeds, coefficients)
    bindings = runtime_bindings(bundle)
    reference = {"sha256": "e" * 64}
    boundary_audit = {
        "schema": "eggroll-es-population-boundary-audit-v4",
        "iteration": 0,
        "phase": "after_complete_population_exact_restore_before_plan",
        "engine_count": 4,
        "reference_generation": 1,
        "reference_identity": reference,
        "current_identity": reference,
        "unselected_origin_sha256": bindings["unselected_origin_sha256"],
        "runtime_mapping": bindings,
        "worker_reports": [{"rank": rank} for rank in range(4)],
        "passed": True,
    }
    boundary_audit["audit_sha256"] = anchor_v4.canonical_sha256(
        boundary_audit,
    )
    plan_id = anchor_v4.canonical_sha256({
        "schema": "eggroll-es-distributed-plan-id-v4",
        "iteration": 0,
        "coefficient_sha256": coefficient_sha,
        "reference_generation": 1,
        "reference_sha256": reference["sha256"],
        "layer_plan_sha256": bundle["plan_sha256"],
        "reward_config_sha256": (
            anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        ),
        "runtime_mapping_sha256": anchor_v4.canonical_sha256(bindings),
        "population_boundary_audit_sha256": boundary_audit["audit_sha256"],
    })
    return {
        "iteration": 0,
        "seeds": seeds,
        "coefficients": coefficients,
        "coefficient_sha256": coefficient_sha,
        "population_boundary_audit_v4": boundary_audit,
        "dense_gold_reward_v4": {
            "config": dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4),
            "reward_config_sha256": (
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
            ),
        },
        "frozen_layer_plan_v4": {
            "file_sha256": bundle["file_sha256"],
            "plan_sha256": bundle["plan_sha256"],
            "model_config_sha256": bundle["model_config_sha256"],
            "runtime_mapping": bindings,
            "runtime_mapping_sha256": anchor_v4.canonical_sha256(bindings),
        },
        "distributed_update_v4": {
            "schema": "eggroll-es-distributed-seed-plan-v4",
            "plan_id": plan_id,
            "reference_generation": 1,
            "reference_identity": reference,
            "layer_plan_sha256": bundle["plan_sha256"],
            "dense_reward_sha256": (
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
            ),
            "runtime_mapping": bindings,
            "runtime_mapping_sha256": anchor_v4.canonical_sha256(bindings),
            "population_boundary_audit_sha256": boundary_audit[
                "audit_sha256"
            ],
        },
        "applications": [],
    }


def test_v4_journal_validation_rejects_plan_reward_or_mapping_relabel(
    tmp_path,
):
    bundle = make_frozen_plan(tmp_path)
    line_search_v4.set_active_layer_plan_bundle_v4(bundle)
    plan = zero_application_v4_plan(bundle)
    journal = {"targets": [0.0]}
    applications, distributed, layer, reward, boundary_audit = (
        line_search_v4._validate_v4_plan_and_applications(journal, plan)
    )
    assert applications == []
    assert distributed["plan_id"]
    assert layer["runtime_mapping"]["source_unit_count"] == 70
    assert reward["reward_config_sha256"]
    assert boundary_audit["passed"] is True

    plan["dense_gold_reward_v4"]["reward_config_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="plan provenance differs"):
        line_search_v4._validate_v4_plan_and_applications(journal, plan)
    plan = zero_application_v4_plan(bundle)
    plan["frozen_layer_plan_v4"]["runtime_mapping"][
        "checkpoint_to_runtime_mapping_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="plan (?:provenance|identity) differs"):
        line_search_v4._validate_v4_plan_and_applications(journal, plan)


def test_v4_wrapper_persists_canonical_coefficient_values(tmp_path):
    bundle = make_frozen_plan(tmp_path)
    plan = zero_application_v4_plan(bundle)
    journal = {
        "seeds": list(plan["seeds"]),
        "coefficient_plan": {
            "coefficient_sha256": plan["coefficient_sha256"],
        },
    }
    line_search_v4.bind_coefficient_values_v4(journal, plan)
    assert journal["coefficient_plan"]["coefficients"] == plan["coefficients"]


class PopulationBoundaryCoordinator(anchor_v4.FrozenLayerDenseRewardMixinV4):
    def __init__(self, bundle, mismatched=False):
        self._v3_reference_generation = 2
        self._v3_reference_identity = {"sha256": "e" * 64}
        self._v4_layer_plan_install = runtime_bindings(bundle)
        self.mismatched = mismatched
        self.calls = []

    def _rpc_all_engines_v4(self, method, args):
        self.calls.append((method, args))
        reports = [{
            "schema": "eggroll-es-post-population-audit-v4",
            "passed": True,
            "rank": rank,
            "world_size": 4,
            "reference_generation": 2,
            "reference_sha256": "e" * 64,
            "current_identity": {"sha256": "e" * 64},
            **self._v4_layer_plan_install,
        } for rank in range(4)]
        if self.mismatched:
            reports[-1]["unselected_origin_sha256"] = "0" * 64
        return reports


def test_post_population_full_partition_audit_is_required_and_bound(tmp_path):
    bundle = make_frozen_plan(tmp_path)
    trainer = PopulationBoundaryCoordinator(bundle)
    audit = trainer._population_boundary_audit_v4(iteration=3)
    assert trainer.calls == [(
        "audit_population_completion_v4",
        (4, 2, "e" * 64),
    )]
    assert audit["passed"] is True
    assert audit["iteration"] == 3
    assert len(audit["worker_reports"]) == 4
    assert audit["audit_sha256"] == anchor_v4.canonical_sha256({
        key: value for key, value in audit.items() if key != "audit_sha256"
    })
    with pytest.raises(RuntimeError, match="worker audit differs"):
        PopulationBoundaryCoordinator(
            bundle, mismatched=True,
        )._population_boundary_audit_v4(iteration=3)


def test_unbound_coefficient_plan_is_not_persisted_before_boundary_audit(
    monkeypatch,
):
    trainer = object.__new__(anchor_v4.FrozenLayerDenseRewardMixinV4)
    trainer._v4_withhold_unbound_plan = True
    trainer._pending_identity_audit = {"passed": True}
    persisted = []
    monkeypatch.setattr(
        anchor_v4.anchor_v3.DistributedAnchoredStepMixinV3,
        "_persist_anchor_plan",
        lambda self, plan: persisted.append(plan.copy()),
    )
    plan = {"coefficient_sha256": "a" * 64}
    assert trainer._persist_anchor_plan(plan) is None
    assert persisted == []
    assert plan["identity_audit"]["passed"] is True
    plan["population_boundary_audit_v4"] = {"passed": True}
    trainer._persist_anchor_plan(plan)
    assert len(persisted) == 1
