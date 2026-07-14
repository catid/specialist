import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import eggroll_es_worker_v4 as worker_v4
from eggroll_es_worker_v3 import canonical_sha256_v3, coefficient_sha256_v3
from eggroll_es_worker_v4 import (
    FrozenLayerPlanAuditWorkerExtensionV4,
    LayerRestrictedExactAuditWorkerExtensionV4,
    checkpoint_runtime_mapping_v4,
    update_manifest_v4,
    validate_frozen_layer_plan_v4,
)


SOURCE_GATE = "model.language_model.layers.0.mlp.gate.weight"
SOURCE_DOWN = (
    "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"
)
RUNTIME_GATE = "language_model.model.layers.0.mlp.gate.weight"
RUNTIME_DOWN = (
    "language_model.model.layers.0.mlp.shared_expert.down_proj.weight"
)
RUNTIME_SELECTED = {RUNTIME_GATE, RUNTIME_DOWN}


class WeightOnly(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(
            values, dtype=torch.bfloat16,
        ))


class TinySharedExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = WeightOnly([1.5, -2.0])


class TinyMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = WeightOnly([0.25, -0.5, 1.0])
        self.shared_expert = TinySharedExpert()


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMlp()


class TinyRuntimeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer()])


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyRuntimeCore()


class TinyPartitionedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unselected = torch.nn.Parameter(torch.tensor(
            [3.0, 4.0, 5.0, 6.0], dtype=torch.bfloat16,
        ))
        self.language_model = TinyLanguageModel()


class FakeInterEngineGroup:
    def __init__(self, rank=0, fail_after=None):
        self.rank = rank
        self.world_size = 4
        self.available = True
        self.disabled = False
        self.calls = []
        self.fail_after = fail_after

    def all_reduce(self, tensor, out_tensor=None, stream=None):
        self.calls.append((tuple(tensor.shape), tensor.dtype, stream))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("synthetic all-reduce failure")
        assert out_tensor is tensor
        return out_tensor


def frozen_plan_artifact(units=None):
    if units is None:
        units = [SOURCE_GATE, SOURCE_DOWN]
    units = sorted(units)
    plan = {
        "schema": "qwen36-es-layer-plan-v1",
        "model_config": "/frozen/test/config.json",
        "model_config_sha256": "1" * 64,
        "plan": "tiny_test",
        "layers": [0],
        "layer_types": {"0": "linear_attention"},
        "groups": ["dense"],
        "num_units": len(units),
        "units": units,
        "include_regex": "^(?:" + "|".join(
            re.escape(name) for name in units
        ) + ")$",
    }
    plan_sha = canonical_sha256_v3(plan)
    plan["plan_sha256"] = plan_sha
    raw = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    file_sha = hashlib.sha256(raw).hexdigest()
    return raw, file_sha, plan_sha, plan


def dense_reward_artifact():
    config = {
        "schema": "eggroll-es-dense-qa-reward-v1",
        "aggregation": "mean_examples_of_mean_answer_token_logprob",
        "include_end_token": False,
        "max_input_tokens": 1024,
    }
    return config, canonical_sha256_v3(config)


def make_uninstalled_worker(group=None):
    worker = object.__new__(LayerRestrictedExactAuditWorkerExtensionV4)
    worker.model_runner = SimpleNamespace(model=TinyPartitionedModel())
    worker.inter_pg = group or FakeInterEngineGroup()
    worker.world_size = 1
    return worker


def freeze_tiny_plan(monkeypatch, file_sha, plan_sha, plan, elements=5):
    mapping = checkpoint_runtime_mapping_v4(plan["units"])
    monkeypatch.setitem(worker_v4.FROZEN_LAYER_PLANS_V4, plan_sha, {
        "plan": "tiny_test",
        "file_sha256": file_sha,
        "source_unit_count": len(plan["units"]),
        "runtime_selected_parameter_count": (
            mapping["runtime_selected_parameter_count"]
        ),
        "selected_element_count": elements,
        "selected_byte_count": 2 * elements,
        "checkpoint_to_runtime_mapping_sha256": mapping["mapping_sha256"],
        "runtime_selected_name_sha256": (
            mapping["runtime_selected_name_sha256"]
        ),
    })
    return mapping


def install_worker(monkeypatch, *, units=None, group=None):
    raw, file_sha, plan_sha, plan = frozen_plan_artifact(units)
    mapping = freeze_tiny_plan(monkeypatch, file_sha, plan_sha, plan)
    reward, reward_sha = dense_reward_artifact()
    worker = make_uninstalled_worker(group)
    report = worker.install_layer_plan_v4(
        raw, file_sha, plan_sha, reward, reward_sha, chunk_bytes=2,
    )
    return worker, report, {
        "raw": raw,
        "file_sha": file_sha,
        "plan_sha": plan_sha,
        "plan": plan,
        "mapping": mapping,
        "reward": reward,
        "reward_sha": reward_sha,
    }


def prepare_worker(worker, target_alpha=0.1):
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    coefficients = [0.5, -0.25, 1.0, -1.5, 0.75, 0.1, -0.4, 0.8]
    bindings = worker._binding_fields_v4()
    return worker.prepare_sharded_seed_update_v4(
        seeds,
        coefficients,
        coefficient_sha256_v3(seeds, coefficients),
        8,
        4,
        worker._v3_reference_generation,
        "plan-a",
        worker._v3_update_sequence + 1,
        worker._v3_accepted_alpha,
        target_alpha,
        worker._v3_current_identity["sha256"],
        bindings["layer_plan_file_sha256"],
        bindings["layer_plan_sha256"],
        bindings["checkpoint_to_runtime_mapping_sha256"],
        bindings["source_unit_count"],
        bindings["runtime_selected_name_sha256"],
        bindings["selected_parameter_manifest_sha256"],
        bindings["runtime_selected_parameter_count"],
        bindings["selected_element_count"],
        bindings["unselected_origin_sha256"],
        bindings["dense_reward_sha256"],
    )


def named_parameters(worker):
    return dict(worker.model_runner.model.named_parameters())


def selected_clones(worker):
    parameters = named_parameters(worker)
    return {
        name: parameters[name].detach().clone()
        for name in sorted(RUNTIME_SELECTED)
    }


def assert_selected_equal(worker, expected):
    current = selected_clones(worker)
    assert set(current) == set(expected)
    for name in expected:
        assert torch.equal(current[name], expected[name])


def test_production_plan_mapping_is_exact_70_to_46_and_expected_size():
    root = Path(__file__).resolve().parent
    paths = [
        root / "experiments/layer_plans/front_back_dense.json",
        root / "experiments/layer_plans/middle_matched_dense.json",
    ]
    for path in paths:
        raw = path.read_bytes()
        plan_sha = json.loads(raw)["plan_sha256"]
        frozen = worker_v4.FROZEN_LAYER_PLANS_V4[plan_sha]
        plan, _ = validate_frozen_layer_plan_v4(
            raw, frozen["file_sha256"], plan_sha,
        )
        mapping = checkpoint_runtime_mapping_v4(plan["units"])
        assert mapping["source_unit_count"] == 70
        assert mapping["runtime_selected_parameter_count"] == 46
        assert mapping["mapping_sha256"] == (
            frozen["checkpoint_to_runtime_mapping_sha256"]
        )
        assert mapping["runtime_selected_name_sha256"] == (
            frozen["runtime_selected_name_sha256"]
        )

    linear_elements = (
        25_165_824 + 131_072 + 8_388_608 + 524_288
        + 1_048_576 + 2_097_152
    )
    full_elements = (
        18_874_368 + 8_388_608 + 524_288 + 1_048_576 + 2_097_152
    )
    assert linear_elements == 37_355_520
    assert full_elements == 30_932_992
    assert 6 * linear_elements + 2 * full_elements == 285_999_104
    assert 2 * 285_999_104 == 571_998_208


def test_partial_packed_checkpoint_selection_is_rejected():
    with pytest.raises(ValueError, match="partial packed runtime parameter"):
        checkpoint_runtime_mapping_v4([
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        ])


def test_install_validates_hashes_mapping_and_exact_runtime_selection(monkeypatch):
    worker, report, artifact = install_worker(monkeypatch)
    assert FrozenLayerPlanAuditWorkerExtensionV4 is (
        LayerRestrictedExactAuditWorkerExtensionV4
    )
    assert report["installed"] is True
    assert report["idempotent"] is False
    assert report["reference_present_before_install"] is False
    assert report["reference_generation_before_install"] == 0
    assert (report["rank"], report["world_size"]) == (0, 4)
    assert report["layer_plan_file_sha256"] == artifact["file_sha"]
    assert report["layer_plan_sha256"] == artifact["plan_sha"]
    assert report["checkpoint_to_runtime_mapping_sha256"] == (
        artifact["mapping"]["mapping_sha256"]
    )
    assert report["source_unit_count"] == 2
    assert report["runtime_selected_parameter_count"] == 2
    assert report["selected_element_count"] == 5
    assert report["selected_byte_count"] == 10
    assert report["initial_identity"]["selected"]["total_bytes"] == 10
    assert report["initial_identity"]["selected"]["parameter_count"] == 2
    assert report["initial_identity"]["unselected"]["parameter_count"] == 1

    repeated = worker.install_layer_plan_v4(
        artifact["raw"], artifact["file_sha"], artifact["plan_sha"],
        artifact["reward"], artifact["reward_sha"],
    )
    assert repeated["idempotent"] is True
    assert repeated["initial_identity"] == report["initial_identity"]
    with torch.no_grad():
        named_parameters(worker)[RUNTIME_GATE][0].add_(1.0)
    with pytest.raises(RuntimeError, match="changed after layer plan installation"):
        worker.install_layer_plan_v4(
            artifact["raw"], artifact["file_sha"], artifact["plan_sha"],
            artifact["reward"], artifact["reward_sha"],
        )


def test_install_rejects_reference_or_generation_and_unknown_runtime(monkeypatch):
    raw, file_sha, plan_sha, plan = frozen_plan_artifact()
    freeze_tiny_plan(monkeypatch, file_sha, plan_sha, plan)
    reward, reward_sha = dense_reward_artifact()

    worker = make_uninstalled_worker()
    worker.exact_reference_weights = {"already": torch.ones(1)}
    with pytest.raises(RuntimeError, match="before exact reference capture"):
        worker.install_layer_plan_v4(
            raw, file_sha, plan_sha, reward, reward_sha,
        )
    worker = make_uninstalled_worker()
    worker._v3_reference_generation = 1
    with pytest.raises(RuntimeError, match="before exact reference capture"):
        worker.install_layer_plan_v4(
            raw, file_sha, plan_sha, reward, reward_sha,
        )

    unknown = "model.language_model.layers.0.linear_attn.out_proj.weight"
    bad_raw, bad_file, bad_sha, bad_plan = frozen_plan_artifact(
        [SOURCE_GATE, unknown],
    )
    freeze_tiny_plan(monkeypatch, bad_file, bad_sha, bad_plan, elements=4)
    worker = make_uninstalled_worker()
    with pytest.raises(RuntimeError, match="unknown runtime parameters"):
        worker.install_layer_plan_v4(
            bad_raw, bad_file, bad_sha, reward, reward_sha,
        )
    assert getattr(worker, "_v4_layer_plan_installed", False) is False


def test_install_rejects_non_bfloat16_selected_runtime_parameters(monkeypatch):
    raw, file_sha, plan_sha, plan = frozen_plan_artifact()
    mapping = freeze_tiny_plan(monkeypatch, file_sha, plan_sha, plan)
    frozen = worker_v4.FROZEN_LAYER_PLANS_V4[plan_sha]
    frozen["selected_byte_count"] = 20
    reward, reward_sha = dense_reward_artifact()
    worker = make_uninstalled_worker()
    for name, parameter in named_parameters(worker).items():
        if name in RUNTIME_SELECTED:
            parameter.data = parameter.data.float()
    assert mapping["runtime_selected_parameter_count"] == 2
    with pytest.raises(RuntimeError, match="frozen BF16"):
        worker.install_layer_plan_v4(
            raw, file_sha, plan_sha, reward, reward_sha,
        )
    assert getattr(worker, "_v4_layer_plan_installed", False) is False


def test_raw_plan_tampering_fails_before_install(monkeypatch):
    raw, file_sha, plan_sha, plan = frozen_plan_artifact()
    freeze_tiny_plan(monkeypatch, file_sha, plan_sha, plan)
    with pytest.raises(ValueError, match="file identity changed"):
        validate_frozen_layer_plan_v4(raw + b" ", file_sha, plan_sha)


def test_selected_only_reference_perturb_restore_and_partition_audit(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    initial_selected = selected_clones(worker)
    initial_unselected = named_parameters(worker)["unselected"].detach().clone()
    reference = worker.save_self_exact_reference(chunk_bytes=2)
    assert set(worker.exact_reference_weights) == RUNTIME_SELECTED
    assert reference["identity"]["unselected"] == (
        worker._v4_unselected_origin_identity
    )

    seed = 123
    worker.perturb_self_weights(seed, 0.1)
    parameters = named_parameters(worker)
    for name, before in initial_selected.items():
        parameter = parameters[name]
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            parameter.shape, dtype=parameter.dtype,
            device=parameter.device, generator=generator,
        )
        assert torch.equal(parameter, before + 0.1 * noise)
    assert torch.equal(parameters["unselected"], initial_unselected)

    assert worker.restore_self_weights_exact() is True
    assert_selected_equal(worker, initial_selected)
    check = worker.verify_self_exact_reference(chunk_bytes=2)
    assert check["passed"] is True
    assert check["current"] == check["reference"]


def test_unselected_origin_is_immutable_and_never_restored(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    with torch.no_grad():
        named_parameters(worker)["unselected"][0].add_(1.0)
    drifted = named_parameters(worker)["unselected"].detach().clone()
    # Population inner-loop restore is selected-only; the dedicated phase
    # boundary catches complement drift once, without PCIe hashing per sample.
    assert worker.restore_self_weights_exact() is True
    assert torch.equal(named_parameters(worker)["unselected"], drifted)
    with pytest.raises(RuntimeError, match="drifted from immutable origin"):
        worker.audit_population_completion_v4(
            4,
            worker._v3_reference_generation,
            worker._v3_reference_identity["sha256"],
            chunk_bytes=2,
        )
    selected_check = worker.verify_self_exact_reference(chunk_bytes=2)
    assert selected_check["passed"] is True
    assert selected_check["unselected_audit"] == (
        "deferred_to_population_completion_v4"
    )
    with pytest.raises(RuntimeError, match="drifted from immutable origin"):
        worker.save_self_exact_reference(chunk_bytes=2)


def test_population_inner_loop_is_selected_only_and_one_boundary_audit(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    reference_sha = worker._v3_reference_identity["sha256"]
    original_partition_identity = worker._partition_identity_v4

    def reject_inner_complement_hash(partition, *args, **kwargs):
        if partition == "unselected":
            raise AssertionError("population inner loop hashed complement")
        return original_partition_identity(partition, *args, **kwargs)

    monkeypatch.setattr(
        worker, "_partition_identity_v4", reject_inner_complement_hash,
    )
    worker.perturb_self_weights(123, 0.1)
    assert worker.restore_self_weights_exact() is True

    monkeypatch.setattr(
        worker, "_partition_identity_v4", original_partition_identity,
    )
    audit = worker.audit_population_completion_v4(
        4, worker._v3_reference_generation, reference_sha, chunk_bytes=2,
    )
    assert audit["passed"] is True
    assert audit["rank"] == 0
    assert audit["world_size"] == 4
    assert audit["current_identity"] == worker._v3_reference_identity


def test_prepare_manifest_preflight_and_update_bind_all_identities(monkeypatch):
    worker, _, artifact = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    prepared = prepare_worker(worker)
    assert prepared["prepared"] is True
    assert prepared["shard_indices"] == [0, 4]
    assert prepared["layer_plan_sha256"] == artifact["plan_sha"]
    assert prepared["dense_reward_sha256"] == artifact["reward_sha"]
    assert prepared["source_unit_count"] == 2
    assert prepared["runtime_selected_parameter_count"] == 2
    assert prepared["selected_element_count"] == 5
    preflight = prepared["allocation_preflight"]
    assert preflight["parameter_count"] == 2
    assert preflight["element_count"] == 5
    assert preflight["collectives_created"] is False
    pending_manifest = worker._v3_pending_update["manifest"]
    assert pending_manifest["schema"] == (
        "eggroll-es-layer-restricted-update-manifest-v4"
    )
    assert canonical_sha256_v3(pending_manifest) == prepared["manifest_sha256"]


def test_sharded_fp32_update_collects_selected_only_and_abort_is_exact(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    initial_selected = selected_clones(worker)
    initial_unselected = named_parameters(worker)["unselected"].detach().clone()
    worker.save_self_exact_reference(chunk_bytes=2)
    prepared = prepare_worker(worker)
    executed = worker.execute_prepared_seed_update_v4(
        prepared["manifest_sha256"],
    )
    assert executed["parameter_count"] == 2
    assert executed["reduced_element_count"] == 5
    assert len(worker.inter_pg.calls) == 2
    assert all(call[1] == torch.float32 for call in worker.inter_pg.calls)
    with pytest.raises(AssertionError):
        assert_selected_equal(worker, initial_selected)
    assert torch.equal(named_parameters(worker)["unselected"], initial_unselected)

    committed = worker.commit_prepared_seed_update_v4(
        prepared["manifest_sha256"], executed["final_identity"]["sha256"],
    )
    assert committed["reference_fresh_for_population"] is False
    with pytest.raises(RuntimeError, match="stale for population"):
        worker.restore_self_weights_exact()
    aborted = worker.abort_distributed_update_v4(
        "plan-a", worker._v3_reference_generation,
    )
    assert aborted["aborted"] is True
    assert_selected_equal(worker, initial_selected)
    assert torch.equal(named_parameters(worker)["unselected"], initial_unselected)


def test_collective_failure_rolls_selected_back_without_touching_complement(monkeypatch):
    worker, _, _ = install_worker(
        monkeypatch, group=FakeInterEngineGroup(fail_after=1),
    )
    initial_selected = selected_clones(worker)
    initial_unselected = named_parameters(worker)["unselected"].detach().clone()
    worker.save_self_exact_reference(chunk_bytes=2)
    prepared = prepare_worker(worker)
    with pytest.raises(RuntimeError, match="synthetic all-reduce failure"):
        worker.execute_prepared_seed_update_v4(prepared["manifest_sha256"])
    assert_selected_equal(worker, initial_selected)
    assert torch.equal(named_parameters(worker)["unselected"], initial_unselected)
    assert worker._v3_pending_update is None
    assert worker._v3_reference_fresh is True


def test_unselected_drift_during_update_makes_rollback_fail_closed(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    initial_selected = selected_clones(worker)
    worker.save_self_exact_reference(chunk_bytes=2)
    prepared = prepare_worker(worker)
    with torch.no_grad():
        named_parameters(worker)["unselected"][0].add_(1.0)
    with pytest.raises(RuntimeError, match="unselected-origin audit also failed"):
        worker.execute_prepared_seed_update_v4(prepared["manifest_sha256"])
    assert_selected_equal(worker, initial_selected)
    assert worker._v3_pending_update is None


def test_commit_rehashes_and_rejects_post_execute_drift(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    prepared = prepare_worker(worker)
    executed = worker.execute_prepared_seed_update_v4(
        prepared["manifest_sha256"],
    )
    with torch.no_grad():
        named_parameters(worker)[RUNTIME_GATE][0].add_(1.0)
    with pytest.raises(RuntimeError, match="changed between execute and commit"):
        worker.commit_prepared_seed_update_v4(
            prepared["manifest_sha256"],
            executed["final_identity"]["sha256"],
        )


def test_execute_rehashes_prepared_manifest_before_collective(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    prepared = prepare_worker(worker)
    worker._v3_pending_update["manifest"]["target_alpha"] = 0.2
    with pytest.raises(RuntimeError, match="manifest payload changed"):
        worker.execute_prepared_seed_update_v4(prepared["manifest_sha256"])
    assert worker.inter_pg.calls == []


def test_prepare_rejects_any_controller_binding_change_before_collective(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    bindings = worker._binding_fields_v4()
    seeds = list(range(8))
    coefficients = [0.1] * 8
    arguments = [
        seeds,
        coefficients,
        coefficient_sha256_v3(seeds, coefficients),
        8,
        4,
        worker._v3_reference_generation,
        "plan-a",
        1,
        0.0,
        0.1,
        worker._v3_current_identity["sha256"],
        bindings["layer_plan_file_sha256"],
        bindings["layer_plan_sha256"],
        bindings["checkpoint_to_runtime_mapping_sha256"],
        bindings["source_unit_count"],
        bindings["runtime_selected_name_sha256"],
        bindings["selected_parameter_manifest_sha256"],
        bindings["runtime_selected_parameter_count"],
        bindings["selected_element_count"],
        bindings["unselected_origin_sha256"],
        "0" * 64,
    ]
    with pytest.raises(RuntimeError, match="bindings changed"):
        worker.prepare_sharded_seed_update_v4(*arguments)
    assert worker.inter_pg.calls == []
    assert worker._v3_pending_update is None


def test_worker_state_rehashes_partitions_and_repeats_bindings(monkeypatch):
    worker, _, _ = install_worker(monkeypatch)
    worker.save_self_exact_reference(chunk_bytes=2)
    state = worker.inspect_distributed_update_state_v4(4)
    assert state["pending"] is False
    for key, value in worker._binding_fields_v4().items():
        assert state[key] == value
    with torch.no_grad():
        named_parameters(worker)[RUNTIME_GATE][0].add_(1.0)
    with pytest.raises(RuntimeError, match="current selected identity changed"):
        worker.inspect_distributed_update_state_v4(4)


@pytest.mark.parametrize("method", [
    "save_self_initial_weights",
    "update_weights_from_seeds",
    "save_self_weights_to_disk",
    "load_weights_from_disk",
    "inspect_distributed_update_state_v3",
    "_allocation_readiness_preflight_v3",
    "prepare_sharded_seed_update_v3",
    "execute_prepared_seed_update_v3",
    "commit_prepared_seed_update_v3",
    "abort_distributed_update_v3",
])
def test_inherited_full_model_paths_are_fail_closed(monkeypatch, method):
    worker, _, _ = install_worker(monkeypatch)
    with pytest.raises(RuntimeError, match="inherited full-model path"):
        getattr(worker, method)()


def test_inherited_full_model_restore_and_post_install_broadcast_are_forbidden(
    monkeypatch,
):
    worker, _, _ = install_worker(monkeypatch)
    with pytest.raises(RuntimeError, match="subtractive perturbation restore"):
        worker.restore_self_weights(123, 0.1)
    with pytest.raises(RuntimeError, match="inherited full-model path"):
        worker.broadcast_all_weights(0)


def test_update_manifest_v4_canonical_payload_includes_every_binding():
    manifest = update_manifest_v4(
        coefficient_sha256="1" * 64,
        population_size=8,
        world_size=4,
        reference_generation=1,
        plan_id="plan-a",
        update_sequence=1,
        previous_alpha=0.0,
        target_alpha=0.1,
        expected_base_sha256="2" * 64,
        layer_plan_file_sha256="3" * 64,
        layer_plan_sha256="4" * 64,
        checkpoint_to_runtime_mapping_sha256="5" * 64,
        source_unit_count=70,
        runtime_selected_name_sha256="6" * 64,
        selected_parameter_manifest_sha256="7" * 64,
        runtime_selected_parameter_count=46,
        selected_element_count=285_999_104,
        unselected_origin_sha256="8" * 64,
        dense_reward_sha256="9" * 64,
    )
    assert manifest["source_unit_count"] == 70
    assert manifest["runtime_selected_parameter_count"] == 46
    assert manifest["selected_element_count"] == 285_999_104
    assert canonical_sha256_v3(manifest)
