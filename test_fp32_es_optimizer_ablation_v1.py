import copy
import hashlib
import math

import pytest
import torch

import build_fp32_es_optimizer_sigma_preregistration_v1 as builder
import fp32_es_optimizer_ablation_v1 as subject


def _key(module, side):
    return f"base_model.model.model.layers.0.{module}.lora_{side}.weight"


def _small_master(a=(1.0, 2.0), b=(3.0,)):
    return {
        _key("unit", "A"): torch.tensor(a, dtype=torch.float32),
        _key("unit", "B"): torch.tensor(b, dtype=torch.float32),
    }


def _resign(plan):
    plan["content_sha256_before_self_field"] = subject.canonical_sha256_v1({
        key: value for key, value in plan.items()
        if key != "content_sha256_before_self_field"
    })
    return plan


@pytest.fixture(scope="session")
def plan():
    return builder.build_preregistration_v1()


def _launch_ready(plan):
    value = copy.deepcopy(plan)
    value["status"] = "launch_ready_after_runtime_dependencies"
    value["authorization"]["gpu_launch"] = True
    value["dependencies"]["all_runtime_dependencies_complete"] = True
    value["dependencies"]["mirrored_es"][
        "accepted_all_four_gpu_activity_receipt_complete"
    ] = True
    value["dependencies"]["mirrored_es"]["status"] = "complete"
    value["dependencies"]["memory_roofline"][
        "optimizer_phase_transfer_and_bandwidth_receipt_complete"
    ] = True
    value["dependencies"]["memory_roofline"]["status"] = "complete"
    return _resign(value)


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _run_receipt(plan, arm_id="adamw__global__raw_pair_difference", seed=1701):
    arm = {item["arm_id"]: item for item in plan["grid"]}[arm_id]
    previous = plan["checkpoint_contract"][
        "initial_checkpoint_sha256_by_arm_and_seed"
    ][arm_id][str(seed)]
    pre_state = _sha(f"{arm_id}:{seed}:optimizer:0")
    updates = []
    for index in range(subject.UPDATES_PER_REPLICATE_V1):
        post_state = _sha(f"{arm_id}:{seed}:optimizer:{index + 1}")
        committed = _sha(f"{arm_id}:{seed}:checkpoint:{index + 1}")
        target = 1.0 + index
        observed = target * (1.0 + 5.0e-7)
        memory = plan["memory_bandwidth_contract"]
        optimizer_memory = memory["optimizers"][arm["optimizer"]]
        host_traffic = optimizer_memory[
            "algorithmic_minimum_host_memory_traffic_bytes_per_update_per_replica"
        ]
        host_elapsed = 0.125
        updates.append({
            "update_index": index,
            "base_sigma": subject.SIGMA_SCHEDULE_V1[index],
            "directions": subject.DIRECTIONS_PER_UPDATE_V1,
            "signed_candidates": subject.SIGNED_CANDIDATES_PER_UPDATE_V1,
            "rollouts": subject.ROLLOUTS_PER_UPDATE_V1,
            "scale_mode": arm["scale_mode"],
            "coefficient_mode": arm["coefficient_mode"],
            "optimizer": arm["optimizer"],
            "candidate_scale_application_count": 1,
            "estimator_inverse_scale_application_count": 1,
            "noise_representation": "unscaled_unit_standard_normal",
            "pair_difference_variance": 0.25,
            "finite_gradient": True,
            "target_update_l2": target,
            "observed_update_l2": observed,
            "update_norm_relative_error": abs(observed - target) / target,
            "budget_ratio": subject.UPDATE_BUDGET_RATIO_V1,
            "optimizer_state_dtype": "torch.float32",
            "optimizer_step_before": index,
            "optimizer_step_after": index + 1,
            "bias_correction_step": (
                index + 1 if arm["optimizer"] == "adamw" else None
            ),
            "pre_optimizer_state_sha256": pre_state,
            "rollback_optimizer_state_sha256": pre_state,
            "post_optimizer_state_sha256": post_state,
            "resume_checkpoint_sha256": previous,
            "previous_checkpoint_sha256": previous,
            "rollback_checkpoint_sha256": previous,
            "candidate_checkpoint_sha256": committed,
            "committed_checkpoint_sha256": committed,
            "replica_checkpoint_sha256": [committed] * 4,
            "direction_seed_set_sha256": subject.canonical_sha256_v1(
                plan["compute_contract"]["direction_seeds_by_replicate_and_update"]
                [str(seed)][index]
            ),
            "train_panel_content_sha256": plan["inputs"]["train_panel"]
            ["content_sha256"],
            "useful_physical_gpus": [0, 1, 2, 3],
            "persistent_optimizer_host_tensor_bytes_per_replica": (
                optimizer_memory[
                    "persistent_optimizer_host_tensor_bytes_per_replica"
                ]
            ),
            "persistent_optimizer_gpu_bytes_per_replica": 0,
            "measured_host_memory_traffic_bytes_per_replica": host_traffic,
            "reduced_gradient_d2h_bytes_per_replica": memory[
                "reduced_gradient_d2h_bytes_per_update_per_replica"
            ],
            "committed_runtime_h2d_bytes_per_replica": memory[
                "committed_runtime_h2d_bytes_per_update_per_replica"
            ],
            "population_candidate_restore_h2d_bytes_all_replicas": memory[
                "fixed_population_candidate_plus_restore_h2d_bytes_per_update_all_replicas"
            ],
            "checkpoint_logical_bytes_per_replica": optimizer_memory[
                "checkpoint_tensor_and_step_bytes_per_replica"
            ],
            "host_optimizer_elapsed_seconds": host_elapsed,
            "achieved_host_bandwidth_bytes_per_second": (
                host_traffic / host_elapsed
            ),
            "peak_phase_vram_bytes_by_gpu": {
                str(gpu): 80_000_000_000 for gpu in range(4)
            },
        })
        previous, pre_state = committed, post_state
    return {
        "schema": "fp32-es-optimizer-sigma-run-receipt-v1",
        "plan_content_sha256": plan["content_sha256_before_self_field"],
        "arm_id": arm_id,
        "replicate_seed": seed,
        "start_checkpoint_sha256": plan["checkpoint_contract"]
        ["initial_checkpoint_sha256_by_arm_and_seed"][arm_id][str(seed)],
        "updates": updates,
        "final_checkpoint_sha256": previous,
        "total_directions": (
            subject.DIRECTIONS_PER_UPDATE_V1 * subject.UPDATES_PER_REPLICATE_V1
        ),
        "total_signed_candidates": (
            subject.SIGNED_CANDIDATES_PER_UPDATE_V1
            * subject.UPDATES_PER_REPLICATE_V1
        ),
        "total_rollouts": subject.ROLLOUTS_PER_REPLICATE_V1,
        "parameter_surface_identity_sha256": plan["parameter_surface"]
        ["identity_sha256"],
        "update_budget_ratio": subject.UPDATE_BUDGET_RATIO_V1,
        "charged_gpu_seconds": 12_000.0,
        "useful_physical_gpus": [0, 1, 2, 3],
        "final_eval_requests": plan["compute_contract"]
        ["exact_final_eval_requests_per_replicate"],
        "train_only_during_updates": True,
        "dev_opened_after_training": True,
        "ood_opened_after_training": True,
        "protected_holdout_opened": False,
        "last_update_monotonic_ns": 100,
        "training_sealed_monotonic_ns": 101,
        "dev_opened_monotonic_ns": 102,
        "ood_opened_monotonic_ns": 103,
    }


def test_module_rms_uses_shape_normalized_rms_not_raw_l2():
    master = {
        _key("small", "A"): torch.ones(1, dtype=torch.float32),
        _key("small", "B"): torch.ones(1, dtype=torch.float32),
        _key("large", "A"): torch.ones(100, dtype=torch.float32),
        _key("large", "B"): torch.ones(100, dtype=torch.float32),
    }
    table = subject.module_sigma_table_v1(
        master, 0.001, "module_fp32_rms_shape_normalized"
    )
    sigmas = {item["module"]: item["sigma"] for item in table["records"]}
    assert len(sigmas) == 2
    assert len(set(sigmas.values())) == 1
    assert next(iter(sigmas.values())) == pytest.approx(0.001)
    assert table["expected_perturbation_l2_squared"] == pytest.approx(
        202 * 0.001 ** 2
    )


def test_module_rms_zero_master_uses_floor_without_nan_or_size_bias():
    master = {
        _key("small", "A"): torch.zeros(1, dtype=torch.float32),
        _key("small", "B"): torch.zeros(1, dtype=torch.float32),
        _key("large", "A"): torch.zeros(10, dtype=torch.float32),
        _key("large", "B"): torch.zeros(10, dtype=torch.float32),
    }
    table = subject.module_sigma_table_v1(
        master, 0.002, "module_fp32_rms_shape_normalized"
    )
    assert {item["sigma"] for item in table["records"]} == {0.002}


def test_antithetic_materialization_scales_unit_noise_exactly_once():
    master = _small_master()
    noise = {key: torch.full_like(value, 2.0) for key, value in master.items()}
    table = subject.module_sigma_table_v1(master, 0.01, "global")
    plus = subject.materialize_antithetic_candidate_v1(master, noise, table, 1)
    minus = subject.materialize_antithetic_candidate_v1(master, noise, table, -1)
    for key in master:
        assert torch.allclose((plus[key] + minus[key]) / 2, master[key])
        assert torch.allclose(plus[key] - minus[key], 0.04 * torch.ones_like(master[key]))


def test_mirrored_gradient_has_one_inverse_sigma_and_correct_algebra():
    master = _small_master(a=(0.0,), b=(0.0,))
    keys = list(master)
    root_two = math.sqrt(2.0)
    noises = [
        {
            keys[0]: torch.tensor([root_two], dtype=torch.float32),
            keys[1]: torch.tensor([0.0], dtype=torch.float32),
        },
        {
            keys[0]: torch.tensor([0.0], dtype=torch.float32),
            keys[1]: torch.tensor([root_two], dtype=torch.float32),
        },
    ]
    true_gradient = {keys[0]: 2.0, keys[1]: 5.0}
    sigma = 0.01
    differences = [
        2 * sigma * root_two * true_gradient[key] for key in keys
    ]
    table = subject.module_sigma_table_v1(master, sigma, "global")
    result = subject.mirrored_gradient_v1(
        master, noises, differences, table, "raw_pair_difference"
    )
    assert result["candidate_scale_application_count"] == 1
    assert result["estimator_inverse_scale_application_count"] == 1
    for key in keys:
        assert result["gradient"][key].item() == pytest.approx(
            true_gradient[key], rel=2e-6
        )


@pytest.mark.parametrize("mode", subject.COEFFICIENT_MODES_V1)
def test_registered_pair_shaping_is_antisymmetric(mode):
    differences = [-3.0, 1.0, 1.0, 0.0]
    forward = subject.shape_pair_coefficients_v1(differences, mode)
    swapped = subject.shape_pair_coefficients_v1(
        [-value for value in differences], mode
    )
    assert swapped == [-value for value in forward]


def test_independent_signed_candidate_rank_is_not_registered():
    with pytest.raises(ValueError, match="not mirrored-pair safe"):
        subject.shape_pair_coefficients_v1([1.0, -1.0], "centered_candidate_rank")


def test_zero_pair_variance_skips_without_state_or_checkpoint_advance():
    master = _small_master()
    noises = [{key: torch.ones_like(value) for key, value in master.items()}] * 2
    table = subject.module_sigma_table_v1(master, 0.01, "global")
    result = subject.mirrored_gradient_v1(
        master, noises, [1.0, 1.0], table, "raw_pair_difference"
    )
    assert result["status"] == "skip_zero_pair_difference_variance"
    assert result["optimizer_state_may_advance"] is False
    assert result["checkpoint_may_change"] is False


def test_nonfinite_pair_difference_fails_before_optimizer():
    master = _small_master()
    noises = [{key: torch.ones_like(value) for key, value in master.items()}] * 2
    table = subject.module_sigma_table_v1(master, 0.01, "global")
    with pytest.raises(ValueError, match="pair difference must be finite"):
        subject.mirrored_gradient_v1(
            master, noises, [1.0, float("nan")], table, "raw_pair_difference"
        )


def test_momentum_accumulates_tiny_fp32_residuals():
    master = _small_master(a=(0.0,), b=(0.0,))
    gradient = {key: torch.full_like(value, 1.0e-8) for key, value in master.items()}
    state = subject.initial_optimizer_state_v1("momentum")
    _, state, _ = subject.optimizer_direction_v1(
        master, gradient, state, "momentum"
    )
    direction, state, receipt = subject.optimizer_direction_v1(
        master, gradient, state, "momentum"
    )
    assert receipt["step_after"] == 2
    assert all(value.dtype == torch.float32 for value in direction.values())
    assert all(value.item() == pytest.approx(1.9e-8, rel=1e-6)
               for value in direction.values())
    assert state["slot_dtype"] == "torch.float32"


def test_adam_bias_correction_uses_new_exact_step_and_fp32_moments():
    master = _small_master(a=(1.0,), b=(-2.0,))
    gradient = {
        list(master)[0]: torch.tensor([2.0], dtype=torch.float32),
        list(master)[1]: torch.tensor([-4.0], dtype=torch.float32),
    }
    state = subject.initial_optimizer_state_v1("adamw")
    direction, state, receipt = subject.optimizer_direction_v1(
        master, gradient, state, "adamw"
    )
    assert receipt["bias_correction_step"] == 1
    config = subject.OPTIMIZER_CONFIGS_V1["adamw"]
    for key in master:
        expected = gradient[key] / (gradient[key].abs() + config["epsilon"])
        expected = expected - config["weight_decay"] * master[key]
        assert torch.allclose(direction[key], expected, rtol=1e-6, atol=1e-7)
    _, state, receipt = subject.optimizer_direction_v1(
        master, gradient, state, "adamw"
    )
    assert receipt["bias_correction_step"] == 2
    assert state["step"] == 2


def test_silent_bf16_optimizer_slot_is_rejected():
    master = _small_master(a=(0.0,), b=(0.0,))
    bad_state = {
        "schema": "fp32-es-optimizer-state-v1",
        "optimizer": "momentum",
        "step": 1,
        "slot_dtype": "torch.float32",
        "slots": {"velocity": {
            key: torch.zeros_like(value, dtype=torch.bfloat16)
            for key, value in master.items()
        }},
    }
    with pytest.raises(RuntimeError, match="CPU FP32"):
        subject.validate_optimizer_state_v1(bad_state, master, "momentum")


def test_sgd_is_direct_fp32_ascent_direction():
    master = _small_master(a=(0.0,), b=(0.0,))
    gradient = {
        key: torch.full_like(value, 0.125) for key, value in master.items()
    }
    direction, state, receipt = subject.optimizer_direction_v1(
        master, gradient, subject.initial_optimizer_state_v1("sgd"), "sgd"
    )
    assert all(torch.equal(direction[key], gradient[key]) for key in master)
    assert state["step"] == 1
    assert receipt["ascent"] is True


def test_zero_budgeted_update_rolls_back_tentative_optimizer_step():
    master = _small_master(a=(0.0,), b=(0.0,))
    zero = {key: torch.zeros_like(value) for key, value in master.items()}
    initial = subject.initial_optimizer_state_v1("momentum")
    result = subject.budgeted_optimizer_update_v1(
        master, zero, initial, "momentum"
    )
    assert result["status"] == "skip_zero_optimizer_direction"
    assert result["candidate_optimizer_state"]["step"] == 0
    assert result["optimizer_receipt"]["state_advance_rolled_back"] is True


def test_noise_shape_broadcast_is_rejected():
    master = _small_master()
    noise = {
        key: torch.ones(1, dtype=torch.float32) for key in master
    }
    table = subject.module_sigma_table_v1(master, 0.01, "global")
    with pytest.raises(RuntimeError, match="tensor shape changed"):
        subject.materialize_antithetic_candidate_v1(master, noise, table, 1)


def test_update_norm_projection_equalizes_different_raw_directions():
    master = _small_master()
    one = {key: torch.ones_like(value) for key, value in master.items()}
    ten = {key: 10 * torch.ones_like(value) for key, value in master.items()}
    # A larger test-only budget avoids making three FP32 rounding quanta carry
    # the full relative-tolerance assertion; production has 4.5M elements.
    first = subject.apply_update_norm_budget_v1(master, one, budget_ratio=0.1)
    second = subject.apply_update_norm_budget_v1(master, ten, budget_ratio=0.1)
    assert first["target_update_l2"] == second["target_update_l2"]
    assert first["observed_update_l2"] == pytest.approx(
        second["observed_update_l2"], rel=1e-6
    )
    assert first["relative_error"] <= subject.UPDATE_NORM_RELATIVE_TOLERANCE_V1


def test_zero_optimizer_direction_does_not_change_checkpoint_candidate():
    master = _small_master()
    zero = {key: torch.zeros_like(value) for key, value in master.items()}
    result = subject.apply_update_norm_budget_v1(master, zero)
    assert result["status"] == "skip_zero_optimizer_direction"
    assert result["optimizer_state_may_advance"] is False
    assert all(torch.equal(result["candidate"][key], master[key]) for key in master)


def test_checkpoint_identity_includes_optimizer_step_and_cursor():
    master = _small_master(a=(0.0,), b=(0.0,))
    initial = subject.initial_optimizer_state_v1("sgd")
    first = subject.checkpoint_identity_v1(
        master, initial, "sgd", {"cursor": 0, "seed": 1}
    )
    advanced = copy.deepcopy(initial)
    advanced["step"] = 1
    second = subject.checkpoint_identity_v1(
        master, advanced, "sgd", {"cursor": 0, "seed": 1}
    )
    moved = subject.checkpoint_identity_v1(
        master, initial, "sgd", {"cursor": 1, "seed": 1}
    )
    assert len({first["checkpoint_sha256"], second["checkpoint_sha256"], moved["checkpoint_sha256"]}) == 3


def test_memory_contract_seals_fp32_host_state_and_zero_optimizer_vram():
    memory = subject.optimizer_memory_contract_v1()
    assert memory["optimizers"]["sgd"][
        "persistent_optimizer_host_tensor_bytes_per_replica"
    ] == 0
    assert memory["optimizers"]["momentum"][
        "persistent_optimizer_host_tensor_bytes_per_replica"
    ] == subject.MASTER_BYTES_V1
    assert memory["optimizers"]["adamw"][
        "persistent_optimizer_host_tensor_bytes_per_replica"
    ] == 2 * subject.MASTER_BYTES_V1
    assert all(item["persistent_optimizer_gpu_bytes_per_replica"] == 0
               for item in memory["optimizers"].values())


def test_preregistration_is_a_launch_ineligible_sealed_cpu_preview(plan):
    result = subject.validate_preregistration_v1(plan)
    assert result["status"] == "sealed_cpu_preview_runtime_dependencies_pending"
    with pytest.raises(RuntimeError, match="dependencies remain incomplete"):
        subject.validate_preregistration_v1(plan, launch=True)


def test_synthetic_completed_dependencies_make_contract_launch_valid(plan):
    value = _launch_ready(plan)
    assert subject.validate_preregistration_v1(value, launch=True)["status"] == "launch_ready"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["sigma_contract"].__setitem__(
                "candidate_scale_application_count", 2
            ),
            "sigma contract changed",
        ),
        (
            lambda value: value["sigma_contract"].__setitem__(
                "module_basis", "fp32_module_l2_without_shape_divisor"
            ),
            "sigma contract changed",
        ),
        (
            lambda value: value["optimizer_contract"].__setitem__(
                "slot_dtype", "torch.bfloat16"
            ),
            "optimizer contract changed",
        ),
        (
            lambda value: value["memory_bandwidth_contract"]["optimizers"]
            ["adamw"].__setitem__(
                "persistent_optimizer_host_tensor_bytes_per_replica", 0
            ),
            "memory/bandwidth accounting changed",
        ),
        (
            lambda value: value["compute_contract"].__setitem__(
                "rollouts_per_replicate", 1024
            ),
            "equal-compute contract changed",
        ),
        (
            lambda value: value["update_budget_contract"].__setitem__(
                "ratio", subject.UPDATE_BUDGET_RATIO_V1 * 2
            ),
            "update budget changed",
        ),
    ],
)
def test_preregistration_rejects_adversarial_contract_drift(plan, mutate, message):
    value = copy.deepcopy(plan)
    mutate(value)
    _resign(value)
    with pytest.raises(RuntimeError, match=message):
        subject.validate_preregistration_v1(value)


def test_preregistration_rejects_module_size_domination_even_if_inner_hash_is_resealed(plan):
    value = copy.deepcopy(plan)
    wrapper = next(
        item for item in value["sigma_contract"]["tables"]
        if item["mode"] == "module_fp32_rms_shape_normalized"
    )
    record = wrapper["table"]["records"][0]
    record["raw_scale_basis"] *= math.sqrt(record["elements"])
    table = wrapper["table"]
    table["content_sha256"] = subject.canonical_sha256_v1({
        key: item for key, item in table.items() if key != "content_sha256"
    })
    _resign(value)
    with pytest.raises(RuntimeError, match="module-size-dominated"):
        subject.validate_preregistration_v1(value)


def test_valid_runtime_receipt_passes_exact_chain_and_gates(plan):
    ready = _launch_ready(plan)
    result = subject.validate_run_receipt_v1(ready, _run_receipt(ready))
    assert result["status"] == "valid_complete_run_receipt"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda receipt: receipt["updates"][0].__setitem__(
                "bias_correction_step", 0
            ),
            "algebra or compute changed",
        ),
        (
            lambda receipt: receipt["updates"][0].__setitem__(
                "estimator_inverse_scale_application_count", 2
            ),
            "algebra or compute changed",
        ),
        (
            lambda receipt: receipt["updates"][0].__setitem__(
                "optimizer_state_dtype", "torch.bfloat16"
            ),
            "algebra or compute changed",
        ),
        (
            lambda receipt: receipt["updates"][0].__setitem__(
                "persistent_optimizer_gpu_bytes_per_replica", 2
            ),
            "memory or bandwidth receipt changed",
        ),
        (
            lambda receipt: receipt["updates"][0].__setitem__(
                "measured_host_memory_traffic_bytes_per_replica", 1
            ),
            "memory or bandwidth receipt changed",
        ),
        (
            lambda receipt: receipt["updates"][0].__setitem__(
                "observed_update_l2", 2.0
            ),
            "unequal update norm budget",
        ),
        (
            lambda receipt: receipt.__setitem__("total_rollouts", 2047),
            "native work",
        ),
        (
            lambda receipt: receipt["updates"][1].__setitem__(
                "resume_checkpoint_sha256", _sha("wrong resume")
            ),
            "checkpoint resume",
        ),
        (
            lambda receipt: receipt["updates"][1].__setitem__(
                "pre_optimizer_state_sha256", _sha("wrong optimizer state")
            ),
            "optimizer resume",
        ),
        (
            lambda receipt: receipt.__setitem__("protected_holdout_opened", True),
            "evaluation gate",
        ),
        (
            lambda receipt: receipt.__setitem__("dev_opened_monotonic_ns", 99),
            "before training was sealed",
        ),
    ],
)
def test_runtime_receipt_rejects_adversarial_drift(plan, mutate, message):
    ready = _launch_ready(plan)
    receipt = _run_receipt(ready)
    mutate(receipt)
    with pytest.raises((RuntimeError, ValueError), match=message):
        subject.validate_run_receipt_v1(ready, receipt)


def test_complete_grid_requires_every_arm_seed_once(plan):
    ready = _launch_ready(plan)
    receipts = [
        _run_receipt(ready, arm["arm_id"], seed)
        for arm in ready["grid"]
        for seed in subject.REPLICATE_SEEDS_V1
    ]
    result = subject.validate_grid_receipts_v1(ready, receipts)
    assert result == {
        "status": "valid_complete_nonadaptive_grid",
        "receipt_count": 24,
        "total_rollouts": 24 * subject.ROLLOUTS_PER_REPLICATE_V1,
    }
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        subject.validate_grid_receipts_v1(ready, receipts[:-1])
