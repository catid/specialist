import copy
import hashlib
import math
import random

import pytest
import torch

import build_structured_es_preregistration_v1 as builder
import structured_es_oracle_v1 as subject


MATRIX_SHAPE = [19, 23]
MATRIX_ELEMENTS = math.prod(MATRIX_SHAPE)
METHODS = [
    ("iid_absolute_index", None),
    *[("structured_outer_product", rank) for rank in subject.STRUCTURED_RANKS_V1],
]
PINNED_STREAM_SHA256 = {
    ("iid_absolute_index", None): (
        "93f71d75714e3606d1910b850b821776f9c61c09c960f192a448c7eed82175c6"
    ),
    ("structured_outer_product", 1): (
        "3b223a7d888e4749785405f516eb320c8095efbbace0942621a7656526ea21de"
    ),
    ("structured_outer_product", 4): (
        "346d537b265881c5e7cb1b1180fb0600990f1e6f2690085eb1dfc17044a3d504"
    ),
    ("structured_outer_product", 8): (
        "e26c47178dab2ed7e09e1e11aadca166e1b3040831e54999362f2221e6b484fc"
    ),
    ("structured_outer_product", 16): (
        "c97bdf663b2fd76455cdc55265062fbf5566f342b3b11cdadb264cd0bcdcffd2"
    ),
}


@pytest.fixture(scope="module")
def plan():
    return builder.build_preregistration_v1()


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _resign_plan(value):
    value["content_sha256_before_self_field"] = subject.canonical_sha256_v1({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    return value


def _resign_reconstruction(value):
    value["content_sha256"] = subject.canonical_sha256_v1({
        key: item for key, item in value.items() if key != "content_sha256"
    })
    return value


def _noise(method, rank, start=0, count=MATRIX_ELEMENTS):
    return subject.noise_chunk_v1(
        method,
        1701,
        "unit.matrix",
        MATRIX_SHAPE,
        start,
        count,
        rank,
    )


def _systems_receipt(plan, arm_id):
    arm = {item["arm_id"]: item for item in plan["arms"]}[arm_id]
    if arm["surface"] == "matched_lora_4528128":
        method_key = (
            "iid_absolute_index"
            if arm["structured_rank"] is None
            else f"structured_rank_{arm['structured_rank']}"
        )
        ceiling = plan["memory_bandwidth_contract"]["lora"]["methods"][
            method_key
        ]["weighted_update_scratch_ceiling_bytes"]
        elements = subject.LORA_ELEMENTS_V1
        candidate_bytes = (
            subject.SIGNED_CANDIDATES_PER_UPDATE_V1
            * subject.LORA_RUNTIME_BF16_BYTES_V1
        )
    else:
        ceiling = plan["memory_bandwidth_contract"][
            "dense_fullweight_system_anchor"
        ]["weighted_update_scratch_ceiling_bytes"]
        elements = subject.FULL_MODEL_ELEMENTS_V1
        candidate_bytes = (
            subject.SIGNED_CANDIDATES_PER_UPDATE_V1
            * subject.FULL_MODEL_BF16_BYTES_V1
        )
    target = 1.0
    observed = target * (1.0 + 1.0e-6)
    return {
        "schema": "structured-es-systems-receipt-v1",
        "plan_content_sha256": plan["content_sha256_before_self_field"],
        "arm_id": arm_id,
        "replicate_seed": plan["compute_contract"]["systems_replicate_seed"],
        "directions": subject.DIRECTIONS_PER_UPDATE_V1,
        "signed_candidates": subject.SIGNED_CANDIDATES_PER_UPDATE_V1,
        "train_units_per_candidate": subject.TRAIN_UNITS_PER_CANDIDATE_V1,
        "rollouts": subject.ROLLOUTS_PER_UPDATE_V1,
        "surface_elements": elements,
        "method": arm["method"],
        "structured_rank": arm["structured_rank"],
        "rng_algorithm": subject.RNG_ALGORITHM_V1,
        "dense_noise_elements_allocated": 0,
        "dense_candidate_elements_allocated": 0,
        "peak_scratch_bytes": ceiling,
        "scratch_ceiling_bytes": ceiling,
        "candidate_bytes_written": candidate_bytes,
        "reward_vector_sha256": _sha(f"reward:{arm_id}"),
        "reward_mean": 0.25,
        "reward_variance": 0.5,
        "stream_update_vs_oracle_max_abs": 0.0,
        "stream_update_vs_oracle_max_ulp": 0,
        "target_update_l2": target,
        "observed_update_l2": observed,
        "update_norm_relative_error": abs(observed - target) / target,
        "canonical_master_before_sha256": _sha("master"),
        "canonical_master_after_sha256": _sha("master"),
        "canonical_runtime_before_sha256": _sha("runtime"),
        "canonical_runtime_after_sha256": _sha("runtime"),
        "rollback_complete": True,
        "poisoned": False,
        "useful_physical_gpus": [0, 1, 2, 3],
        "v66d_telemetry_report_content_sha256": (
            subject.V66D_REPORT_CONTENT_SHA256_V1
        ),
        "protected_or_eval_opened": False,
        "charged_gpu_seconds": 500.0,
    }


def test_absolute_rng_has_pinned_fp32_vectors_and_domain_separation():
    expected = {
        "iid_element": [
            0.39477285742759705,
            -0.6862331032752991,
            0.5891834497451782,
            -0.5949168801307678,
            0.05696180462837219,
            -0.24526812136173248,
            -0.6653454303741455,
            1.5860224962234497,
        ],
        "structured_left_rank_4": [
            -1.2427632808685303,
            1.7084938287734985,
            1.2230257987976074,
            -0.20997877418994904,
            1.0992939472198486,
            -2.065319776535034,
            -0.2704937160015106,
            1.3725193738937378,
        ],
        "structured_right_rank_4": [
            0.8245996832847595,
            -0.11855602264404297,
            1.0870213508605957,
            -0.791411817073822,
            2.3618416786193848,
            0.901041567325592,
            0.23339304327964783,
            1.2871406078338623,
        ],
    }
    observed = {
        domain: [
            subject.absolute_normal_v1(
                1701, "layers.0.q_proj.lora_A.weight", domain, index
            )
            for index in range(8)
        ]
        for domain in expected
    }
    assert observed == expected
    assert len({tuple(values) for values in observed.values()}) == len(observed)
    assert subject.absolute_normal_v1(
        1702, "layers.0.q_proj.lora_A.weight", "iid_element", 0
    ) != expected["iid_element"][0]
    assert subject.absolute_normal_v1(
        1701, "layers.1.q_proj.lora_A.weight", "iid_element", 0
    ) != expected["iid_element"][0]


@pytest.mark.parametrize("method,rank", METHODS)
def test_absolute_slices_are_exactly_chunk_order_independent(method, rank):
    full = _noise(method, rank)
    ranges = [(0, 1), (1, 18), (18, 119), (119, 120), (120, 401), (401, 437)]
    pieces = [_noise(method, rank, start, end - start) for start, end in ranges]
    assert torch.equal(torch.cat(pieces), full)


@pytest.mark.parametrize("method,rank", METHODS)
def test_reconstruction_is_exact_across_world_sizes_and_chunkings(method, rank):
    receipts = [
        subject.build_reconstruction_receipt_v1(
            method,
            1701,
            "unit.matrix",
            MATRIX_SHAPE,
            structured_rank=rank,
            world_size=world_size,
            chunk_elements=chunk_elements,
        )
        for world_size, chunk_elements in ((1, 437), (4, 17), (7, 13))
    ]
    for receipt in receipts:
        validated = subject.validate_reconstruction_receipt_v1(receipt)
        assert validated["status"] == "exact_absolute_index_reconstruction"
    assert {item["canonical_full_stream_sha256"] for item in receipts} == {
        PINNED_STREAM_SHA256[(method, rank)]
    }


def test_reconstruction_accepts_shuffled_receipts_but_rejects_adversarial_coverage():
    receipt = subject.build_reconstruction_receipt_v1(
        "structured_outer_product",
        1701,
        "unit.matrix",
        MATRIX_SHAPE,
        structured_rank=4,
        world_size=4,
        chunk_elements=17,
    )
    shuffled = copy.deepcopy(receipt)
    random.Random(17).shuffle(shuffled["chunks"])
    subject.validate_reconstruction_receipt_v1(_resign_reconstruction(shuffled))

    variants = []
    duplicate = copy.deepcopy(receipt)
    duplicate["chunks"].append(copy.deepcopy(duplicate["chunks"][0]))
    variants.append(duplicate)
    gap = copy.deepcopy(receipt)
    gap["chunks"][1]["start"] += 1
    variants.append(gap)
    wrong_values = copy.deepcopy(receipt)
    wrong_values["chunks"][0]["value_sha256"] = "0" * 64
    variants.append(wrong_values)
    false_ceiling = copy.deepcopy(receipt)
    false_ceiling["chunk_elements"] = 1
    variants.append(false_ceiling)
    wrong_shard = copy.deepcopy(receipt)
    wrong_shard["chunks"][0]["rank"] = 3
    variants.append(wrong_shard)
    for variant in variants:
        with pytest.raises((ValueError, RuntimeError)):
            subject.validate_reconstruction_receipt_v1(
                _resign_reconstruction(variant)
            )


def test_rng_and_identity_reject_ambiguous_or_out_of_domain_inputs():
    with pytest.raises(ValueError):
        subject.absolute_normal_v1(1701, "tensor", "iid", True)
    with pytest.raises(ValueError):
        subject.absolute_normal_v1(1701, "tensor", "iid", 1 << 62)
    with pytest.raises(ValueError):
        subject.noise_identity_v1("iid_absolute_index", 1701, "", [2, 2])
    with pytest.raises(ValueError):
        subject.noise_identity_v1("iid_absolute_index", 1701, "tensor", [2, 2], 1)
    with pytest.raises(ValueError):
        subject.noise_identity_v1(
            "structured_outer_product", 1701, "tensor", [8, 8], 16
        )


def test_dense_iid_oracle_supports_flat_vector_and_expert_tensor_shapes():
    for shape in ([31], [2, 3, 5]):
        total = math.prod(shape)
        whole = subject.noise_chunk_v1(
            "iid_absolute_index", 1701, "dense.tensor", shape, 0, total
        )
        pieces = torch.cat([
            subject.noise_chunk_v1(
                "iid_absolute_index", 1701, "dense.tensor", shape, start, end - start
            )
            for start, end in subject.chunk_ranges_v1(0, total, 7)
        ])
        assert torch.equal(whole, pieces)
        receipt = subject.build_reconstruction_receipt_v1(
            "iid_absolute_index",
            1701,
            "dense.tensor",
            shape,
            world_size=4,
            chunk_elements=7,
        )
        subject.validate_reconstruction_receipt_v1(receipt)

        master = torch.arange(total, dtype=torch.float32).reshape(shape)
        candidates = list(subject.streamed_candidate_chunks_v1(
            master,
            "iid_absolute_index",
            1701,
            "dense.tensor",
            0.0006,
            1,
            chunk_elements=7,
        ))
        assert sum(item["candidate"].numel() for item in candidates) == total
        updates = list(subject.streamed_weighted_update_chunks_v1(
            shape,
            "iid_absolute_index",
            [1, 2],
            [0.5, -0.25],
            "dense.tensor",
            0.0006,
            chunk_elements=7,
        ))
        assert sum(item["update"].numel() for item in updates) == total

    with pytest.raises(ValueError):
        subject.noise_identity_v1(
            "structured_outer_product", 1701, "dense.expert", [2, 3, 5], 1
        )


@pytest.mark.parametrize("rank", subject.STRUCTURED_RANKS_V1)
def test_structured_scale_and_fixed_component_reduction_match_manual_oracle(rank):
    row, column = 3, 5
    dot = 0.0
    for component in range(rank):
        left = subject.absolute_normal_v1(
            1701, "unit.matrix", f"structured_left_rank_{rank}", row * rank + component
        )
        right = subject.absolute_normal_v1(
            1701,
            "unit.matrix",
            f"structured_right_rank_{rank}",
            column * rank + component,
        )
        product = subject._float32_v1(left * right)
        dot = subject._float32_v1(dot + product)
    expected = subject._float32_v1(dot / math.sqrt(rank))
    actual = _noise(method="structured_outer_product", rank=rank)[
        row * MATRIX_SHAPE[1] + column
    ].item()
    assert actual == expected
    theory = subject.structured_moment_theory_v1(rank)
    assert theory["rank_scale"] == 1.0 / math.sqrt(rank)
    assert theory["entry_variance"] == 1.0
    assert theory["entry_fourth_moment"] == 3.0 + 6.0 / rank
    assert theory["entry_excess_kurtosis"] == 6.0 / rank


@pytest.mark.parametrize("method,rank", METHODS)
def test_antithetic_candidates_stream_without_dense_materialization(method, rank):
    master = torch.linspace(-1.0, 1.0, MATRIX_ELEMENTS, dtype=torch.float32).reshape(
        MATRIX_SHAPE
    )
    plus_chunks = list(subject.streamed_candidate_chunks_v1(
        master,
        method,
        1701,
        "unit.matrix",
        0.0006,
        1,
        structured_rank=rank,
        chunk_elements=7,
    ))
    minus_chunks = list(subject.streamed_candidate_chunks_v1(
        master,
        method,
        1701,
        "unit.matrix",
        0.0006,
        -1,
        structured_rank=rank,
        chunk_elements=11,
    ))
    plus = torch.cat([item["candidate"] for item in plus_chunks])
    minus = torch.cat([item["candidate"] for item in minus_chunks])
    noise = torch.cat([item["noise"] for item in plus_chunks])
    torch.testing.assert_close((plus + minus) / 2.0, master.reshape(-1), atol=1e-7, rtol=0)
    torch.testing.assert_close(plus - minus, 2.0 * 0.0006 * noise, atol=2e-7, rtol=2e-5)
    for item in plus_chunks + minus_chunks:
        assert item["candidate"].numel() <= 11
        assert item["dense_candidate_materialized"] is False
        assert item["dense_noise_materialized"] is False


@pytest.mark.parametrize("method,rank", METHODS)
def test_streamed_weighted_update_is_exact_across_chunkings(method, rank):
    seeds = [7001, 7002, 7003]
    coefficients = [0.25, -1.5, 2.0]
    kwargs = {
        "shape": MATRIX_SHAPE,
        "method": method,
        "seeds": seeds,
        "coefficients": coefficients,
        "tensor_key": "unit.matrix",
        "sigma": 0.0006,
        "structured_rank": rank,
    }
    whole_items = list(subject.streamed_weighted_update_chunks_v1(
        **kwargs, chunk_elements=MATRIX_ELEMENTS
    ))
    chunked_items = list(subject.streamed_weighted_update_chunks_v1(
        **kwargs, chunk_elements=13
    ))
    whole = torch.cat([item["update"] for item in whole_items])
    chunked = torch.cat([item["update"] for item in chunked_items])
    assert torch.equal(chunked, whole)

    manual = torch.zeros(MATRIX_ELEMENTS, dtype=torch.float32)
    for seed, coefficient in zip(seeds, coefficients, strict=True):
        manual.add_(subject.noise_chunk_v1(
            method,
            seed,
            "unit.matrix",
            MATRIX_SHAPE,
            0,
            MATRIX_ELEMENTS,
            rank,
        ), alpha=coefficient)
    manual.mul_(1.0 / (2.0 * len(seeds) * 0.0006))
    assert torch.equal(whole, manual)
    assert all(item["dense_update_materialized"] is False for item in chunked_items)
    assert all(item["dense_noise_materialized"] is False for item in chunked_items)


def test_streamed_update_rejects_duplicate_seed_nonfinite_coefficient_and_sigma():
    common = {
        "shape": MATRIX_SHAPE,
        "method": "iid_absolute_index",
        "tensor_key": "unit.matrix",
    }
    with pytest.raises(ValueError):
        list(subject.streamed_weighted_update_chunks_v1(
            **common, seeds=[1, 1], coefficients=[1.0, 2.0], sigma=0.1
        ))
    with pytest.raises(ValueError):
        list(subject.streamed_weighted_update_chunks_v1(
            **common, seeds=[1], coefficients=[float("nan")], sigma=0.1
        ))
    with pytest.raises(ValueError):
        list(subject.streamed_weighted_update_chunks_v1(
            **common, seeds=[1], coefficients=[1.0], sigma=0.0
        ))


def test_transaction_happy_path_and_partial_candidate_full_restore():
    master_sha = _sha("master")
    runtime_sha = _sha("runtime")
    transaction = subject.RuntimeTransactionOracleV1(10, master_sha, runtime_sha)
    transaction.begin_candidate(_sha("noise"))
    transaction.record_candidate_chunk(5, 10)
    transaction.record_candidate_chunk(0, 5)
    transaction.finish_candidate()
    transaction.begin_restore()
    transaction.record_restore_chunk(0, 4)
    transaction.record_restore_chunk(4, 10)
    transaction.finish_restore(master_sha, runtime_sha)
    assert transaction.receipt()["phase"] == "quiescent"
    assert transaction.receipt()["poisoned"] is False

    partial = subject.RuntimeTransactionOracleV1(10, master_sha, runtime_sha)
    partial.begin_candidate(_sha("noise"))
    partial.record_candidate_chunk(0, 3)
    partial.begin_restore()
    partial.record_restore_chunk(0, 10)
    partial.finish_restore(master_sha, runtime_sha)
    assert partial.receipt()["phase"] == "quiescent"


def test_transaction_gap_overlap_and_restore_identity_fail_terminally():
    master_sha = _sha("master")
    runtime_sha = _sha("runtime")

    incomplete = subject.RuntimeTransactionOracleV1(10, master_sha, runtime_sha)
    incomplete.begin_candidate(_sha("noise"))
    incomplete.record_candidate_chunk(0, 9)
    with pytest.raises(RuntimeError):
        incomplete.finish_candidate()
    assert incomplete.poisoned is True
    with pytest.raises(RuntimeError):
        incomplete.begin_candidate(_sha("other"))

    overlap = subject.RuntimeTransactionOracleV1(10, master_sha, runtime_sha)
    overlap.begin_candidate(_sha("noise"))
    overlap.record_candidate_chunk(0, 6)
    with pytest.raises(RuntimeError):
        overlap.record_candidate_chunk(5, 10)
    assert overlap.poisoned is True

    mismatch = subject.RuntimeTransactionOracleV1(10, master_sha, runtime_sha)
    mismatch.begin_candidate(_sha("noise"))
    mismatch.record_candidate_chunk(0, 10)
    mismatch.finish_candidate()
    mismatch.begin_restore()
    mismatch.record_restore_chunk(0, 10)
    with pytest.raises(RuntimeError):
        mismatch.finish_restore(master_sha, _sha("wrong runtime"))
    assert mismatch.poisoned is True
    assert mismatch.phase == "terminal_poison"


def test_lora_scratch_and_random_draw_accounting_is_exact(plan):
    accounting = subject.lora_streaming_accounting_v1(
        plan["surfaces"]["lora"]["tensor_shapes"]
    )
    assert accounting == plan["memory_bandwidth_contract"]["lora"]
    assert accounting["elements"] == 4_528_128
    assert accounting["fp32_master_bytes"] == 18_112_512
    assert accounting["runtime_bf16_view_bytes"] == 9_842_688
    iid = accounting["methods"]["iid_absolute_index"]
    assert iid["unique_random_values_per_direction"] == 4_528_128
    assert iid["candidate_scratch_ceiling_bytes"] == 131_072
    assert iid["weighted_update_scratch_ceiling_bytes"] == 196_608
    expected = {
        1: (143_744, 574_976, 32_896, 9.0),
        4: (574_976, 2_299_904, 131_584, 4.5),
        8: (1_149_952, 4_599_808, 263_168, 3.75),
        16: (2_299_904, 9_199_616, 526_336, 3.375),
    }
    for rank, (values, factor_bytes, cache_bytes, fourth_moment) in expected.items():
        item = accounting["methods"][f"structured_rank_{rank}"]
        assert item["factor_values_per_direction"] == values
        assert item["factor_bytes_per_direction"] == factor_bytes
        assert item["maximum_factor_cache_bytes_per_tensor"] == cache_bytes
        assert item["entry_fourth_moment"] == fourth_moment
        assert item["dense_noise_elements_allocated"] == 0
        assert item["dense_candidate_elements_allocated"] == 0


def test_preregistration_is_valid_but_cannot_authorize_a_launch(plan):
    result = subject.validate_preregistration_v1(plan)
    assert result["status"] == "sealed_cpu_correctness_runtime_dependencies_pending"
    assert result["arm_count"] == 6
    with pytest.raises(RuntimeError):
        subject.validate_preregistration_v1(plan, launch=True)


@pytest.mark.parametrize(
    "path,value",
    [
        (("authorization", "candidate_commit"), True),
        (("rng_contract", "local_chunk_index_rng"), "allowed"),
        (("structured_scale_theory", "rank_scale"), "1/k"),
        (("streaming_contract", "dense_full_surface_noise_materialization"), "allowed"),
        (("memory_bandwidth_contract", "structured_does_not_reduce_runtime_install_bytes_without_fusion"), False),
        (("compute_contract", "fixed_optimizer"), "adamw"),
        (("compute_contract", "quality_rollouts_per_lora_arm_seed"), 1),
        (("source_contracts", "v66d_accepted_telemetry", "report_content_sha256"), "0" * 64),
        (("dependencies", "v66d_accepted_gpu_attribution_complete"), False),
        (("surfaces", "dense_fullweight", "index_file_sha256"), "0" * 64),
    ],
)
def test_semantic_plan_tampering_is_rejected_even_after_resigning(plan, path, value):
    changed = copy.deepcopy(plan)
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    _resign_plan(changed)
    with pytest.raises((ValueError, RuntimeError)):
        subject.validate_preregistration_v1(changed)


def test_every_registered_systems_arm_has_a_valid_equal_work_receipt(plan):
    for arm in plan["arms"]:
        result = subject.validate_systems_receipt_v1(
            plan, _systems_receipt(plan, arm["arm_id"])
        )
        assert result == {
            "status": "valid_structured_systems_receipt",
            "arm_id": arm["arm_id"],
            "replicate_seed": 1701,
        }


@pytest.mark.parametrize(
    "field,value",
    [
        ("replicate_seed", 1702),
        ("directions", 7),
        ("surface_elements", subject.LORA_ELEMENTS_V1 + 1),
        ("dense_noise_elements_allocated", 1),
        ("dense_candidate_elements_allocated", False),
        ("candidate_bytes_written", 1),
        ("reward_variance", 0.0),
        ("stream_update_vs_oracle_max_ulp", 3),
        ("canonical_master_after_sha256", "0" * 64),
        ("rollback_complete", False),
        ("poisoned", True),
        ("useful_physical_gpus", [0, 1, 2]),
        ("v66d_telemetry_report_content_sha256", "0" * 64),
        ("protected_or_eval_opened", True),
        ("charged_gpu_seconds", 14_400.0001),
    ],
)
def test_systems_receipt_rejects_work_memory_restore_and_access_violations(
    plan, field, value
):
    receipt = _systems_receipt(plan, "lora_structured_rank_4")
    receipt[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        subject.validate_systems_receipt_v1(plan, receipt)


def test_systems_receipt_rejects_scratch_overflow_and_update_norm_drift(plan):
    receipt = _systems_receipt(plan, "lora_structured_rank_4")
    receipt["peak_scratch_bytes"] = receipt["scratch_ceiling_bytes"] + 1
    with pytest.raises(RuntimeError):
        subject.validate_systems_receipt_v1(plan, receipt)

    receipt = _systems_receipt(plan, "lora_structured_rank_4")
    receipt["observed_update_l2"] = 1.001
    receipt["update_norm_relative_error"] = 0.001
    with pytest.raises(RuntimeError):
        subject.validate_systems_receipt_v1(plan, receipt)
