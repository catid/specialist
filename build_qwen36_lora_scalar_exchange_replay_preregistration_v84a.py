#!/usr/bin/env python3
"""Build/check the CPU-only V84A scalar-exchange replay preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import eggroll_es_scalar_exchange_replay_v84a as contract


ROOT = Path(__file__).resolve().parent
V82B_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_collective_compression_v82b.json"
)
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_scalar_exchange_replay_v84a.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_lora_scalar_exchange_replay_v84a_cpu_evidence_20260717.md"
)

BOUND_SOURCE_SHA256 = {
    "eggroll_es_worker_lora_v72.py": (
        "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2"
    ),
    "structured_es_oracle_v1.py": (
        "8fca35f89744f292ef0d9327f547196dd26f93268336f3fad4812a065f35f740"
    ),
    "eggroll_es_fused_structured_runtime_v72.py": (
        "357607f3c16b071f67d2bc3adb0317bbbd29f31f7e1db0cf1aa3030ac997df6e"
    ),
    "eggroll_es_fp32_collective_coalescing_v83a.py": (
        "0c1ed3a7e451da20e76d3b0ea971771b4e7acbc17d6e5f34b5d54efe4a7bc0d6"
    ),
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_collective_compression_v82b.json": (
        "c4417708c035198959647a1b3db21dfecf7709b051658d918500793264378e50"
    ),
    "experiments/eggroll_es_hpo/"
    "qwen36_collective_scope_incident_v82b_20260717.md": (
        "949f0a4be2c493a913711af23a52d3379dc8af2d511d6b66c8e2b46ebed7feec"
    ),
    "experiments/eggroll_es_hpo/"
    "qwen36_lora_collective_compression_v82b_cpu_evidence_20260717.md": (
        "ec15fa1d6e3a2fabd7725e1158fa0c5cb013540e01007ca6685fc39c5b3a9e5e"
    ),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_sources() -> dict[str, dict[str, str]]:
    result = {}
    for relative, expected in BOUND_SOURCE_SHA256.items():
        observed = file_sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"V84A bound source changed: {relative}")
        result[relative] = {"file_sha256": observed}
    return result


def build_preregistration() -> dict:
    v82b = json.loads(V82B_PREREG.read_text(encoding="utf-8"))
    records = v82b["canonical_lora_update_scope"]["canonical_master"][
        "ordered_shape_manifest"
    ]
    accounting = contract.byte_rng_accounting_v84a(records)
    body = {
        "schema": "qwen36-lora-scalar-exchange-replay-preregistration-v84a",
        "created_at_utc": "2026-07-17T00:00:00Z",
        "purpose": (
            "prospective comparison of native dense FP32 LoRA all-reduce "
            "against finite seed/coefficient allgather, all-rank digest "
            "consensus, and deterministic full update replay"
        ),
        "status": "prospective_cpu_contract_live_arm_unauthorized",
        "bead": "specialist-0j5.35",
        "bound_sources": bound_sources(),
        "canonical_lora_scope": {
            "v82b_content_sha256": v82b["content_sha256_before_self_field"],
            "ordered_shape_manifest_sha256": v82b[
                "canonical_lora_update_scope"
            ]["canonical_master"]["ordered_shape_manifest_sha256"],
            "tensor_count": contract.TENSOR_COUNT_V84A,
            "module_count": contract.MODULE_COUNT_V84A,
            "elements": contract.TOTAL_ELEMENTS_V84A,
            "fp32_bytes": contract.TOTAL_FP32_BYTES_V84A,
            "world_size": contract.WORLD_SIZE_V84A,
        },
        "wire_and_consensus_contract": {
            "directions": contract.DIRECTIONS_V84A,
            "signed_candidates": contract.SIGNED_CANDIDATES_V84A,
            "antithetic_collapse": (
                "r_plus*epsilon+r_minus*(-epsilon)="
                "(r_plus-r_minus)*epsilon"
            ),
            "pair_wire": "big_endian_uint64_seed_plus_ieee754_binary64_coefficient",
            "pair_bytes": contract.PAIR_BYTES_V84A,
            "global_pair_wire_bytes": (
                contract.DIRECTIONS_V84A * contract.PAIR_BYTES_V84A
            ),
            "global_digest_wire_bytes": (
                contract.WORLD_SIZE_V84A * contract.DIGEST_BYTES_V84A
            ),
            "required_pair_order": "ascending_uint64_seed",
            "required_arithmetic_order": (
                "ascending_origin_rank_then_ascending_seed"
            ),
            "all_rank_digest_consensus_required": True,
            "duplicate_missing_unexpected_nonfinite_rejected": True,
            "negative_zero_coefficient_canonicalized_to_positive_zero": True,
            "resume_retry_identity_binds": [
                "consensus_sha256",
                "original_master_sha256",
                "rank",
                "chunk_elements",
                "rng_algorithm",
                "plan_id",
                "update_sequence",
            ],
        },
        "accounting": accounting,
        "numerical_contract": {
            "fake_four_rank_methods": [
                "iid_absolute_index",
                "structured_outer_product_rank_1",
                "structured_outer_product_rank_4",
                "structured_outer_product_rank_8",
                "structured_outer_product_rank_16",
            ],
            "maximum_allowed_final_update_ulp": 2,
            "global_seed_order_alone_is_not_sufficient": True,
            "reason": (
                "FP32 addition is nonassociative; synthetic global-seed "
                "accumulation exceeded the accepted 2-ULP gate, while "
                "origin-rank-then-seed replay reproduced the registered "
                "fake dense reduction within the gate"
            ),
            "live_collective_reduction_order_still_unproven": True,
        },
        "transaction_contract": {
            "original_master_exact_preflight_and_postflight": True,
            "pending_candidate_not_committed_during_replay": True,
            "provisional_commit_retains_rollback": True,
            "rejection_restores_exact_original": True,
            "partial_failure_discards_pending_and_terminally_poisons": True,
            "stale_retry_and_stale_restore_rejected": True,
            "final_identity_mismatch_terminally_poisons": True,
            "dense_full_noise_allocation_forbidden": True,
            "dense_full_update_scratch_allocation_forbidden": True,
        },
        "prospective_v73d_gate": {
            "accepted_v73d_profile_present": False,
            "dense_collective_ranked_top_three": False,
            "minimum_replicates_with_dense_collective_top_three": 2,
            "required_replicates": 3,
            "live_scalar_arm_authorized": False,
            "paired_live_implementation_or_benchmark_authorized": False,
            "rule": (
                "no live implementation, model load, GPU run, or paired "
                "benchmark unless an accepted V73D profile first ranks the "
                "canonical dense FP32 collective in the top three in at "
                "least two registered replicates"
            ),
        },
        "prospective_close_not_applicable_criteria": [
            (
                "close if accepted V73D does not rank the canonical dense "
                "FP32 collective top-three in at least two of three replicates"
            ),
            (
                "after that prerequisite only, close if measured extra "
                "deterministic replay time is not smaller than removed dense "
                "collective time after both scalar consensus collectives"
            ),
            (
                "close if an implementation cannot retain origin-rank/seed "
                "arithmetic order within the registered 2-ULP gate"
            ),
            (
                "close if cleanup, rollback, poison, retry identity, or "
                "all-rank consensus semantics differ from this contract"
            ),
            (
                "close if measured HBM/RNG overhead negates the memory-"
                "bandwidth benefit; source-equivalent exact-order replay "
                "projects 4x RNG and 724500480 extra explicit HBM bytes per actor"
            ),
        ],
        "decision": {
            "current_path": "retain_native_exact_fp32_v72",
            "scalar_replay_is_prospectively_registered": True,
            "scalar_replay_live_arm": False,
            "quality_claim": False,
            "speed_claim": False,
            "memory_saving_claim": False,
            "promotion_authorized": False,
        },
        "authority": {
            "source_and_synthetic_cpu_only": True,
            "gpu_or_model_opened": False,
            "ray_or_live_communicator_opened": False,
            "dataset_training_eval_ood_holdout_or_probe_opened": False,
            "adapter_update_committed": False,
            "live_arm_executed": False,
            "promotion_authorized": False,
        },
    }
    return {
        **body,
        "content_sha256_before_self_field": contract.canonical_sha256_v84a(body),
    }


def render_report(value: dict) -> str:
    accounting = value["accounting"]
    network = accounting["network_projection"]
    hbm = accounting[
        "explicit_hbm_projection_excluding_rng_and_collective_internals"
    ]
    rng = accounting["rng_and_canonical_work"]
    body = f"""# Qwen3.6 LoRA scalar exchange and deterministic replay V84A

## Decision

Keep the native exact-FP32 V72 update.  V84A registers a source-only scalar
exchange/replay alternative, but authorizes no live implementation or run.
An accepted V73D profile must first rank the canonical dense collective among
the top three bottlenecks in at least two of three replicates.  Even then, a
paired arm is useful only if the removed collective time exceeds the extra
four-rank replay work.

## Exact network projection

The native 70-call, 4,528,128-element FP32 update projects
{network['native_nominal_ring_bus_bytes_per_actor']:,} ring bus bytes per actor.
Eight antithetic directions collapse to eight uint64-seed/binary64-coefficient
pairs.  Their fixed allgather projects {network['pair_allgather_bus_bytes_per_actor']}
bytes per actor; the four SHA-256 digest consensus projects another
{network['digest_allgather_bus_bytes_per_actor']} bytes.  Total scalar traffic is
{network['scalar_total_bus_bytes_per_actor']} bytes per actor, nominally
{network['native_to_scalar_nominal_bus_ratio']:,.0f}x smaller.  These are ring/
allgather projections, not measurements of the canonical communicator.

## RNG, scratch, and HBM tradeoff

Balanced native work generates two directions per actor; replay generates all
eight.  IID normals therefore rise from
{rng['iid_absolute_index:rank=None']['native_balanced_rng_normals_per_actor']:,}
to {rng['iid_absolute_index:rank=None']['replay_rng_normals_per_actor']:,} per
actor.  The same 4x multiplier applies to structured RNG values at ranks 1,
4, 8, and 16.  At 16,384-element chunks, exact-order scratch ranges from
{rng['iid_absolute_index:rank=None']['maximum_streamed_update_scratch_bytes']:,}
bytes for IID to
{rng['structured_outer_product:rank=16']['maximum_streamed_update_scratch_bytes']:,}
bytes for structured rank 16; no full dense noise/update surface is allocated.

The explicit source-equivalent HBM ledger, excluding RNG and collective
internals, is {hbm['native_source_equivalent_bytes_per_actor']:,} bytes per
native actor versus {hbm['replay_source_equivalent_bytes_per_actor']:,} bytes
for exact-order replay, an increment of
{hbm['replay_incremental_bytes_per_actor']:,} bytes.  A fused implementation
could reduce that traffic, but V84A makes no fused-kernel or speed claim.

## Numerical surprise and safety boundary

Globally sorted FP32 accumulation is algebraically correct but exceeded the
accepted two-ULP final-update gate in the synthetic comparison.  The bounded
proof therefore retains each pair's origin rank, sorts seeds canonically for
identity, and replays arithmetic in ascending origin-rank then seed order.
Fake-four-rank tests cover IID and structured ranks 1/4/8/16, all-rank digest
consensus, missing/duplicate/nonfinite rejection, retry identity, provisional
rollback, terminal poison, and finalization.

No model, GPU, Ray, live communicator, dataset, training example, or protected
evaluation source was opened.  No update was committed; quality, speed,
memory-saving, live-arm, and promotion authority are false.

- V84A preregistration content SHA-256:
  `{value['content_sha256_before_self_field']}`
"""
    report_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return body + f"- V84A report body SHA-256: `{report_hash}`\n"


def expected_bytes() -> tuple[bytes, bytes]:
    value = build_preregistration()
    prereg = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report = render_report(value).encode("utf-8")
    return prereg, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.check == args.write:
        parser.error("select exactly one of --check or --write")
    prereg, report = expected_bytes()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(prereg)
        REPORT.write_bytes(report)
        return
    if OUTPUT.read_bytes() != prereg:
        raise RuntimeError("V84A preregistration bytes changed")
    if REPORT.read_bytes() != report:
        raise RuntimeError("V84A report bytes changed")


if __name__ == "__main__":
    main()

