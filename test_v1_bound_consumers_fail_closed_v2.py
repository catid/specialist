"""Synthetic-only quarantine checks for historical V1-bound consumers."""

import pytest

import build_fp32_es_optimizer_sigma_preregistration_v1 as fp32
import build_qwen36_collective_compression_preregistration_v82 as collective
import build_qwen36_lora_rank_surface_preregistration_v1 as rank_surface
import build_qwen36_production_layout_decision_v75 as layout
import build_reward_shaping_ablation_preregistration_v1 as reward
import eggroll_es_decode_robustness_v68 as decode
import eggroll_es_front_tail_topology_v70 as topology
import eggroll_es_moe_targeting_v69 as moe
import eggroll_es_multiobjective_trust_region_v67 as trust


@pytest.mark.parametrize(
    "entrypoint,args,pattern",
    (
        (trust.build_preregistration, (), "historical"),
        (trust.validate_preregistration, ({"synthetic": True},), "nonpromotable"),
        (trust.evaluate_trust_region, ({"synthetic": True},), "nonpromotable"),
        (trust.require_promotion, ({"synthetic": True},), "disabled"),
        (decode.build_preregistration, (), "nonpromotable"),
        (decode.validate_preregistration, ({"synthetic": True},), "nonpromotable"),
        (decode.require_selection, ({"synthetic": True},), "disabled"),
        (moe.build_preregistration, (), "nonpromotable"),
        (moe.validate_preregistration, ({"synthetic": True},), "nonpromotable"),
        (moe.require_selection, ({"synthetic": True},), "disabled"),
        (topology.build_preregistration, (), "nonpromotable"),
        (
            topology.validate_preregistration,
            ({"synthetic": True},),
            "nonpromotable",
        ),
        (topology.require_hpo_selection, ({"synthetic": True},), "disabled"),
        (fp32.build_preregistration_v1, (), "historical"),
        (reward.build_preregistration_v1, (), "historical"),
        (collective.build_preregistration_v82, (), "historical"),
        (
            collective.validate_preregistration_v82,
            ({"synthetic": True},),
            "nonpromotable",
        ),
        (rank_surface.validate_upstream_contracts, (), "quarantined"),
        (rank_surface.build_preregistration, (), "nonpromotable"),
        (
            rank_surface.validate_preregistration,
            ({"synthetic": True},),
            "nonpromotable",
        ),
        (layout.build_decision_v75, (), "historical"),
        (layout.validate_decision_v75, ({"synthetic": True},), "nonpromotable"),
    ),
)
def test_historical_consumer_entrypoint_fails_before_any_input_read(
    entrypoint, args, pattern
):
    with pytest.raises(RuntimeError, match=pattern):
        entrypoint(*args)
