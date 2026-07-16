#!/usr/bin/env python3

import numpy as np

import lora_es_f1_anchor_projection_v43m as subject


def test_v43m_reconstructs_existing_objectives_and_adds_f1_anchor():
    value = subject.build_projection_v43m()
    assert value["source"]["population_resampled"] is False
    assert value["source"]["population_rescored"] is False
    assert value["seeds"] == list(subject.v43i.SEEDS)
    assert value["required_centered_rank_anchors"] == [
        "prose_lm", "qa_answer_logprob", "qa_generation_mean_f1",
    ]
    assert value["objective_fitness"]["qa_generation_mean_f1"][
        "zero_spread"
    ] is False
    assert value["qa_generation_mean_f1_is_projection_constraint"] is True


def test_v43m_projection_satisfies_all_three_halfspaces_and_trust_region():
    value = subject.build_projection_v43m()
    coefficients = np.asarray(value["projection"]["coefficients"])
    objectives = value["objective_fitness"]
    for anchor in value["required_centered_rank_anchors"]:
        assert float(np.dot(
            coefficients, np.asarray(objectives[anchor]["coefficients"])
        )) >= -1e-12
    diagnostics = value["projection"]["diagnostics"]
    assert diagnostics["all_anchor_halfspaces_satisfied"] is True
    assert diagnostics["update_norm_ratio"] <= 0.5 + 1e-12
    assert value["shadow_ood_holdout_or_heldout_opened"] is False
    assert value["current_fixed_holdout_cycle_eligible"] is False


def test_v43m_is_exact_noop_against_already_rejected_v43i_candidate():
    value = subject.build_identity_noop_v43m()
    basis = value["identity_basis"]
    assert basis["coefficient_lists_equal_exactly"] is True
    assert basis["same_coefficient_sha256"] == (
        subject.IDENTICAL_COEFFICIENT_SHA256
    )
    assert basis["deterministic_scaled_tensor_update_equal_exactly"] is True
    assert value["already_evaluated_state"]["candidate_state_sha256"] == (
        subject.V43I_CANDIDATE_STATE_SHA256
    )
    assert value["already_evaluated_state"]["candidate_gate_passed"] is False
    assert value["decision"] == "do_not_launch_duplicate_gpu_evaluation"
    assert value["gpu_model_or_dataset_accessed"] is False
