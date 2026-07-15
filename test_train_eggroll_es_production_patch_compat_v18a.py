#!/usr/bin/env python3
"""Offline tests for the pure V18A production-patch trainer mechanics."""

import copy
import json
import math

import numpy as np
import pytest

import train_eggroll_es_production_patch_compat_v18a as trainer_v18a


@pytest.fixture(scope="module")
def panel_bundle_v18a():
    return trainer_v18a.load_patch_panel_bundle_v18a()


def empty_unit_scores_v18a():
    return {
        arm: np.zeros(
            (
                5,
                2,
                32,
                trainer_v18a.frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm],
            ),
            dtype=np.float64,
        )
        for arm in trainer_v18a.ARMS_V18A
    }


def base_only_unit_scores_v18a(panel_bundle):
    scores = empty_unit_scores_v18a()
    directions = np.arange(32, dtype=np.float64) - 15.5
    for arm in trainer_v18a.ARMS_V18A:
        for panel_index, panel_name in enumerate(trainer_v18a.PANEL_NAMES_V18A):
            batch = panel_bundle["panels"][panel_name]["arms"][arm]
            base_positions = [
                index
                for index, stratum in enumerate(batch["ht_strata"])
                if stratum in trainer_v18a.BASE_CATEGORIES_V18A
            ]
            response = directions * (1.0 + 0.03 * panel_index)
            response += 0.02 * np.sin(
                directions * (panel_index + 1.0) / 7.0
            )
            scores[arm][panel_index, 0][:, base_positions] = response[:, None]
            scores[arm][panel_index, 1][:, base_positions] = -response[:, None]
    return scores


def reseal_v18a(value):
    value["content_sha256_before_self_field"] = trainer_v18a.canonical_sha256({
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    return value


def test_v18a_materializes_exact_variable_patch_batches(panel_bundle_v18a):
    bundle = panel_bundle_v18a
    assert bundle == trainer_v18a.load_patch_panel_bundle_v18a()
    assert bundle["content_sha256_before_self_field"] == (
        trainer_v18a.PANEL_BUNDLE_CONTENT_SHA256_V18A
    )
    assert bundle["contains_evaluation_content"] is False
    assert bundle["preregistration"]["corrected_commit"] == (
        trainer_v18a.CORRECTED_PREREG_COMMIT_V18A
    )
    expected_rows = {
        "production_only": 52,
        "patch_one_third": 53,
        "patch_two_thirds": 54,
        "patch_full": 55,
    }
    maximum_layers = dict(zip(trainer_v18a.ARMS_V18A, range(4)))
    full_joint_ids = []
    for panel in bundle["panels"].values():
        assert set(panel["arms"]) == set(trainer_v18a.ARMS_V18A)
        base_joint_ids = set(panel["arms"]["production_only"]["joint_ids"])
        for arm in trainer_v18a.ARMS_V18A:
            batch = panel["arms"][arm]
            assert len(batch["joint_ids"]) == expected_rows[arm]
            assert len(set(batch["joint_ids"])) == expected_rows[arm]
            assert base_joint_ids.issubset(batch["joint_ids"])
            assert math.isclose(
                math.fsum(batch["weights"]),
                trainer_v18a.frame_v18a.ARM_POPULATIONS_V18A[arm],
                abs_tol=1e-12,
            )
            for category in trainer_v18a.BASE_CATEGORIES_V18A:
                assert batch["ht_strata"].count(category) == 13
            for layer in (1, 2, 3):
                assert batch["ht_strata"].count(
                    f"candidate_only_layer_{layer}"
                ) == int(layer <= maximum_layers[arm])
        full_joint_ids.extend(panel["arms"]["patch_full"]["joint_ids"])
    assert len(full_joint_ids) == 275
    assert len(set(full_joint_ids)) == 275

    tampered = copy.deepcopy(bundle)
    tampered["panels"]["optimization_0"]["arms"]["patch_full"][
        "weights"
    ][0] += 1e-6
    reseal_v18a(tampered)
    with pytest.raises(RuntimeError, match="bundle changed"):
        trainer_v18a.validate_patch_panel_bundle_v18a(tampered)


def test_v18a_schedule_balances_four_arms_and_restores_once():
    schedule = trainer_v18a.resident_signed_wave_schedule_v18a()
    assert len(schedule) == 16
    assert [item["sign"] for item in schedule] == ["plus", "minus"] * 8
    for sign in trainer_v18a.SIGNS_V18A:
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in trainer_v18a.ARMS_V18A:
            assert sorted(
                item["resident_arm_order"].index(arm) for item in signed
            ) == [0, 0, 1, 1, 2, 2, 3, 3]

    events = []
    item = schedule[3]
    captures = trainer_v18a.execute_patch_resident_signed_wave_v18a(
        item,
        perturb=lambda seeds, negate: events.append(("perturb", seeds, negate)),
        score_arm=lambda arm: events.append(("score", arm)) or arm,
        restore=lambda: events.append(("restore",)),
    )
    assert events == [
        ("perturb", item["engine_seeds"], True),
        *(("score", arm) for arm in item["resident_arm_order"]),
        ("restore",),
    ]
    assert tuple(captures) == tuple(item["resident_arm_order"])

    failed_events = []
    failing_arm = schedule[0]["resident_arm_order"][1]
    with pytest.raises(RuntimeError, match="synthetic arm failure"):
        trainer_v18a.execute_patch_resident_signed_wave_v18a(
            schedule[0],
            perturb=lambda _seeds, _negate: failed_events.append("perturb"),
            score_arm=lambda arm: (
                (_ for _ in ()).throw(RuntimeError("synthetic arm failure"))
                if arm == failing_arm
                else failed_events.append(arm)
            ),
            restore=lambda: failed_events.append("restore"),
        )
    assert failed_events == [
        "perturb", schedule[0]["resident_arm_order"][0], "restore"
    ]
    with pytest.raises(RuntimeError, match="schedule item changed"):
        trainer_v18a.execute_patch_resident_signed_wave_v18a(
            None, perturb=lambda *_: None, score_arm=lambda *_: None,
            restore=lambda: None,
        )


def test_v18a_ht_uses_exact_arm_weights_and_denominators(panel_bundle_v18a):
    scores = empty_unit_scores_v18a()
    for arm_index, arm in enumerate(trainer_v18a.ARMS_V18A):
        values = scores[arm]
        for panel_index in range(5):
            for sign_index in range(2):
                for direction_index in range(32):
                    values[panel_index, sign_index, direction_index] = (
                        1000.0 * arm_index
                        + 100.0 * panel_index
                        + 10.0 * sign_index
                        + direction_index
                        + np.arange(values.shape[-1]) / 1000.0
                    )
    observed = trainer_v18a.observed_panel_scores_v18a(
        scores, panel_bundle_v18a
    )
    for arm_index, arm in enumerate(trainer_v18a.ARMS_V18A):
        weights = np.asarray([
            panel_bundle_v18a["panels"][panel]["arms"][arm]["weights"]
            for panel in trainer_v18a.PANEL_NAMES_V18A
        ])
        expected = np.einsum("psdu,pu->psd", scores[arm], weights) / (
            trainer_v18a.frame_v18a.ARM_POPULATIONS_V18A[arm]
        )
        np.testing.assert_allclose(
            observed[arm_index], expected, rtol=0.0, atol=1e-12
        )


def test_v18a_bootstrap_fixes_base_panels_and_shares_q1_role_draws(
    panel_bundle_v18a,
):
    scores = empty_unit_scores_v18a()
    for arm in trainer_v18a.ARMS_V18A:
        for panel_index, panel_name in enumerate(trainer_v18a.PANEL_NAMES_V18A):
            strata = panel_bundle_v18a["panels"][panel_name]["arms"][arm][
                "ht_strata"
            ]
            base_positions = [
                index
                for index, stratum in enumerate(strata)
                if stratum in trainer_v18a.BASE_CATEGORIES_V18A
            ]
            scores[arm][panel_index, :, :, base_positions] = 10 + panel_index
    bootstrapped = trainer_v18a._bootstrap_panel_scores_v18a(
        scores,
        panel_bundle_v18a,
        np.random.default_rng(12345),
        64,
    )
    for arm_index, arm in enumerate(trainer_v18a.ARMS_V18A):
        denominator = trainer_v18a.frame_v18a.ARM_POPULATIONS_V18A[arm]
        for panel_index in range(5):
            np.testing.assert_allclose(
                bootstrapped[arm_index, :, panel_index],
                (10 + panel_index) * 272.0 / denominator,
                rtol=0.0,
                atol=1e-12,
            )

    scores = empty_unit_scores_v18a()
    for arm in trainer_v18a.ARMS_V18A[1:]:
        for panel_index, panel_name in enumerate(trainer_v18a.PANEL_NAMES_V18A):
            strata = panel_bundle_v18a["panels"][panel_name]["arms"][arm][
                "ht_strata"
            ]
            layer_one = strata.index("candidate_only_layer_1")
            scores[arm][panel_index, :, :, layer_one] = 100 + panel_index
    bootstrapped = trainer_v18a._bootstrap_panel_scores_v18a(
        scores,
        panel_bundle_v18a,
        np.random.default_rng(54321),
        512,
    )
    one_third = bootstrapped[1] * 280.0 / 8.0
    two_thirds = bootstrapped[2] * 287.0 / 8.0
    full = bootstrapped[3] * 295.0 / 8.0
    np.testing.assert_allclose(one_third, two_thirds, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(one_third, full, rtol=0.0, atol=1e-12)
    assert set(np.unique(np.rint(one_third[:, :3]).astype(int))) == {
        100, 101, 102,
    }
    assert set(np.unique(np.rint(one_third[:, 3:]).astype(int))) == {103, 104}
    assert np.count_nonzero(bootstrapped[0]) == 0


def test_v18a_exact_50k_bootstrap_36_gates_and_compact_output(
    panel_bundle_v18a,
):
    scores = base_only_unit_scores_v18a(panel_bundle_v18a)
    compact = trainer_v18a.build_compact_estimator_summary_v18a(
        scores, panel_bundle_v18a
    )
    bootstrap = compact["paired_bootstrap"]
    assert bootstrap["seed"] == trainer_v18a.prereg_v18a.BOOTSTRAP_SEED_V18A
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 36
    assert sum(
        len(endpoints)
        for endpoints in bootstrap["comparisons"].values()
    ) == 36
    for endpoints in bootstrap["comparisons"].values():
        assert set(endpoints) == set(trainer_v18a.prereg_v18a.ENDPOINT_NAMES_V18A)
        for endpoint in endpoints.values():
            assert endpoint["patch_minus_production"] == pytest.approx(
                0.0, abs=1e-12
            )
            assert endpoint["familywise_lcb"] == pytest.approx(0.0, abs=1e-12)
            assert endpoint["noninferiority_margin"] == 0.0
    serialized = json.dumps(compact, sort_keys=True)
    for forbidden in (
        "unit_scores", "replicates", "questions", "answers", "row_sha256",
        "joint_ids", "bootstrap_draws",
    ):
        assert forbidden not in serialized
    assert compact["persisted_response_vectors_or_row_content"] is False

    passing = copy.deepcopy(compact)
    passing["runtime_integrity"] = {"all_integrity_audits_passed": True}
    for endpoints in passing["paired_bootstrap"]["comparisons"].values():
        for endpoint in endpoints.values():
            endpoint["patch_minus_production"] = 0.1
            endpoint["familywise_lcb"] = 0.01
    gate = trainer_v18a.evaluate_patch_gate_v18a(passing)
    assert gate["selected_largest_passing_patch"] == "patch_full"
    assert gate["arms"]["patch_full"]["observed_pass_count"] == 12
    assert gate["arms"]["patch_full"]["bootstrap_pass_count"] == 12
    assert gate["dataset_promotion_authorized"] is False
    assert gate["model_update_authorized"] is False
    assert gate["evaluation_authorized"] is False


def test_v18a_pure_trainer_exposes_no_launch_update_or_eval_entrypoint():
    for name in (
        "main", "run", "load_trainer", "make_trainer",
        "apply_seed_coefficients", "train_step", "fit", "eval_step",
        "save_checkpoint",
    ):
        assert not hasattr(trainer_v18a, name)
