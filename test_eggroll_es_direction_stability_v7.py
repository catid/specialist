import copy
import json
from pathlib import Path

import pytest

import run_eggroll_es_anchor_stability_v7 as stability_v7
import train_eggroll_es_specialist_anchor_v7 as anchor_v7


ROOT = Path(__file__).resolve().parent
FRONT_PLAN_SHA = (
    "af9dcf4e5c932aeb192ee0d195e9c4fee9d4a510467d850d7cf26f6db4c2d823"
)


def load_front_bundle():
    spec = anchor_v7.FROZEN_STABILITY_PLANS_V7[FRONT_PLAN_SHA]
    return anchor_v7.load_frozen_layer_plan_v7(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=FRONT_PLAN_SHA,
        expected_model_config_sha256=anchor_v7.MODEL_CONFIG_SHA256_V7,
    )


def front_cli(seed=43, target="0", dry_run=True):
    spec = anchor_v7.FROZEN_STABILITY_PLANS_V7[FRONT_PLAN_SHA]
    args = [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", FRONT_PLAN_SHA,
        "--expected-model-config-sha256", anchor_v7.MODEL_CONFIG_SHA256_V7,
        "--v7-stage", "stability",
        "--v7-smoke-gate-json", str(stability_v7.SMOKE_GATE_PATH_V7),
        "--v7-pilot-family-json", str(stability_v7.PILOT_EVIDENCE_PATH_V7),
        "--population-size", "16", "--batch-size", "64",
        "--seed", str(seed), "--target-alphas", target,
        "--experiment-name",
        f"snapshot794_layer_v7_front_stability_seed{seed}",
    ]
    if dry_run:
        args.append("--v7-dry-run")
    return args


def synthetic_completed_journal():
    source = ROOT / (
        "experiments/eggroll_es_hpo/runs/"
        "snapshot794_layer_v6_front_pilot_seed42/alpha_line_search.json"
    )
    journal = json.loads(source.read_text())
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v7"
    journal["targets"] = [0.0]
    journal["states"] = [journal["states"][0]]
    for key in ("edge_split_family_v6", "pilot_requires_four_smokes_v6"):
        journal["policy"].pop(key)
    journal["policy"].update({
        "direction_stability_family_v7": "front_middle_late_cross_seed",
        "stage_v7": "stability", "target_alpha_zero_only_v7": True,
        "benchmark_selection_forbidden_v7": True,
        "cross_seed_coefficient_cosine_threshold_v7": 0.5,
        "requires_clean_v6_family_v7": True,
    })
    snapshot = journal["snapshot"]
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v7"
    snapshot.pop("edge_split_v6")
    implementation = {
        key: anchor_v7.file_sha256(path)
        for key, path in stability_v7.V7_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"] = implementation
    smoke = stability_v7._clean_smoke_evidence_v7(
        stability_v7.SMOKE_GATE_PATH_V7
    )
    pilots = stability_v7._pilot_family_evidence_v7(
        stability_v7.PILOT_EVIDENCE_PATH_V7
    )
    recipe = stability_v7.frozen_recipe_v7("front", 43, smoke, pilots)
    snapshot["recipe"] = {
        key: recipe[key] for key in (
            "model_name", "checkpoint", "sigma", "population_size",
            "batch_size", "mini_batch_size", "max_tokens", "seed",
            "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
        )
    }
    spec = anchor_v7.FROZEN_STABILITY_PLANS_V7[FRONT_PLAN_SHA]
    snapshot["direction_stability_v7"] = {
        "schema": "eggroll-es-direction-stability-snapshot-v7",
        "family": "front_middle_late_cross_seed", "stage": "stability",
        "arm": "front", "layers": spec["layers"], "seed_pair": [43, 44],
        "seed": 43, "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_direction_only",
        "coefficient_cosine_threshold": 0.5, "recipe": recipe,
        "plan_sha256": FRONT_PLAN_SHA,
        "plan_file_sha256": spec["file_sha256"],
        "implementation_bundle_sha256": (
            stability_v7.driver_v1.canonical_sha256(implementation)
        ),
    }
    journal.pop("content_sha256_before_self_field")
    journal["content_sha256_before_self_field"] = (
        stability_v7.driver_v1.canonical_sha256(journal)
    )
    return journal


def test_v7_only_allows_front_and_middle_late():
    assert {
        item["plan"] for item in anchor_v7.FROZEN_STABILITY_PLANS_V7.values()
    } == {"front", "middle_late"}
    assert anchor_v7.validate_frozen_layer_plan_bundle_v7(
        load_front_bundle()
    )["plan"] == "front"


def test_v7_cli_dry_run_and_rejects_selection_alpha(capsys):
    result = stability_v7.main(front_cli())
    assert result == {
        "schema": "eggroll-es-direction-stability-dry-run-v7",
        "arm": "front", "seed": 43, "stage": "stability",
        "targets": [0.0], "recipe_sha256": (
            "f76fe671065123f25efe1d5ccc2079096cfe29589f10a6b1c3843b48545cedcd"
        ),
    }
    assert "direction-stability-dry-run-v7" in capsys.readouterr().out
    bundle, remaining = anchor_v7.parse_frozen_layer_plan_cli_v7(
        front_cli(target="0,0.000001")
    )
    with pytest.raises(ValueError, match="exactly alpha zero"):
        stability_v7.validate_frozen_execution_cli_v7(remaining, bundle)


def test_v7_completed_contract_is_fail_closed(monkeypatch):
    journal = synthetic_completed_journal()
    monkeypatch.setattr(
        stability_v7, "_validate_inherited_zero_target_v7",
        lambda value: {
            "seed": 43, "state_count": 1,
            "coefficient_sha256": value["coefficient_plan"][
                "coefficient_sha256"
            ],
            "robust_plan_sha256": value["coefficient_plan"][
                "robust_plan_binding_v5"
            ]["robust_plan_sha256"],
            "distributed_update_v4": {},
        },
    )
    audit = stability_v7.validate_completed_journal_v7(journal)
    assert (audit["arm"], audit["seed"], audit["state_count"]) == (
        "front", 43, 1,
    )
    tampered = copy.deepcopy(journal)
    tampered["snapshot"]["direction_stability_v7"][
        "benchmark_treatment_applied"
    ] = True
    tampered["content_sha256_before_self_field"] = (
        stability_v7.driver_v1.canonical_sha256({
            key: value for key, value in tampered.items()
            if key != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="stability snapshot changed"):
        stability_v7.validate_completed_journal_v7(tampered)


def test_v7_keeps_extended_arm_audit_outside_exact_v5_binding():
    source = ROOT / (
        "experiments/eggroll_es_hpo/runs/"
        "snapshot794_layer_v6_front_pilot_seed42/alpha_line_search.json"
    )
    journal = json.loads(source.read_text())
    plan = stability_v7.driver_v5._journal_plan_v5(journal)
    binding_v7 = anchor_v7.validate_robust_plan_v7(
        plan, recompute_numeric=True,
    )
    binding_v5 = anchor_v7.anchor_v6.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    assert binding_v7["direction_stability_arm_v7"] == "front"
    assert binding_v7["edge_split_arm_v6"] == "front"
    assert "direction_stability_arm_v7" not in binding_v5
    assert "edge_split_arm_v6" not in binding_v5
    assert plan["robust_plan_binding_v5"] == binding_v5
