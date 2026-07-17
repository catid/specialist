from __future__ import annotations

import copy

import pytest

import build_curriculum_ablation_preregistration_v1 as builder
import curriculum_ablation_v1 as curriculum


def reseal(plan):
    plan = copy.deepcopy(plan)
    plan.pop("content_sha256_before_self_field", None)
    plan["content_sha256_before_self_field"] = curriculum.canonical_sha256_v1(plan)
    return plan


def reseal_dataset(plan, stage_id):
    plan = copy.deepcopy(plan)
    section = plan["datasets"][stage_id]
    old = section.pop("content_sha256")
    new = curriculum.canonical_sha256_v1(section)
    section["content_sha256"] = new
    for arm in plan["arms"]:
        for stage in arm["stages"]:
            if stage["stage_id"] == stage_id:
                assert stage["dataset_content_sha256"] == old
                stage["dataset_content_sha256"] = new
    required = plan["source_disjoint_extension"][
        "required_new_input_content_sha256s"
    ]
    required[required.index(old)] = new
    required.sort()
    return reseal(plan)


def launch_ready(plan):
    plan = copy.deepcopy(plan)
    extension = plan["source_disjoint_extension"]
    extension.update({
        "status": "passed",
        "audit_file_sha256": "b" * 64,
        "audit_content_sha256": "c" * 64,
        "cross_role_collision_counts": {
            domain: 0 for domain in curriculum.IDENTITY_DOMAINS_V1
        },
        "audit_passed": True,
        "launch_eligible": True,
    })
    plan["authorization"]["gpu_launch"] = True
    plan["authorization"]["new_adaptation_inputs"] = True
    plan["status"] = "sealed_launch_ready"
    return reseal(plan)


def state_sha(*parts):
    return curriculum.canonical_sha256_v1(list(parts))


def valid_receipts(plan):
    initial = plan["checkpoint_contract"]["initial_adapter_state_sha256"]
    receipts = []
    for arm in plan["arms"]:
        checkpoint = initial
        stage_receipts = []
        for stage in arm["stages"]:
            updates = stage["optimizer_updates"]
            midpoint = updates // 2
            intermediate = state_sha(arm["arm_id"], stage["stage_id"], "mid")
            output = state_sha(arm["arm_id"], stage["stage_id"], "final")
            fresh_state = state_sha(arm["arm_id"], stage["stage_id"], "fresh")
            midpoint_state = state_sha(arm["arm_id"], stage["stage_id"], "state-mid")
            final_state = state_sha(arm["arm_id"], stage["stage_id"], "state-final")
            segments = [
                {
                    "segment_index": 0,
                    "input_checkpoint_sha256": checkpoint,
                    "output_checkpoint_sha256": intermediate,
                    "input_training_state_sha256": fresh_state,
                    "output_training_state_sha256": midpoint_state,
                    "starting_update": 0,
                    "ending_update": midpoint,
                },
                {
                    "segment_index": 1,
                    "input_checkpoint_sha256": intermediate,
                    "output_checkpoint_sha256": output,
                    "input_training_state_sha256": midpoint_state,
                    "output_training_state_sha256": final_state,
                    "starting_update": midpoint,
                    "ending_update": updates,
                },
            ]
            stage_receipts.append({
                "stage_id": stage["stage_id"],
                "input_checkpoint_sha256": checkpoint,
                "output_checkpoint_sha256": output,
                "dataset_content_sha256": stage["dataset_content_sha256"],
                "optimizer_updates": updates,
                "nonpadding_tokens": stage["nonpadding_tokens"],
                "generated_rollouts": stage["generated_rollouts"],
                "charged_gpu_seconds": stage["target_gpu_seconds"],
                "resume_segments": segments,
            })
            checkpoint = output
        receipts.append({
            "schema": curriculum.RECEIPT_SCHEMA_V1,
            "arm_id": arm["arm_id"],
            "stage_receipts": stage_receipts,
            "total_charged_gpu_seconds": arm["total_target_gpu_seconds"],
            "useful_physical_gpu_ids": [0, 1, 2, 3],
            "protected_holdout_opened": False,
            "final_checkpoint_sha256": checkpoint,
        })
    return receipts


@pytest.fixture(scope="module")
def plan():
    return builder.build_preregistration_v1()


def test_preview_is_sealed_but_fails_closed_for_launch(plan):
    assert curriculum.validate_plan_v1(plan)["status"] == (
        "sealed_cpu_preview_launch_ineligible"
    )
    with pytest.raises(RuntimeError, match="fresh four-domain source-disjoint audit"):
        curriculum.validate_plan_v1(plan, require_launch_ready=True)


def test_stage_inventories_and_caps_are_exact(plan):
    cpt, sft, es = (plan["datasets"][stage] for stage in ("cpt", "sft", "es"))
    assert len(cpt["units"]) == 29
    assert sum(unit["selected_tokens"] for unit in cpt["units"]) == 99_216
    assert max(unit["selected_tokens"] for unit in cpt["units"]) == 4_845
    assert len(sft["units"]) == 433
    assert sft["exclusion_reasons"] == {"canonical_url_trivia": 15}
    assert sft["base_pass_nonpadding_tokens"] == 33_072
    assert len(es["items"]) == 64
    assert max(
        sum(item["source"] == source for item in es["items"])
        for source in {item["source"] for item in es["items"]}
    ) <= 15


def test_cpt_rejects_assistant_stage_leakage(plan):
    changed = copy.deepcopy(plan)
    changed["datasets"]["cpt"]["units"][0][
        "training_format"
    ] = "masked_prompt_qa_chat"
    changed["datasets"]["cpt"]["units"][0]["assistant_supervision"] = True
    with pytest.raises(ValueError, match="unit schema leaked|eligibility or token cap"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "cpt"))


def test_sft_rejects_raw_document_disguised_as_answer(plan):
    changed = copy.deepcopy(plan)
    changed["datasets"]["sft"]["units"][0]["raw_markdown"] = True
    changed["datasets"]["sft"]["units"][0][
        "training_format"
    ] = "causal_next_token_markdown"
    with pytest.raises(ValueError, match="verification or stage role"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "sft"))


def test_cpt_duplicate_source_document_fails_even_if_totals_are_rebalanced(plan):
    changed = copy.deepcopy(plan)
    units = changed["datasets"]["cpt"]["units"]
    units[1]["source_document_identity_sha256"] = units[0][
        "source_document_identity_sha256"
    ]
    with pytest.raises(ValueError, match="duplicated a source unit"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "cpt"))


def test_sft_duplicate_fact_or_row_fails(plan):
    changed = copy.deepcopy(plan)
    units = changed["datasets"]["sft"]["units"]
    units[1]["fact_id"] = units[0]["fact_id"]
    with pytest.raises(ValueError, match="duplicated a source unit"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "sft"))


def test_url_trivia_cannot_reenter_sft(plan):
    changed = copy.deepcopy(plan)
    unit = changed["datasets"]["sft"]["units"][0]
    unit["url_trivia"] = True
    unit["kind"] = "qa_resource_index"
    with pytest.raises(ValueError, match="verification or stage role"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "sft"))


def test_rights_gap_cannot_enter_cpt(plan):
    changed = copy.deepcopy(plan)
    changed["datasets"]["cpt"]["units"][0][
        "rights_status"
    ] = "legacy_manifest_gap"
    changed["datasets"]["cpt"]["units"][0][
        "promotion_gate"
    ] = "rights_review_required_before_new_snapshot"
    with pytest.raises(ValueError, match="eligibility or token cap"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "cpt"))


def test_cpt_source_token_cap_is_enforced(plan):
    changed = copy.deepcopy(plan)
    units = changed["datasets"]["cpt"]["units"]
    unit = next(item for item in units if item["available_tokens"] > 4_845)
    unit["selected_tokens"] += 1
    unit["token_stop"] += 1
    changed["datasets"]["cpt"]["selected_nonpadding_tokens"] += 1
    changed["datasets"]["cpt"]["target_nonpadding_tokens"] += 1
    with pytest.raises(ValueError, match="eligibility or token cap|stage contract"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "cpt"))


def test_es_duplicate_conflict_unit_fails(plan):
    changed = copy.deepcopy(plan)
    items = changed["datasets"]["es"]["items"]
    items[1]["unit_identity_sha256"] = items[0]["unit_identity_sha256"]
    with pytest.raises(ValueError, match="duplicated a source unit"):
        curriculum.validate_plan_v1(reseal_dataset(changed, "es"))


def test_unequal_native_token_or_rollout_budget_fails(plan):
    changed = copy.deepcopy(plan)
    changed["arms"][1]["stages"][0]["nonpadding_tokens"] += 1
    with pytest.raises(ValueError, match="native token/update/rollout budget"):
        curriculum.validate_plan_v1(reseal(changed))

    changed = copy.deepcopy(plan)
    changed["arms"][3]["stages"][0]["generated_rollouts"] -= 1
    with pytest.raises(ValueError, match="native token/update/rollout budget"):
        curriculum.validate_plan_v1(reseal(changed))


def test_unequal_total_gpu_budget_fails(plan):
    changed = copy.deepcopy(plan)
    changed["arms"][2]["stages"][0]["target_gpu_seconds"] -= 1
    with pytest.raises(ValueError, match="equal total compute"):
        curriculum.validate_plan_v1(reseal(changed))


def test_plan_protected_holdout_authorization_fails(plan):
    changed = copy.deepcopy(plan)
    changed["authorization"]["protected_holdout_access"] = True
    with pytest.raises(ValueError, match="authorization"):
        curriculum.validate_plan_v1(reseal(changed))


def test_tampered_plan_identity_fails(plan):
    changed = copy.deepcopy(plan)
    changed["purpose"] += " tampered"
    with pytest.raises(RuntimeError, match="plan content identity"):
        curriculum.validate_plan_v1(changed)


def test_synthetic_passed_extension_makes_contract_launch_ready(plan):
    ready = launch_ready(plan)
    assert curriculum.validate_plan_v1(
        ready, require_launch_ready=True
    )["status"] == "sealed_launch_ready"


def test_valid_execution_receipts_cover_native_work_compute_and_resume(plan):
    ready = launch_ready(plan)
    result = curriculum.validate_execution_receipts_v1(
        ready, valid_receipts(ready)
    )
    assert result["all_native_budgets_exact"] is True
    assert result["all_checkpoint_and_resume_chains_exact"] is True
    assert result["protected_holdout_opened"] is False


def test_cross_stage_checkpoint_drift_fails(plan):
    ready = launch_ready(plan)
    receipts = valid_receipts(ready)
    receipts[0]["stage_receipts"][1]["input_checkpoint_sha256"] = "d" * 64
    receipts[0]["stage_receipts"][1]["resume_segments"][0][
        "input_checkpoint_sha256"
    ] = "d" * 64
    with pytest.raises(RuntimeError, match="checkpoint chain drifted"):
        curriculum.validate_execution_receipts_v1(ready, receipts)


def test_same_stage_resume_training_state_drift_fails(plan):
    ready = launch_ready(plan)
    receipts = valid_receipts(ready)
    receipts[0]["stage_receipts"][0]["resume_segments"][1][
        "input_training_state_sha256"
    ] = "e" * 64
    with pytest.raises(RuntimeError, match="resume state drifted"):
        curriculum.validate_execution_receipts_v1(ready, receipts)


def test_execution_native_budget_drift_fails(plan):
    ready = launch_ready(plan)
    receipts = valid_receipts(ready)
    receipts[1]["stage_receipts"][0]["nonpadding_tokens"] -= 1
    with pytest.raises(ValueError, match="nonpadding_tokens budget drifted"):
        curriculum.validate_execution_receipts_v1(ready, receipts)


def test_execution_gpu_or_protected_access_drift_fails(plan):
    ready = launch_ready(plan)
    receipts = valid_receipts(ready)
    receipts[2]["useful_physical_gpu_ids"] = [0, 1, 2]
    with pytest.raises(ValueError, match="access/GPU coverage"):
        curriculum.validate_execution_receipts_v1(ready, receipts)

    receipts = valid_receipts(ready)
    receipts[2]["protected_holdout_opened"] = True
    with pytest.raises(ValueError, match="access/GPU coverage"):
        curriculum.validate_execution_receipts_v1(ready, receipts)


def test_execution_compute_mismatch_fails(plan):
    ready = launch_ready(plan)
    receipts = valid_receipts(ready)
    receipts[3]["stage_receipts"][0]["charged_gpu_seconds"] = 14_000
    receipts[3]["total_charged_gpu_seconds"] = 14_000
    with pytest.raises(ValueError, match="stage GPU-second budget"):
        curriculum.validate_execution_receipts_v1(ready, receipts)
