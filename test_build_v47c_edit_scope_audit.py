import json
from pathlib import Path

import build_v47c_edit_scope_audit as subject


def test_v47c_edit_scope_separates_training_from_frozen_shadow():
    value = subject.build()
    scope = value["replay_scope"]
    assert scope["accepted_edit_decisions"] == 51
    assert scope["edited_rows_entering_fold3_train"] == 42
    assert scope["edited_units_entering_fold3_train"] == 34
    assert scope["edited_rows_excluded_in_frozen_fold3_shadow"] == 9
    assert scope["edited_units_excluded_in_frozen_fold3_shadow"] == 3
    assert scope["fold_assignment_changes"] == 0


def test_v47c_eval_policy_keeps_original_targets_and_persists_no_semantics():
    value = subject.build()
    policy = value["evaluation_policy"]
    assert policy["primary_shadow_targets_refreshed_after_training"] is False
    assert policy["refreshed_v430_shadow_authorized_as_selection_input"] is False
    assert value["semantic_access"][
        "manual_question_answer_or_evidence_inspection"
    ] is False
    assert value["semantic_access"]["question_answer_or_evidence_persisted"] is False
    text = json.dumps(value).lower()
    assert "v46d" not in text
    assert "holdout_eval" not in text
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    assert "eval_qa" not in source
    assert "ood_qa" not in source
