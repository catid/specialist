import copy

import pytest

import build_qwen36_tp1_process_group_cleanup_v83 as subject


def test_explicit_future_receipt_accepts_only_two_safe_states():
    never = subject.validate_explicit_cleanup_v83({
        "process_group_initialized_before_shutdown": False,
        "destroy_process_group_attempted": False,
        "process_group_initialized_after_shutdown": False,
    })
    initialized = subject.validate_explicit_cleanup_v83({
        "process_group_initialized_before_shutdown": True,
        "destroy_process_group_attempted": True,
        "process_group_initialized_after_shutdown": False,
    })
    assert never["cleanup_semantics_passed"] is True
    assert initialized["cleanup_semantics_passed"] is True


@pytest.mark.parametrize(
    "value",
    [
        {},
        {
            "process_group_initialized_before_shutdown": False,
            "destroy_process_group_attempted": True,
            "process_group_initialized_after_shutdown": False,
        },
        {
            "process_group_initialized_before_shutdown": True,
            "destroy_process_group_attempted": False,
            "process_group_initialized_after_shutdown": False,
        },
        {
            "process_group_initialized_before_shutdown": True,
            "destroy_process_group_attempted": True,
            "process_group_initialized_after_shutdown": True,
        },
    ],
)
def test_explicit_future_receipt_rejects_missing_or_leaked_states(value):
    with pytest.raises((ValueError, RuntimeError)):
        subject.validate_explicit_cleanup_v83(value)


def test_v83_binds_three_runs_and_preserves_immutable_failure():
    value = subject.build_v83()
    assert set(value["runs"]) == set(subject.RUN_BUNDLES)
    assert value["immutable_parent_result"][
        "literal_torch_process_group_destroyed_clause_passed"
    ] is False
    assert value["immutable_parent_result"]["result_rewritten"] is False
    assert value["additive_v83_verdict"]["cleanup_semantics_passed"] is True
    for run in value["runs"].values():
        assert len(run["actor_receipts"]) == 4
        assert all(
            row["source_bound_interpretation"]["destroy_was_required"] is False
            for row in run["actor_receipts"]
        )
        assert len(run["external_cleanup"]["accepted_final_batches"]) == 3


def test_v83_is_self_hashed_and_never_authorizes_promotion():
    value = subject.build_v83()
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert subject.canonical_sha256_v83(body) == claimed
    assert value["authority"]["dataset_or_protected_content_opened"] is False
    assert value["authority"]["checkpoint_or_layout_promotion_authorized"] is False
    assert value["additive_v83_verdict"][
        "semantic_ood_and_promotion_gates_still_pending"
    ] is True
