"""Synthetic-only tombstone checks for the quarantined V1 contract."""

import pytest

import recipe_evaluation_contract_v1 as subject


@pytest.mark.parametrize(
    "operation",
    (
        subject.build_contract,
        subject.validate_contract,
        subject.assert_adaptation_inputs,
        subject.charge_compute_attempt,
        subject.aggregate_compute_ledger,
        subject.validate_compute_match,
        subject.claim_protected_access_once,
        subject.load_claimed_protected_rows,
        subject.claim_and_load_protected_rows,
        subject.validate_terminal_aggregate_receipt,
    ),
)
def test_every_v1_entry_point_fails_closed_without_a_source_read(operation):
    with pytest.raises(RuntimeError, match="permanently quarantined"):
        operation()


def test_v1_tombstone_records_exact_quarantined_identity():
    assert subject.QUARANTINED_V1 is True
    assert subject.QUARANTINED_SOURCE_SHA256 == (
        "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b"
    )
    assert subject.QUARANTINED_CONTRACT_CONTENT_SHA256 == (
        "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
    )
