#!/usr/bin/env python3
"""Permanent fail-closed tombstone for the quarantined V1 boundary.

The V1 source was irreversibly accessed on 2026-07-17.  Its complete 59-row
source and all 18 legacy heldout candidates are quarantined.  This module must
never rebuild, validate, claim, load, score, or provide compute authorization
for a V1 manifest.  Use recipe_evaluation_contract_v2 for compatible work.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
TRAIN = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/train_v434_fold3_v49d.jsonl"
).resolve()
QUARANTINED_SOURCE = (ROOT / "data/eval_qa_v3.jsonl").resolve()
QUARANTINED_SOURCE_SHA256 = (
    "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b"
)
QUARANTINED_CONTRACT_CONTENT_SHA256 = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
QUARANTINED_V1 = True


def _quarantined(*_args, **_kwargs):
    raise RuntimeError(
        "recipe evaluation contract V1 is permanently quarantined; "
        "no rebuild, validation, compute authorization, or protected read is allowed"
    )


build_contract = _quarantined
validate_contract = _quarantined
assert_adaptation_inputs = _quarantined
charge_compute_attempt = _quarantined
aggregate_compute_ledger = _quarantined
validate_compute_match = _quarantined
claim_protected_access_once = _quarantined
load_claimed_protected_rows = _quarantined
claim_and_load_protected_rows = _quarantined
validate_terminal_aggregate_receipt = _quarantined
audit_role_records = _quarantined
canonical_sha256 = _quarantined
file_sha256 = _quarantined
_read_json = _quarantined
_read_jsonl = _quarantined


def main(_argv: list[str] | None = None) -> int:
    _quarantined()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
