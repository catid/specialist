#!/usr/bin/env python3
"""Resident alpha search using the fail-closed anchored v2 trainer.

The frozen v1 driver owns dataset allowlists, no-resume journaling, and strict
OOD evaluation gates.  This entry point substitutes only the corrected v2
trainer/worker and records those new identities in the snapshot and journal.
"""

from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import train_eggroll_es_specialist_anchor_v2 as anchor_v2


_V1_BUILD_SNAPSHOT = driver_v1.build_snapshot
_V1_EXECUTE_LINE_SEARCH = driver_v1.execute_line_search


def validate_effective_anchor_api(module=anchor_v2):
    """Fail before model launch if the v2 adapter cannot satisfy v1 calls."""
    required = ("coefficient_sha256", "load_anchor_prose", "load_trainer")
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "corrected anchor adapter is missing callable API members: "
            + ", ".join(missing)
        )
    return required


def build_snapshot(*args, **kwargs):
    snapshot = _V1_BUILD_SNAPSHOT(*args, **kwargs)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v2"
    snapshot["implementation"]["corrected_driver"] = (
        anchor_v2.file_sha256(Path(__file__).resolve())
    )
    snapshot["implementation"]["exact_worker"] = anchor_v2.file_sha256(
        anchor_v2.ROOT / "eggroll_es_worker_v2.py"
    )
    return snapshot


def execute_line_search(*args, **kwargs):
    journal = _V1_EXECUTE_LINE_SEARCH(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    audit = trainer._latest_anchor_plan.get("identity_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise RuntimeError(
            "v2 line search completed without a passed identity audit"
        )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v2"
    journal["coefficient_plan"]["identity_audit"] = audit
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main():
    validate_effective_anchor_api()
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    driver_v1.anchor = anchor_v2
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    try:
        driver_v1.main()
    finally:
        driver_v1.anchor = old_anchor
        driver_v1.build_snapshot = old_build
        driver_v1.execute_line_search = old_execute


if __name__ == "__main__":
    main()
