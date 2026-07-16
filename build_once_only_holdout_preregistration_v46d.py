#!/usr/bin/env python3
"""Seal launch-enabled V42I holdout after V45F/V46C resolution."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_once_only_holdout_eval_v46d as runtime


OUTPUT = runtime.PREREG


def build() -> dict:
    prior = runtime.prior_preregistration_v46d()
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "once-only-fixed-sft-v42i-holdout-preregistration-v46d",
        "status": (
            "preregistered_after_v45f_v46c_before_once_only_holdout_access"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "once-only aggregate evaluation of the V45F/V46C-resolved exact "
            "V42I SFT control against three exact base duplicates"
        ),
        "resolution_evidence": runtime.resolution_evidence_v46d(),
        "implementation_bindings": runtime.nonprotected_bindings_v46d(),
        "launch_interlock": {
            "resolved": True,
            "resolution": "V45F and V46C both strict-selected SFT V42I",
            "real_launch_permitted": True,
            "candidate_identity_replaceable_after_seal": False,
        },
        "extends_prepared_preregistration": {
            "path": str(runtime.V46A_PREREG),
            "file_sha256": runtime.V46A_PREREG_FILE_SHA256,
            "content_sha256": runtime.V46A_PREREG_CONTENT_SHA256,
            "only_prior_interlock_resolution_changed": True,
            "holdout_commitment_reused_without_opening_or_hashing": True,
        },
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "report": str(runtime.REPORT),
        },
        "candidate_selection_permitted": False,
        "post_result_tuning_or_selection_permitted": False,
        "holdout_access_authorized_once": True,
        "holdout_access_count_before_preregistration": 0,
        "protected_semantics_accessed_while_building": False,
        "holdout_opened_or_hashed_while_building": False,
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "real_launch_currently_interlocked": False,
        "real_launch_permitted_after_v45f_v46c_resolution": True,
        "all_four_gpus_receive_one_identical_holdout_batch": True,
        "all_four_gpus_must_be_resident_and_positive": True,
        "zero_access_dry_run_required": True,
        "fresh_once_only_runtime_revision": "v46d_v45f_v46c_resolved",
    })
    value["pre_result_commitments"] = dict(value["pre_result_commitments"])
    value["pre_result_commitments"].update({
        "candidate_resolution_reports": ["V45F", "V46C"],
        "candidate_identity_is_irreplaceable_after_this_seal": True,
        "no_post_holdout_candidate_selection": True,
    })
    value["post_result_policy"] = dict(value["post_result_policy"])
    value["post_result_policy"].update({
        "no_post_holdout_tuning_or_selection_without_exception": True,
        "result_cannot_authorize_holdout_reopen": True,
    })
    value["content_sha256_before_self_field"] = (
        runtime.prior.core.canonical_sha256(value)
    )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    runtime.prior.core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.prior.core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "fixed_candidate": "sft_v42i",
        "resolution_evidence": "V45F+V46C",
        "real_launch_permitted": True,
        "holdout_opened_or_hashed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
