#!/usr/bin/env python3
"""V73G target shim binding the sealed V73E runtime to the repaired worker."""

from __future__ import annotations

import sys

import qwen36_v73g_exact_phase_profiler_contract as contract


SUCCESSOR_TARGET_AUTHORITY = {
    "semantic_quality_selection_or_hpo_performed": False,
    "protected_dev_ood_or_holdout_opened": False,
}

# The V73E implementation imports its contract by the historical module name.
# This fresh process aliases that name before importing the implementation, so
# every inherited validation resolves the prospectively sealed V73G surface.
sys.modules["qwen36_v73e_exact_phase_profiler_contract"] = contract

import run_lora_es_v71_v72_profile_calibration_v73e as _implementation


def main(argv=None) -> int:
    return _implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
