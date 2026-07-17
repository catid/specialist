#!/usr/bin/env python3
"""V73G launcher shim binding V73E analysis to fresh successor artifacts."""

from __future__ import annotations

import sys

import qwen36_v73g_exact_phase_profiler_contract as contract


SUCCESSOR_LAUNCHER_AUTHORITY = {
    "checkpoint_snapshot_or_promotion_performed": False,
    "checkpoint_snapshot_or_promotion_authorized": False,
}

sys.modules["qwen36_v73e_exact_phase_profiler_contract"] = contract

import run_qwen36_v73e_exact_phase_profiler as _implementation


def main(argv=None) -> int:
    return _implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
