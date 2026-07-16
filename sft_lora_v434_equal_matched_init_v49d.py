#!/usr/bin/env python3
"""V47A matched schedule with the fresh v434 V49D equal-control weights."""

from __future__ import annotations

import sft_lora_equal_unit_matched_init_v42a as v42a
import sft_lora_equal_unit_matched_init_v47a as v47a
import sft_v434_sampling_midpoint_weighting_v49d as weighting


def main(argv: list[str] | None = None) -> None:
    original = v42a.assign_equal_unit_weights
    v42a.assign_equal_unit_weights = weighting.assign_equal_weights_v49d
    try:
        v47a.main(argv)
    finally:
        v42a.assign_equal_unit_weights = original


if __name__ == "__main__":
    main()
