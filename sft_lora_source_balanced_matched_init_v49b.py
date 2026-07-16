#!/usr/bin/env python3
"""V47A/V42I matched SFT with only V49A row weights substituted."""

from __future__ import annotations

import sft_lora_equal_unit_matched_init_v42a as v42a
import sft_lora_equal_unit_matched_init_v47a as v47a
import sft_source_balanced_weighting_v49b as weighting


def main(argv: list[str] | None = None) -> None:
    original = v42a.assign_equal_unit_weights
    v42a.assign_equal_unit_weights = (
        weighting.assign_source_balanced_weights_v49b
    )
    try:
        v47a.main(argv)
    finally:
        v42a.assign_equal_unit_weights = original


if __name__ == "__main__":
    main()
