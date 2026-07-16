#!/usr/bin/env python3
"""V47A matched-init SFT with an explicit 48-optimizer-step cap.

This keeps the V42B exact canonical adapter loader and the V42A training
implementation intact.  The only runtime extension is a fail-closed
``max_steps=48`` binding so the refreshed-data control consumes exactly the
same optimizer-step budget as V42I even if future Trainer scheduling behavior
changes.
"""

from __future__ import annotations

import argparse
import json

import sft_lora_equal_unit_matched_init_v42a as v42a
import sft_lora_equal_unit_matched_init_v42b as v42b


EXPECTED_MAX_STEPS_V47A = 48
BASE_PARSER_V47A = v42a.parser


def parser() -> argparse.ArgumentParser:
    result = BASE_PARSER_V47A()
    result.add_argument("--max-steps", required=True, type=int)
    return result


def schedule_audit_v47a(max_steps: int) -> dict:
    if max_steps != EXPECTED_MAX_STEPS_V47A:
        raise ValueError("V47A explicit optimizer-step cap changed")
    return {
        "schema": "specialist-explicit-sft-step-cap-v47a",
        "max_steps": EXPECTED_MAX_STEPS_V47A,
        "terminal_authority": "transformers.TrainingArguments.max_steps",
        "overrides_num_train_epochs_if_schedule_drift_occurs": True,
        "matched_v42i_optimizer_steps": True,
    }


def main(argv: list[str] | None = None) -> None:
    effective = parser().parse_args(argv)
    audit = schedule_audit_v47a(effective.max_steps)
    original_parser = v42a.parser
    original_training_arguments = v42a.TrainingArguments
    original_loader = v42a.PeftModel

    def exact_training_arguments_v47a(*args, **kwargs):
        if "max_steps" in kwargs:
            raise ValueError("V47A refuses a second max_steps source")
        kwargs["max_steps"] = effective.max_steps
        return original_training_arguments(*args, **kwargs)

    v42a.parser = parser
    v42a.TrainingArguments = exact_training_arguments_v47a
    v42a.PeftModel = v42b.ExactCanonicalAdapterLoaderV42B
    try:
        print(json.dumps({"schedule_audit_v47a": audit}, sort_keys=True), flush=True)
        v42a.main(argv)
    finally:
        v42a.PeftModel = original_loader
        v42a.TrainingArguments = original_training_arguments
        v42a.parser = original_parser


if __name__ == "__main__":
    main()
