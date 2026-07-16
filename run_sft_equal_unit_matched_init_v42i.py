#!/usr/bin/env python3
"""Fail-closed 5.5e-5 HPO wrapper around the verified V42B runtime."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import run_sft_equal_unit_matched_init_v42a as v42a
import run_sft_equal_unit_matched_init_v42b as v42b
import run_sft_train_only_control_v36a as engine


LEARNING_RATE = 5.5e-5


def parser() -> argparse.ArgumentParser:
    return v42b.parser()


def build_train_command(args: argparse.Namespace) -> list[str]:
    if args.learning_rate != LEARNING_RATE:
        raise ValueError("V42I learning-rate arm changed")
    reference = copy.copy(args)
    reference.learning_rate = 1e-4
    command = v42b.build_train_command(reference)
    command[command.index("--learning-rate") + 1] = str(LEARNING_RATE)
    return command


def validate_preregistration(args, command) -> dict:
    audit = v42b.validate_preregistration(args, command)
    value = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    wrapper = value.get("implementation", {}).get("hpo_launcher_v42i")
    expected = {
        "path": str(Path(__file__).resolve()),
        "sha256": engine.file_sha256(Path(__file__).resolve()),
    }
    if wrapper != expected:
        raise RuntimeError("V42I HPO launcher binding changed")
    if (
        value.get("recipe", {}).get("learning_rate") != LEARNING_RATE
        or value.get("recipe", {}).get("only_change_from_v42b")
        != "peak cosine-schedule learning rate 1e-4 -> 5.5e-5"
    ):
        raise RuntimeError("V42I HPO preregistration recipe changed")
    audit["hpo_launcher_v42i"] = wrapper
    return audit


def rewrite_evidence(args) -> None:
    v42b.rewrite_evidence(args)
    report_path = Path(args.report).resolve()
    attempt_path = Path(args.attempt_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42i",
        "status": "complete_train_only_lr5p5e5_state_sealed_shadow_unopened",
    })
    report["recipe"].update({
        "learning_rate": LEARNING_RATE,
        "hpo_parent": "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load",
        "only_change_from_v42b": "peak cosine-schedule learning rate 1e-4 -> 5.5e-5",
    })
    report["implementation"]["hpo_launcher_v42i"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": engine.file_sha256(Path(__file__).resolve()),
    }
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-matched-init-equal-unit-attempt-v42i",
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    engine.SFT_SCRIPT = v42b.SFT_SCRIPT
    engine.EXPECTED_ENCODING_AUDIT = v42a.EXPECTED_ENCODING_AUDIT_V42A
    engine.parser = parser
    engine.build_train_command = build_train_command
    engine.validate_preregistration = validate_preregistration
    engine.validate_output_artifacts = v42a.validate_output_artifacts
    result = engine.main(argv)
    effective = parser().parse_args(argv) if argv is not None else parser().parse_args()
    if result == 0 and not effective.dry_run:
        rewrite_evidence(effective)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
