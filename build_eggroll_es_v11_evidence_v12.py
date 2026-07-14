#!/usr/bin/env python3
"""Deep-validate completed V11g and mint compact immutable V12 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import run_eggroll_es_anchor_equivalence_v11g as equivalence_v11g
import run_eggroll_es_anchor_line_search as driver_v1
import train_eggroll_es_specialist_anchor_v12 as anchor_v12


ROOT = Path(__file__).resolve().parent
V11G_RUN_V12 = (
    equivalence_v11g.RUNS / equivalence_v11g.EXPERIMENT_NAME_V11G
).resolve()
V11G_ATTEMPT_V12 = equivalence_v11g._attempt_path().resolve()
V11G_JOURNAL_V12 = (
    V11G_RUN_V12 / equivalence_v11g.driver_v11c.driver_v1.JOURNAL_NAME
).resolve()
EVIDENCE_OUTPUT_V12 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11_EVIDENCE_V12.json"
).resolve()
ATTEMPT_KEYS_V12 = {
    "schema", "status", "phase", "experiment_name", "run_directory",
    "source_provenance", "v11c_implementation", "v11f_failure_evidence",
    "effective_cli", "seed_forwarding_audit", "policy_forwarding_audit",
    "intended_recipe_or_data_changed_from_v11f",
    "effective_completion_policy_corrected", "target_alpha_zero_only",
    "model_update_applied", "prior_v11f_baseline_validation_and_ood_scored",
    "v11g_baseline_validation_and_ood_scored",
    "sealed_evaluation_data_opened_or_scored",
    "run_directory_exists_after_attempt", "v11c_journal_exists_after_attempt",
    "journal_binding", "content_sha256_before_self_field",
}


def file_sha256(path):
    return driver_v1.file_sha256(path)


def _assert_safe_path(path, label):
    path = Path(path).resolve()
    equivalence_v11g.driver_v11c.driver_v11b.driver_v11.driver_v8.offline_audit._assert_no_heldout(
        str(path), label,
    )
    return path


def _canonical_without_self(value):
    return driver_v1.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def _validate_source_v12(source):
    expected_relative = "run_eggroll_es_anchor_equivalence_v11g.py"
    if (
        not isinstance(source, dict)
        or set(source) != {"schema", "git_head", "relative_path", "file_sha256"}
        or source.get("schema") != "eggroll-es-v11g-source"
        or source.get("relative_path") != expected_relative
        or not isinstance(source.get("git_head"), str)
        or len(source["git_head"]) != 40
        or not isinstance(source.get("file_sha256"), str)
        or len(source["file_sha256"]) != 64
    ):
        raise RuntimeError("v12 V11g committed source provenance changed")
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{source['git_head']}:{expected_relative}"],
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v12 V11g source commit is unavailable") from error
    if (
        hashlib.sha256(committed).hexdigest() != source["file_sha256"]
        or file_sha256(ROOT / expected_relative) != source["file_sha256"]
    ):
        raise RuntimeError("v12 V11g source differs from its launch commit")
    return copy.deepcopy(source)


def validate_v11g_inputs_v12(attempt_path=V11G_ATTEMPT_V12):
    """Validate the durable launch and perform the one deep ancestry replay."""
    attempt_path = _assert_safe_path(attempt_path, "v12 V11g launch attempt")
    if attempt_path != V11G_ATTEMPT_V12:
        raise RuntimeError("v12 requires the canonical V11g launch attempt path")
    attempt = json.loads(attempt_path.read_text())
    if (
        not isinstance(attempt, dict)
        or set(attempt) != ATTEMPT_KEYS_V12
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v11g"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_v11c_driver_main"
        or attempt.get("experiment_name") != equivalence_v11g.EXPERIMENT_NAME_V11G
        or Path(attempt.get("run_directory", "")).resolve() != V11G_RUN_V12
        or attempt.get("intended_recipe_or_data_changed_from_v11f") is not False
        or attempt.get("effective_completion_policy_corrected") is not True
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("prior_v11f_baseline_validation_and_ood_scored") is not True
        or attempt.get("v11g_baseline_validation_and_ood_scored") is not True
        or attempt.get("sealed_evaluation_data_opened_or_scored") is not False
        or attempt.get("run_directory_exists_after_attempt") is not True
        or attempt.get("v11c_journal_exists_after_attempt") is not True
        or attempt.get("content_sha256_before_self_field")
        != _canonical_without_self(attempt)
    ):
        raise RuntimeError("v12 requires a canonical completed V11g launch attempt")

    source = _validate_source_v12(attempt.get("source_provenance"))
    expected_failure = equivalence_v11g.bind_v11f_failure_v11g()
    expected_implementation = (
        equivalence_v11g.driver_v11e.audit_v11c_implementation_v11e()
    )
    expected_cli = equivalence_v11g.audit_effective_cli_v11g(
        equivalence_v11g.FROZEN_REAL_ARGV_V11G
    )
    expected_seed = equivalence_v11g.driver_v11f.seed_forwarding_audit_v11f(
        equivalence_v11g.driver_v11f.INHERITED_SEEDS_V11F
    )
    expected_policy = equivalence_v11g.policy_forwarding_audit_v11g(
        equivalence_v11g.BASE_POLICY_V11G
    )
    if (
        attempt.get("v11f_failure_evidence") != expected_failure
        or attempt.get("v11c_implementation") != expected_implementation
        or attempt.get("effective_cli") != expected_cli
        or attempt.get("seed_forwarding_audit") != expected_seed
        or attempt.get("policy_forwarding_audit") != expected_policy
    ):
        raise RuntimeError("v12 V11g launch ancestry changed")

    journal_path = _assert_safe_path(V11G_JOURNAL_V12, "v12 V11g journal")
    journal = json.loads(journal_path.read_text())
    prior = equivalence_v11g._patch_v11c_globals()
    try:
        audit = equivalence_v11g.driver_v11c.validate_completed_journal_v11c(
            journal
        )
    finally:
        equivalence_v11g._restore_v11c_globals(prior)
    journal_binding = {
        "schema": "eggroll-es-v11g-journal-binding",
        "path": str(journal_path),
        "file_sha256": file_sha256(journal_path),
        "content_sha256": audit["content_sha256"],
        "journal_schema": journal.get("schema"),
        "seed_sha256": driver_v1.canonical_sha256(journal.get("seeds")),
        "policy_sha256": driver_v1.canonical_sha256(journal.get("policy")),
    }
    if (
        attempt.get("journal_binding") != journal_binding
        or journal_binding.get("path") != str(journal_path)
        or journal_binding.get("file_sha256") != file_sha256(journal_path)
        or journal_binding.get("content_sha256")
        != journal.get("content_sha256_before_self_field")
        or journal_binding.get("content_sha256") != audit.get("content_sha256")
        or audit.get("equivalence", {}).get("all_exact") is not True
    ):
        raise RuntimeError("v12 V11g journal binding or exact equivalence changed")
    return {
        "attempt_path": attempt_path,
        "attempt": attempt,
        "source": source,
        "journal_path": journal_path,
        "journal": journal,
        "journal_binding": journal_binding,
        "audit": audit,
    }


def build_evidence(attempt_path=V11G_ATTEMPT_V12):
    validated = validate_v11g_inputs_v12(attempt_path)
    journal = validated["journal"]
    audit = validated["audit"]
    cross = journal.get("coefficient_plan", {}).get("resident_sign_cross_v11")
    consensus = anchor_v12.consensus_from_resident_cross_v12(cross)
    evidence = {
        "schema": "eggroll-es-v11g-compact-equivalence-evidence-v12",
        "passed": True,
        "validation": "one_deep_V9_V10_V11c_V11g_ancestry_replay_at_mint_time",
        "downstream_validation": "exact_artifact_file_and_content_hashes",
        "v11g_attempt": {
            "path": str(validated["attempt_path"]),
            "file_sha256": file_sha256(validated["attempt_path"]),
            "content_sha256": validated["attempt"][
                "content_sha256_before_self_field"
            ],
            "source_provenance": validated["source"],
            "v11f_failure_binding_sha256": validated["attempt"][
                "v11f_failure_evidence"
            ]["binding_sha256"],
        },
        "v11g_journal": {
            "path": str(validated["journal_path"]),
            "file_sha256": validated["journal_binding"]["file_sha256"],
            "content_sha256": validated["journal_binding"]["content_sha256"],
            "equivalence_binding_sha256": audit["equivalence"]["binding_sha256"],
            "resident_artifact_content_sha256": audit["resident"][
                "content_sha256"
            ],
        },
        "consensus": consensus,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": file_sha256(Path(__file__).resolve()),
            "v11g_driver_path": str(Path(equivalence_v11g.__file__).resolve()),
            "v11g_driver_file_sha256": file_sha256(equivalence_v11g.__file__),
        },
        "contains_validation_ood_or_heldout_content": False,
        "selection_surface": "V11g_train_and_anchor_responses_only",
    }
    evidence["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        evidence
    )
    return evidence


def _exclusive_write_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v12 compact evidence output already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v11g-attempt", default=str(V11G_ATTEMPT_V12))
    parser.add_argument("--output-json", default=str(EVIDENCE_OUTPUT_V12))
    args = parser.parse_args(argv)
    output = _assert_safe_path(args.output_json, "v12 compact evidence output")
    if output != EVIDENCE_OUTPUT_V12:
        raise ValueError("v12 compact evidence must use its canonical output path")
    evidence = build_evidence(args.v11g_attempt)
    _exclusive_write_json(output, evidence)
    print(json.dumps({
        "output": str(output),
        "file_sha256": file_sha256(output),
        "content_sha256": evidence["content_sha256_before_self_field"],
        "consensus_coefficient_sha256": evidence["consensus"][
            "coefficient_sha256"
        ],
    }, sort_keys=True))
    return evidence


if __name__ == "__main__":
    main()
